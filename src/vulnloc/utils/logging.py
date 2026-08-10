"""Logging setup (``docs/Architecture.md`` §8).

Two sinks, one call:

* **console** — human-readable, colourized via ``rich``;
* **file** — one JSON object per line, for machine post-processing.

Every record carries the active ``run_id`` so that lines from different runs can be
separated after the fact. The run id is stored in a :class:`contextvars.ContextVar`
rather than a module-level global (``Rules.md`` §2 bans global mutable state), which
also makes it correct under concurrency in the serving layer.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from vulnloc.utils.exceptions import ConfigError

__all__ = [
    "JsonLogFormatter",
    "RunIdFilter",
    "bind_run_id",
    "current_run_id",
    "get_logger",
    "setup_logging",
]

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("vulnloc_run_id", default="-")

# Attributes present on every LogRecord; anything else was supplied via `extra=`.
_STANDARD_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "run_id",
}


def current_run_id() -> str:
    """Return the run id bound to the current context, or ``"-"`` if unset."""
    return _RUN_ID.get()


@contextmanager
def bind_run_id(run_id: str) -> Iterator[str]:
    """Bind ``run_id`` to every log record emitted inside the block.

    Args:
        run_id: Identifier to attach, typically the MLflow run id.

    Yields:
        The bound run id.
    """
    token = _RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _RUN_ID.reset(token)


class RunIdFilter(logging.Filter):
    """Inject the context-local ``run_id`` onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``run_id`` to ``record`` and always keep the record.

        Args:
            record: The record being emitted.

        Returns:
            Always True — this filter annotates, it does not drop records.
        """
        record.run_id = current_run_id()
        return True


class JsonLogFormatter(logging.Formatter):
    """Render a record as a single-line JSON object.

    Keys passed through ``logger.info(..., extra={...})`` are merged into the payload,
    so structured context (``unit_id``, ``step``, ``n_dropped``) survives to the file.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON object on one line.

        Args:
            record: The record being emitted.

        Returns:
            A JSON string with ``ts``, ``level``, ``logger``, ``run_id``, ``message``
            and, when present, ``extra`` and ``exception``.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: str | int = "INFO",
    *,
    log_dir: str | os.PathLike[str] | None = None,
    json_to_file: bool = True,
    filename: str = "vulnloc.log",
) -> logging.Logger:
    """Configure the ``vulnloc`` logger. Idempotent: safe to call more than once.

    Args:
        level: Logging level name or numeric level for both sinks.
        log_dir: Directory for the JSON log file. Created if missing.
        json_to_file: Whether to attach the JSON file handler at all.
        filename: Basename of the JSON log file inside ``log_dir``.

    Returns:
        The configured ``vulnloc`` package logger.

    Raises:
        ConfigError: If ``level`` is not a valid logging level, or ``json_to_file``
            is requested without a ``log_dir``.
    """
    resolved = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(resolved, int):
        raise ConfigError(f"Unknown logging level: {level!r}.", level=level)

    logger = logging.getLogger("vulnloc")
    logger.setLevel(resolved)
    logger.propagate = False
    for handler in list(logger.handlers):  # idempotency: rebuild handlers, don't stack them
        logger.removeHandler(handler)
        handler.close()

    run_id_filter = RunIdFilter()

    console = RichHandler(rich_tracebacks=True, show_path=False, log_time_format="%H:%M:%S")
    console.setLevel(resolved)
    console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    console.addFilter(run_id_filter)
    logger.addHandler(console)

    if json_to_file:
        if log_dir is None:
            raise ConfigError("json_to_file=True requires a log_dir.")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(JsonLogFormatter())
        file_handler.addFilter(run_id_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``vulnloc`` logger.

    Args:
        name: Usually ``__name__``. A leading ``"vulnloc."`` is not duplicated.

    Returns:
        The namespaced logger.
    """
    if name == "vulnloc" or name.startswith("vulnloc."):
        return logging.getLogger(name)
    return logging.getLogger(f"vulnloc.{name}")
