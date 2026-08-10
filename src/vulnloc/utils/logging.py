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
    "ConsoleFormatter",
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


# Attributes the stdlib refuses to let `extra=` overwrite; passing one raises KeyError
# from inside logging itself. `name` and `module` are the easy ones to hit by accident.
_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
}


class _SafeExtraLogger(logging.Logger):
    """A logger that renames ``extra`` keys colliding with LogRecord attributes.

    The stdlib raises ``KeyError`` when ``extra`` contains a reserved name, which turns
    a logging call into a crash — the exact opposite of what logging is for. Structured
    context here is arbitrary domain data (``name``, ``module``, ``filename`` are all
    natural field names for a corpus audit), so collisions are inevitable. Renaming is
    the only behaviour that keeps the data *and* the process alive.
    """

    # No @typing.override: it is 3.12+, and the Docker image and CI both run 3.11
    # (ADR-000 §D0.3). Revisit if requires-python ever moves past 3.11.
    def makeRecord(  # noqa: N802 - name and signature are fixed by logging.Logger
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: Any,
        exc_info: Any,
        func: str | None = None,
        extra: Any = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        """Build a record, renaming reserved ``extra`` keys to ``extra_<key>``."""
        if extra:
            extra = {
                (f"extra_{key}" if key in _RESERVED_RECORD_KEYS else key): value
                for key, value in extra.items()
            }
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


def current_run_id() -> str:
    """Return the run id bound to the current context, or ``"-"`` if unset."""
    return _RUN_ID.get()


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    """Return only the fields supplied via ``logger.info(..., extra={...})``.

    Args:
        record: The record being emitted.

    Returns:
        The caller-supplied structured context, excluding stdlib record attributes.
    """
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_RECORD_KEYS and not key.startswith("_")
    }


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
        extras = _record_extras(record)
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable rendering that keeps the structured context visible.

    The stdlib default silently discards anything passed via ``extra=``, which made
    console output strictly less informative than the JSON file — a message like
    "Accelerator" is useless without the GPU name attached to it. Extras are appended
    as ``key=value`` pairs in insertion order, so the most relevant field comes first.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as ``[logger] message key=value ...``.

        Args:
            record: The record being emitted.

        Returns:
            The formatted line, with any ``extra=`` fields appended.
        """
        base = f"[{record.name}] {record.getMessage()}"
        extras = _record_extras(record)
        if not extras:
            return base
        rendered = " ".join(f"{key}={value}" for key, value in extras.items())
        return f"{base}  {rendered}"


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
    console.setFormatter(ConsoleFormatter())
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
    qualified = name if name == "vulnloc" or name.startswith("vulnloc.") else f"vulnloc.{name}"
    # Swap the logger class only around our own creation, so third-party loggers keep
    # the stdlib behaviour and we avoid a process-wide side effect at import time.
    previous = logging.getLoggerClass()
    logging.setLoggerClass(_SafeExtraLogger)
    try:
        return logging.getLogger(qualified)
    finally:
        logging.setLoggerClass(previous)
