"""Config resolution, hashing, and provenance (``docs/Architecture.md`` §8).

``Rules.md`` §5.8 requires every checkpoint to record ``git_sha`` and ``config_hash``,
and §5.1 requires one config to correspond to exactly one run. These helpers produce
both identifiers deterministically so a results table row can always be traced back to
the exact code and configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vulnloc.utils.exceptions import ConfigError

__all__ = [
    "config_hash",
    "git_sha",
    "provenance",
    "resolve_config",
    "run_id_text",
    "save_config",
    "strip_comments_enabled",
]


def strip_comments_enabled(cfg: DictConfig) -> bool:
    """Whether a config asks for comments to be stripped from source (``Memory.md`` §1.25).

    The key is ``representation.strip_comments`` and it is **absent from
    ``configs/config.yaml`` on purpose**: adding a key there changes every run's
    ``config_hash`` (``CLAUDE.md`` landmine #38). A config without it — every run
    archived before 2026-10-07 — did not strip, which is what an absent parameter
    means everywhere else in this repository (landmine #39). Turn it on with
    ``+representation.strip_comments=true``.

    Args:
        cfg: A composed config, or a run's archived one.

    Returns:
        ``True`` only when the key is present and true.
    """
    return bool(OmegaConf.select(cfg, "representation.strip_comments"))


def run_id_text(value: Any) -> str:
    """A run id as the caller typed it — or a refusal if the config parser ate it.

    A run id is hexadecimal, so a prefix can be all digits around a single ``e``:
    ``920989e5`` is one of ablation A6's runs, and Hydra reads it as the float
    ``92098900000.0``. The original text cannot be recovered from that, so the only honest
    thing to do is say what happened.

    Args:
        value: The option's value as the config holds it.

    Returns:
        The run id or prefix.

    Raises:
        ConfigError: If no run id was given, or it arrived as a number.
    """
    if value is None or value == "":
        raise ConfigError("no run id was given")
    if not isinstance(value, str):
        raise ConfigError(
            "the run id was parsed as a number, not as text",
            parsed=repr(value),
            hint="A prefix such as 920989e5 is a float to the config parser. Pass the full "
            "32-character run id, or more of it.",
        )
    return value


def resolve_config(cfg: DictConfig) -> dict[str, Any]:
    """Materialize a Hydra config into a plain, fully-interpolated dict.

    Args:
        cfg: The Hydra/OmegaConf config for a run.

    Returns:
        A nested ``dict`` with every ``${...}`` interpolation resolved.

    Raises:
        ConfigError: If an interpolation cannot be resolved (usually a missing
            override or a typo in a config group).
    """
    try:
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:  # OmegaConf raises several unrelated resolution errors
        raise ConfigError(f"Could not resolve config: {exc}") from exc
    if not isinstance(container, dict):
        raise ConfigError(f"Expected a mapping at the config root, got {type(container).__name__}.")
    # OmegaConf types its keys as `str | bytes | int | Enum | float | bool`; a Hydra
    # config root is always string-keyed.
    return {str(key): value for key, value in container.items()}


def config_hash(cfg: DictConfig | dict[str, Any], *, length: int = 12) -> str:
    """Compute a stable content hash of a resolved config.

    The config is serialized as JSON with sorted keys, so the hash depends on the
    config's *content* and not on Hydra's composition order.

    Args:
        cfg: A Hydra config or an already-resolved dict.
        length: Number of leading hex characters to return.

    Returns:
        The truncated SHA-256 hex digest.
    """
    container = resolve_config(cfg) if isinstance(cfg, DictConfig) else cfg
    serialized = json.dumps(container, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def git_sha(*, short: bool = True) -> str:
    """Return the current commit SHA, or ``"unknown"`` outside a git checkout.

    A dirty working tree is marked with a ``-dirty`` suffix so that a run logged from
    uncommitted code is never mistaken for a reproducible one.

    Args:
        short: Return the abbreviated SHA rather than the full 40 characters.

    Returns:
        The SHA (optionally suffixed with ``-dirty``), or ``"unknown"``.
    """
    rev = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        sha = subprocess.run(rev, capture_output=True, text=True, timeout=10).stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def provenance(cfg: DictConfig | dict[str, Any]) -> dict[str, str]:
    """Build the provenance tags attached to every run and checkpoint.

    Args:
        cfg: The run's config.

    Returns:
        A mapping with ``git_sha``, ``config_hash`` and ``vulnloc_version``.
    """
    from vulnloc import __version__

    return {
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "vulnloc_version": __version__,
    }


def save_config(cfg: DictConfig, path: str | Path) -> Path:
    """Write the fully-resolved config into a run directory (``Architecture.md`` §8).

    Args:
        cfg: The Hydra config for the run.
        path: Destination YAML path. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return destination
