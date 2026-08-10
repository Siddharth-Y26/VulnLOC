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

__all__ = ["config_hash", "git_sha", "provenance", "resolve_config", "save_config"]


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
