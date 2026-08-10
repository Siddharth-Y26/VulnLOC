"""Cross-cutting utilities: exceptions, logging, config provenance, registries.

This package sits below every layer in ``docs/Architecture.md`` §3 and may be imported
from anywhere. It must not import from any other ``vulnloc`` subpackage.
"""

from vulnloc.utils.config import config_hash, git_sha, provenance, resolve_config, save_config
from vulnloc.utils.exceptions import (
    AlignmentError,
    ConfigError,
    DataError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
    RegistryError,
    VulnLocError,
)
from vulnloc.utils.logging import bind_run_id, current_run_id, get_logger, setup_logging
from vulnloc.utils.registry import Registry

__all__ = [
    "AlignmentError",
    "ConfigError",
    "DataError",
    "ExplanationError",
    "GraphBuildError",
    "ModelError",
    "ParseError",
    "Registry",
    "RegistryError",
    "VulnLocError",
    "bind_run_id",
    "config_hash",
    "current_run_id",
    "get_logger",
    "git_sha",
    "provenance",
    "resolve_config",
    "save_config",
    "setup_logging",
]
