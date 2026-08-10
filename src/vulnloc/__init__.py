"""VulnLoc: explainable hybrid LLM-GNN line-level vulnerability localization.

Layering (``docs/Architecture.md`` §3). Imports flow strictly downward; the rule is
enforced by ``import-linter`` (see ``[tool.importlinter]`` in ``pyproject.toml``)::

    serving (L6) -> explanation (L5) -> evaluation (L4) -> training (L3)
                 -> models (L2) -> representation (L1) -> data (L0) -> utils

``utils`` sits below every layer and may be imported by all of them.
"""

from vulnloc.utils.exceptions import (
    AlignmentError,
    ConfigError,
    DataError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
    VulnLocError,
)

__version__ = "0.1.0"

__all__ = [
    "AlignmentError",
    "ConfigError",
    "DataError",
    "ExplanationError",
    "GraphBuildError",
    "ModelError",
    "ParseError",
    "VulnLocError",
    "__version__",
]
