"""Typed exception hierarchy (``docs/Architecture.md`` §8, ``docs/Rules.md`` §4).

Every failure raised by VulnLoc inherits :class:`VulnLocError`, so a caller can catch
the project's failures without catching everything. Bare ``except Exception:`` and
``except: pass`` are banned by ``Rules.md`` §4.2-4.3.

Each exception carries optional structured context (``unit_id`` and arbitrary details)
because ``Rules.md`` §4.7 requires API errors to serialize as::

    {"error": {"type": "ParseError", "message": "...", "unit_id": "..."}}
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlignmentError",
    "ConfigError",
    "DataError",
    "ExplanationError",
    "GraphBuildError",
    "ModelError",
    "ParseError",
    "RegistryError",
    "VulnLocError",
]


class VulnLocError(Exception):
    """Base class for every error raised by VulnLoc.

    Args:
        message: Human-readable description of what failed.
        unit_id: Identifier of the :class:`~vulnloc.representation.CodeUnit` being
            processed, when the failure is attributable to one sample.
        **details: Extra structured context to attach to the error payload.
    """

    def __init__(self, message: str, *, unit_id: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.unit_id = unit_id
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the error payload shape required by ``Rules.md`` §4.7.

        Returns:
            A mapping with keys ``type``, ``message``, ``unit_id`` and ``details``.
        """
        return {
            "type": type(self).__name__,
            "message": self.message,
            "unit_id": self.unit_id,
            "details": self.details,
        }

    def __str__(self) -> str:
        parts = [self.message]
        if self.unit_id is not None:
            parts.append(f"unit_id={self.unit_id!r}")
        parts.extend(f"{key}={value!r}" for key, value in sorted(self.details.items()))
        return " | ".join(parts)


class ConfigError(VulnLocError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataError(VulnLocError):
    """Corpus download, loading, labelling, or splitting failed (L0)."""


class ParseError(VulnLocError):
    """Source code could not be parsed into a syntax tree (L1)."""


class GraphBuildError(VulnLocError):
    """An AST/CFG/DFG/CPG could not be constructed from a parsed unit (L1)."""


class AlignmentError(VulnLocError):
    """A token-to-line or node-to-line mapping is missing or invalid.

    Raised whenever the line-alignment invariants of ``Rules.md`` §4.5 are violated.
    This is never recoverable: the line index is the atomic unit of this project
    (``Architecture.md`` §1.1).
    """


class ModelError(VulnLocError):
    """A model failed to build, load, or run (L2/L3)."""


class ExplanationError(VulnLocError):
    """Evidence extraction, explanation generation, or grounding verification failed (L5)."""


class RegistryError(ConfigError):
    """A registry lookup or registration failed (unknown or duplicate key)."""
