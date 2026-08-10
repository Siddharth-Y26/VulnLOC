"""Explainer registry (``docs/Architecture.md`` §9, extension point 5).

Phase 0 note:
    Empty by design. Populated in Phase 7 with the template explainer (7a) and the
    constrained generative explainer (7b).
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["EXPLAINER_REGISTRY", "register_explainer"]

EXPLAINER_REGISTRY: Registry[Any] = Registry("explainer")

register_explainer = EXPLAINER_REGISTRY.register
