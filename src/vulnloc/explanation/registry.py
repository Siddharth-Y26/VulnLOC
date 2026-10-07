"""Explainer registry (``docs/Architecture.md`` §9, extension point 5).

Holds ``template`` — the deterministic template explainer of Phase 7a
(``explanation/generator.py``). The constrained generative explainer (7b) is deferred in
writing (``docs/Phases.md``) and has no key.
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["EXPLAINER_REGISTRY", "register_explainer"]

EXPLAINER_REGISTRY: Registry[Any] = Registry("explainer")

register_explainer = EXPLAINER_REGISTRY.register
