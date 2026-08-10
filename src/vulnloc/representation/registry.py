"""Graph-builder registry (``docs/Architecture.md`` §9, extension point 1).

Phase 0 note:
    Empty by design. Populated in Phase 2 with ``ast``, ``cfg``, ``dfg`` and ``cpg``
    builders, once ADR-002 fixes the representation (Q3) and the CPG tooling (Q6).
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["GRAPH_REGISTRY", "register_graph"]

GRAPH_REGISTRY: Registry[Any] = Registry("graph_builder")

register_graph = GRAPH_REGISTRY.register
