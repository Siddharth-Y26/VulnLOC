"""Model-layer registries (``docs/Architecture.md`` §9, extension points 2, 3).

Adding a GNN, a semantic encoder, a fusion strategy, or a head must require a new file
plus a decorator plus a YAML line — never an edit to an existing module.

Phase 0 note:
    These registries are intentionally empty. They are populated by Phase 3 (semantic),
    Phase 4 (structural), Phase 5 (fusion) and Phase 6 (heads).
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = [
    "FUSION_REGISTRY",
    "HEAD_REGISTRY",
    "SEMANTIC_REGISTRY",
    "STRUCTURAL_REGISTRY",
    "register_fusion",
    "register_head",
    "register_semantic",
    "register_structural",
]

SEMANTIC_REGISTRY: Registry[Any] = Registry("semantic_encoder")
STRUCTURAL_REGISTRY: Registry[Any] = Registry("structural_encoder")
FUSION_REGISTRY: Registry[Any] = Registry("fusion")
HEAD_REGISTRY: Registry[Any] = Registry("head")

register_semantic = SEMANTIC_REGISTRY.register
register_structural = STRUCTURAL_REGISTRY.register
register_fusion = FUSION_REGISTRY.register
register_head = HEAD_REGISTRY.register
