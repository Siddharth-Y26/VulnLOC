"""L2 — Models: semantic and structural encoders, fusion strategies, heads, losses.

Pure computation: this layer performs no I/O and no logging
(``docs/Architecture.md`` §3). Populated from Phase 3 onward.
"""

from vulnloc.models.registry import (
    FUSION_REGISTRY,
    HEAD_REGISTRY,
    SEMANTIC_REGISTRY,
    STRUCTURAL_REGISTRY,
    register_fusion,
    register_head,
    register_semantic,
    register_structural,
)

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
