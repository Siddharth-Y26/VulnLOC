"""Dataset registry (``docs/Architecture.md`` §9, extension point 4).

Phase 0 note:
    Empty by design. Populated in Phase 1, once ADR-001 fixes the corpus (Q1).
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["DATASET_REGISTRY", "register_dataset"]

DATASET_REGISTRY: Registry[Any] = Registry("dataset")

register_dataset = DATASET_REGISTRY.register
