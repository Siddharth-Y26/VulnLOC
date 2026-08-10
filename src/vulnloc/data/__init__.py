"""L0 — Data: download, split, cache and line-label alignment of raw corpora.

Must not import from any other layer (``docs/Architecture.md`` §3): this layer knows
nothing about models. Blocked on ADR-001 (dataset choice, Q1) until Phase 1.
"""

from vulnloc.data.registry import DATASET_REGISTRY, register_dataset

__all__ = ["DATASET_REGISTRY", "register_dataset"]
