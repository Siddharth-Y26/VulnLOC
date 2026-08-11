"""L0 — Data: download, split, cache and line-label alignment of raw corpora.

Must not import from any other layer (``docs/Architecture.md`` §3): this layer knows
nothing about models. Blocked on ADR-001 (dataset choice, Q1) until Phase 1.
"""

from vulnloc.data.registry import DATASET_REGISTRY, register_dataset

# Import order matters: the registry must exist before the loaders that decorate
# themselves into it. Importing loaders here is what makes `DATASET_REGISTRY.get(...)`
# work from a config without the caller knowing which module defines what.
from vulnloc.data import loaders  # noqa: E402  isort: skip

__all__ = ["DATASET_REGISTRY", "loaders", "register_dataset"]
