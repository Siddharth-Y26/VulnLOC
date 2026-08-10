"""L4 — Evaluation: metrics, ablation runners, statistical tests.

Never mutates models (``docs/Architecture.md`` §3). Populated in Phase 8.
"""

from vulnloc.evaluation.registry import METRIC_REGISTRY, register_metric

__all__ = ["METRIC_REGISTRY", "register_metric"]
