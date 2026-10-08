"""L4 — Evaluation: metrics, statistics over seeds, the evaluator.

Never mutates models (``docs/Architecture.md`` §3). Everything here is a function of
stored line scores (``training/line_scores.py``), so no model is loaded to evaluate one.
"""

from vulnloc.evaluation import metrics
from vulnloc.evaluation.registry import METRIC_REGISTRY, register_metric

__all__ = ["METRIC_REGISTRY", "metrics", "register_metric"]
