"""Metric registry (``docs/Architecture.md`` §9, extension point 6).

Populated in Phase 8 by ``vulnloc.evaluation.metrics``. An entry is a **per-unit ranking
metric**: a function of one unit's labels in rank order — ``[n] bool``, most suspicious
line first, at least one ``True`` — returning a float. The evaluator computes every entry
for every unit under each tie order and reports the mean over units, so adding a metric is
one function and one decorator.

Metrics of a pooled set of lines (average precision, precision / recall / F1) have a
different shape and are plain functions in ``metrics/localization.py``.
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["METRIC_REGISTRY", "register_metric"]

METRIC_REGISTRY: Registry[Any] = Registry("metric")

register_metric = METRIC_REGISTRY.register
