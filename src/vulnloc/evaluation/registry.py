"""Metric registry (``docs/Architecture.md`` §9, extension point 6).

Phase 0 note:
    Empty by design. Populated in Phase 8 from the metric suite in
    ``docs/ResearchPlan.md`` §4.
"""

from __future__ import annotations

from typing import Any

from vulnloc.utils.registry import Registry

__all__ = ["METRIC_REGISTRY", "register_metric"]

METRIC_REGISTRY: Registry[Any] = Registry("metric")

register_metric = METRIC_REGISTRY.register
