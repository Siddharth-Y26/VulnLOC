"""Fusion strategies (cross-attention and its comparators) behind ``FusionModule`` (Phase 5).

Importing this package registers every strategy. A ``@register_fusion`` decorator only
runs when its module is imported, so without this a config naming ``concat`` would fail
with "unknown fusion" despite the strategy existing — the same registry-discovery trap
``data/loaders/__init__.py`` documents. Adding a strategy means adding one import here.

``fusion=cross_attention`` against ``fusion=concat`` being a flag flip and nothing else is
a Phase 5 exit criterion, and it is this registry that makes it one.
"""

import inspect
from typing import Any

from vulnloc.models.fusion.base import FusionModule, FusionOutput
from vulnloc.models.fusion.comparators import (
    ConcatFusion,
    GatedFusion,
    GnnOnlyFusion,
    LlmOnlyFusion,
    SumFusion,
)
from vulnloc.models.fusion.cross_attention import (
    CrossAttentionBlock,
    CrossAttentionFusion,
    cross_attend,
)
from vulnloc.models.registry import FUSION_REGISTRY
from vulnloc.utils.exceptions import ModelError

__all__ = [
    "ConcatFusion",
    "CrossAttentionBlock",
    "CrossAttentionFusion",
    "FusionModule",
    "FusionOutput",
    "GatedFusion",
    "GnnOnlyFusion",
    "LlmOnlyFusion",
    "SumFusion",
    "build_fusion",
    "cross_attend",
]


def build_fusion(kind: str, **kwargs: Any) -> FusionModule:
    """Build a fusion strategy through the registry.

    Keyword arguments are forwarded **unfiltered**, so a config key that the chosen
    strategy does not accept is a hard failure rather than a silent no-op. That is the
    point: ``fusion.heads`` belongs to ``cross_attention.yaml`` alone, and a ``heads``
    setting that reached ``concat`` and did nothing would be ``Memory.md`` §7 #22 — a knob
    that looks like it works — inside the comparison A1 rests on.

    Args:
        kind: A ``FUSION_REGISTRY`` key — ``cross_attention``, ``concat``, ``sum``,
            ``gated``, ``llm_only`` or ``gnn_only``.
        **kwargs: Forwarded to the strategy. Every strategy takes ``sem_dim``,
            ``str_dim``, ``hidden_dim`` and ``dropout``; ``cross_attention`` also takes
            ``heads`` and ``n_layers``.

    Returns:
        The strategy.

    Raises:
        RegistryError: If ``kind`` is unknown.
        ModelError: If the arguments do not fit the strategy, or are inconsistent.
    """
    factory = FUSION_REGISTRY.get(kind)
    try:
        strategy: FusionModule = factory(**kwargs)
    except TypeError as error:
        accepted = [
            name
            for name in inspect.signature(factory).parameters
            if name not in {"self", "args", "kwargs"}
        ]
        raise ModelError(
            "fusion strategy does not accept these settings",
            kind=kind,
            given=sorted(kwargs),
            accepted=accepted,
            reason=str(error),
        ) from error
    return strategy
