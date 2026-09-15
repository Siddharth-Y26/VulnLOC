"""The comparator arms of A1 (``Phases.md`` Phase 5, ``Architecture.md`` §4.4).

These exist so that "cross-attention works" is a falsifiable claim rather than a
description. Each is the smallest sensible way to put two line-aligned branches together,
and each shares the base class's projections, masking, output width and line axis with
:class:`~vulnloc.models.fusion.cross_attention.CrossAttentionFusion` — so the only thing
that varies across the arms is the combination rule.

``concat``
    The headline comparator, and the one A1 is stated against. Identical to the
    cross-attention module with the attention blocks removed: same output projection, same
    layer norm, same residual. The parameter difference between the two is exactly the
    multi-head attention.
``sum``
    The cheapest thing that could work. No parameters at all beyond the shared
    projections, which makes it the honest floor for "is the fusion mechanism doing
    anything?".
``gated``
    A learned per-line, per-dimension mixture. It can express "trust the graph on this
    line and the tokens on that one" without any line attending to any *other* line —
    which is precisely the capability cross-attention adds on top. If ``gated`` matches
    ``cross_attention``, the novelty is the mixing and not the cross-line reasoning, and
    that is a result worth being able to state.
``llm_only`` / ``gnn_only``
    ``Phases.md`` lists one comparator called ``none``; ``Architecture.md`` §4.4 splits it
    into these two, which is the more useful form — "no fusion" means something different
    depending on which branch survives. They are **not** re-implementations of Phases 3
    and 4: those baselines have their own models and their own logged numbers. These run
    the single branch through the fused scaffold, which is how you tell a fusion result
    apart from a scaffold result.

One asymmetry is deliberate and is enforced rather than commented. The single-branch arms
narrow the fused line mask to their own branch's coverage. Leaving it as the union would
let ``llm_only`` be scored on lines only the graph reached — inheriting the other branch's
coverage while claiming to be an ablation of it.
"""

from __future__ import annotations

import torch

from vulnloc.models.fusion.base import FusionModule, FusionOutput
from vulnloc.models.registry import register_fusion
from vulnloc.utils.exceptions import ModelError

__all__ = ["ConcatFusion", "GatedFusion", "GnnOnlyFusion", "LlmOnlyFusion", "SumFusion"]


@register_fusion("concat")
class ConcatFusion(FusionModule):
    """Concatenate the two branches per line and project back down.

    A1's control arm. ``LayerNorm(Linear([sem ; str])) + residual``, which is
    :class:`~vulnloc.models.fusion.cross_attention.CrossAttentionFusion` with the
    attention blocks deleted and nothing else changed.

    Args:
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        dropout: Dropout before the output projection.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__(sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.out_proj = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_norm = torch.nn.LayerNorm(hidden_dim)

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Concatenate along the feature axis and project.

        Args:
            sem: ``[B, L, D]`` projected semantic features.
            str_: ``[B, L, D]`` projected structural features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Ignored — this strategy has no attention.

        Returns:
            A :class:`~vulnloc.models.fusion.base.FusionOutput`, ``fused`` ``[B, L, D]``.
        """
        del return_attention
        pair = self.dropout(torch.cat([sem, str_], dim=-1))
        fused = self.out_norm(self.out_proj(pair)) + 0.5 * (sem + str_)
        line_mask = self._line_mask(sem_mask, str_mask)
        return FusionOutput(fused=fused * line_mask.unsqueeze(-1), line_mask=line_mask)


@register_fusion("sum")
class SumFusion(FusionModule):
    """Add the two branches per line.

    The parameter-free floor. If this matches the arms above it, what those arms added was
    capacity rather than mechanism.

    Args:
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        dropout: Dropout on the sum.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__(sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.out_norm = torch.nn.LayerNorm(hidden_dim)

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Sum and normalise.

        Args:
            sem: ``[B, L, D]`` projected semantic features.
            str_: ``[B, L, D]`` projected structural features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Ignored — this strategy has no attention.

        Returns:
            A :class:`~vulnloc.models.fusion.base.FusionOutput`, ``fused`` ``[B, L, D]``.
        """
        del return_attention
        fused = self.out_norm(self.dropout(sem + str_))
        line_mask = self._line_mask(sem_mask, str_mask)
        return FusionOutput(fused=fused * line_mask.unsqueeze(-1), line_mask=line_mask)


@register_fusion("gated")
class GatedFusion(FusionModule):
    """Mix the two branches with a learned per-line, per-dimension gate.

    ``g = sigmoid(Linear([sem ; str]))``, then ``LayerNorm(g * sem + (1 - g) * str)``. The
    gate is computed from both branches, so a line can learn to trust the graph where the
    tokens are uninformative and the reverse — but only from *its own* line's features,
    which is the capability cross-attention extends across lines.

    Args:
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        dropout: Dropout on the mixture.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__(sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gate = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_norm = torch.nn.LayerNorm(hidden_dim)

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Gate, mix and normalise.

        Args:
            sem: ``[B, L, D]`` projected semantic features.
            str_: ``[B, L, D]`` projected structural features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Ignored — this strategy has no attention.

        Returns:
            A :class:`~vulnloc.models.fusion.base.FusionOutput`, ``fused`` ``[B, L, D]``.
        """
        del return_attention
        gate = torch.sigmoid(self.gate(torch.cat([sem, str_], dim=-1)))
        fused = self.out_norm(self.dropout(gate * sem + (1.0 - gate) * str_))
        line_mask = self._line_mask(sem_mask, str_mask)
        return FusionOutput(fused=fused * line_mask.unsqueeze(-1), line_mask=line_mask)


class _SingleBranchFusion(FusionModule):
    """Shared body of the two no-fusion ablations.

    Args:
        branch: ``"sem"`` or ``"str"`` — which branch survives.
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        dropout: Dropout on the surviving branch.

    Raises:
        ModelError: If ``branch`` is neither ``"sem"`` nor ``"str"``.
    """

    def __init__(
        self,
        branch: str,
        *,
        sem_dim: int,
        str_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout)
        if branch not in {"sem", "str"}:
            raise ModelError("branch must be 'sem' or 'str'", branch=branch)
        self.branch = branch
        self.out_norm = torch.nn.LayerNorm(hidden_dim)

    @property
    def uses_semantic(self) -> bool:
        """Only when the semantic branch is the surviving one."""
        return self.branch == "sem"

    @property
    def uses_structural(self) -> bool:
        """Only when the structural branch is the surviving one."""
        return self.branch == "str"

    def _line_mask(self, sem_mask: torch.Tensor, str_mask: torch.Tensor) -> torch.Tensor:
        """The surviving branch's coverage, never the union.

        Args:
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.

        Returns:
            ``[B, L]``.
        """
        return sem_mask if self.branch == "sem" else str_mask

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Pass the surviving branch through and drop the other.

        Args:
            sem: ``[B, L, D]`` projected semantic features.
            str_: ``[B, L, D]`` projected structural features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Ignored — this strategy has no attention.

        Returns:
            A :class:`~vulnloc.models.fusion.base.FusionOutput`, ``fused`` ``[B, L, D]``.
        """
        del return_attention
        kept = sem if self.branch == "sem" else str_
        fused = self.out_norm(self.dropout(kept))
        line_mask = self._line_mask(sem_mask, str_mask)
        return FusionOutput(fused=fused * line_mask.unsqueeze(-1), line_mask=line_mask)

    def extra_repr(self) -> str:
        """Show which branch survives in ``print(model)``."""
        return f"{super().extra_repr()}, branch={self.branch}"


@register_fusion("llm_only")
class LlmOnlyFusion(_SingleBranchFusion):
    """Semantic branch only, inside the fused scaffold.

    Not a replacement for Phase 3's :class:`~vulnloc.models.semantic.baseline.SemanticLineModel`
    — that baseline has its own three logged seeds. This is the arm that isolates what the
    fused *scaffold* contributes independently of fusion, which is the difference between
    "cross-attention helped" and "the extra projection and layer norm helped".

    Args:
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``. Accepted and unused, so the arms stay
            constructible from one config.
        hidden_dim: Common width ``D``.
        dropout: Dropout on the surviving branch.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__(
            "sem", sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout
        )


@register_fusion("gnn_only")
class GnnOnlyFusion(_SingleBranchFusion):
    """Structural branch only, inside the fused scaffold.

    The mirror of :class:`LlmOnlyFusion`, and subject to the same caveat: its line mask is
    the structural branch's coverage, which excludes the 7.84% of flagged lines that carry
    no graph node (``Memory.md`` §7 #17).

    Args:
        sem_dim: Semantic width ``D_sem``. Accepted and unused.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        dropout: Dropout on the surviving branch.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__(
            "str", sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout
        )
