"""Bidirectional cross-attention over the shared line axis — the project's novelty.

``Architecture.md`` §4.4 states the mechanism:

.. code-block:: text

    sem_lines [B, L, D] --Q--+
                             +-> MHA(sem <- str) -> sem'
    str_lines [B, L, D] -K,V-+

    str_lines [B, L, D] --Q--+
                             +-> MHA(str <- sem) -> str'
    sem_lines [B, L, D] -K,V-+

    fused = LayerNorm( Linear([sem' ; str']) ) + residual

Each *line* of the semantic branch asks which *lines* of the structural branch are
relevant to it, and vice versa. Because both branches were projected onto the same line
index by :mod:`vulnloc.models.alignment`, position ``i`` means the same source line on
both sides — so an attention weight is directly readable as "line ``i`` looked at line
``j``", which is what the Phase 7 explanation layer exports as evidence.

Four things in here are not stylistic choices:

**A unit whose structural branch is empty produces NaN, not an error.** ``softmax`` over a
row that is masked everywhere is ``0/0``. ``torch.nn.MultiheadAttention`` does not guard
it, the NaN propagates into every parameter on the first backward pass, and the run dies
several minutes later somewhere unrelated. Units with no keys on one side are real — a
function no grammar parses cleanly yields a graph the builder may reject, and 7.84% of
flagged lines have no node even in a graph that built (``Memory.md`` §7 #17) — so
:func:`cross_attend` handles them explicitly and zeroes their attended output instead.

**The residual is the pre-attention projected features, not the attended ones.** It lets
the fused representation fall back to roughly the ``sum`` comparator when cross-attention
has nothing useful to contribute. A1 is supposed to be able to come back negative; an
architecture that cannot degrade to its own baseline makes that outcome mean something
other than what it says.

**Attention maps are off by default and retrievable on demand.** A map is
``[B, H, L, L]``; at the configured ``L`` = 512 and four heads that is 4 MiB per unit per
direction in float32, and ``need_weights=True`` also disables PyTorch's fused attention
kernel. Nothing in the training loop reads them, and Phase 7 asks for them one batch at a
time.

**The head count has to divide the width.** Otherwise ``MultiheadAttention`` silently
becomes a different-width model and a sweep over ``fusion.heads`` stops comparing heads —
the same failure ``configs/gnn/default.yaml`` records for the GAT head count.
"""

from __future__ import annotations

import torch

from vulnloc.models.fusion.base import FusionModule, FusionOutput
from vulnloc.models.registry import register_fusion
from vulnloc.utils.exceptions import ModelError

__all__ = ["CrossAttentionBlock", "CrossAttentionFusion", "cross_attend"]


def cross_attend(
    attention: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    key_mask: torch.Tensor,
    *,
    need_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One direction of cross-attention, safe against units with no keys.

    ``key_padding_mask`` marks positions to *ignore*. When every position of a unit is
    ignored, the softmax denominator is zero and the whole row comes back ``NaN`` — which
    then poisons every parameter through the backward pass rather than raising. Such units
    exist here by construction, so they are given a permissive mask (so the softmax has
    something to normalise) and their output is zeroed afterwards, which is what "this
    branch contributed nothing for this unit" should look like.

    Args:
        attention: The attention module. Must be ``batch_first``.
        query: ``[B, L, D]`` queries.
        key_value: ``[B, L, D]`` keys and values.
        key_mask: ``[B, L]``, ``True`` where a key is real.
        need_weights: Return per-head attention weights. Costs the fused kernel.

    Returns:
        ``(attended [B, L, D], weights [B, H, L, L] or None)``.
    """
    no_keys = ~key_mask.any(dim=-1)  # [B]
    key_padding_mask = ~key_mask
    if bool(no_keys.any()):
        key_padding_mask = key_padding_mask.masked_fill(no_keys.unsqueeze(-1), False)

    attended, weights = attention(
        query,
        key_value,
        key_value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        average_attn_weights=False,
    )

    if bool(no_keys.any()):
        attended = attended.masked_fill(no_keys[:, None, None], 0.0)
        if weights is not None:
            weights = weights.masked_fill(no_keys[:, None, None, None], 0.0)
    return attended, weights


class CrossAttentionBlock(torch.nn.Module):
    """One bidirectional cross-attention layer: ``sem <- str`` and ``str <- sem``.

    Args:
        hidden_dim: Common width ``D``.
        heads: Attention heads ``H``. Must divide ``hidden_dim``.
        dropout: Dropout inside the attention and on its residual branch.

    Raises:
        ModelError: If ``heads`` does not divide ``hidden_dim``.
    """

    def __init__(self, hidden_dim: int, *, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if heads <= 0 or hidden_dim % heads:
            raise ModelError(
                "heads must be positive and divide hidden_dim",
                hidden_dim=hidden_dim,
                heads=heads,
            )
        self.heads = heads
        self.sem_from_str = torch.nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.str_from_sem = torch.nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.sem_norm = torch.nn.LayerNorm(hidden_dim)
        self.str_norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Attend in both directions.

        Args:
            sem: ``[B, L, D]`` semantic line features.
            str_: ``[B, L, D]`` structural line features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            need_weights: Return the attention maps.

        Returns:
            ``(sem' [B, L, D], str' [B, L, D], attn_sem2str [B, H, L, L] or None,
            attn_str2sem [B, H, L, L] or None)``.
        """
        sem_attended, attn_sem2str = cross_attend(
            self.sem_from_str, sem, str_, str_mask, need_weights=need_weights
        )
        str_attended, attn_str2sem = cross_attend(
            self.str_from_sem, str_, sem, sem_mask, need_weights=need_weights
        )
        sem_out = self.sem_norm(sem + self.dropout(sem_attended))
        str_out = self.str_norm(str_ + self.dropout(str_attended))
        return sem_out, str_out, attn_sem2str, attn_str2sem


@register_fusion("cross_attention")
class CrossAttentionFusion(FusionModule):
    """Bidirectional cross-attention fusion — A1's treatment arm.

    Structurally identical to :class:`~vulnloc.models.fusion.comparators.ConcatFusion`
    apart from the attention blocks: same projections, same output projection, same
    residual, same output width. That is deliberate, and it is what makes the difference
    between the two attributable to the mechanism.

    Args:
        sem_dim: Semantic width ``D_sem``.
        str_dim: Structural width ``D_str``.
        hidden_dim: Common width ``D``.
        heads: Attention heads. Must divide ``hidden_dim``.
        n_layers: Bidirectional blocks to stack. The returned attention maps are the
            **last** block's.
        dropout: Dropout inside attention, on residuals and before the output projection.

    Raises:
        ModelError: If a width is not positive, ``heads`` does not divide ``hidden_dim``,
            or ``n_layers`` is not positive.
    """

    def __init__(
        self,
        *,
        sem_dim: int,
        str_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(sem_dim=sem_dim, str_dim=str_dim, hidden_dim=hidden_dim, dropout=dropout)
        if n_layers <= 0:
            raise ModelError("n_layers must be positive", n_layers=n_layers)
        self.n_layers = n_layers
        self.heads = heads
        self.blocks = torch.nn.ModuleList(
            CrossAttentionBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(n_layers)
        )
        self.out_proj = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_norm = torch.nn.LayerNorm(hidden_dim)

    @property
    def produces_attention(self) -> bool:
        """This strategy has attention maps to export."""
        return True

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Run the blocks, then project the pair down to one line representation.

        Args:
            sem: ``[B, L, D]`` projected semantic features.
            str_: ``[B, L, D]`` projected structural features.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Retain the last block's maps.

        Returns:
            A :class:`~vulnloc.models.fusion.base.FusionOutput` whose ``fused`` is
            ``[B, L, D]``.
        """
        sem_out, str_out = sem, str_
        attn_sem2str: torch.Tensor | None = None
        attn_str2sem: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            last = index == self.n_layers - 1
            sem_out, str_out, sem2str, str2sem = block(
                sem_out,
                str_out,
                sem_mask,
                str_mask,
                need_weights=return_attention and last,
            )
            if last:
                attn_sem2str, attn_str2sem = sem2str, str2sem

        pair = self.dropout(torch.cat([sem_out, str_out], dim=-1))
        # The residual is the *pre-attention* projected pair, so the module can fall back
        # to roughly `sum` when the attention contributes nothing.
        fused = self.out_norm(self.out_proj(pair)) + 0.5 * (sem + str_)
        line_mask = self._line_mask(sem_mask, str_mask)
        return FusionOutput(
            fused=fused * line_mask.unsqueeze(-1),
            line_mask=line_mask,
            attn_sem2str=attn_sem2str,
            attn_str2sem=attn_str2sem,
        )

    def extra_repr(self) -> str:
        """Show the widths, head count and depth in ``print(model)``."""
        return f"{super().extra_repr()}, heads={self.heads}, n_layers={self.n_layers}"
