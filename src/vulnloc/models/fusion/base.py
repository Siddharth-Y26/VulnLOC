"""The interface every A1 arm shares (``Architecture.md`` §4.2, §4.4).

Experiment A1 asks one question — **does bidirectional cross-attention beat
concatenation?** — and the only way that question has an answer is if the two arms differ
in the fusion mechanism and in nothing else. That is what this module is for. Every
comparator inherits the same input projections, the same masking rule, the same output
width and the same line axis, so the difference between ``cross_attention`` and
``concat`` is literally the multi-head attention and the two layer norms around it.

Three decisions here are load-bearing, and each is a way the comparison could have been
confounded silently:

**The projections live in the base class, not in the cross-attention module.** The two
branches arrive at different widths — CodeBERT is 768, the GNN is 256 — so *something*
has to project them into a common space before they can be added, concatenated or
attended. If that projection lived in the cross-attention module alone, ``concat`` would
be comparing "attention plus a projection" against "no projection", and the extra
parameters would sit inside the arm that is supposed to win.

**A line a branch never saw is exactly zero, not a bias vector.**
:func:`~vulnloc.models.alignment.scatter_mean_lines` returns zeros where no token or node
landed, but a ``Linear`` maps zero to its bias — a learnable constant the head could read
as "the semantic branch is silent here", which on a corpus where 7.84% of flagged lines
carry no graph node (``Memory.md`` §7 #17) is a feature correlated with the label. The
projections are re-masked afterwards so silence stays silent.

**The fused line mask is the union of the two branch masks.** A line is scoreable if
*either* branch has evidence for it, which is the concrete thing fusion buys: the
structural branch alone cannot reach the §7 #17 lines and a fused model can. Single-branch
comparators override this — ``llm_only`` must not be allowed to score a line it has no
tokens for merely because a node landed there, or the ablation would quietly be scoring on
the other branch's coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vulnloc.utils.exceptions import ModelError

__all__ = ["FusionModule", "FusionOutput"]


@dataclass(frozen=True)
class FusionOutput:
    """What every fusion strategy returns.

    Attributes:
        fused: ``[B, L, D]`` line features for the head.
        line_mask: ``[B, L]``, ``True`` where the fused representation has evidence.
        attn_sem2str: ``[B, H, L, L]`` attention paid by each semantic line to each
            structural line, or ``None`` when maps were not requested or the strategy has
            none. ``[b, h, i, j]`` is how much line ``i`` of unit ``b`` attended to line
            ``j`` in head ``h``; rows sum to 1 over ``j``.
        attn_str2sem: ``[B, H, L, L]``, the other direction.
    """

    fused: torch.Tensor
    line_mask: torch.Tensor
    attn_sem2str: torch.Tensor | None = None
    attn_str2sem: torch.Tensor | None = None

    @property
    def has_attention(self) -> bool:
        """Whether attention maps came back with this output."""
        return self.attn_sem2str is not None or self.attn_str2sem is not None


class FusionModule(torch.nn.Module):
    """Base class for every fusion strategy, holding what the arms must share.

    Subclasses implement :meth:`_combine` and nothing else. The projections, the
    re-masking, the shape validation and the output contract are fixed here so a new
    comparator cannot accidentally change the experiment's controls.

    Args:
        sem_dim: Width of the semantic line features, ``D_sem``.
        str_dim: Width of the structural line features, ``D_str``.
        hidden_dim: Common width ``D`` both branches are projected onto, and the width the
            head receives.
        dropout: Dropout applied inside the strategy. Zeroed by the overfit config.

    Raises:
        ModelError: If any width is not positive.
    """

    def __init__(
        self, *, sem_dim: int, str_dim: int, hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        widths = (("sem_dim", sem_dim), ("str_dim", str_dim), ("hidden_dim", hidden_dim))
        for name, value in widths:
            if value <= 0:
                raise ModelError("width must be positive", width=name, value=value)
        self.sem_dim = sem_dim
        self.str_dim = str_dim
        self.hidden_dim = hidden_dim
        self.sem_proj = torch.nn.Linear(sem_dim, hidden_dim)
        self.str_proj = torch.nn.Linear(str_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        """Width of the fused features, ``D``. The head is built from this."""
        return self.hidden_dim

    @property
    def produces_attention(self) -> bool:
        """Whether this strategy can return attention maps at all.

        ``False`` for every comparator: they are not worse at explaining, they simply have
        no attention to export, and saying so is more useful than returning zeros that
        look like evidence.
        """
        return False

    @property
    def uses_semantic(self) -> bool:
        """Whether the semantic branch reaches the output at all.

        ``False`` only for the ``gnn_only`` ablation. The model reads this to skip the
        encoder rather than running 125M parameters whose result is then discarded — and,
        more importantly, so that the ablation is a *structural* fact about the model
        rather than a multiplication by zero that a gradient could still flow through.
        """
        return True

    @property
    def uses_structural(self) -> bool:
        """Whether the structural branch reaches the output at all.

        ``False`` only for the ``llm_only`` ablation.
        """
        return True

    def forward(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> FusionOutput:
        """Fuse two line-aligned branches.

        Args:
            sem: ``[B, L, D_sem]`` semantic line features.
            str_: ``[B, L, D_str]`` structural line features.
            sem_mask: ``[B, L]``, ``True`` where a token landed on the line.
            str_mask: ``[B, L]``, ``True`` where a graph node landed on the line.
            return_attention: Retain and return the attention maps. Off during training: a
                map is ``[B, H, L, L]``, which at ``L`` = 512 is 1 MiB per unit per head
                per direction in float32, and nothing in the training loop reads it.

        Returns:
            A :class:`FusionOutput`.

        Raises:
            ModelError: If the shapes or widths disagree.
        """
        self._validate(sem, str_, sem_mask, str_mask)
        # Project, then re-zero: `Linear(0) == bias`, and a bias vector on a line the
        # branch never saw is a learnable "this branch is silent here" feature.
        sem_p = self.sem_proj(sem) * sem_mask.unsqueeze(-1)
        str_p = self.str_proj(str_) * str_mask.unsqueeze(-1)
        return self._combine(sem_p, str_p, sem_mask, str_mask, return_attention=return_attention)

    def _validate(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
    ) -> None:
        """Check both branches arrived on the same line axis.

        Args:
            sem: ``[B, L, D_sem]``.
            str_: ``[B, L, D_str]``.
            sem_mask: ``[B, L]``.
            str_mask: ``[B, L]``.

        Raises:
            ModelError: If the shapes or widths disagree.
        """
        if sem.dim() != 3 or str_.dim() != 3:
            raise ModelError(
                "both branches must be [B, L, D]",
                sem=tuple(sem.shape),
                structural=tuple(str_.shape),
            )
        if sem.shape[:2] != str_.shape[:2]:
            raise ModelError(
                "the two branches are not on the same line axis",
                sem=tuple(sem.shape[:2]),
                structural=tuple(str_.shape[:2]),
            )
        if sem.shape[-1] != self.sem_dim or str_.shape[-1] != self.str_dim:
            raise ModelError(
                "branch width does not match the fusion module",
                expected_sem=self.sem_dim,
                found_sem=int(sem.shape[-1]),
                expected_str=self.str_dim,
                found_str=int(str_.shape[-1]),
            )
        if sem_mask.shape != sem.shape[:2] or str_mask.shape != str_.shape[:2]:
            raise ModelError(
                "masks must be [B, L]",
                lines=tuple(sem.shape[:2]),
                sem_mask=tuple(sem_mask.shape),
                str_mask=tuple(str_mask.shape),
            )

    def _line_mask(self, sem_mask: torch.Tensor, str_mask: torch.Tensor) -> torch.Tensor:
        """Which lines the fused representation may be scored on.

        The union by default: fusion's whole point is that one branch covers what the
        other cannot. Single-branch comparators narrow it to their own branch.

        Args:
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.

        Returns:
            ``[B, L]``.
        """
        return sem_mask | str_mask

    def _combine(
        self,
        sem: torch.Tensor,
        str_: torch.Tensor,
        sem_mask: torch.Tensor,
        str_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> FusionOutput:
        """Strategy-specific combination of two already-projected branches.

        Args:
            sem: ``[B, L, D]`` projected semantic features, zero where unseen.
            str_: ``[B, L, D]`` projected structural features, zero where unseen.
            sem_mask: ``[B, L]`` semantic coverage.
            str_mask: ``[B, L]`` structural coverage.
            return_attention: Whether to retain attention maps.

        Returns:
            A :class:`FusionOutput`.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("a fusion strategy must implement _combine")

    def extra_repr(self) -> str:
        """Show the widths in ``print(model)``."""
        return f"sem_dim={self.sem_dim}, str_dim={self.str_dim}, hidden_dim={self.hidden_dim}"
