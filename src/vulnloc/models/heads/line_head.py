"""Per-line scoring heads (``Architecture.md`` §4.2).

``forward(fused, line_mask) -> [B, L]`` logits. One score per source line, which is the
output the whole project is organised around.

Two things this module is careful about, both of which look like details and are not:

**A line that received no features is not a line the model may score.** Padding lines
beyond a unit's length, and real lines that no token or node landed on — 7.84% of flagged
lines have no AST node at all (``Memory.md`` §7 #21) — carry no evidence. Their logits are
forced to ``-inf`` rather than left to whatever a linear layer makes of a zero vector, so
a masked line cannot win a top-k ranking by accident.

**The function-level score is derived from the line scores, not learned beside them.**
A separate function head would be free to disagree with its own line scores — to call a
function vulnerable while ranking every line as clean — and that contradiction would be
invisible in aggregate metrics. Pooling with a soft maximum keeps the two consistent by
construction, and makes the function decision explainable by the lines that drove it,
which is what ``Architecture.md`` §4.4 promises the explanation layer.
"""

from __future__ import annotations

import torch

from vulnloc.models.registry import register_head
from vulnloc.utils.exceptions import ModelError

__all__ = ["MASKED_LOGIT", "LineHead", "masked_logsumexp_pool"]

MASKED_LOGIT = -1e4
"""Logit assigned to a line with no features.

Finite rather than ``-inf``: ``-inf`` survives the forward pass but produces ``NaN`` the
moment it meets a multiplication by zero in the backward pass, and a NaN gradient poisons
every parameter in one step. ``-1e4`` is ``0`` to within float32 sigmoid precision and
differentiates cleanly. It is also safe under float16 autocast, where ``-1e5`` would
overflow to ``-inf`` and reintroduce the problem.
"""


def masked_logsumexp_pool(logits: torch.Tensor, line_mask: torch.Tensor) -> torch.Tensor:
    """Pool per-line logits into one function-level logit.

    A smooth maximum: dominated by the highest-scoring line, but with gradient reaching
    every line that has evidence. A hard ``max`` would send gradient to exactly one line
    per function, which on 0.795% positive lines is a very thin training signal.

    Args:
        logits: ``[B, L]`` per-line logits.
        line_mask: ``[B, L]``, ``True`` where a line carries features.

    Returns:
        ``[B]`` function-level logits.
    """
    neutral = torch.full_like(logits, MASKED_LOGIT)
    masked = torch.where(line_mask, logits, neutral)
    pooled = torch.logsumexp(masked, dim=-1)
    # A unit with no usable lines at all would otherwise pool to log(L * exp(-1e4)),
    # which is -inf-adjacent; pin it to the neutral value instead.
    empty = ~line_mask.any(dim=-1)
    return torch.where(empty, torch.full_like(pooled, MASKED_LOGIT), pooled)


@register_head("line")
class LineHead(torch.nn.Module):
    """Maps line features to per-line logits, plus a derived function-level logit.

    Args:
        input_dim: Width of the incoming line features, ``D``.
        hidden_dim: Width of the hidden layer. ``0`` makes the head a single linear
            projection — the honest baseline, and what an ablation should compare against.
        dropout: Dropout before the projection.

    Raises:
        ModelError: If ``input_dim`` is not positive.
    """

    def __init__(self, input_dim: int, *, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ModelError("input_dim must be positive", input_dim=input_dim)

        self.input_dim = input_dim
        layers: list[torch.nn.Module] = [torch.nn.Dropout(dropout)]
        if hidden_dim > 0:
            layers += [
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            ]
        else:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, line_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every line.

        Args:
            features: ``[B, L, D]`` line-aligned features.
            line_mask: ``[B, L]``, ``True`` where the line carries features.

        Returns:
            ``(line_logits [B, L], function_logits [B])``. Masked lines hold
            :data:`MASKED_LOGIT`.

        Raises:
            ModelError: If the shapes disagree or the width is wrong.
        """
        if features.dim() != 3:
            raise ModelError("features must be [B, L, D]", shape=tuple(features.shape))
        if features.shape[-1] != self.input_dim:
            raise ModelError(
                "feature width does not match the head",
                expected=self.input_dim,
                found=features.shape[-1],
            )
        if line_mask.shape != features.shape[:2]:
            raise ModelError(
                "line_mask must be [B, L]",
                features=tuple(features.shape),
                line_mask=tuple(line_mask.shape),
            )

        logits = self.net(features).squeeze(-1)
        logits = torch.where(line_mask, logits, torch.full_like(logits, MASKED_LOGIT))
        return logits, masked_logsumexp_pool(logits, line_mask)
