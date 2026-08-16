"""Losses for an extremely imbalanced line-level task.

**0.795% of source lines are vulnerable** (``Memory.md`` §1.4). Plain BCE on that ratio
has a trivial minimum — predict "clean" everywhere — that costs almost nothing in loss and
scores superbly on accuracy. ``Memory.md`` §7 lists this under the pre-recorded traps for
exactly that reason, so the imbalance is handled in the objective rather than discovered
in the metrics.

Two mechanisms, both config-selectable so the choice is an ablation and not an assumption:

``bce``
    Positive-weighted binary cross-entropy. ``pos_weight`` is computed from the training
    split's own scoreable lines (:meth:`LineLabelledDataset.positive_line_weight`), not
    from a corpus-wide constant, so it matches the denominator the loss actually sums over.
``focal``
    Down-weights easy negatives by ``(1 - p)**gamma``. Usually the better choice when the
    negatives are not merely numerous but overwhelmingly easy, which is the case here —
    most lines are braces, declarations and comments.

Both take the **line mask** and reduce over valid positions only. A padded line that is
scored and then averaged in is a real bug and a silent one: it dilutes the loss by however
much padding a batch happens to carry, so the effective learning rate would depend on the
batch's length distribution.
"""

from __future__ import annotations

import torch
from torch.nn import functional as functional

from vulnloc.utils.exceptions import ModelError

__all__ = ["LineLoss", "focal_loss_with_logits", "weighted_bce_with_logits"]


def _reduce_masked(per_element: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked-in positions only.

    Args:
        per_element: Elementwise losses.
        mask: ``True`` where the element counts.

    Returns:
        A scalar. Zero when nothing is valid, so an all-padding batch contributes no
        gradient rather than a NaN.
    """
    valid = mask.sum()
    if valid == 0:
        return per_element.sum() * 0.0
    return (per_element * mask).sum() / valid


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Positive-weighted BCE, reduced over valid positions.

    Args:
        logits: Any shape.
        targets: Same shape, 0.0 or 1.0.
        mask: Same shape, ``True`` where the position counts.
        pos_weight: Multiplier on the positive term.

    Returns:
        A scalar loss.
    """
    per_element = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    return _reduce_masked(per_element, mask)


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss, reduced over valid positions.

    Args:
        logits: Any shape.
        targets: Same shape, 0.0 or 1.0.
        mask: Same shape, ``True`` where the position counts.
        alpha: Weight on the positive class.
        gamma: Focusing exponent; 0 reduces this to weighted BCE.

    Returns:
        A scalar loss.
    """
    per_element = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return _reduce_masked(alpha_t * (1 - p_t).pow(gamma) * per_element, mask)


class LineLoss(torch.nn.Module):
    """Combines the line-level and function-level objectives.

    The function term is not free supervision — it is the same labels seen at a coarser
    grain — but it is the only signal available for the 24.4% of positives with no line
    labels, and it stabilises early training when the line head has no idea what it is
    looking for. ``function_weight=0`` turns it off for a pure line-level ablation.

    Args:
        kind: ``"bce"`` or ``"focal"``.
        pos_weight: Positive weight for ``bce``.
        alpha: Positive weight for ``focal``.
        gamma: Focusing exponent for ``focal``.
        function_weight: Weight on the function-level term.

    Raises:
        ModelError: If ``kind`` is not a known loss.
    """

    def __init__(
        self,
        *,
        kind: str = "focal",
        pos_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        function_weight: float = 0.3,
    ) -> None:
        super().__init__()
        if kind not in {"bce", "focal"}:
            raise ModelError("unknown loss", kind=kind, supported=["bce", "focal"])
        self.kind = kind
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.function_weight = function_weight

    def _line_term(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply the configured loss to the line axis."""
        if self.kind == "bce":
            return weighted_bce_with_logits(logits, targets, mask, pos_weight=self.pos_weight)
        return focal_loss_with_logits(logits, targets, mask, alpha=self.alpha, gamma=self.gamma)

    def forward(
        self,
        line_logits: torch.Tensor,
        line_targets: torch.Tensor,
        line_mask: torch.Tensor,
        function_logits: torch.Tensor,
        function_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined loss.

        Args:
            line_logits: ``[B, L]``.
            line_targets: ``[B, L]``, 0.0 or 1.0.
            line_mask: ``[B, L]``, ``True`` where a line is real and scoreable.
            function_logits: ``[B]``.
            function_targets: ``[B]``, 0.0 or 1.0.

        Returns:
            ``{"loss": scalar, "line_loss": scalar, "function_loss": scalar}``.
        """
        line_loss = self._line_term(line_logits, line_targets, line_mask)
        function_loss = weighted_bce_with_logits(
            function_logits,
            function_targets,
            torch.ones_like(function_targets, dtype=torch.bool),
        )
        return {
            "loss": line_loss + self.function_weight * function_loss,
            "line_loss": line_loss.detach(),
            "function_loss": function_loss.detach(),
        }
