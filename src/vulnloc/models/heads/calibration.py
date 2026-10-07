"""Post-hoc calibration of line scores, and the decision threshold (Phase 6).

The line head is trained with a focal loss against 0.795% positive lines. That is the right
loss for ranking and the wrong one for probabilities: focal loss deliberately down-weights
confident examples, so ``sigmoid(logit)`` is not the chance a line is flagged and ``0.5`` is
not a meaningful cut-off. Every line F1 this project logged before Phase 6 moved with the
threshold rather than with the model for exactly that reason (``Memory.md`` §1.14, §1.22).

Three things live here, all of them functions of logits and labels and nothing else, so
they can be fitted from a score dump and applied at inference without the training layer:

**A calibrator that cannot change the ranking.** ``probability = sigmoid(scale * logit +
shift)`` with ``scale`` constrained positive. A monotone map leaves every Top-k and IFA
number exactly where it was; a calibrator free to pick a negative scale would invert the
ranking on a bad fit and report excellent calibration for it.

**A reliability curve whose bins say how full they are.** With positives this rare, most
bins are almost empty and a curve without counts invites reading noise as miscalibration.

**A threshold chosen by an explicit rule.** Maximum line F1 over the candidate cut-offs the
data actually offers — chosen on validation and nowhere else (``Rules.md`` §5.3).

Fitting is deterministic: full-batch L-BFGS in float64 on the CPU, from a fixed start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vulnloc.utils.exceptions import ModelError

__all__ = [
    "CalibrationBin",
    "LogitCalibrator",
    "ThresholdChoice",
    "best_f1_threshold",
    "expected_calibration_error",
    "fit_calibrator",
    "reliability_bins",
]

CalibratorKind = Literal["temperature", "platt"]


@dataclass(frozen=True)
class LogitCalibrator:
    """A monotone affine map on logits: ``sigmoid(scale * logit + shift)``.

    Attributes:
        scale: Positive multiplier. ``1 / scale`` is the temperature.
        shift: Additive offset. ``0`` for pure temperature scaling.
        kind: How it was fitted — ``"temperature"``, ``"platt"`` or ``"identity"``.
    """

    scale: float = 1.0
    shift: float = 0.0
    kind: str = "identity"

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ModelError(
                "a calibrator's scale must be positive; anything else reorders the lines",
                scale=self.scale,
            )

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated probabilities.

        Args:
            logits: Line logits, any shape.

        Returns:
            Probabilities with the same shape, in the same order.
        """
        return torch.sigmoid(self.scale * logits + self.shift)

    def to_dict(self) -> dict[str, float | str]:
        """Return the parameters as plain JSON-serialisable values."""
        return {"kind": self.kind, "scale": self.scale, "shift": self.shift}


def _check_pairs(values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a flat ``(values, labels)`` pair and return it as float64 / bool."""
    if values.dim() != 1 or values.shape != labels.shape:
        raise ModelError(
            "expected two flat tensors of equal length",
            values=tuple(values.shape),
            labels=tuple(labels.shape),
        )
    if values.numel() == 0:
        raise ModelError("nothing to calibrate on: no scored lines")
    return values.detach().to(torch.float64).cpu(), (labels > 0).cpu()


def fit_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    kind: CalibratorKind = "platt",
    max_iter: int = 200,
) -> LogitCalibrator:
    """Fit a calibrator by minimising the binary cross-entropy of the mapped logits.

    Args:
        logits: ``[N]`` line logits — scoreable lines only; a masked line's sentinel logit
            is not a prediction and must not be fitted to.
        labels: ``[N]`` 0/1 labels.
        kind: ``"temperature"`` fits the scale alone; ``"platt"`` fits scale and shift,
            which a focal-loss model needs because its bias is off as well as its slope.
        max_iter: L-BFGS iteration cap.

    Returns:
        The fitted calibrator.

    Raises:
        ModelError: If the inputs are malformed, hold a single class, or the fit diverges.
    """
    if kind not in ("temperature", "platt"):
        raise ModelError("unknown calibrator kind", kind=kind)
    z, positive = _check_pairs(logits, labels)
    if bool(positive.all()) or not bool(positive.any()):
        raise ModelError(
            "cannot calibrate against a single class",
            positives=int(positive.sum()),
            lines=int(positive.numel()),
        )
    target = positive.to(torch.float64)

    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    shift = torch.zeros((), dtype=torch.float64, requires_grad=kind == "platt")
    parameters = [log_scale, shift] if kind == "platt" else [log_scale]
    optimizer = torch.optim.LBFGS(
        parameters, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-10
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.exp(log_scale) * z + shift, target
        )
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    scale = float(torch.exp(log_scale.detach()))
    offset = float(shift.detach())
    if not (torch.isfinite(torch.tensor([scale, offset])).all() and scale > 0.0):
        raise ModelError("calibration did not converge", scale=scale, shift=offset)
    return LogitCalibrator(scale=scale, shift=offset, kind=kind)


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of a reliability curve.

    Attributes:
        lower: Inclusive lower edge of the predicted-probability bin.
        upper: Exclusive upper edge (inclusive for the last bin).
        count: Lines whose predicted probability fell in the bin.
        mean_probability: Mean predicted probability of those lines; ``0.0`` if empty.
        positive_rate: Fraction of them that are flagged; ``0.0`` if empty.
    """

    lower: float
    upper: float
    count: int
    mean_probability: float
    positive_rate: float


def reliability_bins(
    probabilities: torch.Tensor, labels: torch.Tensor, *, n_bins: int = 15
) -> list[CalibrationBin]:
    """Bin predictions by predicted probability, equal-width on ``[0, 1]``.

    Args:
        probabilities: ``[N]`` predicted probabilities.
        labels: ``[N]`` 0/1 labels.
        n_bins: Number of bins.

    Returns:
        ``n_bins`` bins in ascending order, empty ones included with ``count == 0`` — a
        curve that silently drops its empty bins hides where the model never predicts.

    Raises:
        ModelError: If the inputs are malformed or ``n_bins`` is not positive.
    """
    if n_bins <= 0:
        raise ModelError("n_bins must be positive", n_bins=n_bins)
    p, positive = _check_pairs(probabilities, labels)
    index = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    bins: list[CalibrationBin] = []
    for bin_index in range(n_bins):
        members = index == bin_index
        count = int(members.sum())
        bins.append(
            CalibrationBin(
                lower=bin_index / n_bins,
                upper=(bin_index + 1) / n_bins,
                count=count,
                mean_probability=float(p[members].mean()) if count else 0.0,
                positive_rate=float(positive[members].to(torch.float64).mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(bins: list[CalibrationBin]) -> float:
    """Count-weighted mean gap between predicted probability and observed rate.

    Args:
        bins: A reliability curve from :func:`reliability_bins`.

    Returns:
        The ECE in ``[0, 1]``; ``0.0`` for a curve with no lines at all.
    """
    total = sum(calibration_bin.count for calibration_bin in bins)
    if total == 0:
        return 0.0
    return sum(
        calibration_bin.count
        * abs(calibration_bin.mean_probability - calibration_bin.positive_rate)
        for calibration_bin in bins
    ) / float(total)


@dataclass(frozen=True)
class ThresholdChoice:
    """A decision threshold and what it achieves on the data it was chosen on.

    Attributes:
        threshold: A line is called flagged when its probability is ``>= threshold``.
        precision: Precision at that threshold.
        recall: Recall at that threshold.
        f1: F1 at that threshold.
        predicted_positive: Lines at or above the threshold.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive: int


def best_f1_threshold(probabilities: torch.Tensor, labels: torch.Tensor) -> ThresholdChoice:
    """The cut-off with the highest line F1 among those the data offers.

    Candidates are the distinct predicted probabilities: lowering the threshold between two
    of them changes no prediction. Ties in F1 go to the **higher** threshold — fewer lines
    for a reviewer to read at the same score.

    Args:
        probabilities: ``[N]`` predicted probabilities.
        labels: ``[N]`` 0/1 labels.

    Returns:
        The chosen threshold with its precision, recall and F1.

    Raises:
        ModelError: If the inputs are malformed or there is no positive line, in which case
            recall — and so F1 — is undefined for every threshold.
    """
    p, positive = _check_pairs(probabilities, labels)
    n_positive = int(positive.sum())
    if n_positive == 0:
        raise ModelError("cannot choose an F1 threshold without a positive line")

    order = torch.argsort(p, descending=True, stable=True)
    ranked_p = p[order]
    true_positives = torch.cumsum(positive[order].to(torch.float64), dim=0)
    predicted = torch.arange(1, p.numel() + 1, dtype=torch.float64)
    # Only the last position of each run of equal probabilities is a reachable cut-off:
    # a threshold cannot admit one line of a tie and refuse another.
    is_cut = torch.ones(p.numel(), dtype=torch.bool)
    is_cut[:-1] = ranked_p[:-1] != ranked_p[1:]

    precision = true_positives / predicted
    recall = true_positives / float(n_positive)
    f1 = torch.where(
        true_positives > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    f1 = torch.where(is_cut, f1, torch.full_like(f1, -1.0))
    # argmax returns the first maximum, and the ranking is descending in probability, so
    # the first is the highest threshold among equals.
    best = int(torch.argmax(f1))
    return ThresholdChoice(
        threshold=float(ranked_p[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        f1=float(f1[best]),
        predicted_positive=int(predicted[best]),
    )
