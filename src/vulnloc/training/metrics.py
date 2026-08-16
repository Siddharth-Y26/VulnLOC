"""Line- and function-level metrics (L3, so the training loop may use them).

Phase 8 owns evaluation properly. What lives here is the minimum Phase 3's exit criteria
name — function-level and line-level numbers, logged — kept at L3 because the training
loop cannot import L4.

The metric choices are not neutral, and the reasons are the point:

**Accuracy is not reported at all.** With 0.795% positive lines, predicting "clean"
everywhere scores 99.2%. Including it would invite exactly the mistake ``Memory.md`` §7
pre-records.

**Line ranking is measured per function, not pooled.** The task is "given a function
flagged as vulnerable, where is the flaw" — so Top-k accuracy and IFA (Initial False
Alarms: how many clean lines a reviewer reads before the first true one) are computed
within each function and then averaged. Pooling every line in the corpus into one ranking
would let long clean functions dominate and would answer a question nobody asked.

**Coverage travels with recall.** ``flagged_lines_without_features`` counts flagged lines
the model had nothing to score. It is a ceiling, not a failure, and reporting recall
without it overstates what the model could possibly have achieved (``Memory.md`` §7 #21).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["LineMetrics", "MetricAccumulator", "initial_false_alarms", "top_k_accuracy"]


def top_k_accuracy(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float | None:
    """Whether any true line appears in the top ``k`` ranked lines.

    Args:
        scores: ``[L]`` per-line scores for one function.
        targets: ``[L]`` 0/1 labels.
        k: How many lines a reviewer would read.

    Returns:
        1.0 or 0.0, or ``None`` when the function has no positive line — in which case
        the question is undefined and averaging a 0 into it would be wrong.
    """
    if targets.sum() == 0:
        return None
    order = torch.argsort(scores, descending=True)
    return float(targets[order[:k]].sum() > 0)


def initial_false_alarms(scores: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Clean lines ranked above the first true line.

    The reviewer-effort metric: 0 means the top-ranked line is genuinely vulnerable.

    Args:
        scores: ``[L]`` per-line scores for one function.
        targets: ``[L]`` 0/1 labels.

    Returns:
        The count, or ``None`` when the function has no positive line.
    """
    if targets.sum() == 0:
        return None
    order = torch.argsort(scores, descending=True)
    ranked = targets[order]
    first_hit = int(torch.argmax(ranked))
    return float(first_hit)


@dataclass
class LineMetrics:
    """One evaluation's worth of numbers."""

    n_units: int = 0
    n_scored_units: int = 0
    function_tp: int = 0
    function_fp: int = 0
    function_fn: int = 0
    function_tn: int = 0
    line_tp: int = 0
    line_fp: int = 0
    line_fn: int = 0
    flagged_lines: int = 0
    flagged_lines_without_features: int = 0
    top_1: list[float] = field(default_factory=list)
    top_5: list[float] = field(default_factory=list)
    ifa: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a flat mapping for MLflow.

        Returns:
            Metric name to value. Precision/recall/F1 are 0.0 when undefined, which is
            reported alongside the counts they came from so the reader can tell an
            honest zero from an empty denominator.
        """

        def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return precision, recall, f1

        def _mean(values: list[float]) -> float:
            """Mean of a possibly-empty list; 0.0 rather than a division by zero."""
            return sum(values) / len(values) if values else 0.0

        fn_p, fn_r, fn_f1 = _f1(self.function_tp, self.function_fp, self.function_fn)
        ln_p, ln_r, ln_f1 = _f1(self.line_tp, self.line_fp, self.line_fn)

        return {
            "function/precision": fn_p,
            "function/recall": fn_r,
            "function/f1": fn_f1,
            "function/tp": float(self.function_tp),
            "function/fp": float(self.function_fp),
            "function/fn": float(self.function_fn),
            "function/tn": float(self.function_tn),
            "line/precision": ln_p,
            "line/recall": ln_r,
            "line/f1": ln_f1,
            "line/tp": float(self.line_tp),
            "line/fp": float(self.line_fp),
            "line/fn": float(self.line_fn),
            "line/top_1_accuracy": _mean(self.top_1),
            "line/top_5_accuracy": _mean(self.top_5),
            "line/ifa_mean": _mean(self.ifa),
            "line/scored_units": float(self.n_scored_units),
            # The ceiling, carried with the recall it bounds.
            "coverage/flagged_lines": float(self.flagged_lines),
            "coverage/flagged_lines_without_features": float(self.flagged_lines_without_features),
            "coverage/flagged_line_coverage_pct": (
                100.0
                * (self.flagged_lines - self.flagged_lines_without_features)
                / self.flagged_lines
                if self.flagged_lines
                else 0.0
            ),
        }


class MetricAccumulator:
    """Folds batches into a :class:`LineMetrics`.

    Args:
        threshold: Probability above which a line or function is called vulnerable.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.metrics = LineMetrics()

    def update(
        self,
        line_logits: torch.Tensor,
        line_targets: torch.Tensor,
        line_mask: torch.Tensor,
        line_valid: torch.Tensor,
        function_logits: torch.Tensor,
        function_targets: torch.Tensor,
    ) -> None:
        """Fold one batch in.

        Args:
            line_logits: ``[B, L]``.
            line_targets: ``[B, L]``.
            line_mask: ``[B, L]`` lines that received features.
            line_valid: ``[B, L]`` lines that exist in the source.
            function_logits: ``[B]``.
            function_targets: ``[B]``.
        """
        metrics = self.metrics
        metrics.n_units += int(function_targets.numel())

        function_pred = (torch.sigmoid(function_logits) >= self.threshold).long()
        function_true = function_targets.long()
        metrics.function_tp += int(((function_pred == 1) & (function_true == 1)).sum())
        metrics.function_fp += int(((function_pred == 1) & (function_true == 0)).sum())
        metrics.function_fn += int(((function_pred == 0) & (function_true == 1)).sum())
        metrics.function_tn += int(((function_pred == 0) & (function_true == 0)).sum())

        # A line is judged only where it both exists and could be scored. Counting
        # unscoreable lines as false negatives would fold the coverage ceiling into the
        # model's recall and make the two impossible to separate later.
        judged = line_valid & line_mask
        line_pred = (torch.sigmoid(line_logits) >= self.threshold) & judged
        line_true = (line_targets > 0) & judged
        metrics.line_tp += int((line_pred & line_true).sum())
        metrics.line_fp += int((line_pred & ~line_true).sum())
        metrics.line_fn += int((~line_pred & line_true).sum())

        flagged = (line_targets > 0) & line_valid
        metrics.flagged_lines += int(flagged.sum())
        metrics.flagged_lines_without_features += int((flagged & ~line_mask).sum())

        for index in range(line_logits.shape[0]):
            usable = line_valid[index]
            if not usable.any():
                continue
            scores = line_logits[index][usable]
            targets = line_targets[index][usable]
            if targets.sum() == 0:
                continue
            metrics.n_scored_units += 1
            for k, bucket in ((1, metrics.top_1), (5, metrics.top_5)):
                value = top_k_accuracy(scores, targets, k)
                if value is not None:
                    bucket.append(value)
            ifa = initial_false_alarms(scores, targets)
            if ifa is not None:
                metrics.ifa.append(ifa)

    def result(self) -> dict[str, float]:
        """Return the accumulated metrics.

        Returns:
            Metric name to value.
        """
        return self.metrics.to_dict()
