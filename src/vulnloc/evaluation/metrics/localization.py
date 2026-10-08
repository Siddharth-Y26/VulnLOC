"""Line-level localization metrics (Phase 8; ``docs/ResearchPlan.md`` §4.2).

Two kinds of metric live here, and the difference decides how each is aggregated.

**Per-unit ranking metrics** answer "given this function, where is the flaw": the lines of
one unit are ranked, the metric reads the ranking, and the partition's number is the mean
over units. They are the entries of ``METRIC_REGISTRY``: each is a function of one unit's
labels *in rank order* and nothing else, so a new one is a function and a decorator.
Pooling every line of the corpus into one ranking would let long clean functions dominate
(``training/metrics.py`` says why); these never do.

**Pooled line metrics** — average precision, and precision / recall / F1 at a threshold —
are properties of a set of scored lines. The caller decides which lines; the evaluator
passes the lines the model could score and reports the ones it could not beside them, as
the training loop's line metrics do.

**Ties are not noise to be ignored.** The stored logits are half-precision values, so in a
few percent of units the best flagged line is exactly tied with an unflagged one, and
``torch.argsort`` promises no order among equals (``Memory.md`` §7 #63). Every ranking
metric is therefore computed under three orders: ``source`` (ties keep source order — the
deterministic value), ``best`` (every tie broken for the flagged line) and ``worst`` (every
tie broken against it). The last two bound what any tie order could have reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vulnloc.evaluation.registry import METRIC_REGISTRY, register_metric
from vulnloc.utils.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "LOC_BUDGETS_PCT",
    "TIE_ORDERS",
    "TOP_KS",
    "average_precision",
    "average_rank",
    "first_rank",
    "initial_false_alarms",
    "precision_recall_f1",
    "ranked_labels",
    "unit_ranking_metrics",
]

TIE_ORDERS = ("source", "best", "worst")
"""How equal scores are ordered: as in the source, for the flagged line, against it."""

TOP_KS = (1, 3, 5, 10)
"""The k of Top-k accuracy (``ResearchPlan.md`` §4.2)."""

LOC_BUDGETS_PCT = (1, 5, 20)
"""Review budgets, in percent of the unit's lines, for effort-aware recall."""


def ranked_labels(
    scores: np.ndarray, labels: np.ndarray, candidates: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """One unit's labels in rank order, under each tie order.

    Args:
        scores: ``[n_lines]`` — higher is more suspicious.
        labels: ``[n_lines]`` — non-zero where the line is flagged.
        candidates: ``[n_lines]`` bool — the lines that may be ranked. ``None`` ranks every
            line, which is what the training loop does: a line the model could not score
            holds the head's masked sentinel and so ranks last.

    Returns:
        ``{tie_order: [n_candidates] bool}`` for each of :data:`TIE_ORDERS`, most
        suspicious line first.

    Raises:
        DataError: If the arrays disagree in shape or a candidate's score is NaN — a NaN
            has no rank, and ``argsort`` would give it one silently.
    """
    values = np.asarray(scores, dtype=np.float64)
    flagged = np.asarray(labels) > 0
    if values.ndim != 1 or values.shape != flagged.shape:
        raise DataError(
            "a unit's scores and labels must be flat and of one length",
            scores=values.shape,
            labels=flagged.shape,
        )
    if candidates is None:
        index = np.arange(values.shape[0])
    else:
        mask = np.asarray(candidates, dtype=bool)
        if mask.shape != values.shape:
            raise DataError(
                "the candidate mask must cover every line",
                candidates=mask.shape,
                scores=values.shape,
            )
        index = np.flatnonzero(mask)
    ranked_scores = values[index]
    ranked_flags = flagged[index]
    if np.isnan(ranked_scores).any():
        raise DataError("a line score is NaN; it cannot be ranked")
    descending = -ranked_scores
    # lexsort sorts by its last key first and is stable, so each order is "by score, then
    # by the tie rule, then by source position".
    return {
        "source": ranked_flags[np.argsort(descending, kind="stable")],
        "best": ranked_flags[np.lexsort((~ranked_flags, descending))],
        "worst": ranked_flags[np.lexsort((ranked_flags, descending))],
    }


def _top_k(k: int) -> Callable[[np.ndarray], float]:
    """Build the Top-k metric for one ``k``."""

    def top_k(ranked: np.ndarray) -> float:
        return float(ranked[:k].any())

    top_k.__name__ = top_k.__qualname__ = f"top_{k}"
    top_k.__doc__ = (
        f"Whether a flagged line is among the {k} most suspicious.\n\n"
        "    Args:\n        ranked: ``[n]`` bool, the unit's labels in rank order.\n\n"
        "    Returns:\n        ``1.0`` or ``0.0``.\n"
    )
    return top_k


for _k in TOP_KS:
    register_metric(f"top_{_k}")(_top_k(_k))


@register_metric("ifa")
def initial_false_alarms(ranked: np.ndarray) -> float:
    """Clean lines a reviewer reads before the first flagged one.

    Args:
        ranked: ``[n]`` bool, the unit's labels in rank order, at least one ``True``.

    Returns:
        The count; ``0.0`` when the top-ranked line is flagged.
    """
    return float(np.argmax(ranked))


@register_metric("first_rank")
def first_rank(ranked: np.ndarray) -> float:
    """1-indexed rank of the best-ranked flagged line. Its mean over units is MFR.

    Always ``ifa + 1``: every line above the first flagged one is clean. Reported because
    ``ResearchPlan.md`` §4.2 names it, not because it says anything IFA does not.

    Args:
        ranked: ``[n]`` bool, the unit's labels in rank order, at least one ``True``.

    Returns:
        The rank.
    """
    return float(np.argmax(ranked)) + 1.0


@register_metric("average_rank")
def average_rank(ranked: np.ndarray) -> float:
    """Mean 1-indexed rank of the unit's flagged lines. Its mean over units is MAR.

    Args:
        ranked: ``[n]`` bool, the unit's labels in rank order, at least one ``True``.

    Returns:
        The mean rank. Unlike the first rank it is pulled down by every flagged line the
        model ranks low, so it grows with the length of the unit.
    """
    return float((np.flatnonzero(ranked) + 1).mean())


def _recall_at_loc(percent: int) -> Callable[[np.ndarray], float]:
    """Build the effort-aware recall metric for one review budget."""

    def recall_at_loc(ranked: np.ndarray) -> float:
        # Integer arithmetic: ceil(percent * n / 100) without a float that rounds 3.0000001
        # up to 4. A reviewer always reads at least one line.
        budget = max(1, (percent * ranked.shape[0] + 99) // 100)
        return float(ranked[:budget].sum() / ranked.sum())

    recall_at_loc.__name__ = recall_at_loc.__qualname__ = f"recall_at_{percent}pct_loc"
    recall_at_loc.__doc__ = (
        f"Share of the unit's flagged lines within its top {percent}% of ranked lines.\n\n"
        "    The budget is per unit — ``ceil(percent * n / 100)`` lines, never fewer than\n"
        "    one — because the ranking is per unit.\n\n"
        "    Args:\n        ranked: ``[n]`` bool, the unit's labels in rank order, at least "
        "one ``True``.\n\n    Returns:\n        Recall in ``[0, 1]``.\n"
    )
    return recall_at_loc


for _percent in LOC_BUDGETS_PCT:
    register_metric(f"recall_at_{_percent}pct_loc")(_recall_at_loc(_percent))


def unit_ranking_metrics(
    scores: np.ndarray, labels: np.ndarray, candidates: np.ndarray | None = None
) -> dict[str, dict[str, float]] | None:
    """Every registered ranking metric for one unit, under each tie order.

    Args:
        scores: ``[n_lines]`` — higher is more suspicious.
        labels: ``[n_lines]`` — non-zero where the line is flagged.
        candidates: ``[n_lines]`` bool — the lines that may be ranked; ``None`` for all.

    Returns:
        ``{metric: {tie_order: value}}``, or ``None`` when no candidate line is flagged —
        the question is then undefined for this unit and it must not be averaged in as a
        zero.
    """
    orders = ranked_labels(scores, labels, candidates)
    if not orders["source"].any():
        return None
    return {
        name: {tie: float(metric(orders[tie])) for tie in TIE_ORDERS}
        for name, metric in METRIC_REGISTRY.items().items()
    }


def _flat_pair(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a flat ``(values, labels)`` pair; return float64 values and bool labels."""
    numbers = np.asarray(values, dtype=np.float64)
    flagged = np.asarray(labels) > 0
    if numbers.ndim != 1 or numbers.shape != flagged.shape:
        raise DataError(
            "expected two flat arrays of equal length",
            values=numbers.shape,
            labels=flagged.shape,
        )
    if np.isnan(numbers).any():
        raise DataError("a line score is NaN; it cannot be ranked")
    return numbers, flagged


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, as average precision.

    ``sum_n (R_n - R_{n-1}) * P_n`` over the distinct score thresholds, most suspicious
    first. Lines with equal scores enter together, so the value does not depend on how
    ties are ordered. A ranker with no information scores the positive rate, which is why
    the evaluator reports that rate beside it.

    Args:
        scores: ``[N]`` — higher is more suspicious.
        labels: ``[N]`` — non-zero where the line is flagged.

    Returns:
        Average precision in ``(0, 1]``.

    Raises:
        DataError: If the arrays are malformed or no line is flagged, in which case recall
            is undefined at every threshold.
    """
    values, flagged = _flat_pair(scores, labels)
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        raise DataError("average precision is undefined without a flagged line")
    order = np.argsort(-values, kind="stable")
    ranked_values = values[order]
    true_positives = np.cumsum(flagged[order])
    predicted = np.arange(1, values.shape[0] + 1)
    # Only the last line of a run of equal scores is a reachable threshold.
    reachable = np.ones(values.shape[0], dtype=bool)
    reachable[:-1] = ranked_values[:-1] != ranked_values[1:]
    precision = true_positives[reachable] / predicted[reachable]
    recall = true_positives[reachable] / n_flagged
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def precision_recall_f1(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    """Precision, recall and F1 of "flagged when the probability is at least ``threshold``".

    Args:
        probabilities: ``[N]`` calibrated probabilities.
        labels: ``[N]`` — non-zero where the line is flagged.
        threshold: The decision threshold.

    Returns:
        ``{"precision", "recall", "f1", "tp", "fp", "fn"}``. A ratio with an empty
        denominator is ``0.0``; the counts beside it say which kind of zero it is.
    """
    values, flagged = _flat_pair(probabilities, labels)
    predicted = values >= threshold
    tp = int((predicted & flagged).sum())
    fp = int((predicted & ~flagged).sum())
    fn = int((~predicted & flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
