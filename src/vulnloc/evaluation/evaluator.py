"""Evaluate stored line scores: one checkpoint, then an arm over its seeds (Phase 8).

Everything here is a function of a score file (``training/line_scores.py``), the
checkpoint's validation-fitted calibrator and threshold, and — for the two ceilings a
score file cannot see — the partition's own text and labels. No model is loaded.

``docs/EvaluationProtocol.md`` fixes what is computed; this module is that protocol as
code. Four conventions it inherits rather than chooses:

* **Ranking metrics rank every line of a unit**, as ``training/metrics.py`` does. A line
  the model could not score holds the head's masked sentinel and ranks last; a unit whose
  flagged lines are all unscoreable is a miss, not an exclusion.
* **Pooled metrics use the lines the model could score**, with the flagged lines it could
  not reported beside them. Folding the coverage ceiling into recall would make the two
  impossible to separate afterwards.
* **Labels are never edited.** A flagged line that is blank once comments are stripped
  stays a target in the primary reading and is counted as a ceiling; a second reading
  withholds blank lines from targets and candidates alike.
* **A threshold is applied, not chosen.** The calibrator and threshold come from
  validation. On validation itself that makes the thresholded numbers in-sample, so they
  are labelled as such and the calibration report's out-of-fold estimate is the one to
  quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vulnloc.data.cache import iter_shard
from vulnloc.evaluation.metrics.localization import (
    TIE_ORDERS,
    average_precision,
    precision_recall_f1,
    unit_ranking_metrics,
)
from vulnloc.evaluation.statistics import bootstrap_mean_interval, compare_arms, spread
from vulnloc.models.heads.calibration import expected_calibration_error, reliability_bins
from vulnloc.representation.line_map import split_lines
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.text import normalize_source_whitespace, strip_comments

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from vulnloc.models.heads.calibration import LogitCalibrator
    from vulnloc.training.line_scores import UnitScores

__all__ = [
    "BOOTSTRAPPED",
    "COMPARISONS",
    "LOGGED_METRICS",
    "PartitionUnit",
    "aggregate_arm",
    "arm_name",
    "compare",
    "evaluate_checkpoint",
    "label_ceiling",
    "load_partition",
    "scalars",
]

BOOTSTRAPPED = ("top_5", "ifa")
"""Per-unit metrics that get a bootstrap interval over units (the protocol's §6)."""

COMPARISONS = (
    ("cross_attention", "concat"),
    ("cross_attention", "llm_only"),
    ("concat", "llm_only"),
)
"""The pre-stated comparisons between arms, first minus second — and no others."""

LOGGED_METRICS = {
    "top_1": ("line/top_1_accuracy", True),
    "top_5": ("line/top_5_accuracy", True),
    "ifa": ("line/ifa_mean", False),
}
"""``metric -> (name the scoring pass logged it under, higher is better)``."""


@dataclass(frozen=True)
class PartitionUnit:
    """What the partition itself says about one line-labelled unit.

    Attributes:
        lines: The unit's lines, as the scored model saw them.
        flagged: 1-indexed numbers of the flagged lines — all of them, including any past
            the line cap that a score file never holds.
    """

    lines: tuple[str, ...]
    flagged: frozenset[int]


def load_partition(
    cache_dir: Path, split: str, *, normalize: bool, strip: bool
) -> dict[str, PartitionUnit]:
    """The partition's line-labelled units: their text and every flagged line number.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: The partition.
        normalize: Apply ADR-006 whitespace normalization, as every reported run did.
        strip: Strip comments first (ADR-010), as the scores being evaluated were made.

    Returns:
        ``unit_id -> PartitionUnit`` for every unit with at least one flagged line.
    """
    units: dict[str, PartitionUnit] = {}
    for unit in iter_shard(cache_dir, split):
        if unit.vulnerable_lines:
            source = strip_comments(unit.source) if strip else unit.source
            source = normalize_source_whitespace(source) if normalize else source
            units[unit.unit_id] = PartitionUnit(
                lines=tuple(split_lines(source)), flagged=frozenset(unit.vulnerable_lines)
            )
    return units


def arm_name(manifest: Mapping[str, Any]) -> str:
    """The group a checkpoint's seeds are averaged in.

    Args:
        manifest: A score file's manifest.

    Returns:
        The fusion kind, with the attention direction when the run named one, ``_frozen``
        for a frozen encoder and ``_retrained`` for a run trained on comment-stripped
        text. The same rule as ``scripts/audit_score_leakage.py``: a frozen run is never
        averaged with a fine-tuned one, nor a model that never saw a comment with one that
        was only evaluated without them.
    """
    direction = str(manifest.get("fusion_direction") or "")
    return (
        f"{manifest['fusion_kind']}"
        f"{'_' + direction if direction else ''}"
        f"{'_frozen' if manifest['frozen'] else ''}"
        f"{'_retrained' if manifest.get('strip_comments_at_training') else ''}"
    )


def _blank_mask(unit: UnitScores, partition: Mapping[str, PartitionUnit]) -> np.ndarray:
    """``[n_lines]`` bool: which of the unit's scored lines are blank in the text."""
    entry = partition.get(unit.unit_id)
    n = int(unit.logits.shape[0])
    if entry is None or len(entry.lines) < n:
        raise DataError(
            "a scored unit does not match the partition's text",
            unit_id=unit.unit_id,
            scored_lines=n,
            text_lines=None if entry is None else len(entry.lines),
        )
    return np.array([not text.strip() for text in entry.lines[:n]], dtype=bool)


def _ranking(
    units: Sequence[UnitScores],
    *,
    source_order: bool,
    withheld: Mapping[str, np.ndarray] | None,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """The ranking metrics of one set of units, under each tie order.

    Args:
        units: The checkpoint's units.
        source_order: Rank by position instead of by the model's score — the rule-only
            baseline, on the same units.
        withheld: ``unit_id -> [n_lines] bool`` of lines removed from targets and
            candidates; ``None`` to rank every line.
        resamples: Bootstrap resamples; ``0`` skips the intervals.
        bootstrap_seed: Seed of the bootstrap.

    Returns:
        ``{"units", "tied_units", "metrics": {name: {tie_order: mean}}, "ifa_median",
        "bootstrap_95": {name: [low, high]}}``.
    """
    per_unit: dict[str, dict[str, list[float]]] = {}
    for unit in units:
        n = int(unit.logits.shape[0])
        scores = -np.arange(n, dtype=np.float64) if source_order else unit.logits
        candidates = None if withheld is None else ~withheld[unit.unit_id]
        values = unit_ranking_metrics(scores, unit.labels, candidates)
        if values is None:
            continue
        for name, by_tie in values.items():
            bucket = per_unit.setdefault(name, {tie: [] for tie in TIE_ORDERS})
            for tie in TIE_ORDERS:
                bucket[tie].append(by_tie[tie])
    if not per_unit:
        raise DataError("no unit has a flagged line among its candidates", units=len(units))

    ifa = per_unit["ifa"]
    report: dict[str, Any] = {
        "units": len(ifa["source"]),
        # A unit is tied when the order of equal scores decides where its first flagged
        # line lands — the units behind the gap between the two bounds.
        "tied_units": int(np.sum(np.asarray(ifa["best"]) != np.asarray(ifa["worst"]))),
        "metrics": {
            name: {tie: float(np.mean(values[tie])) for tie in TIE_ORDERS}
            for name, values in sorted(per_unit.items())
        },
        "ifa_median": float(np.median(ifa["source"])),
    }
    if resamples:
        report["bootstrap_95"] = {
            name: list(
                bootstrap_mean_interval(
                    per_unit[name]["source"], resamples=resamples, seed=bootstrap_seed
                )
            )
            for name in BOOTSTRAPPED
        }
    return report


def _pooled(
    units: Sequence[UnitScores], *, calibrator: LogitCalibrator, threshold: float, n_bins: int
) -> dict[str, Any]:
    """Threshold-free and thresholded metrics over the lines the model could score."""
    logits = np.concatenate([unit.logits[unit.scoreable] for unit in units]).astype(np.float64)
    labels = np.concatenate([unit.labels[unit.scoreable] for unit in units])
    probabilities = calibrator.probabilities(torch.as_tensor(logits, dtype=torch.float64))
    bins = reliability_bins(probabilities, torch.as_tensor(labels), n_bins=n_bins)
    flagged = int((labels > 0).sum())
    return {
        "lines": int(labels.shape[0]),
        "flagged_lines": flagged,
        # What a ranker with no information scores on average precision.
        "base_rate": flagged / int(labels.shape[0]),
        # On logits, not probabilities: the calibrator is monotone, so the value is the
        # same, and it then cannot depend on a calibrator at all.
        "pr_auc": average_precision(logits, labels),
        "calibrator": calibrator.to_dict(),
        "threshold": threshold,
        **precision_recall_f1(probabilities.numpy(), labels, threshold),
        "ece": expected_calibration_error(bins),
        "reliability": [
            {
                "lower": calibration_bin.lower,
                "upper": calibration_bin.upper,
                "count": calibration_bin.count,
                "mean_probability": calibration_bin.mean_probability,
                "positive_rate": calibration_bin.positive_rate,
            }
            for calibration_bin in bins
        ],
    }


def evaluate_checkpoint(
    units: Sequence[UnitScores],
    *,
    calibrator: LogitCalibrator,
    threshold: float,
    partition: Mapping[str, PartitionUnit] | None = None,
    n_bins: int = 15,
    resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Everything the protocol reports for one checkpoint on one partition.

    Args:
        units: The checkpoint's stored line scores.
        calibrator: Its validation-fitted calibrator.
        threshold: Its validation-chosen threshold, on calibrated probabilities.
        partition: The partition's text and labels. With it the report gains the
            blank-lines-withheld reading; without it only the primary reading is made.
        n_bins: Reliability-curve bins.
        resamples: Bootstrap resamples over units; ``0`` skips the intervals.
        bootstrap_seed: Seed of the bootstrap.

    Returns:
        ``{"ranking", "source_order", "pooled", "coverage"}``. ``ranking`` and
        ``source_order`` each hold ``all_lines`` and, with a partition,
        ``blank_lines_withheld``.

    Raises:
        DataError: If there are no units, or a scored unit does not match the partition.
    """
    if not units:
        raise DataError("no units to evaluate")
    readings: dict[str, Mapping[str, np.ndarray] | None] = {"all_lines": None}
    coverage: dict[str, Any] = {
        "units": len(units),
        "flagged_lines": int(sum(int((unit.labels > 0).sum()) for unit in units)),
        "flagged_lines_without_features": int(
            sum(int(((unit.labels > 0) & ~unit.scoreable).sum()) for unit in units)
        ),
    }
    if partition is not None:
        blank = {unit.unit_id: _blank_mask(unit, partition) for unit in units}
        readings["blank_lines_withheld"] = blank
        coverage["flagged_lines_blank"] = int(
            sum(int(((unit.labels > 0) & blank[unit.unit_id]).sum()) for unit in units)
        )
        coverage["units_whose_flagged_lines_are_all_blank"] = int(
            sum(
                bool((unit.labels > 0).any())
                and not bool(((unit.labels > 0) & ~blank[unit.unit_id]).any())
                for unit in units
            )
        )

    def rank(*, source_order: bool) -> dict[str, Any]:
        return {
            reading: _ranking(
                units,
                source_order=source_order,
                withheld=withheld,
                # The baseline is a rule, not a sample of anything a seed could change;
                # its interval would only measure the units, which the model's does too.
                resamples=0 if source_order else resamples,
                bootstrap_seed=bootstrap_seed,
            )
            for reading, withheld in readings.items()
        }

    ranking = rank(source_order=False)
    coverage["units_ranked"] = ranking["all_lines"]["units"]
    return {
        "ranking": ranking,
        "source_order": rank(source_order=True),
        "pooled": _pooled(units, calibrator=calibrator, threshold=threshold, n_bins=n_bins),
        "coverage": coverage,
    }


def label_ceiling(
    units: Sequence[UnitScores], partition: Mapping[str, PartitionUnit]
) -> dict[str, int]:
    """What the score file cannot see: labels past the line cap, and units never scored.

    It is also the check that the score file belongs to this partition: within the lines
    it holds, its labels must be exactly the partition's.

    Args:
        units: A checkpoint's stored line scores.
        partition: The partition's text and labels.

    Returns:
        ``{"labelled_units", "labelled_units_not_scored", "flagged_lines_in_partition",
        "flagged_lines_past_the_line_cap", "units_with_no_flagged_line_inside_the_cap"}``.

    Raises:
        DataError: If a scored unit is not in the partition, or its stored labels differ
            from the partition's on the lines it holds.
    """
    past_cap = 0
    nothing_inside = 0
    for unit in units:
        entry = partition.get(unit.unit_id)
        if entry is None:
            raise DataError(
                "a scored unit is not a line-labelled unit of this partition",
                unit_id=unit.unit_id,
            )
        n = int(unit.logits.shape[0])
        inside = {number for number in entry.flagged if number <= n}
        stored = {int(position) + 1 for position in np.flatnonzero(unit.labels > 0)}
        if stored != inside:
            raise DataError(
                "a score file's labels differ from the partition's",
                unit_id=unit.unit_id,
                only_in_scores=sorted(stored - inside)[:5],
                only_in_partition=sorted(inside - stored)[:5],
            )
        past_cap += len(entry.flagged) - len(inside)
        nothing_inside += not inside
    scored = {unit.unit_id for unit in units}
    return {
        "labelled_units": len(partition),
        "labelled_units_not_scored": len(set(partition) - scored),
        "flagged_lines_in_partition": sum(len(entry.flagged) for entry in partition.values()),
        "flagged_lines_past_the_line_cap": past_cap,
        "units_with_no_flagged_line_inside_the_cap": nothing_inside,
    }


def scalars(report: Mapping[str, Any]) -> dict[str, float]:
    """One checkpoint's report as a flat ``name -> number`` mapping.

    The names are what the per-arm tables and the comparisons are keyed by.

    Args:
        report: An :func:`evaluate_checkpoint` report. If its ``pooled`` section carries
            ``parameters == "fitted_on_this_partition"``, the thresholded numbers are
            in-sample and are named ``..._in_sample``; an ``out_of_fold`` block beside
            them is flattened as ``..._out_of_fold``.

    Returns:
        ``ranking/<reading>/<metric>`` (source tie order) with ``/tie_best`` and
        ``/tie_worst`` beside each, ``source_order/<reading>/<metric>``, and
        ``pooled/<metric>``.
    """
    flat: dict[str, float] = {}
    for reading, section in report["ranking"].items():
        for name, by_tie in section["metrics"].items():
            flat[f"ranking/{reading}/{name}"] = float(by_tie["source"])
            flat[f"ranking/{reading}/{name}/tie_best"] = float(by_tie["best"])
            flat[f"ranking/{reading}/{name}/tie_worst"] = float(by_tie["worst"])
        flat[f"ranking/{reading}/ifa_median"] = float(section["ifa_median"])
    for reading, section in report["source_order"].items():
        for name, by_tie in section["metrics"].items():
            flat[f"source_order/{reading}/{name}"] = float(by_tie["source"])
    pooled = report["pooled"]
    flat["pooled/pr_auc"] = float(pooled["pr_auc"])
    flat["pooled/base_rate"] = float(pooled["base_rate"])
    in_sample = pooled.get("parameters") == "fitted_on_this_partition"
    suffix = "_in_sample" if in_sample else ""
    for name in ("precision", "recall", "f1", "ece"):
        flat[f"pooled/{name}{suffix}"] = float(pooled[name])
        if "out_of_fold" in pooled:
            flat[f"pooled/{name}_out_of_fold"] = float(pooled["out_of_fold"][name])
    return flat


def aggregate_arm(by_seed: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    """Mean and sample standard deviation over an arm's seeds, per scalar.

    Args:
        by_seed: ``seed -> scalars(report)`` for one arm.

    Returns:
        ``{"n_seeds", "seeds", "metrics": {name: {"mean", "std", "values"}}}`` with the
        values in seed order.

    Raises:
        DataError: If the arm has no seed, or its seeds do not report the same scalars.
    """
    if not by_seed:
        raise DataError("an arm needs at least one seed")
    seeds = sorted(by_seed)
    names = set(by_seed[seeds[0]])
    if any(set(by_seed[seed]) != names for seed in seeds):
        raise DataError("an arm's seeds do not report the same metrics", seeds=seeds)
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metrics": {
            name: spread([by_seed[seed][name] for seed in seeds]) for name in sorted(names)
        },
    }


def compare(
    arms: Mapping[str, Mapping[int, Mapping[str, float]]],
    kinds: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """The pre-stated comparisons, between arms that differ in fusion kind alone.

    Args:
        arms: ``arm -> seed -> scalars(report)``.
        kinds: ``arm -> fusion kind``. Two arms are compared when their kinds are one of
            :data:`COMPARISONS` and the rest of their names — frozen, retrained, attention
            direction — is the same.

    Returns:
        ``{"<first> - <second>": {metric: compare_arms(...)}}``. Tie bounds and the
        source-order baseline are not compared: neither is a property of an arm.
    """
    result: dict[str, dict[str, Any]] = {}
    for first_kind, second_kind in COMPARISONS:
        for first in sorted(arms):
            if kinds[first] != first_kind:
                continue
            second = second_kind + first[len(first_kind) :]
            if second not in arms or kinds[second] != second_kind:
                continue
            names = sorted(
                name
                for name in next(iter(arms[first].values()))
                if not name.startswith("source_order/")
                and not name.endswith(("/tie_best", "/tie_worst"))
                and name != "pooled/base_rate"
            )
            result[f"{first} - {second}"] = {
                name: compare_arms(
                    {seed: values[name] for seed, values in arms[first].items()},
                    {seed: values[name] for seed, values in arms[second].items()},
                )
                for name in names
            }
    return result
