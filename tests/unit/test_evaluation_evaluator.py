"""Guards the evaluator's conventions (``docs/EvaluationProtocol.md`` §5).

The metric functions are pinned in ``test_evaluation_localization.py``. What is pinned
here is which lines and which units they are given, because that is where a held-out
table can be flattered without a wrong formula anywhere: a unit the model could not score
dropped instead of counted as a miss, a blank label quietly deleted, a score file from
another partition lining up by accident, an in-sample threshold quoted as if applied.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnloc.evaluation.evaluator import (
    PartitionUnit,
    aggregate_arm,
    arm_name,
    compare,
    evaluate_checkpoint,
    label_ceiling,
    scalars,
)
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.models.heads.line_head import MASKED_LOGIT
from vulnloc.training.line_scores import UnitScores
from vulnloc.utils.exceptions import DataError

REPO_ROOT = Path(__file__).resolve().parents[2]
IDENTITY = LogitCalibrator()


def scored(
    unit_id: str,
    logits: list[float],
    flagged: tuple[int, ...] = (),
    unscoreable: tuple[int, ...] = (),
) -> UnitScores:
    """A unit's scores; ``flagged`` and ``unscoreable`` are 0-indexed line positions."""
    values = np.array(logits, dtype=np.float32)
    labels = np.zeros(len(logits), dtype=np.int8)
    labels[list(flagged)] = 1
    scoreable = np.ones(len(logits), dtype=np.bool_)
    scoreable[list(unscoreable)] = False
    values[list(unscoreable)] = MASKED_LOGIT
    return UnitScores(unit_id=unit_id, logits=values, labels=labels, scoreable=scoreable)


def text(lines: list[str], flagged: set[int]) -> PartitionUnit:
    """A partition entry; ``flagged`` holds 1-indexed line numbers, as the corpus does."""
    return PartitionUnit(lines=tuple(lines), flagged=frozenset(flagged))


def run(units: list[UnitScores], **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("calibrator", IDENTITY)
    kwargs.setdefault("threshold", 0.5)
    kwargs.setdefault("resamples", 0)
    return evaluate_checkpoint(units, **kwargs)


def _load_audit() -> Any:
    """Import ``scripts/audit_score_leakage.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_audit_score_leakage_for_evaluator",
        REPO_ROOT / "scripts" / "audit_score_leakage.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- which units and which lines ------------------------------------------------------


def test_a_unit_whose_flagged_line_could_not_be_scored_is_a_miss_not_a_missing_unit():
    """It ranks last behind the sentinel — what ``training/metrics.py`` logs for it."""
    units = [
        scored("found", [3.0, 1.0, 2.0], flagged=(0,)),
        scored("unscoreable", [1.0, 2.0, 0.0], flagged=(2,), unscoreable=(2,)),
    ]

    report = run(units)

    section = report["ranking"]["all_lines"]
    assert section["units"] == 2
    assert section["metrics"]["top_1"]["source"] == 0.5
    assert section["metrics"]["ifa"]["source"] == 1.0  # 0 and 2
    assert report["coverage"]["flagged_lines"] == 2
    assert report["coverage"]["flagged_lines_without_features"] == 1


def test_pooled_metrics_use_only_the_lines_the_model_could_score():
    units = [
        scored("found", [3.0, 1.0, 2.0], flagged=(0,)),
        scored("unscoreable", [1.0, 2.0, 0.0], flagged=(2,), unscoreable=(2,)),
    ]

    pooled = run(units)["pooled"]

    assert pooled["lines"] == 5
    assert pooled["flagged_lines"] == 1
    assert pooled["base_rate"] == pytest.approx(0.2)
    assert pooled["pr_auc"] == 1.0  # the one scoreable flagged line is ranked first
    # Every sigmoid(logit) here is above 0.5: five predictions, one of them right.
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (1.0, 4.0, 0.0)
    assert sum(calibration_bin["count"] for calibration_bin in pooled["reliability"]) == 5


def test_a_unit_with_nothing_flagged_is_scored_but_not_ranked():
    units = [scored("a", [2.0, 1.0], flagged=(0,)), scored("clean", [2.0, 1.0])]

    report = run(units)

    assert report["coverage"]["units"] == 2
    assert report["coverage"]["units_ranked"] == 1
    assert report["ranking"]["all_lines"]["metrics"]["top_1"]["source"] == 1.0


def test_nothing_to_rank_at_all_is_an_error_not_a_table_of_zeros():
    with pytest.raises(DataError, match="no unit has a flagged line"):
        run([scored("clean", [2.0, 1.0])])


def test_no_units_is_an_error():
    with pytest.raises(DataError, match="no units"):
        run([])


def test_the_tie_bounds_reach_the_report():
    report = run([scored("tied", [1.0, 1.0], flagged=(1,))])

    section = report["ranking"]["all_lines"]
    assert section["tied_units"] == 1
    assert section["metrics"]["top_1"] == {"source": 0.0, "best": 1.0, "worst": 0.0}


def test_the_threshold_is_applied_to_calibrated_probabilities():
    units = [scored("a", [2.0, 0.0, -2.0], flagged=(0,))]
    calibrator = LogitCalibrator(scale=1.0, shift=-1.0, kind="platt")  # 0.73, 0.27, 0.05

    pooled = run(units, calibrator=calibrator, threshold=0.5)["pooled"]

    assert (pooled["tp"], pooled["fp"]) == (1.0, 0.0)
    assert pooled["calibrator"] == {"kind": "platt", "scale": 1.0, "shift": -1.0}


def test_pr_auc_cannot_be_moved_by_a_calibrator():
    units = [
        scored("a", [2.0, 0.5, -1.0, 0.1], flagged=(1,)),
        scored("b", [0.3, 0.9], flagged=(1,)),
    ]

    plain = run(units)["pooled"]["pr_auc"]
    calibrated = run(units, calibrator=LogitCalibrator(scale=0.3, shift=-4.0, kind="platt"))[
        "pooled"
    ]["pr_auc"]

    assert plain == calibrated


# --- blank lines and the partition's text ------------------------------------------------


def test_blank_flagged_lines_stay_targets_and_are_also_reported_withheld():
    """Labels are not edited: the primary reading keeps the blank line as a miss."""
    units = [
        scored("only_blank", [1.0, 5.0, 2.0], flagged=(1,)),
        scored("kept", [1.0, 3.0], flagged=(1,)),
    ]
    partition = {
        "only_blank": text(["int x;", "", "return x;"], {2}),
        "kept": text(["int y;", "free(y);"], {2}),
    }

    report = run(units, partition=partition)

    assert report["ranking"]["all_lines"]["units"] == 2
    assert report["ranking"]["blank_lines_withheld"]["units"] == 1
    assert report["coverage"]["flagged_lines_blank"] == 1
    assert report["coverage"]["units_whose_flagged_lines_are_all_blank"] == 1


def test_withholding_a_blank_clean_line_removes_it_from_the_candidates_too():
    units = [scored("a", [5.0, 4.0, 1.0], flagged=(1,))]
    partition = {"a": text(["", "strcpy(a, b);", "}"], {2})}

    report = run(units, partition=partition)

    assert report["ranking"]["all_lines"]["metrics"]["ifa"]["source"] == 1.0
    assert report["ranking"]["blank_lines_withheld"]["metrics"]["ifa"]["source"] == 0.0


def test_without_the_partition_only_the_primary_reading_is_made():
    report = run([scored("a", [2.0, 1.0], flagged=(0,))])

    assert list(report["ranking"]) == ["all_lines"]
    assert "flagged_lines_blank" not in report["coverage"]


@pytest.mark.parametrize(
    "partition",
    [{}, {"a": PartitionUnit(lines=("only one line",), flagged=frozenset({1}))}],
    ids=["unit missing", "text shorter than the scores"],
)
def test_a_scored_unit_that_does_not_match_the_text_is_refused(partition):
    with pytest.raises(DataError, match="does not match the partition's text"):
        run([scored("a", [2.0, 1.0], flagged=(0,))], partition=partition)


def test_the_source_order_baseline_never_reads_a_score():
    low = run([scored("a", [0.0, 0.0, 9.0], flagged=(2,))])["source_order"]["all_lines"]
    high = run([scored("a", [9.0, 9.0, 0.0], flagged=(2,))])["source_order"]["all_lines"]

    assert low["metrics"] == high["metrics"]
    assert low["metrics"]["ifa"]["source"] == 2.0
    assert "bootstrap_95" not in low


def test_the_model_gets_a_bootstrap_interval_for_the_headline_metrics():
    units = [scored(f"u{index}", [float(index % 3), 1.0, 0.5], flagged=(0,)) for index in range(30)]

    section = run(units, resamples=300)["ranking"]["all_lines"]

    assert sorted(section["bootstrap_95"]) == ["ifa", "top_5"]
    low, high = section["bootstrap_95"]["ifa"]
    assert low <= section["metrics"]["ifa"]["source"] <= high


# --- what a score file cannot see ---------------------------------------------------------


def test_labels_past_the_line_cap_are_counted_as_a_ceiling():
    units = [scored("capped", [1.0, 2.0, 3.0], flagged=(1,)), scored("all_past", [1.0, 2.0])]
    partition = {
        "capped": text(["a", "b", "c", "d", "e"], {2, 5}),
        "all_past": text(["a", "b", "c"], {3}),
        "never_scored": text(["a"], {1}),
    }

    ceiling = label_ceiling(units, partition)

    assert ceiling == {
        "labelled_units": 3,
        "labelled_units_not_scored": 1,
        "flagged_lines_in_partition": 4,
        "flagged_lines_past_the_line_cap": 2,
        "units_with_no_flagged_line_inside_the_cap": 1,
    }


def test_a_score_file_whose_labels_are_not_the_partitions_is_refused():
    """The check that the scores belong to this partition and this label set."""
    units = [scored("a", [1.0, 2.0, 3.0], flagged=(0,))]

    with pytest.raises(DataError, match="labels differ"):
        label_ceiling(units, {"a": text(["a", "b", "c"], {2})})


def test_a_scored_unit_that_is_not_in_the_partition_is_refused():
    with pytest.raises(DataError, match="not a line-labelled unit"):
        label_ceiling([scored("stranger", [1.0], flagged=(0,))], {})


# --- arms, seeds and comparisons ------------------------------------------------------------


def manifest(**overrides: Any) -> dict[str, Any]:
    base = {"fusion_kind": "cross_attention", "frozen": False}
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"frozen": True},
        {"fusion_direction": "sem_from_str", "frozen": True},
        {"strip_comments_at_training": True},
        {"fusion_kind": "llm_only", "fusion_direction": "", "strip_comments_at_training": False},
    ],
)
def test_arms_are_named_exactly_as_the_score_audit_names_them(fields):
    """One rule for which checkpoints are seeds of each other, in both instruments."""
    audit = _load_audit()

    assert arm_name(manifest(**fields)) == audit.arm_name(manifest(**fields))


def test_a_retrained_checkpoint_is_its_own_arm():
    assert arm_name(manifest(strip_comments_at_training=True)) == "cross_attention_retrained"
    assert arm_name(manifest()) == "cross_attention"


def test_on_validation_the_thresholded_numbers_are_named_in_sample():
    report = run([scored("a", [2.0, 1.0], flagged=(0,))])
    report["pooled"]["parameters"] = "fitted_on_this_partition"
    report["pooled"]["out_of_fold"] = {"precision": 0.1, "recall": 0.2, "f1": 0.3, "ece": 0.4}

    flat = scalars(report)

    assert "pooled/f1" not in flat
    assert flat["pooled/f1_in_sample"] == report["pooled"]["f1"]
    assert flat["pooled/f1_out_of_fold"] == 0.3
    assert flat["pooled/ece_out_of_fold"] == 0.4


def test_on_a_held_out_partition_they_are_the_applied_numbers():
    report = run([scored("a", [2.0, 1.0], flagged=(0,))])
    report["pooled"]["parameters"] = "applied_from_validation"

    flat = scalars(report)

    assert flat["pooled/f1"] == report["pooled"]["f1"]
    assert not any(name.endswith(("_in_sample", "_out_of_fold")) for name in flat)


def test_scalars_carry_the_tie_bounds_and_the_baseline_beside_each_metric():
    flat = scalars(run([scored("tied", [1.0, 1.0], flagged=(1,))]))

    assert flat["ranking/all_lines/top_1"] == 0.0
    assert flat["ranking/all_lines/top_1/tie_best"] == 1.0
    assert flat["ranking/all_lines/top_1/tie_worst"] == 0.0
    assert flat["source_order/all_lines/top_1"] == 0.0
    assert flat["pooled/pr_auc"] == 0.5


def test_an_arm_is_its_mean_and_sample_sd_over_seeds():
    summary = aggregate_arm({42: {"m": 1.0}, 1: {"m": 2.0}, 2: {"m": 3.0}})

    assert summary["n_seeds"] == 3
    assert summary["seeds"] == [1, 2, 42]
    assert summary["metrics"]["m"] == {"mean": 2.0, "std": 1.0, "values": [2.0, 3.0, 1.0]}


def test_seeds_that_report_different_metrics_are_not_one_arm():
    with pytest.raises(DataError, match="same metrics"):
        aggregate_arm({42: {"m": 1.0}, 1: {"other": 2.0}})


def test_only_the_pre_stated_comparisons_are_made_and_only_within_one_population():
    values = {
        "ranking/all_lines/top_5": 0.5,
        "ranking/all_lines/top_5/tie_best": 0.5,
        "source_order/all_lines/top_5": 0.3,
        "pooled/base_rate": 0.07,
        "pooled/pr_auc": 0.2,
    }
    seeds = {42: values, 1: values}
    arms = {
        "cross_attention_retrained": seeds,
        "concat_retrained": seeds,
        "llm_only_retrained": seeds,
        "concat": seeds,  # trained with comments: a different population
        "gated_retrained": seeds,  # not in a pre-stated comparison
    }
    kinds = {
        "cross_attention_retrained": "cross_attention",
        "concat_retrained": "concat",
        "llm_only_retrained": "llm_only",
        "concat": "concat",
        "gated_retrained": "gated",
    }

    result = compare(arms, kinds)

    assert sorted(result) == [
        "concat_retrained - llm_only_retrained",
        "cross_attention_retrained - concat_retrained",
        "cross_attention_retrained - llm_only_retrained",
    ]
    # Neither a tie bound nor the baseline nor the base rate is a property of an arm.
    assert sorted(result["cross_attention_retrained - concat_retrained"]) == [
        "pooled/pr_auc",
        "ranking/all_lines/top_5",
    ]
