"""Guards the model-side leak audit: the arithmetic that says how much a tell is worth.

The audit's numbers decide whether headline results stand, so its groups are pinned on a
case small enough to check by hand — one unit whose flagged line carries the tell, one
whose flagged line does not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnloc.training.line_scores import UnitScores
from vulnloc.utils.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "audit_score_leakage.py"


def _load() -> Any:
    """Import ``scripts/audit_score_leakage.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_audit_score_leakage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def lost_slash(line: str) -> bool:
    return line.endswith(" *")


def intact(line: str) -> bool:
    return line.endswith("*/")


def scores(unit_id: str, logits: list[float], flagged: set[int]) -> UnitScores:
    n = len(logits)
    return UnitScores(
        unit_id=unit_id,
        logits=np.array(logits, dtype=np.float32),
        labels=np.array([1 if line in flagged else 0 for line in range(1, n + 1)], dtype=np.int8),
        scoreable=np.ones(n, dtype=bool),
    )


TEXTS = {
    # The flagged line (2) carries the tell, and the model ranks it first.
    "marked": ["a();", "b();  /* why *", "c();  /* fine */", "d();"],
    # The flagged line (3) is plain code; the model ranks it second.
    "plain": ["e();", "f();", "g();", "h();"],
}
SCORED = [
    scores("marked", [0.1, 3.0, 0.5, 0.2], {2}),
    scores("plain", [0.1, 2.0, 1.0, 0.2], {3}),
]


def test_topk_hit_counts_only_units_with_a_usable_flagged_line():
    logits = np.array([1.0, 2.0, 3.0])
    labels = np.array([True, False, False])

    assert audit.topk_hit(logits, labels, np.array([True, True, True]), 1) == (True, False)
    assert audit.topk_hit(logits, labels, np.array([True, True, True]), 3) == (True, True)
    assert audit.topk_hit(logits, labels, np.array([False, True, True]), 1) == (False, False)


def test_a_unit_whose_flagged_line_could_not_be_scored_is_a_miss_not_a_missing_unit():
    # `training/metrics.py` ranks every line a unit has; the masked sentinel ranks last.
    # A structural-only arm has such units, and dropping them would report a Top-k the
    # run never logged.
    masked = UnitScores(
        unit_id="marked",
        logits=np.array([0.1, -1.0e4, 0.5, 0.2], dtype=np.float32),
        labels=np.array([0, 1, 0, 0], dtype=np.int8),
        scoreable=np.array([True, False, True, True]),
    )

    report = audit.audit_run([masked], TEXTS, tell=lost_slash, mirror=intact)

    assert report["all"] == {"units": 1, "top_1": 0.0, "top_5": 1.0}
    assert report["tell_units"]["units"] == 1, "the split is by text and label, not by coverage"
    assert report["top1_is_tell_line"] == 0


def test_ties_keep_source_order():
    logits = np.zeros(3)
    labels = np.array([False, True, False])

    assert audit.topk_hit(logits, labels, np.ones(3, dtype=bool), 1) == (True, False)
    assert audit.topk_hit(logits, labels, np.ones(3, dtype=bool), 2) == (True, True)


def test_units_are_split_by_whether_the_tell_points_at_a_flagged_line():
    report = audit.audit_run(SCORED, TEXTS, tell=lost_slash, mirror=intact)

    assert report["all"] == {"units": 2, "top_1": 0.5, "top_5": 1.0}
    assert report["tell_units"] == {"units": 1, "top_1": 1.0, "top_5": 1.0}
    assert report["other_units"] == {"units": 1, "top_1": 0.0, "top_5": 1.0}
    assert report["top1_is_tell_line"] == 1


def test_withholding_the_tell_lines_drops_a_unit_whose_only_flagged_line_was_one():
    report = audit.audit_run(SCORED, TEXTS, tell=lost_slash, mirror=intact)

    assert report["tell_lines_withheld"] == {"units": 1, "top_1": 0.0, "top_5": 1.0}


def test_blank_flagged_lines_are_counted_and_can_be_withheld():
    # Comment-stripped text: unit "gone" has one flagged line and it is blank.
    texts = {"gone": ["a();", "", "b();"], "kept": ["c();", "d();", ""]}
    scored = [scores("gone", [0.5, 0.1, 0.9], {2}), scores("kept", [0.2, 0.9, 0.1], {2, 3})]

    report = audit.audit_run(scored, texts, tell=lost_slash, mirror=intact)

    assert report["all"] == {"units": 2, "top_1": 0.5, "top_5": 1.0}
    assert report["blank_lines_withheld"] == {"units": 1, "top_1": 1.0, "top_5": 1.0}
    assert report["flagged_lines"] == 3
    assert report["flagged_lines_blank"] == 2
    assert report["units_whose_flagged_lines_are_all_blank"] == 1


def test_the_rule_only_baselines_never_read_a_score():
    report = audit.audit_run(SCORED, TEXTS, tell=lost_slash, mirror=intact)

    # Source order: the first line of each unit is unflagged.
    assert report["source_order"]["top_1"] == 0.0
    # The tell first: answers the marked unit, not the plain one.
    assert report["tell_first"]["top_1"] == 0.5
    assert report["tell_first_intact_last"]["top_1"] == 0.5


def test_a_tell_on_an_unflagged_line_does_not_make_a_tell_unit():
    texts = {"u": ["a();  /* why *", "b();"]}
    scored = [scores("u", [2.0, 1.0], {2})]

    report = audit.audit_run(scored, texts, tell=lost_slash, mirror=intact)

    assert report["tell_units"]["units"] == 0
    assert report["other_units"]["units"] == 1


def test_a_scored_unit_without_matching_text_is_refused():
    with pytest.raises(ConfigError, match="does not match"):
        audit.audit_run(SCORED, {"marked": TEXTS["marked"]}, tell=lost_slash, mirror=intact)


def test_aggregate_reports_mean_and_sample_standard_deviation():
    first = audit.audit_run(SCORED, TEXTS, tell=lost_slash, mirror=intact)
    second = audit.audit_run(
        [scores("marked", [5.0, 3.0, 0.5, 0.2], {2}), SCORED[1]],
        TEXTS,
        tell=lost_slash,
        mirror=intact,
    )

    summary = audit.aggregate([first, second])

    assert summary["n_seeds"] == 2
    assert summary["tell_units"]["top_1"]["mean"] == pytest.approx(0.5)
    assert summary["tell_units"]["top_1"]["std"] == pytest.approx(0.7071, abs=1e-4)
    assert summary["top1_is_tell_line"] == [1, 0]


def test_arms_keep_frozen_runs_and_attention_directions_apart():
    fine_tuned = {"fusion_kind": "cross_attention", "frozen": False}
    frozen = {"fusion_kind": "cross_attention", "frozen": True}
    one_way = {"fusion_kind": "cross_attention", "frozen": True, "fusion_direction": "sem_from_str"}

    assert audit.arm_name(fine_tuned) == "cross_attention"
    assert audit.arm_name(frozen) == "cross_attention_frozen"
    assert audit.arm_name(one_way) == "cross_attention_sem_from_str_frozen"
    assert audit.arm_name({**frozen, "fusion_direction": ""}) == "cross_attention_frozen"


def test_a_single_seed_has_no_standard_deviation():
    summary = audit.aggregate([audit.audit_run(SCORED, TEXTS, tell=lost_slash, mirror=intact)])

    assert summary["all"]["top_5"] == {"mean": 1.0, "std": None}
