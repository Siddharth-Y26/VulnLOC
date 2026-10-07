"""The calibration report's procedure, on scores small or synthetic enough to check.

The math is pinned in ``test_calibration.py``. What is pinned here is the procedure around
it, which is where the honest-versus-flattering distinction lives: which lines are pooled,
that folds are by unit and reproducible, that the numbers called out-of-fold really come
from units the parameters never saw, and that no partition but validation is accepted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vulnloc.training.line_scores import UnitScores
from vulnloc.utils.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "calibrate.py"


def _load() -> Any:
    """Import ``scripts/calibrate.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_calibrate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calibrate = _load()


def _units(n_units: int = 120, lines: int = 40, *, scale: float = 0.5, shift: float = -2.0):
    """Units whose labels are drawn from ``sigmoid(scale * logit + shift)``."""
    generator = torch.Generator().manual_seed(7)
    units = []
    for index in range(n_units):
        logits = torch.randn(lines, generator=generator, dtype=torch.float64) * 2.0
        flagged = torch.rand(lines, generator=generator, dtype=torch.float64) < torch.sigmoid(
            scale * logits + shift
        )
        units.append(
            UnitScores(
                unit_id=f"unit-{index}",
                logits=logits.numpy().astype(np.float32),
                labels=flagged.numpy().astype(np.int8),
                scoreable=np.ones(lines, dtype=np.bool_),
            )
        )
    return units


# --- which partition, which lines, which fold -----------------------------------------


@pytest.mark.parametrize("split", ["test", "clean_eval", "train"])
def test_only_validation_may_be_calibrated_on(split):
    """A threshold chosen on the partition it is reported on is a result, not a threshold."""
    with pytest.raises(ConfigError, match="validation only"):
        calibrate.require_validation(split)


def test_validation_is_accepted():
    calibrate.require_validation("val")


def test_a_line_the_model_could_not_score_is_not_calibrated_on():
    unit = UnitScores(
        unit_id="u",
        logits=np.array([1.0, -10000.0, -2.0], dtype=np.float32),
        labels=np.array([1, 1, 0], dtype=np.int8),
        scoreable=np.array([True, False, True]),
    )

    logits, labels = calibrate.flatten([unit])

    assert logits.tolist() == [1.0, -2.0]
    assert labels.tolist() == [1.0, 0.0]


def test_a_units_fold_is_a_function_of_its_id_alone():
    ids = [f"bigvul-{index}" for index in range(500)]

    first = [calibrate.fold_of(unit_id, folds=5, seed=42) for unit_id in ids]
    again = [calibrate.fold_of(unit_id, folds=5, seed=42) for unit_id in reversed(ids)]

    assert first == list(reversed(again))
    assert set(first) == {0, 1, 2, 3, 4}
    assert min(first.count(fold) for fold in range(5)) > 60


def test_a_different_seed_is_a_different_partition():
    ids = [f"bigvul-{index}" for index in range(200)]

    one = [calibrate.fold_of(unit_id, folds=5, seed=42) for unit_id in ids]
    other = [calibrate.fold_of(unit_id, folds=5, seed=43) for unit_id in ids]

    assert one != other


# --- counting -------------------------------------------------------------------------


def test_confusion_and_prf_by_hand():
    """Scores .9(+) .6(-) .5(+) .2(+) cut at 0.5: three predicted, two right, one missed."""
    probabilities = torch.tensor([0.9, 0.6, 0.5, 0.2], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    counts = calibrate.confusion(probabilities, labels, 0.5)

    assert counts == {"tp": 2, "fp": 1, "fn": 1}
    assert calibrate.prf(counts) == pytest.approx(
        {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    )


def test_prf_with_nothing_predicted_is_zero_not_an_error():
    assert calibrate.prf({"tp": 0, "fp": 0, "fn": 5}) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


# --- the procedure --------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    return calibrate.calibrate_run(_units(), kind="platt", n_bins=15, folds=5, fold_seed=42)


def test_the_parameters_of_record_recover_the_generating_map(result):
    assert result["calibrator"]["kind"] == "platt"
    assert result["calibrator"]["scale"] == pytest.approx(0.5, abs=0.08)
    assert result["calibrator"]["shift"] == pytest.approx(-2.0, abs=0.15)
    assert (result["units"], result["lines"]) == (120, 4800)


def test_calibration_helps_out_of_fold_not_just_in_sample(result):
    """The raw sigmoid is badly off by construction; held-out units must show the repair."""
    assert result["raw"]["ece"] > 0.15
    assert result["out_of_fold"]["ece"] < 0.03
    assert result["out_of_fold"]["nll"] < result["raw"]["nll"]


def test_the_raw_view_is_the_plain_sigmoid_at_the_logged_threshold(result):
    units = _units()
    logits, labels = calibrate.flatten(units)
    expected = calibrate.confusion(torch.sigmoid(logits), labels, 0.5)

    assert {key: result["raw"][key] for key in ("tp", "fp", "fn")} == expected


def test_the_in_sample_threshold_is_at_least_as_good_as_the_logged_one(result):
    """Calibration is monotone, so the best cut-off cannot score below any fixed one."""
    assert result["in_sample"]["f1"] >= result["raw"]["f1"]


def test_every_line_is_held_out_exactly_once(result):
    held = result["out_of_fold"]["tp"] + result["out_of_fold"]["fn"]

    assert held == result["flagged_lines"]
    assert sum(b["count"] for b in result["out_of_fold"]["reliability"]) == result["lines"]
    assert len(result["out_of_fold"]["fold_thresholds"]) == 5


def test_the_procedure_is_reproducible(result):
    again = calibrate.calibrate_run(_units(), kind="platt", n_bins=15, folds=5, fold_seed=42)

    assert again == result


def test_more_folds_than_units_is_refused():
    with pytest.raises(ConfigError, match="fold is empty"):
        calibrate.calibrate_run(_units(n_units=2), kind="platt", n_bins=15, folds=5, fold_seed=42)


# --- across seeds ---------------------------------------------------------------------


def test_the_summary_reports_spread_and_claims_none_for_one_run(result):
    two = calibrate.summarise({"a": result, "b": result})
    one = calibrate.summarise({"a": result})

    assert two["threshold"]["std"] == pytest.approx(0.0)
    assert two["out_of_fold/f1"]["mean"] == pytest.approx(result["out_of_fold"]["f1"])
    assert one["threshold"]["std"] is None


def test_the_parameters_of_record_can_be_read_back(result):
    report = {"per_run": {"abc": result}}

    calibrator, threshold = calibrate.calibrator_from_report(report, "abc")

    assert calibrator.scale == result["calibrator"]["scale"]
    assert threshold == result["threshold"]
    with pytest.raises(ConfigError, match="does not cover this run"):
        calibrate.calibrator_from_report(report, "missing")
