"""Guards ``scripts/evaluate_scores.py`` — the one place a results table may come from.

The metrics and the evaluator have their own files. What is pinned here is the script's
refusals, because each one stands where a table could otherwise be written from the wrong
inputs: a held-out partition read early, scores made from the other text, a threshold that
was not chosen on validation, an arm one seed short, numbers that do not reconcile with
the scoring pass they came from.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import open_dict

from vulnloc.data.cache import write_shards
from vulnloc.data.labeling import derive_line_labels
from vulnloc.evaluation.evaluator import evaluate_checkpoint
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.training.line_scores import UnitScores, save_scores
from vulnloc.utils.exceptions import ConfigError, DataError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "evaluate_scores.py"

SOURCE = "int f(int n) {\nchar buf[8];\nstrcpy(buf, n);\nreturn 0;\n}"
UNIT_IDS = [f"u{index}" for index in range(8)]


def _load() -> Any:
    """Import ``scripts/evaluate_scores.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_evaluate_scores", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate_scores = _load()


# --- refusals that need no files --------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "clean_eval"])
def test_a_held_out_partition_is_refused_until_the_caller_says_final(split):
    with pytest.raises(ConfigError, match="once, at the end"):
        evaluate_scores.require_final(split, final=False)

    evaluate_scores.require_final(split, final=True)


def test_validation_needs_no_such_statement():
    evaluate_scores.require_final("val", final=False)


def test_scores_made_from_the_other_text_variant_are_refused():
    manifests = [{"source_run": "a" * 32, "strip_comments": True}, {"source_run": "b" * 32}]

    with pytest.raises(ConfigError, match="other text variant") as stripped:
        evaluate_scores.refuse_other_text(manifests, strip=True)
    with pytest.raises(ConfigError, match="other text variant") as shipped:
        evaluate_scores.refuse_other_text(manifests, strip=False)

    # A manifest written before the switch existed was made with comments.
    assert stripped.value.details["runs"] == ["b" * 32]
    assert shipped.value.details["runs"] == ["a" * 32]


# --- calibration reports ------------------------------------------------------------------


def write_report(path: Path, runs: list[str], *, split: str = "val") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "calibrator": {"kind": "platt", "scale": 1.0, "shift": 0.0},
        "threshold": 0.5,
        "out_of_fold": {"precision": 0.11, "recall": 0.22, "f1": 0.33, "ece": 0.044},
    }
    path.write_text(
        json.dumps({"split": split, "per_run": dict.fromkeys(runs, entry)}), encoding="utf-8"
    )


def test_calibrations_are_collected_from_every_report_under_the_directory(tmp_path):
    write_report(tmp_path / "calibration_report.json", ["a"])
    write_report(tmp_path / "arms" / "concat" / "calibration_report.json", ["b", "c"])

    assert sorted(evaluate_scores.load_calibrations(tmp_path)) == ["a", "b", "c"]


def test_no_calibration_report_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="no calibration report"):
        evaluate_scores.load_calibrations(tmp_path)


def test_a_threshold_chosen_on_another_partition_is_refused(tmp_path):
    write_report(tmp_path / "calibration_report.json", ["a"], split="test")

    with pytest.raises(ConfigError, match="not fitted on validation"):
        evaluate_scores.load_calibrations(tmp_path)


def test_two_reports_covering_one_run_are_refused(tmp_path):
    """Which threshold applies would otherwise be decided by file order."""
    write_report(tmp_path / "calibration_report.json", ["a"])
    write_report(tmp_path / "stripped" / "calibration_report.json", ["a"])

    with pytest.raises(ConfigError, match="two calibration reports"):
        evaluate_scores.load_calibrations(tmp_path)


# --- reconciling with the scoring pass -------------------------------------------------


def tied_report() -> dict[str, Any]:
    """One unit whose flagged line is tied with a clean one: Top-1 is 0 or 1 by tie order."""
    unit = UnitScores(
        unit_id="tied",
        logits=np.array([1.0, 1.0], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int8),
        scoreable=np.ones(2, dtype=np.bool_),
    )
    return evaluate_checkpoint([unit], calibrator=LogitCalibrator(), threshold=0.5, resamples=0)


@pytest.mark.parametrize("top_1", [0.0, 1.0])
def test_a_logged_value_anywhere_inside_the_tie_bounds_reconciles(top_1):
    logged = {
        "line/top_1_accuracy": top_1,
        "line/top_5_accuracy": 1.0,
        "line/ifa_mean": 1.0 - top_1,
        "line/scored_units": 1.0,
    }

    assert evaluate_scores.reconcile(tied_report(), logged) == []


def test_a_logged_value_outside_the_tie_bounds_does_not():
    logged = {"line/top_1_accuracy": 0.0, "line/top_5_accuracy": 0.5, "line/ifa_mean": 3.0}

    problems = evaluate_scores.reconcile(tied_report(), logged)

    assert [problem.split(":")[0] for problem in problems] == ["top_5", "ifa"]


def test_a_different_number_of_ranked_units_does_not_reconcile():
    logged = {
        "line/top_1_accuracy": 0.0,
        "line/top_5_accuracy": 1.0,
        "line/ifa_mean": 1.0,
        "line/scored_units": 2.0,
    }

    assert evaluate_scores.reconcile(tied_report(), logged) == ["units ranked: 1 here, 2 logged"]


def test_a_manifest_without_the_metric_is_a_disagreement_not_a_pass():
    problems = evaluate_scores.reconcile(tied_report(), {})

    assert len(problems) == 3 and all("the manifest has no" in problem for problem in problems)


# --- end to end, on a partition of eight five-line functions --------------------------------


def build_inputs(
    tmp_path: Path, *, arms: dict[str, list[int]], tamper: str | None = None
) -> dict[str, str]:
    """Write a shard, score files with manifests and a calibration report under ``tmp_path``.

    Every unit is :data:`SOURCE` with its third line flagged. Each checkpoint's manifest
    logs exactly what the evaluator computes for it, unless ``tamper`` names a run prefix
    whose logged Top-5 is to be made impossible.
    """
    cache_dir = tmp_path / "cache"
    units = [
        derive_line_labels(
            unit_id=unit_id,
            source=SOURCE,
            target=1,
            flaw_line_index="2",
            flaw_line="strcpy(buf, n);",
            metadata={"project": "p", "CVE ID": "CVE-0", "commit_id": None},
        )
        for unit_id in UNIT_IDS
    ]
    write_shards(units, dict.fromkeys(UNIT_IDS, "val"), cache_dir)

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    runs: list[str] = []
    generator = np.random.default_rng(0)
    for kind, seeds in arms.items():
        for seed in seeds:
            run_id = f"{len(runs):032x}"
            runs.append(run_id)
            scored = []
            for unit_id in UNIT_IDS:
                labels = np.zeros(5, dtype=np.int8)
                labels[2] = 1
                scored.append(
                    UnitScores(
                        unit_id=unit_id,
                        # Half precision first, as the stored logits are: ties are possible.
                        logits=generator.normal(size=5).astype(np.float16).astype(np.float32),
                        labels=labels,
                        scoreable=np.ones(5, dtype=np.bool_),
                    )
                )
            save_scores(scores_dir / f"{run_id}_val.npz", scored)
            section = evaluate_checkpoint(
                scored, calibrator=LogitCalibrator(), threshold=0.5, resamples=0
            )["ranking"]["all_lines"]
            logged = {
                "line/top_1_accuracy": section["metrics"]["top_1"]["source"],
                "line/top_5_accuracy": section["metrics"]["top_5"]["source"],
                "line/ifa_mean": section["metrics"]["ifa"]["source"],
                "line/scored_units": float(section["units"]),
            }
            if tamper is not None and run_id.startswith(tamper):
                logged["line/top_5_accuracy"] = 0.0  # every unit has five lines: Top-5 is 1
            manifest = {
                "source_run": run_id,
                "seed": seed,
                "fusion_kind": kind,
                "fusion_direction": "",
                "frozen": False,
                "split": "val",
                "metrics": logged,
            }
            (scores_dir / f"{run_id}_val.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
    write_report(tmp_path / "calibration" / "calibration_report.json", runs)
    return {
        "cache": str(cache_dir),
        "scores_dir": str(scores_dir),
        "calibration_dir": str(tmp_path / "calibration"),
        "output_dir": str(tmp_path / "phase8"),
    }


def configure(isolated_cfg: Any, paths: dict[str, str], **options: Any) -> Any:
    cfg = isolated_cfg()
    with open_dict(cfg):
        cfg.data.cache.output_dir = paths["cache"]
        cfg.evaluate = {
            "scores_dir": paths["scores_dir"],
            "calibration_dir": paths["calibration_dir"],
            "output_dir": paths["output_dir"],
            "resamples": 200,
            **options,
        }
    return cfg


def test_the_report_and_its_tables_are_written_for_every_arm(tmp_path, isolated_cfg):
    paths = build_inputs(tmp_path, arms={"cross_attention": [42, 1, 2], "concat": [42, 1, 2]})

    payload = evaluate_scores.evaluate(configure(isolated_cfg, paths))

    assert sorted(payload["arms"]) == ["concat", "cross_attention"]
    assert all(arm["n_seeds"] == 3 for arm in payload["arms"].values())
    assert list(payload["comparisons"]) == ["cross_attention - concat"]
    assert len(payload["runs"]) == 6
    assert len(payload["run_id"]) == 32
    # Five lines a unit: any ranking finds the flagged line within five.
    assert payload["arms"]["concat"]["metrics"]["ranking/all_lines/top_5"]["mean"] == 1.0
    # On validation the thresholded numbers of record are the calibration's out-of-fold ones.
    out_of_fold = payload["arms"]["concat"]["metrics"]["pooled/f1_out_of_fold"]["mean"]
    assert out_of_fold == pytest.approx(0.33)
    assert "pooled/f1" not in payload["arms"]["concat"]["metrics"]
    first = next(iter(payload["runs"].values()))
    assert first["coverage"]["labelled_units"] == len(UNIT_IDS)
    assert first["coverage"]["flagged_lines_past_the_line_cap"] == 0

    written = json.loads(
        (Path(paths["output_dir"]) / "evaluation_val.json").read_text(encoding="utf-8")
    )
    assert written["run_id"] == payload["run_id"]
    tables = (Path(paths["output_dir"]) / "evaluation_val.md").read_text(encoding="utf-8")
    assert "Line F1 (out of fold)" in tables
    assert "cross_attention - concat" in tables
    assert payload["run_id"] in tables


def test_numbers_that_do_not_reconcile_with_the_scoring_pass_are_not_written(
    tmp_path, isolated_cfg
):
    paths = build_inputs(tmp_path, arms={"concat": [42, 1, 2]}, tamper="0" * 31 + "1")

    with pytest.raises(DataError, match="does not reconcile"):
        evaluate_scores.evaluate(configure(isolated_cfg, paths))

    assert not (Path(paths["output_dir"]) / "evaluation_val.json").exists()


def test_an_arm_one_seed_short_is_refused(tmp_path, isolated_cfg):
    paths = build_inputs(tmp_path, arms={"cross_attention": [42, 1, 2], "concat": [42, 1]})

    with pytest.raises(ConfigError, match="required number of seeds") as refusal:
        evaluate_scores.evaluate(configure(isolated_cfg, paths))

    assert refusal.value.details["arms"] == {"concat": [1, 42]}
    assert not (Path(paths["output_dir"]) / "evaluation_val.json").exists()


def test_fewer_seeds_can_be_evaluated_only_by_saying_so(tmp_path, isolated_cfg):
    paths = build_inputs(tmp_path, arms={"concat": [42]})

    payload = evaluate_scores.evaluate(configure(isolated_cfg, paths, require_seeds=1))

    assert payload["arms"]["concat"]["n_seeds"] == 1
    assert payload["arms"]["concat"]["metrics"]["pooled/pr_auc"]["std"] is None


def test_a_run_no_calibration_report_covers_is_refused(tmp_path, isolated_cfg):
    paths = build_inputs(tmp_path, arms={"concat": [42, 1, 2]})
    write_report(Path(paths["calibration_dir"]) / "calibration_report.json", ["someone-else"])

    with pytest.raises(ConfigError, match="no calibration report covers this run"):
        evaluate_scores.evaluate(configure(isolated_cfg, paths))


def test_the_calibration_directory_has_no_default(tmp_path, isolated_cfg):
    paths = build_inputs(tmp_path, arms={"concat": [42, 1, 2]})
    cfg = configure(isolated_cfg, paths)
    with open_dict(cfg):
        del cfg.evaluate["calibration_dir"]

    with pytest.raises(ConfigError, match="no calibration to apply"):
        evaluate_scores.evaluate(cfg)


def test_the_held_out_tables_name_the_applied_numbers(tmp_path):
    """No model, no files: render a held-out payload and read its column headers."""
    unit = UnitScores(
        unit_id="a",
        logits=np.array([2.0, 1.0, 0.0], dtype=np.float32),
        labels=np.array([1, 0, 0], dtype=np.int8),
        scoreable=np.ones(3, dtype=np.bool_),
    )
    report = evaluate_checkpoint([unit], calibrator=LogitCalibrator(), threshold=0.5, resamples=0)
    report["pooled"]["parameters"] = "applied_from_validation"
    report.update({"arm": "concat", "seed": 42})
    flat = evaluate_scores.scalars(report)
    payload = {
        "split": "test",
        "run_id": "f" * 32,
        "strip_comments": True,
        "scores_dir": "scores",
        "arms": {"concat": evaluate_scores.aggregate_arm({42: flat})},
        "comparisons": {},
        "runs": {"a" * 32: report},
    }

    tables = evaluate_scores.render_markdown(payload)

    assert "# Line-level evaluation — `test`" in tables
    assert "| Line F1 |" in tables and "out of fold" not in tables
    assert "Pre-stated comparisons" not in tables
