"""The seed report is mostly refusals, so the refusals are what these tests pin.

A mean over three runs is arithmetic and cannot really break. What can break is the set of
checks deciding whether those three runs are replicates of one experiment — and every one
of those failures is silent by nature: an incomparable run folded into a mean looks exactly
like a comparable one afterwards. So each guard gets a test that the bad input raises, and
the override that exists gets a test that it still leaves a trace.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vulnloc.utils.exceptions import ConfigError

RUN_A = "a" * 32
RUN_B = "b" * 32

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_seeds.py"


def _load() -> Any:
    """Import ``scripts/report_seeds.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_report_seeds", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_seeds = _load()


def _run(
    run_id: str = RUN_A,
    *,
    seed: int = 42,
    config_hash: str = "cafe1234",
    git_sha: str = "abc1234",
    normalized: bool = True,
    overfit: bool = False,
    void: str | None = None,
    watch_metric: str = "line/top_5_accuracy",
    watch_mode: str = "max",
    checkpoint_dir: str = "./artifacts/checkpoints",
) -> SimpleNamespace:
    """A stand-in for an MLflow ``Run``, carrying only what the script reads."""
    tags = {"config_hash": config_hash, "git_sha": git_sha}
    if void is not None:
        tags["void"] = void
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=0, end_time=3_600_000),
        data=SimpleNamespace(
            params={
                "seed": str(seed),
                "normalize_whitespace": "True" if normalized else "False",
                "train.overfit": "True" if overfit else "False",
                "train.name": "semantic",
                "train.watch_metric": watch_metric,
                "train.watch_mode": watch_mode,
                "train.checkpoint_dir": checkpoint_dir,
                "train.line_labelled_only": "True",
            },
            tags=tags,
        ),
    )


class _Client:
    """A metric-history stub: ``{run_id: {metric: [(step, value), ...]}}``."""

    def __init__(self, history: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
        self._history = history

    def get_metric_history(self, run_id: str, key: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(step=step, value=value)
            for step, value in self._history.get(run_id, {}).get(key, [])
        ]


# --- the guards -------------------------------------------------------------------


def test_differing_configuration_is_refused() -> None:
    """Two runs with different hyperparameters are not two seeds of one experiment."""
    runs = [_run(RUN_A, seed=42, config_hash="one"), _run(RUN_B, seed=1, config_hash="two")]
    with pytest.raises(ConfigError, match="do not share a configuration"):
        report_seeds._check_comparable(runs)


def test_a_repeated_seed_is_refused() -> None:
    """The same seed twice is one measurement recorded twice; its spread is a fiction."""
    runs = [_run(RUN_A, seed=42), _run(RUN_B, seed=42)]
    with pytest.raises(ConfigError, match="more than once"):
        report_seeds._check_comparable(runs)


def test_un_normalized_text_is_refused_when_named_explicitly() -> None:
    """ADR-006: a run on the raw corpus is measuring the indentation, not the code."""
    runs = [_run(RUN_A, seed=42), _run(RUN_B, seed=1, normalized=False)]
    with pytest.raises(ConfigError, match="un-normalized"):
        report_seeds._check_comparable(runs)


def test_source_drift_is_refused_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model change between two seeds makes them different experiments."""
    monkeypatch.setattr(
        report_seeds,
        "_src_changed_between",
        lambda a, b: ["src/vulnloc/models/semantic/encoder.py"],
    )
    runs = [_run(RUN_A, seed=42, git_sha="aaa1111"), _run(RUN_B, seed=1, git_sha="bbb2222")]
    with pytest.raises(ConfigError, match="different source trees"):
        report_seeds._check_comparable(runs)


def test_source_drift_override_records_what_it_let_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override with no trace in the artefact is not an override, it is a hole."""
    changed = ["src/vulnloc/models/structural/encoder.py"]
    monkeypatch.setattr(report_seeds, "_src_changed_between", lambda a, b: changed)
    runs = [_run(RUN_A, seed=42, git_sha="aaa1111"), _run(RUN_B, seed=1, git_sha="bbb2222")]

    provenance = report_seeds._check_comparable(runs, allow_source_drift=True)

    assert provenance["src_vulnloc_drift_accepted"] == {"aaa1111..bbb2222": changed}
    assert provenance["src_vulnloc_compared"] == ["aaa1111..bbb2222"]


def test_identical_source_needs_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A docs commit between two seeds is not a reason to refuse them."""
    monkeypatch.setattr(report_seeds, "_src_changed_between", lambda a, b: [])
    runs = [_run(RUN_A, seed=42, git_sha="aaa1111"), _run(RUN_B, seed=1, git_sha="bbb2222")]

    provenance = report_seeds._check_comparable(runs)

    assert provenance["src_vulnloc_drift_accepted"] == {}
    assert provenance["seeds"] == [42, 1]


def test_uncommitted_changes_are_recorded_rather_than_refused() -> None:
    """A dirty tree is worth knowing about, but it is not grounds to reject the run."""
    runs = [_run(RUN_A, seed=42, git_sha="abc1234-dirty")]
    assert report_seeds._check_comparable(runs)["runs_with_uncommitted_changes"] == ["aaaaaaaa"]


# --- discovery --------------------------------------------------------------------


def test_discovery_steps_over_the_runs_that_must_never_be_averaged() -> None:
    """Void, overfit and un-normalized runs are skipped with a reason, not silently."""
    assert report_seeds._excluded_because(_run(void="ADR-006")) is not None
    assert "overfit" in (report_seeds._excluded_because(_run(overfit=True)) or "")
    assert "ADR-006" in (report_seeds._excluded_because(_run(normalized=False)) or "")
    assert report_seeds._excluded_because(_run()) is None


# --- reading the numbers ----------------------------------------------------------


def test_the_best_step_comes_from_the_run_s_own_watched_metric() -> None:
    """A run that early-stopped on its own metric must report the checkpoint it kept."""
    client = _Client(
        {
            RUN_A: {
                "eval/line/top_5_accuracy": [(0, 0.51), (1, 0.61), (2, 0.59)],
                "eval/line/f1": [(0, 0.11), (1, 0.37), (2, 0.40)],
            }
        }
    )
    step, values = report_seeds._best_step_metrics(client, _run())

    assert step == 1
    # Epoch 2 has the higher F1 and is deliberately not the one reported: the checkpoint
    # on disk is the one the watched metric chose.
    assert values["eval/line/f1"] == pytest.approx(0.37)


def test_a_minimising_watch_metric_picks_the_minimum() -> None:
    client = _Client({RUN_A: {"eval/loss": [(0, 0.9), (1, 0.2), (2, 0.5)]}})
    step, _ = report_seeds._best_step_metrics(client, _run(watch_metric="loss", watch_mode="min"))
    assert step == 1


def test_a_run_with_no_watched_history_raises() -> None:
    """Selecting a best epoch from nothing would be inventing one."""
    with pytest.raises(ConfigError, match="no history"):
        report_seeds._best_step_metrics(_Client({}), _run())


def test_the_tracking_store_and_the_checkpoint_must_agree(tmp_path: Path) -> None:
    """If they disagree, one is wrong about which epoch was kept — and we do not guess."""
    run = _run(checkpoint_dir=str(tmp_path))
    directory = tmp_path / run.info.run_id
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"best": {"line/f1": 0.99}}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="disagree"):
        report_seeds._cross_check(run, {"eval/line/f1": 0.37})


def test_cross_check_agrees_and_tolerates_an_absent_summary(tmp_path: Path) -> None:
    run = _run(checkpoint_dir=str(tmp_path))
    assert report_seeds._cross_check(run, {"eval/line/f1": 0.37}) == "absent"

    directory = tmp_path / run.info.run_id
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"best": {"line/f1": 0.37}}), encoding="utf-8"
    )
    assert report_seeds._cross_check(run, {"eval/line/f1": 0.37}) == "agrees"


# --- aggregation ------------------------------------------------------------------


def test_aggregate_reports_spread_and_n() -> None:
    summary = report_seeds._aggregate(
        [{"eval/line/f1": 0.30}, {"eval/line/f1": 0.40}, {"eval/line/f1": 0.35}], "eval/line/f1"
    )
    assert summary is not None
    assert summary["mean"] == pytest.approx(0.35)
    assert summary["std"] == pytest.approx(0.05)
    assert (summary["min"], summary["max"], summary["n"]) == (0.30, 0.40, 3)


def test_a_single_seed_has_no_standard_deviation() -> None:
    """One measurement has no spread, and a zero would claim otherwise."""
    summary = report_seeds._aggregate([{"eval/line/f1": 0.38}], "eval/line/f1")
    assert summary is not None
    assert summary["std"] is None


def test_a_metric_no_run_logged_is_absent_rather_than_zero() -> None:
    assert report_seeds._aggregate([{"eval/line/f1": 0.38}], "eval/line/ifa_mean") is None


# --- what the report must always carry --------------------------------------------


def test_the_three_ceilings_travel_with_the_report() -> None:
    """Every line-level number this project reports is quoted beside all three."""
    assert set(report_seeds.CEILINGS) == {
        "line_label_noise",
        "flagged_lines_without_an_ast_node",
        "line_cap_512",
    }
    for ceiling in report_seeds.CEILINGS.values():
        assert ceiling["source"].startswith("Memory.md")


def test_function_level_metrics_are_not_headlined() -> None:
    """`line_labelled_only` makes function F1 1.000 by construction; it is never a result."""
    assert not set(report_seeds.HEADLINE_METRICS) & set(
        report_seeds.DEGENERATE_UNDER_LINE_LABELLED_ONLY
    )
    assert all(key.startswith("eval/line/") for key in report_seeds.HEADLINE_METRICS)
