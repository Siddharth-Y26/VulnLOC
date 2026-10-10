"""The ablation tables refuse what they cannot vouch for (Phase 8).

``scripts/make_ablation_tables.py`` turns ``docs/ablations.yaml`` into the tables the paper
quotes. A table that renders is a claim that its runs were made under the conditions its
heading states — so the tests here are mostly about the refusals: a run that never logged
"comments stripped" in a table that says it did, a report selected on another partition, a
probe passed off as a seed, a seed read off the wrong list.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from vulnloc.evaluation.statistics import compare_arms
from vulnloc.utils.exceptions import ConfigError, DataError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "make_ablation_tables.py"
CATALOGUE = REPO / "docs" / "ablations.yaml"

TOP_5, TOP_1, IFA = "eval/line/top_5_accuracy", "eval/line/top_1_accuracy", "eval/line/ifa_mean"


def _load() -> Any:
    """Import ``scripts/make_ablation_tables.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_make_ablation_tables", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tables = _load()


def _store(path: Path, runs: dict[str, tuple[dict[str, str], dict[str, str]]]) -> None:
    """Write an MLflow-shaped SQLite store: ``run_id -> (params, tags)``."""
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE runs (run_uuid TEXT);"
        "CREATE TABLE params (key TEXT, value TEXT, run_uuid TEXT);"
        "CREATE TABLE tags (key TEXT, value TEXT, run_uuid TEXT);"
    )
    for run_id, (params, tags) in runs.items():
        connection.execute("INSERT INTO runs VALUES (?)", (run_id,))
        connection.executemany(
            "INSERT INTO params VALUES (?, ?, ?)", [(k, v, run_id) for k, v in params.items()]
        )
        connection.executemany(
            "INSERT INTO tags VALUES (?, ?, ?)", [(k, v, run_id) for k, v in tags.items()]
        )
    connection.commit()
    connection.close()


def _seed_report(path: Path, values: dict[int, float], *, prefix: str, split: str = "val") -> None:
    """Write a seed report whose Top-5 per seed is ``values``; Top-1 and IFA follow from it."""
    runs = [
        {
            "run_id": f"{prefix}{seed:02d}",
            "seed": seed,
            "metrics": {TOP_5: v, TOP_1: v / 2, IFA: 20 - 10 * v},
        }
        for seed, v in values.items()
    ]

    def headline(key: str) -> dict[str, float]:
        return {"mean": sum(run["metrics"][key] for run in runs) / len(runs)}  # type: ignore[index]

    path.write_text(
        json.dumps(
            {
                "eval_split": split,
                "n_seeds": len(runs),
                "runs": runs,
                "headline": {key: headline(key) for key in (TOP_5, TOP_1, IFA)},
                "provenance": {"src_vulnloc_drift_accepted": {}},
            }
        ),
        encoding="utf-8",
    )


def _params(seed: int, **extra: str) -> dict[str, str]:
    return {
        "seed": str(seed),
        "frozen": "True",
        "fusion_kind": "cross_attention",
        "train.limit": "0",
        **extra,
    }


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, Any]:
    """Two arms of three seeds, their reports and their store; a catalogue table over them."""
    full = {42: 0.60, 1: 0.62, 2: 0.61}
    control = {42: 0.50, 1: 0.51, 2: 0.52}
    _seed_report(tmp_path / "full.json", full, prefix="aa")
    _seed_report(tmp_path / "control.json", control, prefix="bb")
    runs = {f"aa{seed:02d}": (_params(seed), {"git_sha": "abc1234"}) for seed in full}
    runs |= {
        f"bb{seed:02d}": (_params(seed, fusion_kind="concat"), {"git_sha": "abc1234"})
        for seed in control
    }
    runs["aa01"][1]["resumed_at"] = "epoch=1,batch=2,step=3"
    table = {
        "key": "t",
        "title": "A table",
        "store": "here",
        "encoder": "frozen",
        "text": "as_shipped",
        "status": "run",
        "memory": "§0",
        "reference": "full",
        "arms": {
            "full": {"seed_report": "full.json", "expect": {"fusion_kind": "cross_attention"}},
            "control": {"seed_report": "control.json", "expect": {"fusion_kind": "concat"}},
        },
        "comparisons": [{"first": "full", "second": "control", "expects": "first_higher"}],
    }
    stores = {"here": {"label": "a machine", "mlflow_db": "store.db"}}
    return {"root": tmp_path, "runs": runs, "table": table, "stores": stores}


def _build(world: dict[str, Any]) -> dict[str, Any]:
    database = world["root"] / "store.db"
    database.unlink(missing_ok=True)
    _store(database, world["runs"])
    built: dict[str, Any] = tables.build_table(world["table"], world["stores"], world["root"])
    return built


def test_a_table_carries_its_runs_its_seeds_and_a_paired_comparison(world: dict[str, Any]) -> None:
    built = _build(world)
    full = built["arms"]["full"]
    assert [run["seed"] for run in full["runs"]] == [1, 2, 42]
    assert full["runs"][0]["resumed_at"] == "epoch=1,batch=2,step=3"
    comparison = built["comparisons"][0]
    expected = compare_arms({42: 0.60, 1: 0.62, 2: 0.61}, {42: 0.50, 1: 0.51, 2: 0.52})
    assert comparison["metrics"]["top_5"]["difference"] == pytest.approx(expected["difference"])
    assert comparison["metrics"]["top_5"]["by_seed"] == expected["by_seed"]
    assert comparison["outcome"] == "supported"


def test_a_table_marked_stripped_refuses_runs_that_never_logged_it(world: dict[str, Any]) -> None:
    world["table"]["text"] = "stripped"
    with pytest.raises(DataError, match="not what the catalogue says") as caught:
        _build(world)
    assert caught.value.details["problems"] == {
        "strip_comments": {"expected": "True", "logged": "absent"}
    }


def test_a_table_as_shipped_refuses_runs_trained_without_comments(world: dict[str, Any]) -> None:
    for params, _ in world["runs"].values():
        params["strip_comments"] = "True"
    with pytest.raises(DataError, match="not what the catalogue says"):
        _build(world)


def test_an_arm_refuses_a_run_of_another_arm(world: dict[str, Any]) -> None:
    world["runs"]["bb42"][0]["fusion_kind"] = "sum"
    with pytest.raises(DataError) as caught:
        _build(world)
    assert caught.value.details["run_id"] == "bb42"


@pytest.mark.parametrize(
    ("params", "tags"),
    [({"train.limit": "600"}, {}), ({"train.overfit": "True"}, {}), ({}, {"void": "leak"})],
)
def test_a_probe_an_overfit_check_and_a_void_run_are_refused(
    world: dict[str, Any], params: dict[str, str], tags: dict[str, str]
) -> None:
    world["runs"]["aa02"][0].update(params)
    world["runs"]["aa02"][1].update(tags)
    with pytest.raises(DataError, match="not a result"):
        _build(world)


def test_a_seed_the_report_and_the_run_disagree_on_is_refused(world: dict[str, Any]) -> None:
    world["runs"]["aa02"][0]["seed"] = "7"
    with pytest.raises(DataError) as caught:
        _build(world)
    assert caught.value.details["problems"] == {"seed": {"expected": "2", "logged": "7"}}


def test_a_report_selected_on_another_partition_is_refused(world: dict[str, Any]) -> None:
    _seed_report(world["root"] / "full.json", {42: 0.6, 1: 0.6, 2: 0.6}, prefix="aa", split="test")
    with pytest.raises(DataError, match="validation-only"):
        _build(world)


def test_a_headline_that_is_not_the_mean_of_its_runs_is_refused(world: dict[str, Any]) -> None:
    path = world["root"] / "full.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["headline"][TOP_5]["mean"] += 0.01
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataError, match="not the mean of its runs"):
        _build(world)


def test_one_run_cannot_stand_in_two_arms(world: dict[str, Any]) -> None:
    world["table"]["arms"]["control"] = dict(world["table"]["arms"]["full"])
    with pytest.raises(DataError, match="two arms"):
        _build(world)


def test_status_must_agree_with_what_is_on_disk(world: dict[str, Any]) -> None:
    world["table"]["status"] = "pending"
    with pytest.raises(ConfigError, match="marked pending has reports"):
        _build(world)
    world["table"]["status"] = "run"
    (world["root"] / "control.json").unlink()
    with pytest.raises(ConfigError, match="missing a report"):
        _build(world)


def test_a_pending_table_with_no_reports_renders_as_not_yet_run(world: dict[str, Any]) -> None:
    world["table"]["status"] = "pending"
    (world["root"] / "full.json").unlink()
    (world["root"] / "control.json").unlink()
    built = _build(world)
    assert built["arms"] == {}
    assert "**Not yet run.**" in "\n".join(tables._render_table(built))


def test_phase4_seeds_come_from_the_runs_not_from_the_reports_sorted_list(
    world: dict[str, Any],
) -> None:
    # report_phase4.py sorts `seeds` and does not sort `run_ids`: zipping them mislabels runs.
    group = {
        "kind": "gcn",
        "depth": 4,
        "use_subtokens": False,
        "seeds": [1, 2, 42],
        "run_ids": ["cc42", "cc01", "cc02"],
        "metrics": {key: {"mean": 0.4, "std": 0.02, "n": 3} for key in (TOP_5, TOP_1, IFA)},
    }
    (world["root"] / "phase4.json").write_text(json.dumps({"groups": [group]}), encoding="utf-8")
    structural = {"gnn_kind": "gcn", "gnn_depth": "4", "use_subtokens": "False", "train.limit": "0"}
    world["runs"] = {
        f"cc{seed:02d}": ({"seed": str(seed), **structural}, {"git_sha": "abc-dirty"})
        for seed in (42, 1, 2)
    }
    world["table"].update(
        encoder="none",
        reference="gcn",
        comparisons=[],
        arms={
            "gcn": {
                "phase4_group": {"kind": "gcn", "depth": 4, "use_subtokens": False},
                "report": "phase4.json",
            }
        },
    )
    runs = _build(world)["arms"]["gcn"]["runs"]
    assert [(run["seed"], run["run_id"]) for run in runs] == [
        (1, "cc01"),
        (2, "cc02"),
        (42, "cc42"),
    ]
    assert all(run["dirty"] for run in runs)


def test_an_absent_parameter_is_a_knob_that_was_not_enabled() -> None:
    assert tables.param_mismatches({}, {"strip_comments": "False"}) == {}
    assert tables.param_mismatches({}, {"strip_comments": "True"}) != {}
    assert tables.param_mismatches({"loss": "bce"}, {"loss": "focal"}) == {
        "loss": {"expected": "focal", "logged": "bce"}
    }


@pytest.mark.parametrize(
    ("scaled", "n_seeds", "expects", "verdict"),
    [
        (2.5, 3, "first_higher", "supported"),
        (-2.5, 3, "first_higher", "contradicted"),
        (1.99, 3, "first_higher", "not resolved"),
        (-3.0, 3, "none", "differs"),
        (9.0, 2, "first_higher", "not compared"),
        (None, 3, "none", "not compared"),
    ],
)
def test_an_outcome_needs_three_seeds_and_two_pooled_sd(
    scaled: float | None, n_seeds: int, expects: str, verdict: str
) -> None:
    assert tables.outcome(scaled, n_seeds, expects) == verdict


def test_summary_statistics_scale_a_difference_the_way_paired_values_do() -> None:
    first, second = {42: 0.60, 1: 0.62, 2: 0.61}, {42: 0.50, 1: 0.51, 2: 0.52}
    paired = compare_arms(first, second)
    unpaired = tables.compare_summaries(
        {"mean": 0.61, "std": 0.01, "n": 3}, {"mean": 0.51, "std": 0.01, "n": 3}
    )
    assert unpaired["difference_in_pooled_sd"] == pytest.approx(paired["difference_in_pooled_sd"])
    assert unpaired["by_seed"] is None
    one_seed = tables.compare_summaries(
        {"mean": 0.6, "std": None, "n": 1}, {"mean": 0.5, "std": 0.01, "n": 3}
    )
    assert one_seed["difference_in_pooled_sd"] is None
    assert not one_seed["resolved"]


def test_a_comparison_two_ablations_share_is_counted_once(world: dict[str, Any]) -> None:
    built = _build(world)
    again = {**built, "key": "t2"}
    ablations = [{"id": "A1", "tables": [built]}, {"id": "A2", "tables": [again]}]
    summary = tables.summarise(ablations)
    assert summary["comparisons"] == 3
    assert summary["claim_comparisons"] == 1
    assert {item["ablation"] for item in summary["crossings"]} == {"A1"}
    assert 0.05 < summary["chance"] < 0.10


def test_the_rendered_document_cites_runs_and_counts_its_comparisons(world: dict[str, Any]) -> None:
    built = _build(world)
    ablation = {
        "id": "A1",
        "name": "Fusion",
        "claim": "A claim.",
        "note": None,
        "tables": [built],
        "not_run": [],
    }
    empty = {
        "id": "A3",
        "name": "Graphs",
        "claim": "B.",
        "note": None,
        "tables": [],
        "not_run": [{"what": "All.", "why": "Because."}],
    }
    payload = {
        "ablations": [ablation, empty],
        "summary": tables.summarise([ablation, empty]),
        "run_id": "r",
        "catalogue": "c",
    }
    text = tables.render_markdown(payload)
    assert "`aa01`†" in text
    assert "† resumed" in text
    assert "it carries the comment-terminator tell" in text
    assert "makes 3 scaled comparisons" in text
    assert "| A3 | Graphs | B. | not run |" in text
    assert "**Not run — All.** Because." in text


def test_the_checked_in_catalogue_is_well_formed() -> None:
    """A clean clone has no artefacts, and the catalogue must still be checkable there.

    It has to name conditions the script knows, arms its comparisons use, and one key per
    table.
    """
    catalogue: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(CATALOGUE), resolve=True)  # type: ignore[assignment]
    assert catalogue["schema"] == 1
    assert [entry["id"] for entry in catalogue["ablations"]] == [f"A{n}" for n in range(1, 11)]
    keys: list[str] = []
    for entry in catalogue["ablations"]:
        assert entry["tables"] or entry["not_run"], f"{entry['id']} says nothing"
        for table in entry["tables"]:
            keys.append(table["key"])
            tables.table_expectations(table)
            assert table["status"] in tables.STATUSES
            assert table["store"] in catalogue["stores"]
            assert table["reference"] in table["arms"]
            for arm in table["arms"].values():
                assert ("seed_report" in arm) != ("phase4_group" in arm)
            for comparison in table["comparisons"]:
                assert {comparison["first"], comparison["second"]} <= set(table["arms"])
                assert comparison["expects"] in tables.EXPECTATIONS
    assert len(keys) == len(set(keys))
