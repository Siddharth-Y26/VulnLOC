"""The trace of Phase 8's tables (``scripts/trace_phase8_tables.py``).

A trace that always says "0 did not trace" is worth nothing, so what is pinned here is the
failures: a run a table names that the store does not hold, a run that never finished, a
per-seed value the run did not log, a checkpoint with no stored scores.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "trace_phase8_tables.py"

EVALUATION_RUN = "e" * 32
ABLATION_RUN = "a" * 32
FAILURE_RUN = "f" * 32
CHECKPOINTS = {"1" * 32: 0.50, "2" * 32: 0.54, "3" * 32: 0.52}


def _load() -> Any:
    """Import ``scripts/trace_phase8_tables.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_trace_phase8_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tracing = _load()


def _store(path: Path, *, unfinished: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(path)
    store.execute("create table runs (run_uuid text, status text)")
    store.execute("create table metrics (run_uuid text, key text, value real)")
    for run_id in (EVALUATION_RUN, ABLATION_RUN, FAILURE_RUN, *CHECKPOINTS):
        status = "FAILED" if run_id == unfinished else "FINISHED"
        store.execute("insert into runs values (?, ?)", (run_id, status))
    for run_id, best in CHECKPOINTS.items():
        # An earlier, worse epoch too: the table must quote the best one, not any one.
        for value in (best - 0.05, best):
            store.execute("insert into metrics values (?, ?, ?)", (run_id, tracing.TOP5_KEY, value))
    store.commit()
    store.close()


def _tree(root: Path, **store_options: Any) -> Path:
    """A minimal repository root whose tables all trace."""
    _store(root / tracing.LAPTOP_STORE, **store_options)
    phase8 = root / tracing.PHASE8
    phase8.mkdir(parents=True)
    scores = root / "artifacts" / "scores"
    scores.mkdir(parents=True)
    values = list(CHECKPOINTS.values())
    names = " ".join(run_id[:8] for run_id in (EVALUATION_RUN, *CHECKPOINTS))
    for split in tracing.SPLITS:
        for run_id in CHECKPOINTS:
            (scores / f"{run_id}_{split}.npz").write_bytes(b"")
        evaluation = {
            "run_id": EVALUATION_RUN,
            "scores_dir": "artifacts/scores",
            "runs": dict.fromkeys(CHECKPOINTS, {}),
            "arms": {
                "arm": {
                    "n_seeds": 3,
                    "metrics": {"top_5": {"values": values, "mean": sum(values) / 3}},
                }
            },
        }
        (phase8 / f"evaluation_{split}.json").write_text(json.dumps(evaluation), encoding="utf-8")
        (phase8 / f"evaluation_{split}.md").write_text(names, encoding="utf-8")
    seeds = dict(zip(("42", "1", "2"), CHECKPOINTS.items(), strict=True))
    ablations = {
        "run_id": ABLATION_RUN,
        "ablations": [
            {"id": "A3", "tables": []},
            {
                "id": "A1",
                "tables": [
                    {
                        "key": "a1",
                        "store": "laptop_1",
                        "arms": {
                            "cross_attention": {
                                "source": "",
                                "runs": [
                                    {"run_id": run_id, "seed": int(seed)}
                                    for seed, (run_id, _) in seeds.items()
                                ],
                                "by_seed": {
                                    "top_5": {seed: best for seed, (_, best) in seeds.items()}
                                },
                                "metrics": {"top_5": {"mean": sum(values) / 3}},
                            }
                        },
                    }
                ],
            },
        ],
    }
    (phase8 / "ablations.json").write_text(json.dumps(ablations), encoding="utf-8")
    (phase8 / "ablations.md").write_text(names, encoding="utf-8")
    (phase8 / "failure_cases.json").write_text(json.dumps({"run_id": FAILURE_RUN}), "utf-8")
    return root


def _rewrite(path: Path, change: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tables_whose_runs_are_all_logged_and_finished_trace(tmp_path: Path) -> None:
    trace = tracing.trace_tables(_tree(tmp_path))

    assert trace.problems == []
    assert any("not run: A3" in line for line in trace.lines)


def test_a_run_the_store_does_not_hold_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    def name_a_stranger(payload: dict[str, Any]) -> None:
        payload["ablations"][1]["tables"][0]["arms"]["cross_attention"]["runs"][0]["run_id"] = (
            "9" * 32
        )

    _rewrite(root / tracing.PHASE8 / "ablations.json", name_a_stranger)

    problems = tracing.trace_tables(root).problems

    assert any("99999999 is not a finished run" in problem for problem in problems)


def test_a_run_that_never_finished_does_not_count(tmp_path: Path) -> None:
    unfinished = next(iter(CHECKPOINTS))

    problems = tracing.trace_tables(_tree(tmp_path, unfinished=unfinished)).problems

    assert any(f"{unfinished[:8]} is not a finished run" in problem for problem in problems)


def test_a_value_the_run_did_not_log_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    def flatter(payload: dict[str, Any]) -> None:
        payload["ablations"][1]["tables"][0]["arms"]["cross_attention"]["by_seed"]["top_5"][
            "42"
        ] = 0.99

    _rewrite(root / tracing.PHASE8 / "ablations.json", flatter)

    problems = tracing.trace_tables(root).problems

    assert any("seed 42 Top-5 is 0.9900 in the table" in problem for problem in problems)


def test_a_checkpoint_with_no_stored_scores_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    missing = next(iter(CHECKPOINTS))
    (root / "artifacts" / "scores" / f"{missing}_test.npz").unlink()

    problems = tracing.trace_tables(root).problems

    assert problems == [f"evaluation_test: no stored scores for checkpoint {missing[:8]}"]


def test_the_tables_of_record_trace_when_the_tree_has_them() -> None:
    if not (REPO_ROOT / tracing.PHASE8 / "ablations.json").is_file():
        pytest.skip("no Phase 8 tables in this tree (a fresh clone has none)")

    assert tracing.trace_tables(REPO_ROOT).problems == []
