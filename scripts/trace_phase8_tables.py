r"""Trace every row of Phase 8's generated tables back to a logged, finished run.

``docs/Phases.md`` asks that every table cell be traceable to a run id. The tables are
written by scripts that log a run of their own and name the runs they read; this script
checks that the names lead somewhere, without trusting the tables' word for it:

* **the evaluations** (``artifacts/phase8/evaluation_<split>.json``): the evaluation's own
  run is ``FINISHED``; every checkpoint it names is a ``FINISHED`` run with a stored score
  file for that partition and is named in the ``.md`` beside it; every arm's value for
  every metric is the mean of its per-seed values;
* **the ablation tables** (``artifacts/phase8/ablations.json``): the document's own run is
  ``FINISHED``; every run behind every arm is ``FINISHED`` in the store its table declares
  — this laptop's or the second machine's — and is named in the ``.md``; a per-seed Top-5,
  where the table carries one, equals the best epoch the run logged, and an arm's mean
  Top-5 equals the mean of its runs' best logged epochs;
* **the failure sheet** (``artifacts/phase8/failure_cases.json``): its run is ``FINISHED``.

Read-only. It opens the MLflow stores in read-only mode, loads no model, scores nothing and
opens no partition — the held-out files are read as the JSON they already are. It logs no
run: it writes nothing a later number could be quoted from. Exit status 1 if anything does
not trace.

Usage::

    python scripts/trace_phase8_tables.py
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PHASE8 = Path("artifacts/phase8")
LAPTOP_STORE = Path("artifacts/mlflow.db")
MACHINE2_GLOB = "artifacts/machine2/*/mlflow_*.db"
SPLITS = ("val", "test", "clean_eval")
TOP5_KEY = "eval/line/top_5_accuracy"
TOLERANCE = 1e-9


@dataclass
class Trace:
    """What was checked and what did not trace."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def open_store(path: Path) -> sqlite3.Connection:
    """Open an MLflow SQLite store without the ability to write to it."""
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def run_status(store: sqlite3.Connection, run_id: str) -> str | None:
    """A run's status in a store, or ``None`` if the store has no such run."""
    row = store.execute("select status from runs where run_uuid = ?", (run_id,)).fetchone()
    return str(row[0]) if row else None


def best_top5(store: sqlite3.Connection, run_id: str) -> float | None:
    """The best validation Top-5 a run logged — the value early stopping selected."""
    rows = store.execute(
        "select value from metrics where run_uuid = ? and key = ?", (run_id, TOP5_KEY)
    ).fetchall()
    return max(float(value) for (value,) in rows) if rows else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def trace_evaluation(root: Path, split: str, store: sqlite3.Connection, trace: Trace) -> None:
    """Check one partition's evaluation against the store and the stored scores."""
    name = f"evaluation_{split}"
    payload = json.loads((root / PHASE8 / f"{name}.json").read_text(encoding="utf-8"))
    rendered = (root / PHASE8 / f"{name}.md").read_text(encoding="utf-8")
    own = str(payload["run_id"])
    if run_status(store, own) != "FINISHED":
        trace.problems.append(f"{name}: its own run {own[:8]} is {run_status(store, own)}")
    if own[:8] not in rendered:
        trace.problems.append(f"{name}.md does not name its own run {own[:8]}")
    scores = root / str(payload["scores_dir"])
    for run_id in payload["runs"]:
        if run_status(store, run_id) != "FINISHED":
            trace.problems.append(f"{name}: checkpoint {run_id[:8]} is not a finished run")
        if not (scores / f"{run_id}_{split}.npz").is_file():
            trace.problems.append(f"{name}: no stored scores for checkpoint {run_id[:8]}")
        if run_id[:8] not in rendered:
            trace.problems.append(f"{name}.md does not name checkpoint {run_id[:8]}")
    cells = 0
    for arm, record in payload["arms"].items():
        for metric, entry in record["metrics"].items():
            cells += 1
            values = [float(value) for value in entry["values"] if value is not None]
            if len(entry["values"]) != int(record["n_seeds"]):
                trace.problems.append(f"{name}/{arm}/{metric}: values and seeds disagree")
            elif (
                values
                and entry["mean"] is not None
                and not math.isclose(_mean(values), float(entry["mean"]), abs_tol=TOLERANCE)
            ):
                trace.problems.append(f"{name}/{arm}/{metric}: not the mean of its seeds")
    trace.lines.append(
        f"{name}: run {own[:8]}; {len(payload['runs'])} checkpoints, each finished, scored and "
        f"named; {cells} arm-metric cells are the mean of their per-seed values"
    )


def _top5_key(metrics: dict[str, Any]) -> str | None:
    return next((key for key in metrics if "top_5" in key), None)


def _trace_arm(
    key: str,
    arm: str,
    record: dict[str, Any],
    store: sqlite3.Connection,
    rendered: str,
    trace: Trace,
) -> tuple[int, int]:
    """Check one arm of one table. Returns (run references, per-seed values compared)."""
    by_seed = record["by_seed"].get(_top5_key(record["by_seed"]) or "", {})
    logged: list[float] = []
    per_seed = 0
    for run in record["runs"]:
        run_id = str(run["run_id"])
        if run_status(store, run_id) != "FINISHED":
            trace.problems.append(f"{key}/{arm}: run {run_id[:8]} is not a finished run")
        if run_id[:8] not in rendered:
            trace.problems.append(f"{key}/{arm}: run {run_id[:8]} is not named in the .md")
        best = best_top5(store, run_id)
        if best is None:
            trace.problems.append(f"{key}/{arm}: run {run_id[:8]} logged no Top-5")
            continue
        logged.append(best)
        stated = by_seed.get(str(run["seed"]))
        if stated is not None:
            per_seed += 1
            if not math.isclose(best, float(stated), abs_tol=TOLERANCE):
                trace.problems.append(
                    f"{key}/{arm}: seed {run['seed']} Top-5 is {float(stated):.4f} in the "
                    f"table and {best:.4f} in run {run_id[:8]}"
                )
    mean_key = _top5_key(record["metrics"])
    if mean_key is not None and logged and len(logged) == len(record["runs"]):
        stated_mean = float(record["metrics"][mean_key]["mean"])
        if not math.isclose(_mean(logged), stated_mean, abs_tol=TOLERANCE):
            trace.problems.append(f"{key}/{arm}: mean Top-5 is not the mean of its runs")
    return len(record["runs"]), per_seed


def trace_ablations(root: Path, stores: dict[str, sqlite3.Connection], trace: Trace) -> None:
    """Check every arm of every ablation table against the store its table declares."""
    payload = json.loads((root / PHASE8 / "ablations.json").read_text(encoding="utf-8"))
    rendered = (root / PHASE8 / "ablations.md").read_text(encoding="utf-8")
    own = str(payload["run_id"])
    if run_status(stores["laptop_1"], own) != "FINISHED":
        trace.problems.append(f"ablations: its own run {own[:8]} is not a finished run")
    tables = arms = references = per_seed = 0
    not_run = [str(ablation["id"]) for ablation in payload["ablations"] if not ablation["tables"]]
    for ablation in payload["ablations"]:
        for table in ablation["tables"]:
            tables += 1
            key = str(table["key"])
            store = stores.get(str(table["store"]))
            if store is None:
                trace.problems.append(f"{key}: no MLflow store here for {table['store']}")
                continue
            for arm, record in table["arms"].items():
                arms += 1
                source = str(record.get("source") or "").split(" ")[0]
                if source and not (root / source).is_file():
                    trace.problems.append(f"{key}/{arm}: its source {source} is missing")
                found, compared = _trace_arm(key, arm, record, store, rendered, trace)
                references += found
                per_seed += compared
    trace.lines.append(
        f"ablations: run {own[:8]}; {tables} tables, {arms} arm rows, {references} run "
        f"references, each finished in its table's store and named; {per_seed} per-seed Top-5 "
        f"values equal the run's best logged epoch; every arm's mean Top-5 is its runs' mean; "
        f"not run: {', '.join(not_run) or 'none'}"
    )


def trace_failures(root: Path, store: sqlite3.Connection, trace: Trace) -> None:
    """Check that the failure sheet names a finished run."""
    payload = json.loads((root / PHASE8 / "failure_cases.json").read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or "")
    if not run_id or run_status(store, run_id) != "FINISHED":
        trace.problems.append("failure_cases: no finished run behind it")
    trace.lines.append(f"failure_cases: run {run_id[:8]}")


def trace_tables(root: Path = REPO) -> Trace:
    """Trace all of Phase 8's generated tables.

    Args:
        root: The repository root.

    Returns:
        One line per document checked, and every reference that did not trace.
    """
    trace = Trace()
    laptop = open_store(root / LAPTOP_STORE)
    stores = {"laptop_1": laptop}
    second = sorted(root.glob(MACHINE2_GLOB))
    if second:
        stores["machine_2"] = open_store(second[-1])
    for split in SPLITS:
        if (root / PHASE8 / f"evaluation_{split}.json").is_file():
            trace_evaluation(root, split, laptop, trace)
        else:
            trace.problems.append(f"evaluation_{split}.json is absent")
    trace_ablations(root, stores, trace)
    trace_failures(root, laptop, trace)
    return trace


def main() -> int:
    """Print the trace; return 1 if anything did not lead to a finished run."""
    trace = trace_tables()
    for line in trace.lines:
        print(line)
    print(f"did not trace: {len(trace.problems)}")
    for problem in trace.problems:
        print(f"  - {problem}")
    return 1 if trace.problems else 0


if __name__ == "__main__":
    sys.exit(main())
