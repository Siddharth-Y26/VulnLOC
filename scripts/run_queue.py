"""Run the remaining experiment slate unattended, in order, without a network.

The GPU is serial — one 4 GiB card, one run at a time — so the overnight question is not
"which experiment" but "in what order, and what happens when one of them dies at 3am".

Three properties matter more than anything clever here:

**A failure must not end the night.** Every step is isolated in a subprocess; a step that
raises is recorded with its exit code and its log path, and the queue moves to the next
one. Losing one run to a bad config is a bad hour. Losing seven to it is a bad night.

**Re-running must be safe.** Completed steps are skipped, so an interrupted queue is
restarted with the same command rather than a hand-edited one. State lives in
``artifacts/queue/state.json`` and survives a reboot.

**The morning has to be readable without re-deriving anything.** Each step writes its own
log, the state file carries the run id and wall clock of every step, and
``--status`` prints the whole slate at a glance.

Ordering is cheapest-and-completes-a-phase first. Phase 4's slate is minutes per run and
closes three open exit criteria; Phase 3's remaining seeds are hours each and close one.
If the night is cut short, finishing Phase 4 is worth more than half-finishing Phase 3.

Usage::

    python scripts/run_queue.py --status      # what is planned, done, failed
    python scripts/run_queue.py --dry-run     # print the commands, run nothing
    python scripts/run_queue.py               # run everything not yet done
    python scripts/run_queue.py --only phase4 # one group
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO / "artifacts" / "queue"
STATE_PATH = QUEUE_DIR / "state.json"
LOG_DIR = QUEUE_DIR / "logs"

PYTHON = str(Path(sys.executable))


@dataclass
class Step:
    """One queued run.

    Attributes:
        name: Stable identifier; also the log filename and the state key. Renaming a step
            makes the queue forget it ran, so names are treated as permanent.
        group: Coarse bucket for ``--only``.
        why: What this step is for, in one line. Printed by ``--status`` so the morning
            does not have to reconstruct intent from a command line.
        args: Arguments after the script name.
        script: Which entrypoint.
    """

    name: str
    group: str
    why: str
    args: list[str]
    script: str = "scripts/train_structural.py"
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        """The full argv for this step."""
        return [PYTHON, str(REPO / self.script), *self.args]


def build_plan() -> list[Step]:
    """The slate, in the order it should run.

    Returns:
        Every step, cheapest-and-phase-closing first.
    """
    steps: list[Step] = []

    # --- Phase 4: GNN-only metrics over >=3 seeds --------------------------------
    # The headline structural baseline. GCN is the default convolution; the other two
    # follow as the "all three run" criterion rather than as three headline numbers.
    for seed in (42, 1, 2):
        steps.append(
            Step(
                name=f"p4-gcn-seed{seed}",
                group="phase4",
                why=f"Phase 4 exit criterion: GNN-only metrics, seed {seed} of 3.",
                args=["+gnn=gcn", "train=structural", f"seed={seed}", "device.kind=cuda"],
            )
        )

    # --- Phase 4: all three variants run ------------------------------------------
    for kind in ("gat", "ggnn"):
        steps.append(
            Step(
                name=f"p4-{kind}-seed42",
                group="phase4",
                why=f"Phase 4 exit criterion: {kind} runs and is config-selectable, on real data.",
                args=["+gnn=" + kind, "train=structural", "seed=42", "device.kind=cuda"],
            )
        )

    # --- Phase 4: the over-smoothing sweep ----------------------------------------
    # Depth 4 is the default and is already covered by p4-gcn-seed42, so only 2 and 6 are
    # run here. Same seed and same everything else, or the sweep measures the seed.
    for depth in (2, 6):
        steps.append(
            Step(
                name=f"p4-gcn-depth{depth}",
                group="phase4",
                why=f"Phase 4 exit criterion: performance vs depth, {depth} layers "
                f"(depth 4 = p4-gcn-seed42).",
                args=[
                    "+gnn=gcn",
                    "train=structural",
                    f"gnn.depth={depth}",
                    "seed=42",
                    "device.kind=cuda",
                ],
            )
        )

    # --- Phase 4: assemble the criteria while they are fresh ----------------------
    # A report step, not a training one. It runs here so the morning finds an artefact
    # rather than four criteria scattered across seven runs.
    steps.append(
        Step(
            name="p4-report",
            group="phase4",
            why="Assemble Phase 4's four exit criteria from the runs above.",
            script="scripts/report_phase4.py",
            args=[],
        )
    )
    steps.append(
        Step(
            name="p4-seed-report",
            group="phase4",
            why="Aggregate the GNN seeds, with the comparability refusals applied.",
            script="scripts/report_seeds.py",
            args=["train=structural", "+seeds.allow_source_drift=true"],
        )
    )

    # --- Phase 3: the two seeds that close it -------------------------------------
    # Hours each. Last, because a cut-short night should lose these rather than Phase 4's
    # three criteria. Seed 1 restarts from scratch: its earlier run was stopped at epoch 1
    # and `resume_from` would inherit a config this queue does not control.
    for seed in (1, 2):
        steps.append(
            Step(
                name=f"p3-semantic-seed{seed}",
                group="phase3",
                why=f"Phase 3 exit criterion: LLM-only metrics, seed {seed} (42 is logged).",
                script="scripts/train_semantic.py",
                args=["train=semantic", f"seed={seed}", "device.kind=cuda"],
            )
        )

    # Phase 3's own aggregation, last, once its seeds exist. `allow_source_drift` is
    # required and not a shortcut: src/vulnloc genuinely changed between seed 42 and
    # tonight (Phase 4 landed), and the override writes the changed file list into the
    # report rather than hiding it.
    steps.append(
        Step(
            name="p3-seed-report",
            group="phase3",
            why="Aggregate the LLM-only seeds; Phase 3's exit criterion needs >=3.",
            script="scripts/report_seeds.py",
            args=["train=semantic", "+seeds.allow_source_drift=true"],
        )
    )

    return steps


def load_state() -> dict[str, Any]:
    """Read the queue state, or an empty one.

    Returns:
        ``{step_name: {status, ...}}`` under ``"steps"``.
    """
    if STATE_PATH.exists():
        loaded: dict[str, Any] = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return loaded
    return {"steps": {}, "started": None}


def save_state(state: dict[str, Any]) -> None:
    """Write the queue state atomically enough for a crash mid-write.

    Args:
        state: The state to persist.
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def run_id_from_log(log_path: Path) -> str | None:
    """Recover the MLflow run id a step produced, by reading its log.

    The run id is the only thing tying a queue step to a result, and asking MLflow later
    means guessing by timestamp.

    Args:
        log_path: The step's log.

    Returns:
        The 32-character run id, or ``None``.
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in ("checkpoints\\", "checkpoints/"):
        index = text.find(marker)
        while index != -1:
            candidate = text[index + len(marker) : index + len(marker) + 32]
            if len(candidate) == 32 and all(c in "0123456789abcdef" for c in candidate):
                return candidate
            index = text.find(marker, index + 1)
    return None


def print_status(plan: list[Step], state: dict[str, Any]) -> None:
    """Print the whole slate, one line per step.

    Args:
        plan: The steps.
        state: Current state.
    """
    symbols = {"done": "[done]", "failed": "[FAIL]", "running": "[.... ]"}
    print(f"\n  Queue: {len(plan)} steps   state: {STATE_PATH}\n")
    for step in plan:
        record = state["steps"].get(step.name, {})
        status = record.get("status", "pending")
        mark = symbols.get(status, "[    ]")
        detail = ""
        if status == "done":
            detail = (
                f"  {record.get('minutes', 0):.1f} min  run {(record.get('run_id') or '?')[:8]}"
            )
        elif status == "failed":
            detail = f"  exit {record.get('exit_code')}  log {record.get('log')}"
        print(f"  {mark} {step.name:22s} {step.group:8s}{detail}")
        print(f"           {step.why}")
    counts: dict[str, int] = {}
    for step in plan:
        key = state["steps"].get(step.name, {}).get("status", "pending")
        counts[key] = counts.get(key, 0) + 1
    print("\n  " + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n")


def run_step(step: Step, state: dict[str, Any]) -> bool:
    """Run one step to completion, recording whatever happens.

    Args:
        step: The step.
        state: Queue state, updated in place and saved.

    Returns:
        True if the step exited zero.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.name}.log"
    started = time.time()
    state["steps"][step.name] = {
        "status": "running",
        "started": datetime.now(UTC).isoformat(),
        "command": " ".join(step.command()),
        "log": str(log_path),
    }
    save_state(state)

    print(f"\n=== {step.name} — {step.why}")
    print(f"    {' '.join(step.command()[1:])}")
    print(f"    log: {log_path}", flush=True)

    environment = dict(os.environ)
    # No network after 02:00, and the model is already in the HF cache. Offline mode turns
    # a silent 30-second timeout per run into an immediate cache read.
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **step.env})

    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            step.command(),
            cwd=str(REPO),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )

    minutes = (time.time() - started) / 60.0
    ok = completed.returncode == 0
    state["steps"][step.name] = {
        "status": "done" if ok else "failed",
        "started": state["steps"][step.name]["started"],
        "finished": datetime.now(UTC).isoformat(),
        "minutes": round(minutes, 2),
        "exit_code": completed.returncode,
        "command": " ".join(step.command()),
        "log": str(log_path),
        "run_id": run_id_from_log(log_path),
    }
    save_state(state)
    print(
        f"    {'done' if ok else 'FAILED'} in {minutes:.1f} min"
        f"{'' if ok else f' (exit {completed.returncode})'}",
        flush=True,
    )
    return ok


def main() -> int:
    """Entry point.

    Returns:
        0 when every attempted step succeeded, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Print the slate and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    parser.add_argument("--only", default=None, help="Run one group (phase3, phase4).")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also re-run steps that previously failed.",
    )
    args = parser.parse_args()

    plan = build_plan()
    state = load_state()
    if args.only:
        plan = [s for s in plan if s.group == args.only]

    if args.status:
        print_status(plan, state)
        return 0

    if args.dry_run:
        for step in plan:
            print(" ".join(step.command()))
        return 0

    state["started"] = state.get("started") or datetime.now(UTC).isoformat()
    save_state(state)

    failures = 0
    for step in plan:
        previous = state["steps"].get(step.name, {}).get("status")
        if previous == "done":
            print(f"--- {step.name}: already done, skipping")
            continue
        if previous == "failed" and not args.retry_failed:
            print(f"--- {step.name}: failed previously, skipping (--retry-failed to retry)")
            failures += 1
            continue
        if not run_step(step, state):
            failures += 1

    print_status(plan, state)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
