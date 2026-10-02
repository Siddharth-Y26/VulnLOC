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

**A run of hours must survive the machine being needed.** Since D9.7 the slate holds
fine-tuned runs of three to four hours on a laptop that is not free for a whole one of
them. A ``resumable`` step that is paused, killed or crashed is continued from its
``last.pt`` — same MLflow run, same learning curve, no batch trained on twice
(``vulnloc.training.loop``) — rather than restarted. ``--until`` pauses the running step at
a clock time and ``--stop`` pauses it now; both cost seconds, not the run.

Usage::

    python scripts/run_queue.py --status      # what is planned, done, failed — with results
    python scripts/run_queue.py --dry-run     # print the commands, run nothing
    python scripts/run_queue.py               # run everything not yet done
    python scripts/run_queue.py --only phase4 # one group
    python scripts/run_queue.py --only d97 --until 08:30   # stop cleanly at 08:30
    python scripts/run_queue.py --stop        # from another terminal: pause the queue now
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO / "artifacts" / "queue"
STATE_PATH = QUEUE_DIR / "state.json"
LOG_DIR = QUEUE_DIR / "logs"
STOP_PATH = QUEUE_DIR / "STOP"
CHECKPOINT_DIR = REPO / "artifacts" / "checkpoints"

# Mirrors of `vulnloc.training.loop`, pinned equal by tests/unit/test_run_queue.py. Not
# imported: that pulls in torch, and `--status` should answer before torch has loaded.
STOP_FILE_ENV = "VULNLOC_STOP_FILE"
PAUSED_EXIT_CODE = 75

POLL_SECONDS = 15.0

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
        resumable: Whether an interrupted run of this step is continued from its
            ``last.pt`` rather than restarted. Off by default: for a run of minutes a
            restart is cheaper than reasoning about a resume, and report steps have
            nothing to resume.
    """

    name: str
    group: str
    why: str
    args: list[str]
    script: str = "scripts/train_structural.py"
    env: dict[str, str] = field(default_factory=dict)
    resumable: bool = False

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

    # --- A4: what ADR-002 D2.5's second half is actually worth ---------------------
    # Section 1.12 predicts the gain is LARGE at depth 2 (27.91% of flagged lines
    # indistinguishable without subtokens) and SMALL at depth 4 (7.35%). That is a sharp,
    # falsifiable prediction, so both depths are run rather than only the default one. The
    # comparison partners already exist: p4-gcn-seed{42,1,2} and p4-gcn-depth2.
    for seed in (42, 1, 2):
        steps.append(
            Step(
                name=f"a4-gcn-subtok-seed{seed}",
                group="phase4",
                why=f"A4: node-type + subtoken features, depth 4, seed {seed}. "
                f"Partner: p4-gcn-seed{seed}.",
                args=[
                    "+gnn=gcn",
                    "train=structural",
                    "gnn.use_subtokens=true",
                    f"seed={seed}",
                    "device.kind=cuda",
                ],
            )
        )
    steps.append(
        Step(
            name="a4-gcn-subtok-depth2",
            group="phase4",
            why="A4 at depth 2, where section 1.12 predicts the largest gain. "
            "Partner: p4-gcn-depth2.",
            args=[
                "+gnn=gcn",
                "train=structural",
                "gnn.use_subtokens=true",
                "gnn.depth=2",
                "seed=42",
                "device.kind=cuda",
            ],
        )
    )

    # --- Seeds for the two claims that rested on one run each ---------------------
    # The depth sweep's over-smoothing signal (6 vs 4) and A4's depth-2 gain were both
    # single-seed against a depth-4 spread of +/-0.037 — enough to state a direction, not
    # a magnitude. Three seeds each makes them reportable or kills them.
    for depth in (2, 6):
        for seed in (1, 2):
            steps.append(
                Step(
                    name=f"p4-gcn-depth{depth}-seed{seed}",
                    group="phase4",
                    why=f"Third seed for the depth-{depth} arm; the sweep was n=1 there.",
                    args=[
                        "+gnn=gcn",
                        "train=structural",
                        f"gnn.depth={depth}",
                        f"seed={seed}",
                        "device.kind=cuda",
                    ],
                )
            )
    for seed in (1, 2):
        steps.append(
            Step(
                name=f"a4-gcn-subtok-depth2-seed{seed}",
                group="phase4",
                why="Third seed for A4 at depth 2; the +0.0436 gain was n=1.",
                args=[
                    "+gnn=gcn",
                    "train=structural",
                    "gnn.use_subtokens=true",
                    "gnn.depth=2",
                    f"seed={seed}",
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
            why="Aggregate the GNN seeds for the headline group (gcn, depth 4).",
            script="scripts/report_seeds.py",
            # Scoped to one group on purpose. `structural` covers gcn/gat/ggnn and depths
            # 2/4/6, and asking to average across them is exactly what the comparability
            # guard refuses — correctly. The first version of this step did, and failed.
            args=[
                "train=structural",
                "+seeds.allow_source_drift=true",
                # `use_subtokens` is part of the group too, now that A4 runs exist with
                # the same kind and depth. Without it this filter would match both and
                # the comparability guard would refuse — correctly, but after the fact.
                "+seeds.where={gnn_kind: gcn, gnn_depth: 4, use_subtokens: False}",
            ],
        )
    )

    steps.append(
        Step(
            name="a4-seed-report",
            group="phase4",
            why="Aggregate the A4 subtoken seeds, for comparison against p4-seed-report.",
            script="scripts/report_seeds.py",
            args=[
                "train=structural",
                "+seeds.allow_source_drift=true",
                "+seeds.where={gnn_kind: gcn, gnn_depth: 4, use_subtokens: True}",
                "+seeds.output_dir=./artifacts/phase4",
            ],
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

    # --- Phase 5: experiment A1 -------------------------------------------------------
    # Added 2026-09-15, after the owner signed the Phase 3 and 4 gates and accepted
    # ADR-008 and ADR-009. Every arm runs on the FROZEN encoder, reading the semantic
    # feature cache (build it first: scripts/build_semantic_cache.py), over AST graphs with
    # identifier subtokens — the best Phase 4 structural config, chosen by the owner the
    # same day. These args are A1's controlled constants: change one and every logged seed
    # stops being comparable with the rest (Memory.md §7 #38).
    a1_constants = [
        "+gnn=gcn",
        "gnn.use_subtokens=true",
        "llm.frozen=true",
        "device.kind=cuda",
    ]
    arms = ("cross_attention", "concat", "sum", "gated", "llm_only", "gnn_only")

    # Overfit first, for EVERY arm: an arm with a broken branch trains, converges, and
    # A1 comparing it against a working one would look exactly like a result.
    for arm in arms:
        steps.append(
            Step(
                name=f"p5-overfit-{arm}",
                # Its own group, because an overfit run that FAILS to memorise still exits
                # zero. Run `--only p5check`, read the metrics, then `--only phase5`.
                group="p5check",
                why=f"Sanity: {arm} must memorise 32 functions on the A1 config. Never a result.",
                script="scripts/train_fused.py",
                args=[f"+fusion={arm}", "train=overfit_fused", *a1_constants],
            )
        )

    # The core comparison before the ablations, seed by seed, so a run cut short still
    # leaves A1's two arms at equal seed counts.
    for seed in (42, 1, 2):
        for arm in ("cross_attention", "concat"):
            steps.append(
                Step(
                    name=f"a1-{arm}-seed{seed}",
                    group="phase5",
                    why=f"A1 core ({'treatment' if arm == 'cross_attention' else 'control'}), "
                    f"frozen encoder, seed {seed} of 3.",
                    script="scripts/train_fused.py",
                    args=[f"+fusion={arm}", "train=fused", f"seed={seed}", *a1_constants],
                )
            )
    for seed in (42, 1, 2):
        for arm in ("sum", "gated", "llm_only", "gnn_only"):
            steps.append(
                Step(
                    name=f"a1-{arm}-seed{seed}",
                    group="phase5",
                    why=f"A1 ablation arm {arm}, frozen encoder, seed {seed} of 3.",
                    script="scripts/train_fused.py",
                    args=[f"+fusion={arm}", "train=fused", f"seed={seed}", *a1_constants],
                )
            )

    # --- Phase 5: ADR-009 D9.7, the fine-tuned confirmation --------------------------
    # Added 2026-10-02 (owner decision, Memory.md §5 Q11). The frozen screen found
    # cross_attention ~= concat (0.07 sd), which is the no-margin case D9.7 exists for.
    # Same constants as A1 with exactly two changes: the encoder trains, and the live
    # path replaces the cache (train_fused.py refuses the cache for an unfrozen encoder).
    #
    # `llm_only` is added to the two arms D9.7 names. Without it "fusion beats fine-tuned
    # LLM-only" would be measured against Phase 3's 0.6141, which ran under a different
    # scaffold, batch budget and population (ADR-008 §D8.6, Memory.md §7 #42/#43).
    d97_constants = [
        "+gnn=gcn",
        "gnn.use_subtokens=true",
        "llm.frozen=false",
        "train.semantic_cache.enabled=false",
        "device.kind=cuda",
    ]
    d97_arms = ("cross_attention", "concat", "llm_only")

    # The frozen overfit check proved nothing about this path: the encoder never received
    # a gradient there. Same rule as p5check — it exits zero even when it fails to
    # memorise, so it is its own group and its Top-1 is read before `--only d97`.
    for arm in d97_arms:
        steps.append(
            Step(
                name=f"d97-overfit-{arm}",
                group="d97check",
                why=f"Sanity: {arm} must memorise 32 functions with the encoder TRAINING. "
                "Never a result.",
                script="scripts/train_fused.py",
                args=[f"+fusion={arm}", "train=overfit_fused", *d97_constants],
            )
        )

    # Hours each, so the order is what an interrupted week leaves behind. The two arms
    # D9.7 names go seed by seed, so any stopping point leaves them at equal seed counts;
    # the scaffold control follows, because the confirmation is reportable without it and
    # not the other way round.
    for seed in (42, 1, 2):
        for arm in ("cross_attention", "concat"):
            steps.append(
                Step(
                    name=f"d97-{arm}-seed{seed}",
                    group="d97",
                    why=f"D9.7 ({'treatment' if arm == 'cross_attention' else 'control'}), "
                    f"fine-tuned encoder, seed {seed} of 3.",
                    script="scripts/train_fused.py",
                    args=[f"+fusion={arm}", "train=fused", f"seed={seed}", *d97_constants],
                    resumable=True,
                )
            )
    for seed in (42, 1, 2):
        steps.append(
            Step(
                name=f"d97-llm_only-seed{seed}",
                group="d97",
                why=f"D9.7 scaffold control: llm_only fine-tuned in the fused scaffold, "
                f"seed {seed} of 3.",
                script="scripts/train_fused.py",
                args=["+fusion=llm_only", "train=fused", f"seed={seed}", *d97_constants],
                resumable=True,
            )
        )

    # `frozen` is part of the group: the A1 screen logged the same arms under the same
    # `train.name`, and a filter on the arm alone would hand both to the comparability
    # guard, which would refuse them — correctly, but only after the runs were done.
    for arm in d97_arms:
        steps.append(
            Step(
                name=f"d97-seed-report-{arm}",
                group="d97",
                why=f"Aggregate the fine-tuned {arm} seeds (needs all 3).",
                script="scripts/report_seeds.py",
                args=[
                    "train=fused",
                    f"+seeds.where={{fusion_kind: {arm}, frozen: False}}",
                    f"+seeds.tag=finetuned_{arm}",
                    "+seeds.output_dir=./artifacts/phase5",
                ],
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
    if not log_path.is_file():
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


def resume_checkpoint(step: Step, record: dict[str, Any]) -> Path | None:
    """The ``last.pt`` an earlier attempt at this step left behind, if it can be continued.

    Args:
        step: The step about to run.
        record: Its entry in the queue state; empty if it never ran.

    Returns:
        The checkpoint, or ``None`` — never ran, not resumable, already done, or it died
        before writing one, in which case there is nothing to lose by starting over.
    """
    if not step.resumable or record.get("status") in {None, "done"}:
        return None
    run_id = record.get("run_id") or run_id_from_log(Path(record.get("log", "")))
    if not run_id:
        return None
    checkpoint = CHECKPOINT_DIR / run_id / "last.pt"
    return checkpoint if checkpoint.exists() else None


def parse_until(text: str, now: datetime) -> datetime:
    """Turn ``HH:MM`` into the next time the local clock reads it.

    Args:
        text: A 24-hour local time.
        now: The current local time.

    Returns:
        Today at that time, or tomorrow if today's has passed — "until 08:30" typed at
        ten in the evening means the morning.

    Raises:
        ValueError: If ``text`` is not ``HH:MM``.
    """
    hour, _, minute = text.partition(":")
    deadline = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    return deadline if deadline > now else deadline + timedelta(days=1)


def keep_awake(on: bool) -> None:
    """Stop Windows idle-sleeping under a run, or let it again.

    The frozen screen lost wall clock to exactly this (Memory.md, 2026-09-15: asleep at
    11:41 mid-run). It covers the idle timer only — closing the lid or pressing the power
    button still sleeps the machine, and a paused run is what `--until` is for.

    Args:
        on: Hold the system awake, or release it.
    """
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(
        es_continuous | (es_system_required if on else 0)
    )


def result_line(run_id: str | None) -> str:
    """A finished training step's held-out result, read from its ``summary.json``.

    Args:
        run_id: The step's run.

    Returns:
        ``"  Top-5 0.5408  Top-1 0.3012"``, or an empty string for a step that has no
        summary — a report step, or a run that has not finished.
    """
    if not run_id:
        return ""
    summary_path = CHECKPOINT_DIR / run_id / "summary.json"
    if not summary_path.exists():
        return ""
    best = json.loads(summary_path.read_text(encoding="utf-8")).get("best") or {}
    parts = [
        f"{label} {best[key]:.4f}"
        for label, key in (("Top-5", "line/top_5_accuracy"), ("Top-1", "line/top_1_accuracy"))
        if key in best
    ]
    return "  " + "  ".join(parts) if parts else ""


def log_warnings(log_path: Path) -> str:
    """Flag a run that only fitted because Windows paged VRAM to host memory.

    Args:
        log_path: The step's log.

    Returns:
        A marker for ``--status``, or an empty string. Memory.md §7 #24: such a run
        finishes with valid numbers and a wall clock 10-25x what it should be.
    """
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "  !! VRAM SPILLED TO HOST RAM" if "paging to host" in text else ""


def print_status(plan: list[Step], state: dict[str, Any]) -> None:
    """Print the whole slate, one line per step.

    Args:
        plan: The steps.
        state: Current state.
    """
    symbols = {
        "done": "[done]",
        "failed": "[FAIL]",
        "running": "[.... ]",
        "paused": "[paus]",
        "interrupted": "[paus]",
    }
    print(f"\n  Queue: {len(plan)} steps   state: {STATE_PATH}\n")
    for step in plan:
        record = state["steps"].get(step.name, {})
        status = record.get("status", "pending")
        mark = symbols.get(status, "[    ]")
        detail = ""
        if status == "done":
            detail = (
                f"  {record.get('minutes', 0):.1f} min  run {(record.get('run_id') or '?')[:8]}"
                f"{result_line(record.get('run_id'))}"
            )
        elif status == "failed":
            detail = f"  exit {record.get('exit_code')}  log {record.get('log')}"
        elif status in {"paused", "interrupted"}:
            detail = (
                f"  {record.get('minutes', 0):.1f} min so far  "
                f"run {(record.get('run_id') or '?')[:8]}  resumes on the next start"
            )
        if record.get("attempts", 1) > 1:
            detail += f"  ({record['attempts']} attempts)"
        detail += log_warnings(Path(record.get("log", ""))) if record else ""
        print(f"  {mark} {step.name:22s} {step.group:8s}{detail}")
        print(f"           {step.why}")
    counts: dict[str, int] = {}
    for step in plan:
        key = state["steps"].get(step.name, {}).get("status", "pending")
        counts[key] = counts.get(key, 0) + 1
    print("\n  " + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n")


def run_step(step: Step, state: dict[str, Any], deadline: datetime | None = None) -> str:
    """Run one step until it ends or is asked to pause, recording whatever happens.

    Args:
        step: The step.
        state: Queue state, updated in place and saved.
        deadline: Local time at which a resumable step is asked to pause.

    Returns:
        ``"done"``, ``"paused"`` or ``"failed"``.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.name}.log"
    before = state["steps"].get(step.name, {})
    checkpoint = resume_checkpoint(step, before)
    command = step.command()
    if checkpoint is not None:
        # Relative and forward-slashed: a Windows path in a Hydra override is a backslash
        # escape and a drive colon waiting to be misparsed.
        command.append(f"train.resume_from=./{checkpoint.relative_to(REPO).as_posix()}")
    attempts = int(before.get("attempts", 1)) + 1 if checkpoint is not None else 1
    minutes_before = float(before.get("minutes", 0.0)) if checkpoint is not None else 0.0
    started = time.time()
    state["steps"][step.name] = {
        "status": "running",
        "started": before.get("started") if checkpoint is not None else None,
        "command": " ".join(command),
        "log": str(log_path),
        "attempts": attempts,
        "minutes": minutes_before,
        "run_id": before.get("run_id") if checkpoint is not None else None,
    }
    state["steps"][step.name]["started"] = (
        state["steps"][step.name]["started"] or datetime.now(UTC).isoformat()
    )
    save_state(state)

    print(f"\n=== {step.name} — {step.why}")
    print(f"    {' '.join(command[1:])}")
    if checkpoint is not None:
        print(f"    RESUMING attempt {attempts} from {checkpoint}")
    print(f"    log: {log_path}", flush=True)

    environment = dict(os.environ)
    # No network after 02:00, and the model is already in the HF cache. Offline mode turns
    # a silent 30-second timeout per run into an immediate cache read.
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **step.env})
    if step.resumable:
        # Only a step that can be continued is told how to stop. Everything else is
        # minutes long and simply finishes.
        environment[STOP_FILE_ENV] = str(STOP_PATH)

    interrupted = False
    # Appended to on a resume: the earlier attempt's log is the only place its run id is
    # written until the step finishes, and it is the first half of the same run.
    with log_path.open("a" if checkpoint is not None else "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if deadline is not None and datetime.now() >= deadline:
                        request_stop(f"--until {deadline:%H:%M} reached")
        except KeyboardInterrupt:
            # Ctrl+C reaches the child too. Its rolling checkpoint is at most
            # `checkpoint_every_n_steps` old, so this is a rough pause, not a loss.
            returncode = process.wait()
            interrupted = True

    minutes = minutes_before + (time.time() - started) / 60.0
    if interrupted:
        status = "interrupted"
    elif returncode == 0:
        status = "done"
    elif returncode == PAUSED_EXIT_CODE:
        status = "paused"
    else:
        status = "failed"
    state["steps"][step.name] = {
        "status": status,
        "started": state["steps"][step.name]["started"],
        "finished": datetime.now(UTC).isoformat(),
        "minutes": round(minutes, 2),
        "exit_code": returncode,
        "command": " ".join(command),
        "log": str(log_path),
        "attempts": attempts,
        "run_id": run_id_from_log(log_path),
    }
    save_state(state)
    print(
        f"    {status.upper() if status != 'done' else 'done'} after {minutes:.1f} min"
        f"{f' (exit {returncode})' if status == 'failed' else ''}",
        flush=True,
    )
    if interrupted:
        raise KeyboardInterrupt
    return "paused" if status == "interrupted" else status


def request_stop(reason: str) -> None:
    """Ask the running step to pause and the queue to start nothing further.

    Args:
        reason: Written into the stop file, for whoever finds it.
    """
    if STOP_PATH.exists():
        return
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    STOP_PATH.write_text(f"{datetime.now().isoformat()} {reason}\n", encoding="utf-8")
    print(f"    stop requested: {reason}", flush=True)


def main() -> int:
    """Entry point.

    Returns:
        0 when every attempted step succeeded, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Print the slate and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    parser.add_argument(
        "--only",
        default=None,
        help="Run one group (phase3, phase4, p5check, phase5, d97check, d97).",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="HH:MM",
        help="Pause the running step at this local time and start nothing after it.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Ask a queue running elsewhere to pause its step and exit.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also re-run steps that previously failed.",
    )
    args = parser.parse_args()

    if args.stop:
        request_stop("--stop")
        print("  The running step pauses at its next optimizer step; restart to resume it.")
        return 0

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
    deadline = parse_until(args.until, datetime.now()) if args.until else None
    if deadline is not None:
        print(f"  Pausing at {deadline:%a %H:%M}.", flush=True)
    # A stop file left by the last session would end this one before it began.
    STOP_PATH.unlink(missing_ok=True)
    keep_awake(True)

    # The plan is re-read after every pass. A long queue is exactly the situation in
    # which someone adds a step while it is running, and the alternative is killing a
    # four-hour run to pick the new step up.
    failures = 0
    attempted: set[str] = set()
    stopped = False
    try:
        while not stopped:
            plan = build_plan()
            if args.only:
                plan = [s for s in plan if s.group == args.only]
            todo = [
                step
                for step in plan
                if step.name not in attempted
                and state["steps"].get(step.name, {}).get("status") != "done"
                and (
                    state["steps"].get(step.name, {}).get("status") != "failed" or args.retry_failed
                )
            ]
            if not todo:
                break
            for step in todo:
                if STOP_PATH.exists() or (deadline is not None and datetime.now() >= deadline):
                    stopped = True
                    break
                attempted.add(step.name)
                outcome = run_step(step, state, deadline)
                failures += outcome == "failed"
                state = load_state()
                if outcome == "paused":
                    stopped = True
                    break
    except KeyboardInterrupt:
        print("\n  Interrupted. The same command resumes the step where it stopped.")
        state = load_state()
    finally:
        keep_awake(False)

    shown = build_plan()
    if args.only:
        shown = [s for s in shown if s.group == args.only]
    print_status(shown, state)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
