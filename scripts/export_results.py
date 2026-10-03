r"""Package one machine's queue results so another machine can read them — and not merge them.

Written for the owner's second laptop (``docs/SecondMachine.md``). The results come back to
the first machine as a folder of files, never as rows in its MLflow store: ``resolved_device``
cannot tell two GPUs apart (``Memory.md`` §7 #49), so a second machine's runs inside the first
machine's tracking store would be aggregated with its own the first time a filter matched
both. A folder under ``artifacts/machine2/`` cannot be.

What the folder carries is what a write-up needs to cite a number honestly:

* ``manifest.json`` — which machine (host, GPU, VRAM), which commit, and per step its status,
  run id, wall clock, attempts and best held-out metrics.
* ``RESULTS.md`` — the same, as a table to read.
* ``runs/<run_id>/summary.json`` — the training loop's own record of each best epoch.
* ``logs/``, ``queue_state.json``, the seed reports, and a copy of the MLflow store for
  inspection with ``mlflow ui``.
* ``--with-checkpoints`` adds each finished run's ``best.pt`` — ~1.5 GB a fine-tuned run, so
  only when a checkpoint is needed on the other machine (e.g. to evaluate it on ``test``).

Usage::

    python scripts/export_results.py --groups m2check m2 --out E:\\transfer\\results
    python scripts/export_results.py --groups m2ftcheck m2ft --out E:\\transfer\\results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

HEADLINE = {
    "top5": "line/top_5_accuracy",
    "top1": "line/top_1_accuracy",
    "line_f1": "line/f1",
    "ifa": "line/ifa_mean",
}


def _load_queue() -> Any:
    """Import ``scripts/run_queue.py`` by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_run_queue", REPO / "scripts" / "run_queue.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def describe_gpu() -> str:
    """The GPU this machine trains on, as ``nvidia-smi`` names it.

    Returns:
        ``"<name>, <memory>"`` per card, or a sentence saying why it could not be read. Never
        an empty string: a number with no machine attached is the failure this file exists
        to prevent.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: nvidia-smi could not run ({type(error).__name__}: {error})"
    if completed.returncode != 0:
        return f"unavailable: nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"
    return "; ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())


def describe_commit(repo: Path) -> str:
    """The commit the results were produced from, marked ``-dirty`` if the tree was.

    Args:
        repo: The working tree.

    Returns:
        ``"<sha>"`` or ``"<sha>-dirty"``, or a sentence saying why git could not answer.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"unavailable: git could not answer ({type(error).__name__}: {error})"
    return f"{sha}-dirty" if dirty else sha


def _metrics(summary_path: Path) -> dict[str, float]:
    if not summary_path.is_file():
        return {}
    best = json.loads(summary_path.read_text(encoding="utf-8")).get("best") or {}
    return {short: float(best[key]) for short, key in HEADLINE.items() if key in best}


def export(
    groups: list[str],
    out: Path,
    *,
    repo: Path,
    plan: list[Any],
    state: dict[str, Any],
    reports_dir: Path | None,
    with_checkpoints: bool = False,
    gpu: str | None = None,
    commit: str | None = None,
) -> Path:
    """Copy one machine's results for ``groups`` into a new folder under ``out``.

    Args:
        groups: Queue groups to export.
        out: Parent folder; a new ``<host>_<timestamp>`` folder is created inside it, so
            nothing earlier is ever overwritten.
        repo: The working tree the queue ran in.
        plan: The queue's steps (``run_queue.build_plan()``).
        state: The queue's state (``run_queue.load_state()``).
        reports_dir: Where this machine's seed reports were written, if any.
        with_checkpoints: Also copy each finished run's ``best.pt``.
        gpu: Override for the GPU description (tests).
        commit: Override for the commit description (tests).

    Returns:
        The folder written.

    Raises:
        ValueError: If no step belongs to the requested groups.
    """
    steps = [step for step in plan if step.group in set(groups)]
    if not steps:
        raise ValueError(f"no queue step belongs to {groups}")
    host = platform.node() or "unknown-host"
    folder = out / f"{host}_{datetime.now():%Y%m%d-%H%M%S}"
    (folder / "logs").mkdir(parents=True)
    checkpoints = repo / "artifacts" / "checkpoints"

    records: list[dict[str, Any]] = []
    for step in steps:
        record = dict(state.get("steps", {}).get(step.name, {}))
        run_id = record.get("run_id")
        entry: dict[str, Any] = {
            "name": step.name,
            "group": step.group,
            "status": record.get("status", "pending"),
            "run_id": run_id,
            "minutes": record.get("minutes"),
            "attempts": record.get("attempts", 1 if record else 0),
            "exit_code": record.get("exit_code"),
            "command": record.get("command"),
        }
        log = repo / "artifacts" / "queue" / "logs" / f"{step.name}.log"
        if log.is_file():
            shutil.copy2(log, folder / "logs" / log.name)
        if run_id:
            summary = checkpoints / run_id / "summary.json"
            entry.update(_metrics(summary))
            if summary.is_file():
                target = folder / "runs" / run_id
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, target / "summary.json")
                best = checkpoints / run_id / "best.pt"
                if with_checkpoints and best.is_file():
                    shutil.copy2(best, target / "best.pt")
        records.append(entry)

    (folder / "queue_state.json").write_text(
        json.dumps(
            {s["name"]: state.get("steps", {}).get(s["name"], {}) for s in records}, indent=2
        ),
        encoding="utf-8",
    )
    if reports_dir is not None and reports_dir.is_dir():
        shutil.copytree(reports_dir, folder / "reports")
    store = repo / "artifacts" / "mlflow.db"
    if store.is_file():
        # A copy to *read*, with `mlflow ui --backend-store-uri sqlite:///<path>`. Never to
        # be opened as the importing machine's tracking store.
        shutil.copy2(store, folder / f"mlflow_{host}.db")

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "host": host,
        "gpu": gpu if gpu is not None else describe_gpu(),
        "commit": commit if commit is not None else describe_commit(repo),
        "python": platform.python_version(),
        "torch": _version("torch"),
        "groups": groups,
        "steps": records,
        "note": "Results from ONE machine. Read beside another machine's, never pooled with "
        "them (Memory.md §7 #49). Per-run numbers are not a comparison until every arm "
        "has three FINISHED seeds (§7 #37).",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "RESULTS.md").write_text(_table(manifest), encoding="utf-8")
    return folder


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _table(manifest: dict[str, Any]) -> str:
    def cell(value: Any, digits: int = 4) -> str:
        if value is None:
            return "—"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    lines = [
        f"# Results from {manifest['host']}",
        "",
        f"- GPU: {manifest['gpu']}",
        f"- Commit: `{manifest['commit']}` · Python {manifest['python']}"
        f" · torch {manifest['torch']}",
        f"- Exported: {manifest['exported_at']}",
        "",
        f"> {manifest['note']}",
        "",
        "| Step | Status | Run | Min | Attempts | Top-5 | Top-1 | Line F1 | IFA |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for step in manifest["steps"]:
        lines.append(
            f"| {step['name']} | {step['status']} | {(step['run_id'] or '—')[:8]} | "
            f"{cell(step['minutes'], 1)} | {step['attempts']} | {cell(step.get('top5'))} | "
            f"{cell(step.get('top1'))} | {cell(step.get('line_f1'))} | {cell(step.get('ifa'), 2)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Entry point.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--groups", nargs="+", required=True, help="Queue groups to export.")
    parser.add_argument("--out", required=True, type=Path, help="Parent folder for the export.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO / "artifacts" / "m2",
        help="Where this machine's seed reports are (default artifacts/m2).",
    )
    parser.add_argument(
        "--with-checkpoints", action="store_true", help="Also copy each finished run's best.pt."
    )
    args = parser.parse_args()

    queue = _load_queue()
    folder = export(
        args.groups,
        args.out,
        repo=REPO,
        plan=queue.build_plan(),
        state=queue.load_state(),
        reports_dir=args.reports_dir,
        with_checkpoints=args.with_checkpoints,
    )
    print(f"\n  Exported to {folder}\n")
    print((folder / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
