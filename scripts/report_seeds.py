"""Aggregate the seed replicates Phase 3's exit criterion requires — and refuse to, when.

Phase 3 does not close on a run; it closes on **LLM-only metrics over at least three
seeds**. Averaging three numbers is trivial. The part worth automating is everything that
decides whether those three numbers may be averaged *at all*, because a seed report is
exactly where an incomparable run gets quietly folded into a mean and stops being visible.

So this script is mostly guards, and every one of them is a hard failure rather than a
warning:

* **Same configuration.** Every logged parameter must match, excluding the seed and the
  run's own identity. Two runs that differ in the learning rate are not two seeds of one
  experiment, and the offending parameter is *named* rather than merely detected.
  Deliberately **not** ``config_hash``: that digest changes with the seed, which
  replicates must differ on, and changes again when a key moves between config files
  without altering a single effective value — so as a guard it refuses everything
  legitimate and admits nothing.
* **Distinct seeds.** The same seed twice is one measurement recorded twice, and it makes
  the standard deviation a fiction.
* **Normalized text.** ``normalize_whitespace`` must be true on every run (ADR-006). A run
  on the un-normalized corpus does not measure the same task; run ``1843412a…`` scored
  0.96 on it by reading the indentation.
* **Same code.** A differing ``git_sha`` is allowed **only** when ``git diff`` reports
  ``src/vulnloc`` byte-identical between the two commits — a docs commit between two seeds
  is fine, a model change is not, and the difference is checked rather than assumed. A
  sweep that spans days will cross commits touching code it never runs, so
  ``+seeds.allow_source_drift=true`` overrides this one; the changed files are then written
  into the report, because an override that leaves no trace in the artefact is not one.
* **Enough seeds.** Below ``min_seeds`` the report is still written, because it is useful
  while a sweep is in flight, but it is stamped ``sufficient_for_phase_3: false`` and says
  so in its own warning field.

The per-run numbers come from MLflow's metric history at the **best** step, located with
the run's own ``watch_metric`` / ``watch_mode`` rather than a hardcoded key, and they are
cross-checked against the ``summary.json`` the training loop wrote. A mismatch between the
two raises: it means the checkpoint directory and the tracking store disagree about which
epoch was kept, and silently preferring either one would be reporting a number that
nothing produced.

Two things are reported but deliberately **not** headlined:

* ``function/*`` under ``line_labelled_only``, where the evaluation set is all-positive so
  ``fp = tn = 0`` and F1 is 1.000 by construction. It is emitted flagged as degenerate.
* Line-level metrics without the three partition-specific ceilings beside them — label
  noise, no-node coverage and the line cap. They travel with every line number this
  project reports, so they travel inside this artefact too.

Reads the tracking store only; it trains nothing and opens no corpus.

Usage::

    python scripts/report_seeds.py
    python scripts/report_seeds.py '+seeds.runs=[fc902107...,7ea077a0...]'
    python scripts/report_seeds.py +seeds.min_seeds=5
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from vulnloc.training.tracking import ensure_experiment, start_run
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

HEADLINE_METRICS = (
    "eval/line/precision",
    "eval/line/recall",
    "eval/line/f1",
    "eval/line/top_1_accuracy",
    "eval/line/top_5_accuracy",
    "eval/line/ifa_mean",
)
"""The line-level numbers Phase 3 reports. Every one of them carries the ceilings below."""

SUPPORTING_METRICS = (
    "eval/loss",
    "eval/line/tp",
    "eval/line/fp",
    "eval/line/fn",
    "eval/line/scored_units",
    "eval/coverage/flagged_lines",
    "eval/coverage/flagged_lines_without_features",
    "eval/coverage/flagged_line_coverage_pct",
)

DEGENERATE_UNDER_LINE_LABELLED_ONLY = (
    "eval/function/precision",
    "eval/function/recall",
    "eval/function/f1",
)
"""Reported, never headlined: `line_labelled_only` leaves the eval set all-positive."""

CEILINGS = {
    "line_label_noise": {
        "value": "~30%",
        "interval": "95% CI [14.5%, 51.9%], n=20",
        "source": "Memory.md §1.6 — human label audit",
        "means": (
            "No line-level metric can exceed what this permits. A held-out score above it "
            "is a bug report, not a result."
        ),
    },
    "flagged_lines_without_an_ast_node": {
        "value": "7.84%",
        "source": "Memory.md §7 #17",
        "means": (
            "The structural branch cannot score these lines at all. 84.8% of them sit in "
            "units no grammar parses cleanly, so a better grammar does not fix it."
        ),
    },
    "line_cap_512": {
        "value": "val 17.07%, train 0.70%",
        "source": "Memory.md §7 #26",
        "means": (
            "Flagged lines past `llm.n_lines`. It is 24x worse on the reported split than "
            "on train, so the cap must never be tuned on train."
        ),
    },
}
"""Three separate, partition-specific limits. They are never combined into one number."""


def _src_changed_between(sha_a: str, sha_b: str) -> list[str]:
    """Which files under ``src/vulnloc`` differ between two commits?

    Args:
        sha_a: One commit-ish.
        sha_b: The other.

    Returns:
        The changed paths — empty when the two source trees are identical.

    Raises:
        ConfigError: If git cannot answer — an unknown commit, or no repository. The
            answer is load-bearing, so an unavailable answer is a failure rather than a
            permissive default.
    """
    command = ["git", "diff", "--name-only", sha_a, sha_b, "--", "src/vulnloc"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ConfigError(
            "git could not compare the two runs' source trees",
            command=" ".join(command),
            returncode=completed.returncode,
            stderr=completed.stderr.strip()[:400],
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _excluded_because(run: Any) -> str | None:
    """Why a discovered run must not enter a seed report, if it must not.

    Discovery is the only place these are *skipped* rather than raised on. A run the
    caller named explicitly is their claim that it belongs, and a bad claim should fail
    loudly; a run merely found lying in the tracking store should be stepped over with a
    log line, or the void run would block every report forever.

    Args:
        run: A candidate run.

    Returns:
        A reason, or ``None`` when the run is usable.
    """
    if run.data.tags.get("void"):
        return f"tagged void: {run.data.tags['void']}"
    if run.data.params.get("train.overfit", "False") == "True":
        # An overfit run is a debugging instrument (Top-5 = 1.0 on 32 samples) and is
        # never a result, even when it shares a train name with a real run.
        return "train.overfit=true — a debugging instrument, never a result"
    if run.data.params.get("normalize_whitespace") != "True":
        return "trained on un-normalized text (ADR-006); it measures a different task"
    return None


def _collect(
    client: MlflowClient, experiment_id: str, cfg: DictConfig
) -> tuple[list[Any], list[dict[str, str]]]:
    """Find the runs to aggregate.

    Args:
        client: Tracking client.
        experiment_id: Experiment to search.
        cfg: Composed config; reads ``seeds.runs`` and ``train.name``.

    Returns:
        ``(runs, excluded)`` — finished runs oldest first, and what discovery stepped
        over and why. ``excluded`` is empty when runs were named explicitly, because
        nothing is skipped on the caller's behalf in that case.

    Raises:
        ConfigError: If an explicitly named run is missing, or nothing matches.
    """
    explicit = OmegaConf.select(cfg, "seeds.runs") or []
    if explicit:
        runs = []
        for run_id in [str(r) for r in explicit]:
            try:
                runs.append(client.get_run(run_id))
            except Exception as exc:  # noqa: BLE001 - re-raised with the id that failed
                raise ConfigError("run not found in the tracking store", run_id=run_id) from exc
        return sorted(runs, key=lambda r: r.info.start_time), []

    train_name = str(cfg.train.name)
    found = client.search_runs(
        experiment_ids=[experiment_id],
        # The run NAME matters as well as the param: this script's own report run carries
        # `train.name` too, so filtering on the param alone would make each report fold the
        # previous one into the next mean.
        filter_string=(
            f"params.`train.name` = '{train_name}' "
            f"and tags.`mlflow.runName` = '{train_name}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time ASC"],
    )
    runs, excluded = [], []
    for run in found:
        reason = _excluded_because(run)
        if reason is None:
            runs.append(run)
            continue
        excluded.append({"run_id": run.info.run_id, "reason": reason})
        _LOGGER.warning(
            "Run excluded from the seed report",
            extra={"excluded_run": run.info.run_id, "reason": reason},
        )
    if not runs:
        raise ConfigError(
            "no finished runs to aggregate",
            train_name=train_name,
            experiment_id=experiment_id,
            excluded=excluded,
            hint="Pass '+seeds.runs=[<id>,...]' or check train.name.",
        )
    return runs, excluded


EXPECTED_TO_DIFFER = frozenset(
    {
        # The one thing replicates are *supposed* to disagree about.
        "seed",
        "data.splits.seed",
        "data.label_audit.seed",
        # Run identity, not configuration.
        "train.resume_from",
        "train.checkpoint_dir",
    }
)
"""Params a set of replicates may differ on without ceasing to be replicates."""


def _differing_params(runs: list[Any]) -> dict[str, set[str]]:
    """Which logged parameters disagree across the runs?

    **Not** ``config_hash``, which is the obvious choice and the wrong one. That tag is a
    digest of the whole resolved config, so it changes with the ``seed`` — the one value
    replicates must differ on — and it changes when a key is moved between two config
    files without altering a single effective value. Using it as the comparability guard
    refuses every legitimate seed aggregation and accepts nothing; it was doing exactly
    that before this function existed. Parameters are what the run actually ran with, and
    a disagreement can be named instead of merely detected.

    Args:
        runs: The candidate runs.

    Returns:
        ``{param: {values seen}}`` for every param that is not identical across runs,
        excluding :data:`EXPECTED_TO_DIFFER`. Empty means the runs are replicates.
    """
    keys = {key for run in runs for key in run.data.params}
    differing: dict[str, set[str]] = {}
    for key in keys - EXPECTED_TO_DIFFER:
        # A param logged by one run and absent from another counts as a difference: the
        # two runs then did not record the same things about themselves.
        values = {run.data.params.get(key, "<absent>") for run in runs}
        if len(values) > 1:
            differing[key] = values
    return differing


def _check_comparable(runs: list[Any], *, allow_source_drift: bool = False) -> dict[str, Any]:
    """Refuse to average runs that are not replicates of one experiment.

    Args:
        runs: The candidate runs.
        allow_source_drift: Permit a ``src/vulnloc`` difference between the runs'
            commits. Off by default. It exists because a seed sweep spanning days will
            cross commits that touch code the run never executes — a new encoder for the
            next phase, say — and refusing those forever would mean restarting the sweep
            for an unrelated change. Turning it on does **not** hide the drift: the
            changed files are written into the report's provenance, so an override is
            visible in the artefact rather than only in someone's shell history.

    Returns:
        The shared provenance every run agreed on.

    Raises:
        ConfigError: On any disagreement that would make the mean meaningless.
    """
    unnormalized = [
        r.info.run_id for r in runs if r.data.params.get("normalize_whitespace") != "True"
    ]
    if unnormalized:
        raise ConfigError(
            "a run was trained on un-normalized text and cannot be aggregated (ADR-006)",
            runs=unnormalized,
            hint="The corpus writes the flaw label into the indentation; see Memory.md §1.10.",
        )

    # Checked BEFORE the generic parameter comparison. `normalize_whitespace` differing
    # would trip that check too, but "these configs differ" is a far worse thing to be
    # told than "one of these runs measured the indentation".
    differing = _differing_params(runs)
    if differing:
        raise ConfigError(
            "runs do not share a configuration; they are not seeds of one experiment",
            differing_params={k: sorted(v) for k, v in sorted(differing.items())[:8]},
            n_differing=len(differing),
        )

    seeds = [r.data.params.get("seed") for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ConfigError(
            "a seed appears more than once; that is one measurement recorded twice",
            seeds=seeds,
            hint="Variance across duplicated seeds is not variance.",
        )

    shas = sorted({r.data.tags.get("git_sha", "?").removesuffix("-dirty") for r in runs})
    compared: list[str] = []
    drift: dict[str, list[str]] = {}
    for other in shas[1:]:
        span = f"{shas[0]}..{other}"
        compared.append(span)
        changed = _src_changed_between(shas[0], other)
        if not changed:
            continue
        if not allow_source_drift:
            raise ConfigError(
                "runs were produced by different source trees",
                git_shas=shas,
                differing=span,
                changed_files=changed,
                hint=(
                    "src/vulnloc differs between these commits. If the change cannot affect "
                    "these runs, re-run with '+seeds.allow_source_drift=true' — the diff is "
                    "then recorded in the report."
                ),
            )
        drift[span] = changed
        _LOGGER.warning(
            "Source drift accepted by override",
            extra={"span": span, "changed_files": changed},
        )
    dirty = [r.info.run_id[:8] for r in runs if r.data.tags.get("git_sha", "").endswith("-dirty")]

    return {
        "comparable_params": "identical across every run, excluding seed and run identity",
        "config_hashes": sorted({r.data.tags.get("config_hash", "?") for r in runs}),
        "git_shas": shas,
        "src_vulnloc_compared": compared,
        "src_vulnloc_drift_accepted": drift,
        "runs_with_uncommitted_changes": dirty,
        "seeds": [int(s) for s in seeds if s is not None],
    }


def _best_step_metrics(client: MlflowClient, run: Any) -> tuple[int, dict[str, float]]:
    """Read every tracked ``eval/`` metric at the run's own best step.

    The best step is located with the run's ``watch_metric`` and ``watch_mode``, not a
    hardcoded key, so a run that early-stopped on a different metric still reports the
    checkpoint it actually kept.

    Args:
        client: Tracking client.
        run: The run.

    Returns:
        ``(best_step, {metric: value})``.

    Raises:
        ConfigError: If the watched metric has no logged history.
    """
    watch = f"eval/{run.data.params.get('train.watch_metric', 'line/top_5_accuracy')}"
    mode = run.data.params.get("train.watch_mode", "max")
    history = client.get_metric_history(run.info.run_id, watch)
    if not history:
        raise ConfigError(
            "the watched metric has no history; the run logged nothing to select on",
            run_id=run.info.run_id,
            watch_metric=watch,
        )
    pick = max if mode == "max" else min
    best = pick(history, key=lambda m: m.value)

    values: dict[str, float] = {}
    for key in (*HEADLINE_METRICS, *SUPPORTING_METRICS, *DEGENERATE_UNDER_LINE_LABELLED_ONLY):
        points = [m for m in client.get_metric_history(run.info.run_id, key) if m.step == best.step]
        if points:
            values[key] = float(points[-1].value)
    return int(best.step), values


def _cross_check(run: Any, values: dict[str, float]) -> str:
    """Compare the tracking store against the checkpoint directory's ``summary.json``.

    Args:
        run: The run.
        values: Metrics read from MLflow at the best step.

    Returns:
        ``"agrees"``, or ``"absent"`` when no summary was written.

    Raises:
        ConfigError: If the two disagree. One of them is then wrong about which epoch was
            kept, and preferring either would be reporting a number nothing produced.
    """
    directory = run.data.params.get("train.checkpoint_dir", "./artifacts/checkpoints")
    summary_path = Path(directory) / run.info.run_id / "summary.json"
    if not summary_path.exists():
        return "absent"
    best = json.loads(summary_path.read_text(encoding="utf-8")).get("best", {})
    for key, value in values.items():
        local = best.get(key.removeprefix("eval/"))
        if local is None:
            continue
        if abs(float(local) - value) > 1e-6:
            raise ConfigError(
                "the tracking store and summary.json disagree about the best checkpoint",
                run_id=run.info.run_id,
                metric=key,
                mlflow=value,
                summary_json=local,
                path=str(summary_path),
            )
    return "agrees"


def _aggregate(per_run: list[dict[str, float]], key: str) -> dict[str, Any] | None:
    """Mean, sample standard deviation and range for one metric.

    Args:
        per_run: One metric dict per run.
        key: Which metric.

    Returns:
        The summary, or ``None`` when no run logged it.
    """
    values = [run[key] for run in per_run if key in run]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        # Sample standard deviation (n-1). Over three seeds this is a wide estimate, and
        # it is reported as a spread rather than dressed up as a confidence interval.
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values": values,
    }


def build(cfg: DictConfig) -> dict[str, Any]:
    """Aggregate the seed replicates and write the report.

    Args:
        cfg: Composed Hydra config.

    Returns:
        The report, as written to disk.

    Raises:
        ConfigError: If the runs are not comparable — see :func:`_check_comparable`.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = ensure_experiment(str(cfg.mlflow.experiment_name), cfg.mlflow.artifact_location)
    client = MlflowClient()

    runs, excluded = _collect(client, experiment_id, cfg)
    provenance = _check_comparable(
        runs, allow_source_drift=bool(OmegaConf.select(cfg, "seeds.allow_source_drift"))
    )

    # Named from the runs, not from the config: the report describes what it aggregated,
    # which is not necessarily what this invocation happened to compose.
    train_name = str(runs[0].data.params.get("train.name", cfg.train.name))

    min_seeds = int(OmegaConf.select(cfg, "seeds.min_seeds") or 3)
    line_labelled_only = runs[0].data.params.get("train.line_labelled_only") == "True"

    per_run: list[dict[str, float]] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        step, values = _best_step_metrics(client, run)
        per_run.append(values)
        run_rows.append(
            {
                "run_id": run.info.run_id,
                "seed": int(run.data.params["seed"]),
                "best_step": step,
                "cross_check": _cross_check(run, values),
                "wall_clock_hours": round(
                    (run.info.end_time - run.info.start_time) / 3_600_000.0, 3
                ),
                "metrics": values,
            }
        )

    sufficient = len(runs) >= min_seeds
    report: dict[str, Any] = {
        "generated_by": "scripts/report_seeds.py",
        "experiment": str(cfg.mlflow.experiment_name),
        "train_name": train_name,
        "excluded_runs": excluded,
        "eval_split": runs[0].data.params.get("train.eval_split"),
        "n_seeds": len(runs),
        "min_seeds": min_seeds,
        "sufficient_for_phase_3": sufficient,
        "provenance": provenance,
        "runs": run_rows,
        "headline": {key: _aggregate(per_run, key) for key in HEADLINE_METRICS},
        "supporting": {key: _aggregate(per_run, key) for key in SUPPORTING_METRICS},
        "ceilings": CEILINGS,
        "function_level": {
            "degenerate": line_labelled_only,
            "why": (
                (
                    "train.line_labelled_only=true, so the evaluation set is all-positive: "
                    "fp=tn=0 and F1 is 1.000 by construction. Never quote it."
                )
                if line_labelled_only
                else None
            ),
            "metrics": {
                key: _aggregate(per_run, key) for key in DEGENERATE_UNDER_LINE_LABELLED_ONLY
            },
        },
    }
    if not sufficient:
        report["warning"] = (
            f"{len(runs)} seed(s) < {min_seeds}. Phase 3's exit criterion is not met and "
            "nothing in this report is citable."
        )

    output_dir = Path(str(OmegaConf.select(cfg, "seeds.output_dir") or "./artifacts/phase3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_report_{train_name}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with start_run(cfg, run_name="seed-report"):
        mlflow.log_artifact(str(output_path))
        mlflow.log_params(
            {
                "aggregated_runs": ",".join(r["run_id"][:8] for r in run_rows),
                "seeds": ",".join(str(r["seed"]) for r in run_rows),
                "sufficient_for_phase_3": sufficient,
            }
        )
        for key, summary in report["headline"].items():
            if summary is None:
                continue
            mlflow.log_metric(f"{key}/mean", summary["mean"])
            if summary["std"] is not None:
                mlflow.log_metric(f"{key}/std", summary["std"])

    _LOGGER.info(
        "Seed report written",
        extra={
            "path": str(output_path),
            "n_seeds": len(runs),
            "sufficient": sufficient,
            "seeds": provenance["seeds"],
        },
    )
    for key, summary in report["headline"].items():
        if summary is None:
            continue
        spread = "n/a" if summary["std"] is None else f"{summary['std']:.4f}"
        _LOGGER.info(
            "Aggregated",
            extra={"metric": key, "mean": round(summary["mean"], 4), "spread": spread},
        )
    return report


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
