r"""Calibrate the line scores and choose the decision threshold — on validation, and only there.

Phase 6's two measurable exit criteria: *line scores calibrated (reliability curve
reported)* and *threshold chosen on validation only*. Both are computed here from score
files (``scripts/score_checkpoint.py``), so no model is loaded and the numbers are functions
of exactly the scores whose metrics were checked against the run that produced them.

For each run, three views of the same validation lines:

**Raw** — ``sigmoid(logit)`` cut at the 0.5 the training loop logged with. This is the
baseline, and its F1 must equal the run's logged line F1: a second reproduction check.

**In-sample** — the calibrator and the threshold fitted on every validation unit. These are
the *parameters of record*: what Phase 7 and the serving layer apply.

**Out-of-fold** — the same procedure fitted on four fifths of the validation units and
applied to the remaining fifth, five times. These are the *numbers of record*. A threshold
tuned on the lines it is then scored on reports its own optimism; with only validation to
choose on, cross-validation is the honest estimate that does not require touching `test`.
Folds are assigned by content hash of the ``unit_id`` (``utils.hashing``), and by unit, not
by line — lines of one function are not independent draws.

It refuses any partition but validation: ``Rules.md`` §5.3, and a threshold chosen on the
partition it is reported on is not a threshold, it is a result.

Usage::

    python scripts/calibrate.py '+calibrate.runs=[80fc5238...,c193e3bc...,829c3555...]'
    python scripts/calibrate.py '+calibrate.runs=[...]' +calibrate.kind=temperature
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vulnloc.models.heads.calibration import (
    LogitCalibrator,
    best_f1_threshold,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
)
from vulnloc.training.line_scores import UnitScores, load_scores
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.hashing import unit_interval
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

CALIBRATION_SPLIT = "val"
"""The only partition a calibrator or a threshold may be chosen on (``Rules.md`` §5.3)."""

LOGGED_THRESHOLD = 0.5
"""The cut-off every run before Phase 6 logged its line precision, recall and F1 at."""


def require_validation(split: str) -> None:
    """Refuse to choose a calibrator or a threshold on anything but validation.

    Args:
        split: The partition asked for.

    Raises:
        ConfigError: If it is not validation.
    """
    if split != CALIBRATION_SPLIT:
        raise ConfigError(
            "a calibrator and a threshold are chosen on validation only",
            split=split,
            hint="Rules.md §5.3. Apply the validation-fitted parameters to other partitions.",
        )


def fold_of(unit_id: str, *, folds: int, seed: int) -> int:
    """Which cross-validation fold a unit belongs to — a pure function of its id.

    Args:
        unit_id: The unit.
        folds: Number of folds.
        seed: Fold seed.

    Returns:
        A fold index in ``[0, folds)``.
    """
    return min(int(unit_interval(seed, unit_id) * folds), folds - 1)


def flatten(units: list[UnitScores]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the scoreable lines of ``units`` into flat tensors.

    A line the model had no features for holds a sentinel logit, not a prediction, and is
    excluded — exactly as the training loop's line metrics exclude it.

    Args:
        units: Units to pool.

    Returns:
        ``(logits [N] float64, labels [N] float64)`` over scoreable lines.
    """
    logits = np.concatenate([unit.logits[unit.scoreable] for unit in units]) if units else []
    labels = np.concatenate([unit.labels[unit.scoreable] for unit in units]) if units else []
    return (
        torch.as_tensor(np.asarray(logits), dtype=torch.float64),
        torch.as_tensor(np.asarray(labels), dtype=torch.float64),
    )


def confusion(
    probabilities: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, int]:
    """Count true positives, false positives and false negatives at ``threshold``.

    Args:
        probabilities: ``[N]`` probabilities.
        labels: ``[N]`` 0/1 labels.
        threshold: A line is predicted flagged when its probability is ``>= threshold``.

    Returns:
        ``{"tp", "fp", "fn"}``.
    """
    predicted = probabilities >= threshold
    true = labels > 0
    return {
        "tp": int((predicted & true).sum()),
        "fp": int((predicted & ~true).sum()),
        "fn": int((~predicted & true).sum()),
    }


def prf(counts: dict[str, int]) -> dict[str, float]:
    """Precision, recall and F1 from a confusion count; ``0.0`` where undefined.

    Args:
        counts: ``{"tp", "fp", "fn"}``.

    Returns:
        ``{"precision", "recall", "f1"}``.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def nll(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean negative log-likelihood of the labels under the probabilities.

    Args:
        probabilities: ``[N]`` probabilities.
        labels: ``[N]`` 0/1 labels.

    Returns:
        The mean binary cross-entropy, with probabilities clamped away from 0 and 1.
    """
    clamped = probabilities.clamp(1e-12, 1.0 - 1e-12)
    return float(torch.nn.functional.binary_cross_entropy(clamped, labels))


def _view(
    probabilities: torch.Tensor, labels: torch.Tensor, counts: dict[str, int], *, n_bins: int
) -> dict[str, Any]:
    """One view's worth of numbers: calibration error, likelihood, decision quality."""
    bins = reliability_bins(probabilities, labels, n_bins=n_bins)
    return {
        "ece": expected_calibration_error(bins),
        "nll": nll(probabilities, labels),
        **prf(counts),
        **counts,
        "reliability": [asdict(calibration_bin) for calibration_bin in bins],
    }


def calibrate_run(
    units: list[UnitScores], *, kind: str, n_bins: int, folds: int, fold_seed: int
) -> dict[str, Any]:
    """Calibrate one run's validation scores and estimate what that is worth out of fold.

    Args:
        units: The run's validation units.
        kind: ``"platt"`` or ``"temperature"``.
        n_bins: Reliability-curve bins.
        folds: Cross-validation folds, over units.
        fold_seed: Seed for the fold assignment.

    Returns:
        ``{"calibrator", "threshold", "lines", "raw", "in_sample", "out_of_fold"}``. The
        calibrator and threshold are fitted on every unit; ``out_of_fold`` is the estimate
        of how the same procedure performs on units it was not fitted on.

    Raises:
        ConfigError: If a fold would be empty.
    """
    logits, labels = flatten(units)
    raw_probabilities = torch.sigmoid(logits)
    raw = _view(
        raw_probabilities,
        labels,
        confusion(raw_probabilities, labels, LOGGED_THRESHOLD),
        n_bins=n_bins,
    )

    calibrator = fit_calibrator(logits, labels, kind=kind)  # type: ignore[arg-type]
    probabilities = calibrator.probabilities(logits)
    choice = best_f1_threshold(probabilities, labels)
    in_sample = _view(
        probabilities, labels, confusion(probabilities, labels, choice.threshold), n_bins=n_bins
    )

    assignment = [fold_of(unit.unit_id, folds=folds, seed=fold_seed) for unit in units]
    held_probabilities: list[torch.Tensor] = []
    held_labels: list[torch.Tensor] = []
    pooled = {"tp": 0, "fp": 0, "fn": 0}
    fold_thresholds: list[float] = []
    for fold in range(folds):
        fit_units = [unit for unit, where in zip(units, assignment, strict=True) if where != fold]
        held_units = [unit for unit, where in zip(units, assignment, strict=True) if where == fold]
        if not fit_units or not held_units:
            raise ConfigError(
                "a cross-validation fold is empty", fold=fold, folds=folds, units=len(units)
            )
        fit_logits, fit_labels = flatten(fit_units)
        fold_calibrator = fit_calibrator(fit_logits, fit_labels, kind=kind)  # type: ignore[arg-type]
        fold_choice = best_f1_threshold(fold_calibrator.probabilities(fit_logits), fit_labels)
        fold_thresholds.append(fold_choice.threshold)
        logits_held, labels_held = flatten(held_units)
        probabilities_held = fold_calibrator.probabilities(logits_held)
        for key, value in confusion(probabilities_held, labels_held, fold_choice.threshold).items():
            pooled[key] += value
        held_probabilities.append(probabilities_held)
        held_labels.append(labels_held)
    out_of_fold = _view(
        torch.cat(held_probabilities), torch.cat(held_labels), pooled, n_bins=n_bins
    )
    out_of_fold["fold_thresholds"] = fold_thresholds

    return {
        "calibrator": calibrator.to_dict(),
        "threshold": choice.threshold,
        "lines": int(labels.numel()),
        "flagged_lines": int((labels > 0).sum()),
        "units": len(units),
        "raw": raw,
        "in_sample": in_sample,
        "out_of_fold": out_of_fold,
    }


def _spread(values: list[float]) -> dict[str, Any]:
    """Mean and sample standard deviation; no deviation is claimed for a single value."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def summarise(per_run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-run results over seeds.

    Args:
        per_run: ``{run_id: calibrate_run(...)}``.

    Returns:
        Mean and spread, across runs, of the quantities a write-up quotes.
    """
    results = list(per_run.values())
    summary: dict[str, Any] = {
        "threshold": _spread([float(r["threshold"]) for r in results]),
        "calibrator_scale": _spread([float(r["calibrator"]["scale"]) for r in results]),
        "calibrator_shift": _spread([float(r["calibrator"]["shift"]) for r in results]),
    }
    for view in ("raw", "in_sample", "out_of_fold"):
        for key in ("ece", "nll", "precision", "recall", "f1"):
            summary[f"{view}/{key}"] = _spread([float(r[view][key]) for r in results])
    return summary


def build(cfg: DictConfig) -> dict[str, Any]:
    """Calibrate every named run and write the report.

    Args:
        cfg: Composed config. Reads ``calibrate.runs``, ``calibrate.split``,
            ``calibrate.kind``, ``calibrate.n_bins``, ``calibrate.folds``,
            ``calibrate.fold_seed``, ``calibrate.scores_dir`` and ``calibrate.output_dir``.

    Returns:
        The report, also written to ``<output_dir>/calibration_report.json``.

    Raises:
        ConfigError: If no run is named, or the partition is not validation.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    runs = [str(run) for run in (OmegaConf.select(cfg, "calibrate.runs") or [])]
    if not runs:
        raise ConfigError("no runs to calibrate", hint="Pass '+calibrate.runs=[<run id>,...]'.")
    split = str(OmegaConf.select(cfg, "calibrate.split") or CALIBRATION_SPLIT)
    require_validation(split)
    kind = str(OmegaConf.select(cfg, "calibrate.kind") or "platt")
    n_bins = int(OmegaConf.select(cfg, "calibrate.n_bins") or 15)
    folds = int(OmegaConf.select(cfg, "calibrate.folds") or 5)
    fold_seed = int(OmegaConf.select(cfg, "calibrate.fold_seed") or 42)
    scores_dir = Path(
        str(OmegaConf.select(cfg, "calibrate.scores_dir") or "./artifacts/phase6/scores")
    )
    output_dir = Path(str(OmegaConf.select(cfg, "calibrate.output_dir") or "./artifacts/phase6"))

    per_run: dict[str, dict[str, Any]] = {}
    seeds: dict[str, int] = {}
    for run_id in runs:
        units = load_scores(scores_dir / f"{run_id}_{split}.npz")
        manifest = json.loads(
            (scores_dir / f"{run_id}_{split}.manifest.json").read_text(encoding="utf-8")
        )
        seeds[run_id] = int(manifest["seed"])
        per_run[run_id] = calibrate_run(
            units, kind=kind, n_bins=n_bins, folds=folds, fold_seed=fold_seed
        )
        per_run[run_id]["logged_f1_at_0.5"] = float(manifest["metrics"]["line/f1"])

    report: dict[str, Any] = {
        "generated_by": "scripts/calibrate.py",
        "split": split,
        "kind": kind,
        "n_bins": n_bins,
        "folds": folds,
        "fold_seed": fold_seed,
        "runs": runs,
        "seeds": seeds,
        "summary": summarise(per_run),
        "per_run": per_run,
        "note": (
            "calibrator and threshold are fitted on every validation unit; the numbers to "
            "quote are out_of_fold. Calibration is monotone, so Top-k and IFA are unchanged."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with start_run(cfg, run_name="calibration"):
        mlflow.set_tags({"artefact": "calibration", "source_runs": ",".join(r[:8] for r in runs)})
        # `calibration.*`, not `calibrate.*`: start_run has already logged the overrides
        # under their own keys, and MLflow refuses a second value for a logged parameter.
        # These are the values actually used, defaults included.
        mlflow.log_params(
            {
                "calibration.source_runs": ",".join(r[:8] for r in runs),
                "calibration.split": split,
                "calibration.kind": kind,
                "calibration.n_bins": n_bins,
                "calibration.folds": folds,
                "calibration.fold_seed": fold_seed,
            }
        )
        for key, spread in report["summary"].items():
            mlflow.log_metric(f"{key}/mean", float(spread["mean"]))
            if spread["std"] is not None:
                mlflow.log_metric(f"{key}/std", float(spread["std"]))
        mlflow.log_artifact(str(report_path), artifact_path="calibration")

    summary = report["summary"]
    _LOGGER.info(
        "Calibration report written",
        extra={
            "path": str(report_path),
            "runs": len(runs),
            "kind": kind,
            "ece_raw": round(summary["raw/ece"]["mean"], 4),
            "ece_out_of_fold": round(summary["out_of_fold/ece"]["mean"], 4),
            "f1_raw_at_0.5": round(summary["raw/f1"]["mean"], 4),
            "f1_out_of_fold": round(summary["out_of_fold/f1"]["mean"], 4),
            "threshold": round(summary["threshold"]["mean"], 4),
        },
    )
    return report


def calibrator_from_report(report: dict[str, Any], run_id: str) -> tuple[LogitCalibrator, float]:
    """The calibrator and threshold of record for one run.

    Args:
        report: A calibration report.
        run_id: The run.

    Returns:
        ``(calibrator, threshold)``.

    Raises:
        ConfigError: If the report does not cover the run.
    """
    entry = (report.get("per_run") or {}).get(run_id)
    if entry is None:
        raise ConfigError("the calibration report does not cover this run", run_id=run_id)
    parameters = entry["calibrator"]
    return (
        LogitCalibrator(
            scale=float(parameters["scale"]),
            shift=float(parameters["shift"]),
            kind=str(parameters["kind"]),
        ),
        float(entry["threshold"]),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
