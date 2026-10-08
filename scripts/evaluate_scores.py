r"""The Phase 8 metric suite, from stored line scores — one partition, every arm, by script.

``docs/EvaluationProtocol.md`` is the specification; this script is the one place a table
in the paper may come from. It loads no model. Its inputs are the score files
``score_checkpoint.py`` wrote (each already checked against the metrics its run logged),
the validation-fitted calibrators and thresholds ``calibrate.py`` wrote, and the
partition's own text and labels for the ceilings a score file cannot see.

Per checkpoint it reports:

* **ranking** — Top-1/3/5/10, IFA, MFR, MAR and Recall@1/5/20% LOC, per unit and averaged,
  each with its **tie bounds** (the stored logits are half-precision, so the order of
  exactly equal scores can move a unit; ``Memory.md`` §7 #63) and a bootstrap interval over
  units for Top-5 and IFA. Twice: with every line a candidate, and with blank lines
  withheld from targets and candidates.
* **source order** — the same metrics for "read the function from the top", on the same
  units. A model that does not beat it has not learned where to look.
* **pooled** — line-level PR-AUC, and precision / recall / F1 / ECE at the checkpoint's
  validation-chosen threshold on calibrated probabilities.
* **coverage** — the ceilings: flagged lines past the line cap, without features, blank.

Per arm: mean ± sample standard deviation over seeds. Between arms: the three pre-stated
comparisons, paired by seed, in pooled standard deviations.

**It refuses to write** if its Top-1, Top-5 or IFA cannot be reconciled with what the
scoring pass logged for the same file — the logged value must lie inside the tie bounds —
or if an arm has a different number of seeds from ``evaluate.require_seeds``.

**On validation the thresholded numbers are in-sample** (the threshold was chosen on these
lines) and are labelled so; the calibration report's out-of-fold estimate is copied beside
them and is the one to quote. On a held-out partition the parameters are applied unchanged.

``test`` and ``clean_eval`` are refused without ``+evaluate.final=true``
(``Rules.md`` §5.3: once, at the end).

Usage::

    python scripts/evaluate_scores.py +representation.strip_comments=true \
        +evaluate.scores_dir=./artifacts/phase6/scores_retrained \
        +evaluate.calibration_dir=./artifacts/phase6/retrained
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from vulnloc.evaluation.evaluator import (
    LOGGED_METRICS,
    aggregate_arm,
    arm_name,
    compare,
    evaluate_checkpoint,
    label_ceiling,
    load_partition,
    scalars,
)
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.training.line_scores import load_scores
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import run_id_text, strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError, DataError
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LOGGER = get_logger(__name__)

HELD_OUT_SPLITS = frozenset({"test", "clean_eval"})
CALIBRATION_SPLIT = "val"
"""The only partition a calibrator or a threshold may have been fitted on."""

HEADLINE = ("top_5", "ifa")
"""With PR-AUC, the paper's headline metrics (``ResearchPlan.md`` §4.2)."""


def require_final(split: str, *, final: bool) -> None:
    """Refuse a held-out partition unless the caller says this is the one evaluation.

    Args:
        split: The partition asked for.
        final: Whether ``+evaluate.final=true`` was passed.

    Raises:
        ConfigError: For ``test`` or ``clean_eval`` without ``final``.
    """
    if split in HELD_OUT_SPLITS and not final:
        raise ConfigError(
            "this partition is evaluated once, at the end, on the frozen configuration",
            split=split,
            hint="Rules.md §5.3. Add '+evaluate.final=true' only when that moment has come.",
        )


def refuse_other_text(manifests: Sequence[Mapping[str, Any]], *, strip: bool) -> None:
    """Refuse score files made from the text variant this evaluation is not reading.

    Line counts are the same with and without comments, so scores from one variant line
    up against the other's text without an error and every blank-line figure is then
    computed on the wrong lines.

    Args:
        manifests: The score files' manifests.
        strip: Whether this evaluation reads comment-stripped text.

    Raises:
        ConfigError: If any score file was made from the other variant.
    """
    other = [
        str(manifest["source_run"])
        for manifest in manifests
        if bool(manifest.get("strip_comments", False)) != strip
    ]
    if other:
        raise ConfigError(
            "these scores were made from the other text variant",
            runs=other,
            evaluation_strips_comments=strip,
            hint="Add or drop '+representation.strip_comments=true' so the evaluation "
            "reads the text the scores were made from.",
        )


def load_calibrations(directory: Path) -> dict[str, dict[str, Any]]:
    """Every run's calibration entry, from the calibration reports under a directory.

    Args:
        directory: Searched recursively for ``calibration_report.json``.

    Returns:
        ``run_id -> the report's per-run entry``.

    Raises:
        ConfigError: If there is no report, a report was not fitted on validation, or two
            reports cover one run — which of the two thresholds applies is then a choice,
            and it must not be made by file order.
    """
    paths = sorted(directory.rglob("calibration_report.json"))
    if not paths:
        raise ConfigError("no calibration report under this directory", directory=str(directory))
    entries: dict[str, dict[str, Any]] = {}
    origin: dict[str, str] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("split") != CALIBRATION_SPLIT:
            raise ConfigError(
                "a calibration report was not fitted on validation",
                path=str(path),
                split=report.get("split"),
                hint="Rules.md §5.3.",
            )
        for run_id, entry in (report.get("per_run") or {}).items():
            if run_id in entries:
                raise ConfigError(
                    "two calibration reports cover one run",
                    run_id=run_id,
                    first=origin[run_id],
                    second=str(path),
                    hint="Point '+evaluate.calibration_dir' at the directory of the "
                    "calibration that belongs to these scores.",
                )
            entries[run_id] = entry
            origin[run_id] = str(path)
    return entries


def reconcile(report: Mapping[str, Any], logged: Mapping[str, float]) -> list[str]:
    """Name the metrics on which this evaluation disagrees with the scoring pass.

    The scoring pass ranked with ``torch.argsort``, which fixes no order among equal
    scores, so its value need not equal ours — but it must lie between the two tie bounds.

    Args:
        report: An ``evaluate_checkpoint`` report.
        logged: The metrics the score file's manifest recorded.

    Returns:
        One line per disagreement, empty when everything reconciles.
    """
    section = report["ranking"]["all_lines"]
    problems: list[str] = []
    units = logged.get("line/scored_units")
    if units is not None and int(units) != section["units"]:
        problems.append(f"units ranked: {section['units']} here, {int(units)} logged")
    for name, (key, higher_is_better) in LOGGED_METRICS.items():
        if key not in logged:
            problems.append(f"{name}: the manifest has no {key}")
            continue
        bounds = section["metrics"][name]
        low, high = (
            (bounds["worst"], bounds["best"])
            if higher_is_better
            else (bounds["best"], bounds["worst"])
        )
        if not low - 1e-9 <= float(logged[key]) <= high + 1e-9:
            problems.append(
                f"{name}: logged {float(logged[key]):.6f} is outside the tie bounds "
                f"[{low:.6f}, {high:.6f}]"
            )
    return problems


def _pm(entry: Mapping[str, Any], digits: int = 4) -> str:
    """``mean ± std`` for a table cell; the mean alone for a single seed."""
    if entry["std"] is None:
        return f"{entry['mean']:.{digits}f}"
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A Markdown table as lines."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The report as Markdown tables — generated, never typed (``Phases.md``, Phase 8).

    Args:
        payload: The report :func:`evaluate` writes.

    Returns:
        The Markdown text.
    """
    split = payload["split"]
    arms: Mapping[str, Any] = payload["arms"]
    f1_key = "pooled/f1_out_of_fold" if split == CALIBRATION_SPLIT else "pooled/f1"
    ece_key = "pooled/ece_out_of_fold" if split == CALIBRATION_SPLIT else "pooled/ece"
    f1_label = "Line F1 (out of fold)" if split == CALIBRATION_SPLIT else "Line F1"
    ece_label = "ECE (out of fold)" if split == CALIBRATION_SPLIT else "ECE"

    def cell(arm: str, key: str) -> str:
        return _pm(arms[arm]["metrics"][key])

    lines = [
        f"# Line-level evaluation — `{split}`",
        "",
        f"Generated by `scripts/evaluate_scores.py`, MLflow run `{payload['run_id']}`. "
        f"Comments stripped: {payload['strip_comments']}. "
        f"Scores: `{payload['scores_dir']}`. Mean ± sample sd over seeds; ties in source order.",
        "",
        "## Ranking — every line a candidate",
        "",
    ]
    ranking = "ranking/all_lines"
    lines += _table(
        ["Arm", "Seeds", "Top-1", "Top-3", "Top-5", "Top-10", "IFA", "MFR", "MAR", "PR-AUC"],
        [
            [
                arm,
                str(arms[arm]["n_seeds"]),
                cell(arm, f"{ranking}/top_1"),
                cell(arm, f"{ranking}/top_3"),
                cell(arm, f"{ranking}/top_5"),
                cell(arm, f"{ranking}/top_10"),
                _pm(arms[arm]["metrics"][f"{ranking}/ifa"], 2),
                _pm(arms[arm]["metrics"][f"{ranking}/first_rank"], 2),
                _pm(arms[arm]["metrics"][f"{ranking}/average_rank"], 2),
                cell(arm, "pooled/pr_auc"),
            ]
            for arm in arms
        ],
    )
    lines += ["", "## Effort-aware recall, and the thresholded operating point", ""]
    lines += _table(
        ["Arm", "Recall@1% LOC", "Recall@5% LOC", "Recall@20% LOC", f1_label, ece_label],
        [
            [
                arm,
                cell(arm, f"{ranking}/recall_at_1pct_loc"),
                cell(arm, f"{ranking}/recall_at_5pct_loc"),
                cell(arm, f"{ranking}/recall_at_20pct_loc"),
                cell(arm, f1_key),
                cell(arm, ece_key),
            ]
            for arm in arms
        ],
    )
    withheld = "ranking/blank_lines_withheld"
    if all(f"{withheld}/top_5" in arms[arm]["metrics"] for arm in arms):
        lines += ["", "## Ranking — blank lines withheld from targets and candidates", ""]
        lines += _table(
            ["Arm", "Top-1", "Top-5", "IFA"],
            [
                [
                    arm,
                    cell(arm, f"{withheld}/top_1"),
                    cell(arm, f"{withheld}/top_5"),
                    _pm(arms[arm]["metrics"][f"{withheld}/ifa"], 2),
                ]
                for arm in arms
            ],
        )
    lines += [
        "",
        "## Tie bounds — arm means with every tie broken against / for the flagged line",
        "",
    ]
    lines += _table(
        ["Arm", "Top-1", "Top-5", "IFA"],
        [
            [
                arm,
                f"{arms[arm]['metrics'][f'{ranking}/top_1/tie_worst']['mean']:.4f} … "
                f"{arms[arm]['metrics'][f'{ranking}/top_1/tie_best']['mean']:.4f}",
                f"{arms[arm]['metrics'][f'{ranking}/top_5/tie_worst']['mean']:.4f} … "
                f"{arms[arm]['metrics'][f'{ranking}/top_5/tie_best']['mean']:.4f}",
                f"{arms[arm]['metrics'][f'{ranking}/ifa/tie_worst']['mean']:.2f} … "
                f"{arms[arm]['metrics'][f'{ranking}/ifa/tie_best']['mean']:.2f}",
            ]
            for arm in arms
        ],
    )
    lines += ["", "## Per checkpoint — bootstrap 95% intervals over units", ""]
    rows = []
    for run_id, run in payload["runs"].items():
        section = run["ranking"]["all_lines"]
        intervals = section.get("bootstrap_95", {})

        def interval(name: str, digits: int, intervals: Mapping[str, Any] = intervals) -> str:
            if name not in intervals:
                return "—"
            return f"[{intervals[name][0]:.{digits}f}, {intervals[name][1]:.{digits}f}]"

        rows.append(
            [
                f"`{run_id[:8]}`",
                run["arm"],
                str(run["seed"]),
                str(section["units"]),
                str(section["tied_units"]),
                f"{section['metrics']['top_5']['source']:.4f} {interval('top_5', 4)}",
                f"{section['metrics']['ifa']['source']:.2f} {interval('ifa', 2)}",
                f"{run['pooled']['threshold']:.4f}",
            ]
        )
    lines += _table(
        ["Run", "Arm", "Seed", "Units", "Tied units", "Top-5", "IFA", "Threshold"], rows
    )
    lines += ["", "## Source order — the rule-only baseline, same units", ""]
    first_run = next(iter(payload["runs"].values()))
    baseline = first_run["source_order"]["all_lines"]["metrics"]
    lines += _table(
        ["Top-1", "Top-3", "Top-5", "Top-10", "IFA"],
        [
            [
                f"{baseline['top_1']['source']:.4f}",
                f"{baseline['top_3']['source']:.4f}",
                f"{baseline['top_5']['source']:.4f}",
                f"{baseline['top_10']['source']:.4f}",
                f"{baseline['ifa']['source']:.2f}",
            ]
        ],
    )
    if payload["comparisons"]:
        lines += ["", "## Pre-stated comparisons — first minus second, paired by seed", ""]
        rows = []
        for pair, metrics in payload["comparisons"].items():
            for key, label in (
                (f"{ranking}/top_5", "Top-5"),
                (f"{ranking}/top_1", "Top-1"),
                (f"{ranking}/ifa", "IFA"),
                ("pooled/pr_auc", "PR-AUC"),
                (f1_key, f1_label),
            ):
                entry = metrics[key]
                scaled = entry["difference_in_pooled_sd"]
                rows.append(
                    [
                        pair,
                        label,
                        f"{entry['difference']:+.4f}",
                        "—" if scaled is None else f"{scaled:+.2f}",
                        "resolved" if entry["resolved"] else "not resolved",
                        ", ".join(
                            f"{seed}: {value:+.4f}" for seed, value in entry["by_seed"].items()
                        ),
                    ]
                )
        lines += _table(["Comparison", "Metric", "Difference", "In pooled sd", "", "By seed"], rows)
    coverage = first_run["coverage"]
    lines += ["", "## Ceilings — what no ranking of these scores could reach", ""]
    ceiling_rows = [
        ["Line-labelled units in the partition", str(coverage.get("labelled_units", "—"))],
        ["— of them scored", str(coverage["units"])],
        ["— of them ranked (a flagged line inside the line cap)", str(coverage["units_ranked"])],
        ["Flagged lines in the partition", str(coverage.get("flagged_lines_in_partition", "—"))],
        ["— past the line cap", str(coverage.get("flagged_lines_past_the_line_cap", "—"))],
        ["— inside the cap", str(coverage["flagged_lines"])],
        [
            "— inside the cap, blank in the text",
            str(coverage.get("flagged_lines_blank", "—")),
        ],
    ]
    lines += _table(["", "Count"], ceiling_rows)
    without = sorted(
        {int(run["coverage"]["flagged_lines_without_features"]) for run in payload["runs"].values()}
    )
    lines += [
        "",
        f"Flagged lines the model had no features for, by checkpoint: "
        f"{without[0]}–{without[-1]}.",
        "",
        "About 30% of the corpus's line labels are wrong (`Memory.md` §1.6, n = 20, "
        "95% CI 14.5–51.9%): a ceiling on every number above that no count here measures.",
        "",
    ]
    return "\n".join(lines)


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate every scored checkpoint of one partition and write the report.

    Args:
        cfg: Composed config. Reads ``evaluate.split`` (default ``val``),
            ``evaluate.final``, ``evaluate.scores_dir``, ``evaluate.calibration_dir``,
            ``evaluate.runs``, ``evaluate.output_dir``, ``evaluate.require_seeds``
            (default 3), ``evaluate.n_bins``, ``evaluate.resamples`` and
            ``evaluate.bootstrap_seed``.

    Returns:
        The report, as written to ``<output_dir>/evaluation_<split>.json``.

    Raises:
        ConfigError: For a held-out partition without ``evaluate.final``, missing inputs,
            scores from the other text variant, or an arm with the wrong number of seeds.
        DataError: If a checkpoint's metrics cannot be reconciled with what its scoring
            pass logged, or its score file does not match the partition.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: Any) -> Any:
        value = OmegaConf.select(cfg, f"evaluate.{name}")
        return default if value is None else value

    split = str(option("split", "val"))
    require_final(split, final=bool(option("final", False)))
    strip = strip_comments_enabled(cfg)
    variant = "_stripped" if strip else ""
    scores_dir = Path(str(option("scores_dir", f"./artifacts/phase6/scores{variant}")))
    calibration_dir = option("calibration_dir", None)
    if calibration_dir is None:
        raise ConfigError(
            "no calibration to apply",
            hint="Pass '+evaluate.calibration_dir=<directory of calibration_report.json>'.",
        )
    output_dir = Path(str(option("output_dir", "./artifacts/phase8")))
    require_seeds = int(option("require_seeds", 3))
    n_bins = int(option("n_bins", 15))
    resamples = int(option("resamples", 10_000))
    bootstrap_seed = int(option("bootstrap_seed", 42))
    wanted = [run_id_text(run) for run in option("runs", [])]

    manifests = []
    for path in sorted(scores_dir.glob(f"*_{split}.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not wanted or any(manifest["source_run"].startswith(prefix) for prefix in wanted):
            manifests.append(manifest)
    if not manifests:
        raise ConfigError(
            "no score files to evaluate", scores_dir=str(scores_dir), split=split, runs=wanted
        )
    refuse_other_text(manifests, strip=strip)
    calibrations = load_calibrations(Path(str(calibration_dir)))
    partition = load_partition(
        Path(cfg.data.cache.output_dir),
        split,
        normalize=bool(cfg.representation.normalize_whitespace),
        strip=strip,
    )

    runs: dict[str, dict[str, Any]] = {}
    by_arm: dict[str, dict[int, dict[str, float]]] = {}
    kinds: dict[str, str] = {}
    for manifest in manifests:
        run_id = str(manifest["source_run"])
        entry = calibrations.get(run_id)
        if entry is None:
            raise ConfigError(
                "no calibration report covers this run",
                run_id=run_id,
                calibration_dir=str(calibration_dir),
            )
        calibrator = LogitCalibrator(
            scale=float(entry["calibrator"]["scale"]),
            shift=float(entry["calibrator"]["shift"]),
            kind=str(entry["calibrator"]["kind"]),
        )
        units = load_scores(scores_dir / f"{run_id}_{split}.npz")
        report = evaluate_checkpoint(
            units,
            calibrator=calibrator,
            threshold=float(entry["threshold"]),
            partition=partition,
            n_bins=n_bins,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
        )
        report["coverage"].update(label_ceiling(units, partition))
        if split == CALIBRATION_SPLIT:
            report["pooled"]["parameters"] = "fitted_on_this_partition"
            report["pooled"]["out_of_fold"] = {
                name: float(entry["out_of_fold"][name])
                for name in ("precision", "recall", "f1", "ece")
            }
        else:
            report["pooled"]["parameters"] = "applied_from_validation"
        problems = reconcile(report, manifest.get("metrics") or {})
        if problems:
            raise DataError(
                "this evaluation does not reconcile with what the scoring pass logged",
                run_id=run_id,
                problems=problems,
            )
        arm = arm_name(manifest)
        seed = int(manifest["seed"])
        if seed in by_arm.setdefault(arm, {}):
            raise ConfigError("an arm has two checkpoints at one seed", arm=arm, seed=seed)
        by_arm[arm][seed] = scalars(report)
        kinds[arm] = str(manifest["fusion_kind"])
        report.update({"arm": arm, "seed": seed, "fusion_kind": kinds[arm]})
        runs[run_id] = report
        _LOGGER.info(
            "Checkpoint evaluated",
            extra={
                "source_run": run_id,
                "arm": arm,
                "seed": seed,
                "top_5": round(report["ranking"]["all_lines"]["metrics"]["top_5"]["source"], 4),
                "ifa": round(report["ranking"]["all_lines"]["metrics"]["ifa"]["source"], 2),
                "pr_auc": round(report["pooled"]["pr_auc"], 4),
                "tied_units": report["ranking"]["all_lines"]["tied_units"],
            },
        )

    short = {arm: sorted(seeds) for arm, seeds in by_arm.items() if len(seeds) != require_seeds}
    if short:
        raise ConfigError(
            "an arm does not have the required number of seeds",
            arms=short,
            required=require_seeds,
            hint="Memory.md §7 #37. '+evaluate.require_seeds=<n>' only for a stated reason.",
        )

    payload: dict[str, Any] = {
        "generated_by": "scripts/evaluate_scores.py",
        "split": split,
        "held_out": split in HELD_OUT_SPLITS,
        "strip_comments": strip,
        "scores_dir": str(scores_dir).replace("\\", "/"),
        "calibration_dir": str(calibration_dir).replace("\\", "/"),
        "n_bins": n_bins,
        "resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "tie_order": "source",
        "arms": {arm: aggregate_arm(seeds) for arm, seeds in sorted(by_arm.items())},
        "comparisons": compare(by_arm, kinds),
        "runs": runs,
        "note": (
            "Ranking metrics rank every line of a unit; pooled metrics use scoreable lines. "
            "On validation the thresholded numbers are in-sample and the out_of_fold block "
            "is the one to quote. Tie bounds: every tie broken for / against the flagged line."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"evaluation_{split}.json"
    markdown_path = output_dir / f"evaluation_{split}.md"
    with start_run(cfg, run_name=f"evaluation-{split}") as run_id:
        payload["run_id"] = run_id
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
        mlflow.set_tags({"artefact": "evaluation", "split": split})
        mlflow.log_params(
            {
                "evaluation.split": split,
                "evaluation.strip_comments": str(strip),
                "evaluation.checkpoints": len(runs),
                "evaluation.source_runs": ",".join(run[:8] for run in runs),
                "evaluation.resamples": resamples,
                "evaluation.bootstrap_seed": bootstrap_seed,
                "evaluation.n_bins": n_bins,
            }
        )
        metrics: dict[str, float] = {}
        for arm, summary in payload["arms"].items():
            for name, entry in summary["metrics"].items():
                metrics[f"{arm}/{name}/mean"] = float(entry["mean"])
                if entry["std"] is not None:
                    metrics[f"{arm}/{name}/std"] = float(entry["std"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(json_path), artifact_path="evaluation")
        mlflow.log_artifact(str(markdown_path), artifact_path="evaluation")

    for arm, summary in payload["arms"].items():
        _LOGGER.info(
            "Arm evaluated",
            extra={
                "arm": arm,
                "n_seeds": summary["n_seeds"],
                **{name: _pm(summary["metrics"][f"ranking/all_lines/{name}"]) for name in HEADLINE},
                "pr_auc": _pm(summary["metrics"]["pooled/pr_auc"]),
            },
        )
    _LOGGER.info(
        "Report written",
        extra={"path": str(json_path), "tables": str(markdown_path), "run_id": run_id},
    )
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()
