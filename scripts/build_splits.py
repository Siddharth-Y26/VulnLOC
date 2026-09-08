"""Split-building entrypoint — Phase 1.

Turns the raw corpus into the versioned partition every later phase reads. Runs under a
tracked MLflow run because the split's composition is reported in ``DATA_CARD.md`` and
`Rules.md` §1.3 admits no number that did not come from one.

Usage::

    python scripts/build_splits.py                             # project-wise, ADR-003 applied
    python scripts/build_splits.py data.splits.strategy=random # reproduction split
    python scripts/build_splits.py data.splits.overwrite=true  # accept a deliberate change

Re-running without ``overwrite`` is the reproducibility check: an assignment that does
not match the one on disk is a failure, not a new result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.pipeline import (
    Adr003Membership,
    build_splits,
    clean_eval_overlap,
    compose_report,
    load_adr003_membership,
    load_units,
    project_breakdown,
    split_composition,
    write_split_artifacts,
)
from vulnloc.data.splits import SplitSpec
from vulnloc.training.tracking import start_run
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

# Metrics worth tracking per partition. The rest of the composition lives in the JSON
# report; MLflow holds what we would plot or compare across split strategies.
_TRACKED = ("functions", "vulnerable", "line_labelled_functions", "flagged_lines")


def build(cfg: DictConfig) -> dict[str, Any]:
    """Build, verify and persist the dataset split.

    Args:
        cfg: Composed Hydra config. Requires ``data.raw_dir`` and a ``data.splits``
            section.

    Returns:
        The split report.

    Raises:
        DataError: If the corpus is missing, the ADR-003 membership is unavailable, or
            the resulting split differs from the one already on disk.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.data.splits
    dataset = str(settings.dataset)

    membership = (
        load_adr003_membership(settings.audit_report_path)
        if settings.apply_adr003
        else Adr003Membership()
    )
    if not settings.apply_adr003:
        _LOGGER.warning(
            "ADR-003 hold-out disabled: this split has no clean evaluation set and "
            "retains the functions PrimeVul contradicts. Not a reportable split."
        )

    units = load_units(
        dataset,
        cfg.data.raw_dir,
        splits=tuple(str(s) for s in settings.source_splits),
        normalize_whitespace=bool(cfg.data.normalize_whitespace),
    )
    spec = SplitSpec(
        train=float(settings.train),
        val=float(settings.val),
        test=float(settings.test),
        seed=int(settings.seed),
    )
    result = build_splits(
        units,
        strategy=str(settings.strategy),
        spec=spec,
        membership=membership,
        dedup_across_splits=bool(settings.dedup_across_splits),
    )
    compositions = split_composition(units, result.assignment)

    with start_run(cfg, run_name=f"splits-{settings.strategy}") as run_id:
        report = compose_report(
            result,
            compositions,
            dataset=dataset,
            membership=membership,
            extra={
                "run_id": run_id,
                "raw_dir": str(cfg.data.raw_dir),
                "adr003_applied": bool(settings.apply_adr003),
                "clean_eval_overlap": clean_eval_overlap(units, result.assignment),
                "top_projects": project_breakdown(units, result.assignment),
            },
        )
        report = write_split_artifacts(
            settings.output_dir, result, report, overwrite=bool(settings.overwrite)
        )

        mlflow.set_tag("split_fingerprint", result.fingerprint())
        mlflow.log_metrics(
            {
                f"split/{name}/{metric}": float(stats[metric])
                for name, stats in report["composition"].items()
                for metric in _TRACKED
            }
        )
        mlflow.log_metric("split/leaked_projects", len(result.leakage.shared_projects))
        mlflow.log_metric("split/duplicate_hashes", len(result.leakage.duplicate_hashes))
        mlflow.log_artifact(
            str(Path(settings.output_dir) / "split_report.json"), artifact_path="data"
        )

    _LOGGER.info(
        "Split built",
        extra={
            "fingerprint": result.fingerprint(),
            "reproduced": report["reproduced"],
            "counts": json.dumps(result.counts()),
            "is_clean": result.leakage.is_clean,
        },
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
