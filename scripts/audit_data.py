"""Corpus audit entrypoint — Phase 1 / experiment E0.

Produces the measured statistics and the label-noise estimate that ADR-003 reports, and
writes them to a JSON report so every figure in `Memory.md` §1.4 and `DATA_CARD.md`
traces back to a run rather than to a scratch script (`Rules.md` §1.3).

Usage::

    python scripts/audit_data.py                     # full audit, all splits
    python scripts/audit_data.py data.audit.splits=[bigvul_val]
    python scripts/audit_data.py data.audit.skip_overlap=true
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from vulnloc.data.audit import load_primevul_labels, overlap_with_primevul, scan_split
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def audit(cfg: DictConfig) -> dict[str, Any]:
    """Run the corpus audit and write a JSON report.

    Args:
        cfg: Composed Hydra config. Requires ``data.raw_dir`` and a ``data.audit``
            section with ``splits``, ``report_path`` and ``skip_overlap``.

    Returns:
        The report contents.

    Raises:
        DataError: If a required corpus file is missing.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )

    raw = Path(cfg.data.raw_dir)
    split_names = [str(s) for s in cfg.data.audit.splits]
    report: dict[str, Any] = {"splits": {}, "totals": {}}

    totals = {
        "rows": 0,
        "vulnerable": 0,
        "vulnerable_with_lines": 0,
        "vulnerable_without_lines": 0,
        "nonvulnerable_with_lines": 0,
        "flagged_lines": 0,
    }
    for name in split_names:
        stats = scan_split(raw / f"{name}.csv").as_dict()
        report["splits"][name] = stats
        for key in totals:
            totals[key] += int(stats[key])
    totals["vulnerable_pct"] = (
        round(100 * totals["vulnerable"] / totals["rows"], 2) if totals["rows"] else 0.0
    )
    report["totals"] = totals

    if not cfg.data.audit.skip_overlap:
        primevul_dir = raw / "primevul"
        if not primevul_dir.exists():
            raise DataError(
                f"PrimeVul is required for the label-noise estimate but is missing: "
                f"{primevul_dir}. Run scripts/prepare_data.py, or pass "
                f"data.audit.skip_overlap=true to audit Big-Vul alone.",
                path=str(primevul_dir),
            )
        labels = load_primevul_labels(primevul_dir)
        overlap = overlap_with_primevul([raw / f"{name}.csv" for name in split_names], labels)
        report["primevul"] = {
            "unique_functions": len(labels),
            "labelled_vulnerable": sum(labels.values()),
        }
        report["overlap"] = overlap.as_dict()
        # ADR-003 §D3.1/§D3.2: the concrete membership the data layer consumes.
        report["clean_eval_hashes"] = sorted(overlap.clean_eval_hashes)
        report["excluded_hashes"] = sorted(overlap.excluded_hashes)

    destination = Path(cfg.data.audit.report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = {k: v for k, v in report.get("overlap", {}).items()}
    _LOGGER.info("Audit complete", extra={"report": str(destination), **totals, **summary})
    return report


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    audit(cfg)


if __name__ == "__main__":
    main()
