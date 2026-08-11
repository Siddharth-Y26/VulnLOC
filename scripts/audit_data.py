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
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from vulnloc.data.audit import load_primevul_labels, overlap_with_primevul, scan_split
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def _concentration(projects: Counter[str], with_lines: Counter[str]) -> dict[str, Any]:
    """Summarize how unevenly the corpus is spread across projects.

    A project-wise split is only meaningful if no single project dominates. These are
    the numbers that decide whether the strategy promised in ADR-001 §D1.3 is viable.

    Args:
        projects: Function counts per project.
        with_lines: Line-labelled vulnerable counts per project.

    Returns:
        Head shares, the count of projects needed to reach 50/80/95% of the corpus,
        and how many projects carry no line-level supervision at all.
    """
    total = sum(projects.values())
    total_lines = sum(with_lines.values())
    ranked = projects.most_common()

    def _share(n: int) -> float:
        return round(100 * sum(c for _, c in ranked[:n]) / total, 2) if total else 0.0

    cumulative = 0
    thresholds: dict[str, int] = {}
    for rank, (_, count) in enumerate(ranked, start=1):
        cumulative += count
        for label, fraction in (("p50", 0.50), ("p80", 0.80), ("p95", 0.95)):
            if label not in thresholds and cumulative >= fraction * total:
                thresholds[label] = rank

    return {
        "distinct_projects": len(projects),
        "top1_pct": _share(1),
        "top2_pct": _share(2),
        "top5_pct": _share(5),
        "top10_pct": _share(10),
        "projects_for_50pct": thresholds.get("p50"),
        "projects_for_80pct": thresholds.get("p80"),
        "projects_for_95pct": thresholds.get("p95"),
        "projects_with_line_labels": len(with_lines),
        "projects_without_line_labels": len(projects) - len(with_lines),
        "line_labelled_functions": total_lines,
    }


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
    corpus_projects: Counter[str] = Counter()
    corpus_project_vuln: Counter[str] = Counter()
    corpus_project_lines: Counter[str] = Counter()

    for name in split_names:
        split_stats = scan_split(raw / f"{name}.csv")
        stats = split_stats.as_dict()
        report["splits"][name] = stats
        for key in totals:
            totals[key] += int(stats[key])
        corpus_projects.update(split_stats.projects)
        corpus_project_vuln.update(split_stats.project_vulnerable)
        corpus_project_lines.update(split_stats.project_with_lines)
    totals["vulnerable_pct"] = (
        round(100 * totals["vulnerable"] / totals["rows"], 2) if totals["rows"] else 0.0
    )
    report["totals"] = totals

    # Corpus-wide project distribution. Needed in full — not just the head — because
    # Chrome + linux dominate Big-Vul and the split strategy of ADR-001 §D1.3 depends on
    # the shape of the long tail (Memory.md §7 issue #9).
    report["projects"] = [
        {
            "project": project,
            "functions": count,
            "vulnerable": corpus_project_vuln.get(project, 0),
            "vulnerable_with_lines": corpus_project_lines.get(project, 0),
        }
        for project, count in corpus_projects.most_common()
    ]
    report["project_concentration"] = _concentration(corpus_projects, corpus_project_lines)

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
