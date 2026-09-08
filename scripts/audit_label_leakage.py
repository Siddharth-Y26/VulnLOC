"""Can the line label be recovered from formatting alone? — the ADR-006 audit.

**The finding this script exists to reproduce.** The LineVul release of Big-Vul strips
leading whitespace from the lines of ``processed_func`` but re-inserts the flagged lines
verbatim from its ``flaw_line`` column, indentation intact. The label is therefore
written into the *shape* of the input, and a rule with no access to the code at all —
"does this line begin with a space or tab?" — recovers it at **precision 1.0000, zero
false positives**, on every partition.

A model trained on that text learns the formatting. Run ``1843412a...`` did exactly this:
its held-out line metrics matched this predictor to three decimals and were flat from the
first epoch (``Memory.md`` §1.10).

The audit is adversarial by design: each "tell" is a one-feature predictor that reads
**no code semantics**, only surface shape. Any of them scoring materially above the
partition's base rate is a leak. They are scored twice — on the corpus text as shipped,
and after :func:`~vulnloc.representation.normalization.normalize_source_whitespace` — so
the report shows both that the leak is real and that the fix closes it.

Precision is the number that matters here, not F1. A tell that fires on few lines but is
never wrong is a perfect oracle for those lines, and F1 would hide that behind low recall.

Reads the preprocessed shards only; it never opens a Big-Vul CSV.

Usage::

    python scripts/audit_label_leakage.py
    python scripts/audit_label_leakage.py 'graph.report.splits=[val,clean_eval]'
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation.line_map import split_lines
from vulnloc.representation.normalization import normalize_source_whitespace
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

# Each tell reads one line of text and returns "I claim this line is flagged". None of
# them can see the code's meaning, the function, or any label.
TELLS: dict[str, Callable[[str], bool]] = {
    "leading_whitespace": lambda line: line[:1] in (" ", "\t"),
    "trailing_whitespace": lambda line: bool(line) and line[-1:] in (" ", "\t"),
    "contains_tab": lambda line: "\t" in line,
    "inner_double_space": lambda line: "  " in line.lstrip(" \t"),
    "longer_than_60_chars": lambda line: len(line) > 60,
    "is_blank": lambda line: not line.strip(),
}

# A tell is called a leak when it is right far more often than guessing and has fired
# enough times for that to mean something. Precision, not F1: a tell that fires rarely
# and is never wrong is a perfect oracle for the lines it does fire on.
_LEAK_PRECISION = 0.5
_LEAK_MIN_HITS = 25


@dataclass
class TellScore:
    """Confusion counts for one tell on one partition."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_dict(self, base_rate: float) -> dict[str, Any]:
        """Return the tell's section of the report.

        Args:
            base_rate: Percentage of judged lines that are flagged. A tell whose
                precision merely matches this has found nothing.

        Returns:
            A JSON-serializable mapping.
        """
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fires_on_lines": self.tp + self.fp,
            "true_positives": self.tp,
            "false_positives": self.fp,
            # The honest comparison: precision against guessing at the base rate.
            "lift_over_base_rate": round(precision / (base_rate / 100), 2) if base_rate else 0.0,
            "leaks": precision >= _LEAK_PRECISION and self.tp >= _LEAK_MIN_HITS,
        }


def score_split(
    units: list[Any], *, normalize: bool, n_lines: int
) -> tuple[dict[str, TellScore], int, int]:
    """Score every tell over one partition's line-labelled units.

    Args:
        units: The partition's units carrying line labels.
        normalize: Apply the ADR-006 fix before scoring.
        n_lines: Line-axis cap; lines past it are not judged, matching training.

    Returns:
        ``(scores, judged_lines, flagged_lines)``.
    """
    scores = {name: TellScore() for name in TELLS}
    judged = 0
    flagged_total = 0
    for unit in units:
        source = normalize_source_whitespace(unit.source) if normalize else unit.source
        lines = split_lines(source)
        limit = min(unit.n_lines, n_lines, len(lines))
        for index in range(1, limit + 1):
            line = lines[index - 1]
            flagged = index in unit.vulnerable_lines
            judged += 1
            flagged_total += flagged
            for name, tell in TELLS.items():
                fired = tell(line)
                score = scores[name]
                if fired and flagged:
                    score.tp += 1
                elif fired:
                    score.fp += 1
                elif flagged:
                    score.fn += 1
    return scores, judged, flagged_total


def _leaking(tells: dict[str, Any]) -> list[str]:
    """Names of the tells that qualify as leaks.

    Args:
        tells: The tell sections of one partition.

    Returns:
        Sorted tell names.
    """
    return sorted(name for name, tell in tells.items() if tell["leaks"])


def audit(cfg: DictConfig) -> dict[str, Any]:
    """Measure formatting-based label leakage per partition.

    Args:
        cfg: Composed Hydra config. Uses ``graph.report`` for paths and ``llm.n_lines``.

    Returns:
        The report, as written to disk.

    Raises:
        DataError: If the cache holds no shards, or a requested shard is missing.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    n_lines = int(cfg.llm.n_lines)
    requested = [str(s) for s in settings.splits]
    splits = sorted(requested) if requested else sorted(p.stem for p in cache_dir.glob("*.jsonl"))
    if not splits:
        raise DataError(
            f"No shards in {cache_dir}. Run scripts/build_cache.py first.",
            path=str(cache_dir),
        )

    shards: dict[str, Any] = {}
    with start_run(cfg, run_name="label-leakage-audit") as run_id:
        for split in splits:
            path = shard_path(cache_dir, split)
            if not path.exists():
                raise DataError(f"Shard not found: {path}", path=str(path))

            units = [u for u in iter_shard(cache_dir, split) if u.vulnerable_lines]
            if not units:
                _LOGGER.info("Shard has no line-labelled units", extra={"split": split})
                continue

            section: dict[str, Any] = {"line_labelled_units": len(units)}
            for label, normalize in (("as_shipped", False), ("normalized", True)):
                scores, judged, flagged = score_split(units, normalize=normalize, n_lines=n_lines)
                base_rate = 100 * flagged / max(judged, 1)
                section[label] = {
                    "judged_lines": judged,
                    "flagged_lines": flagged,
                    "base_rate_pct": round(base_rate, 2),
                    "tells": {
                        name: score.to_dict(base_rate) for name, score in sorted(scores.items())
                    },
                }
            section["tells_leaking_as_shipped"] = _leaking(section["as_shipped"]["tells"])
            section["tells_leaking_after_normalization"] = _leaking(section["normalized"]["tells"])
            shards[split] = section

            _LOGGER.info(
                "Leakage audited",
                extra={
                    "split": split,
                    "units": len(units),
                    "base_rate_pct": section["as_shipped"]["base_rate_pct"],
                    "leaking_as_shipped": ",".join(section["tells_leaking_as_shipped"]) or "none",
                    "leaking_after_fix": ",".join(section["tells_leaking_after_normalization"])
                    or "none",
                },
            )
            for stage in ("as_shipped", "normalized"):
                for name, tell in section[stage]["tells"].items():
                    mlflow.log_metric(f"{split}/{stage}/{name}/precision", tell["precision"])

        payload = {
            "run_id": run_id,
            "n_lines_cap": n_lines,
            "shards": shards,
            # The headline: a partition appearing here had its labels readable off the
            # page formatting, with no model involved.
            "partitions_leaking_as_shipped": sorted(
                s for s, v in shards.items() if v["tells_leaking_as_shipped"]
            ),
            "partitions_leaking_after_normalization": sorted(
                s for s, v in shards.items() if v["tells_leaking_after_normalization"]
            ),
        }
        output_path = Path(settings.output_path).with_name("label_leakage.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output_path))
        _LOGGER.info("Report written", extra={"path": str(output_path), "run_id": run_id})

    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    audit(cfg)


if __name__ == "__main__":
    main()
