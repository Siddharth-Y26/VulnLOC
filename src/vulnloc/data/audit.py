"""Corpus audit: measured statistics and label-noise estimation (L0).

`Rules.md` §1.3 forbids reporting a number that did not come from a logged run, and
§5.7 forbids adopting a figure from a paper as if it were ours. The 19.2% Big-Vul
label-noise rate in ADR-003 is a *reported result*, so the code that produces it lives
here rather than in a scratch script.

Two independent measurements:

* :func:`scan_split` — per-split composition of the primary corpus.
* :func:`overlap_with_primevul` — agreement between Big-Vul's labels and PrimeVul's
  higher-accuracy ones, which yields both the clean evaluation set (ADR-003 §D3.1) and
  the noise estimate (§D3.3).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

__all__ = [
    "BIGVUL_AUDIT_COLUMNS",
    "OverlapReport",
    "SplitStats",
    "flaw_line_indices",
    "function_hash",
    "load_primevul_labels",
    "overlap_with_primevul",
    "scan_split",
]

_LOGGER = get_logger(__name__)

# PrimeVul normalizes by deleting these before hashing. Verified empirically: this
# reproduces the published func_hash on 300/300 sampled records. Do not "improve" it —
# any change silently breaks the join and yields a near-empty intersection.
_HASH_STRIP_CHARS = (" ", "\t", "\n", "\r")

BIGVUL_AUDIT_COLUMNS = [
    "target",
    "flaw_line",
    "flaw_line_index",
    "processed_func",
    "project",
    "CVE ID",
    "CWE ID",
]

_DEFAULT_CHUNK = 20_000


def function_hash(source: str) -> int:
    """Compute PrimeVul's ``func_hash`` for a function body.

    Args:
        source: Raw function text.

    Returns:
        MD5 of the whitespace-stripped source, as a 128-bit integer.
    """
    for char in _HASH_STRIP_CHARS:
        source = source.replace(char, "")
    return int(hashlib.md5(source.encode("utf-8")).hexdigest(), 16)


def flaw_line_indices(raw: object) -> list[int]:
    """Parse Big-Vul's ``flaw_line_index`` cell into line indices.

    The column is a comma-separated string, empty or NaN when a vulnerable function
    carries no line annotation at all — which is true of 24.4% of positives.

    Args:
        raw: Cell value from ``flaw_line_index``.

    Returns:
        Parsed indices; empty when the cell is blank or missing.

    Raises:
        DataError: If a non-empty cell contains a non-integer token, which would mean
            the schema is not what the loader assumes.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise DataError(f"Malformed flaw_line_index cell: {text!r}") from exc


@dataclass
class SplitStats:
    """Composition of one Big-Vul split."""

    name: str
    rows: int = 0
    vulnerable: int = 0
    vulnerable_with_lines: int = 0
    vulnerable_without_lines: int = 0
    nonvulnerable_with_lines: int = 0
    flagged_lines: int = 0
    projects: Counter[str] = field(default_factory=Counter)
    cwes: Counter[str] = field(default_factory=Counter)
    func_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary."""
        return {
            "name": self.name,
            "rows": self.rows,
            "vulnerable": self.vulnerable,
            "vulnerable_pct": round(100 * self.vulnerable / self.rows, 2) if self.rows else 0.0,
            "vulnerable_with_lines": self.vulnerable_with_lines,
            "vulnerable_without_lines": self.vulnerable_without_lines,
            "nonvulnerable_with_lines": self.nonvulnerable_with_lines,
            "flagged_lines": self.flagged_lines,
            "distinct_projects": len(self.projects),
            "distinct_cwes": len(self.cwes),
            "mean_func_chars": round(self.func_chars / self.rows) if self.rows else 0,
            "top_projects": self.projects.most_common(10),
        }


def scan_split(path: Path, *, chunksize: int = _DEFAULT_CHUNK) -> SplitStats:
    """Measure the composition of one Big-Vul CSV.

    Chunked: the training split is 8.3 GB and must never be materialized whole.

    Args:
        path: CSV to scan.
        chunksize: Rows per chunk.

    Returns:
        Measured statistics for the split.

    Raises:
        DataError: If the file is missing or lacks the expected columns.
    """
    if not path.exists():
        raise DataError(f"Corpus split not found: {path}. Run scripts/prepare_data.py.")

    stats = SplitStats(name=path.stem)
    try:
        reader = pd.read_csv(
            path, usecols=BIGVUL_AUDIT_COLUMNS, chunksize=chunksize, low_memory=False
        )
    except ValueError as exc:
        raise DataError(
            f"{path.name} does not have the expected Big-Vul schema. The LineVul release "
            f"is required — the original CSV and the bstee615/bigvul mirror lack "
            f"flaw_line/flaw_line_index (ADR-001 §D1.1).",
            path=str(path),
        ) from exc

    for chunk in reader:
        stats.rows += len(chunk)
        is_vuln = chunk["target"] == 1
        stats.vulnerable += int(is_vuln.sum())

        indices = chunk["flaw_line_index"]
        has_lines = indices.notna() & (indices.astype(str).str.strip() != "")
        stats.vulnerable_with_lines += int((is_vuln & has_lines).sum())
        stats.vulnerable_without_lines += int((is_vuln & ~has_lines).sum())
        stats.nonvulnerable_with_lines += int((~is_vuln & has_lines).sum())

        for cell in chunk.loc[is_vuln & has_lines, "flaw_line_index"]:
            stats.flagged_lines += len(flaw_line_indices(cell))

        stats.projects.update(chunk["project"].dropna().astype(str))
        stats.cwes.update(chunk.loc[is_vuln, "CWE ID"].dropna().astype(str))
        stats.func_chars += int(chunk["processed_func"].dropna().astype(str).str.len().sum())

    _LOGGER.info("Scanned split", extra=stats.as_dict())
    return stats


def load_primevul_labels(primevul_dir: Path) -> dict[int, int]:
    """Build ``func_hash -> target`` over PrimeVul's unpaired splits.

    A hash seen as vulnerable in any split stays vulnerable: PrimeVul's own
    deduplication means the same function can appear with differing context, and the
    stronger claim is the informative one.

    Args:
        primevul_dir: Directory holding ``primevul_{train,valid,test}.jsonl``.

    Returns:
        Mapping from function hash to binary label.

    Raises:
        DataError: If an expected split file is missing.
    """
    labels: dict[int, int] = {}
    for name in ("primevul_train.jsonl", "primevul_valid.jsonl", "primevul_test.jsonl"):
        path = primevul_dir / name
        if not path.exists():
            raise DataError(f"PrimeVul split not found: {path}.", path=str(path))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = int(record["func_hash"])
                labels[key] = max(labels.get(key, 0), int(record["target"]))
    _LOGGER.info(
        "Loaded PrimeVul labels",
        extra={"functions": len(labels), "vulnerable": sum(labels.values())},
    )
    return labels


@dataclass
class OverlapReport:
    """Agreement between Big-Vul's line-labelled positives and PrimeVul's labels.

    ``absent`` is deliberately *not* treated as disagreement: PrimeVul deduplicated
    aggressively, so absence conflates "rejected" with "removed as duplicate"
    (ADR-003).
    """

    candidates: int = 0
    agrees_vulnerable: int = 0
    says_benign: int = 0
    absent: int = 0
    retained_flagged_lines: int = 0
    clean_eval_hashes: set[int] = field(default_factory=set)
    excluded_hashes: set[int] = field(default_factory=set)

    @property
    def decided(self) -> int:
        """Number of candidates PrimeVul holds an opinion on."""
        return self.agrees_vulnerable + self.says_benign

    @property
    def noise_rate(self) -> float:
        """Measured Big-Vul label-noise rate over the decided subset (ADR-003 §D3.3)."""
        return self.says_benign / self.decided if self.decided else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary (hash sets omitted)."""
        return {
            "candidates": self.candidates,
            "agrees_vulnerable": self.agrees_vulnerable,
            "says_benign": self.says_benign,
            "absent": self.absent,
            "decided": self.decided,
            "noise_rate_pct": round(100 * self.noise_rate, 2),
            "clean_eval_functions": len(self.clean_eval_hashes),
            "retained_flagged_lines": self.retained_flagged_lines,
        }


def overlap_with_primevul(
    bigvul_paths: list[Path],
    primevul_labels: dict[int, int],
    *,
    chunksize: int = _DEFAULT_CHUNK,
) -> OverlapReport:
    """Partition Big-Vul's line-labelled positives against PrimeVul's labels.

    Produces the clean evaluation set and the excluded set of ADR-003 §D3.1/§D3.2, and
    the noise estimate of §D3.3.

    Args:
        bigvul_paths: Big-Vul CSVs to scan.
        primevul_labels: Mapping from :func:`load_primevul_labels`.
        chunksize: Rows per chunk.

    Returns:
        The measured overlap report.
    """
    report = OverlapReport()

    for path in bigvul_paths:
        for chunk in pd.read_csv(
            path,
            usecols=["target", "flaw_line_index", "processed_func"],
            chunksize=chunksize,
            low_memory=False,
        ):
            indices = chunk["flaw_line_index"]
            keep = (
                (chunk["target"] == 1)
                & indices.notna()
                & (indices.astype(str).str.strip() != "")
                & chunk["processed_func"].notna()
            )
            subset = chunk.loc[keep, ["processed_func", "flaw_line_index"]]
            report.candidates += len(subset)

            for func, cell in zip(
                subset["processed_func"].astype(str), subset["flaw_line_index"], strict=True
            ):
                key = function_hash(func)
                label = primevul_labels.get(key)
                if label is None:
                    report.absent += 1
                elif label == 1:
                    report.agrees_vulnerable += 1
                    report.clean_eval_hashes.add(key)
                    report.retained_flagged_lines += len(flaw_line_indices(cell))
                else:
                    report.says_benign += 1
                    report.excluded_hashes.add(key)
        _LOGGER.info("Overlap scanned", extra={"path": path.name, **report.as_dict()})

    return report
