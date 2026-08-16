"""Preprocessed shards: the corpus in the form later phases actually consume (L0).

Big-Vul ships as 11 GB of CSV, most of it columns we never read — whole pre- and
post-fix file contents. A full pass costs minutes and every phase after this one would
pay it repeatedly. Shards are that pass, done once: one JSONL file per partition
holding exactly the fields :class:`~vulnloc.data.labeling.LabelledFunction` carries,
which is ~2% of the raw size.

The shards are the boundary between "reading a corpus" and "training on a dataset".
Nothing downstream should open a Big-Vul CSV again.

Determinism: records are written in corpus order, which is itself fixed by the file
contents, and each shard's SHA-256 goes into the manifest. A shard whose digest changed
means the corpus, the loader or the split changed — the same failure the split
fingerprint catches, one layer up.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vulnloc.data.labeling import LabelIssue, LabelledFunction
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

__all__ = [
    "MANIFEST_FILENAME",
    "ShardStats",
    "iter_shard",
    "shard_path",
    "write_shards",
]

_LOGGER = get_logger(__name__)

MANIFEST_FILENAME = "shards.json"


def shard_path(cache_dir: Path | str, split: str) -> Path:
    """Return the path of one partition's shard.

    Args:
        cache_dir: Directory holding the shards.
        split: Partition name.

    Returns:
        The shard file path.
    """
    return Path(cache_dir) / f"{split}.jsonl"


def _encode(unit: LabelledFunction) -> str:
    return json.dumps(
        {
            "unit_id": unit.unit_id,
            "source": unit.source,
            "language": unit.language,
            "target": unit.target,
            # Sorted, so a shard line is a canonical form of the record rather than a
            # snapshot of set iteration order.
            "vulnerable_lines": sorted(unit.vulnerable_lines),
            "n_lines": unit.n_lines,
            "issues": [str(issue) for issue in unit.issues],
            # Missing values are kept as explicit nulls rather than dropped. Absence and
            # "recorded as absent" are different facts, and a shard that cannot
            # round-trip its input exactly is a shard nobody can verify.
            "metadata": dict(unit.metadata),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _decode(line: str) -> LabelledFunction:
    record = json.loads(line)
    return LabelledFunction(
        unit_id=record["unit_id"],
        source=record["source"],
        language=record["language"],
        target=int(record["target"]),
        vulnerable_lines=frozenset(record["vulnerable_lines"]),
        n_lines=int(record["n_lines"]),
        issues=tuple(LabelIssue(issue) for issue in record["issues"]),
        metadata=record.get("metadata", {}),
    )


def iter_shard(cache_dir: Path | str, split: str) -> Iterator[LabelledFunction]:
    """Read one partition's shard back.

    Args:
        cache_dir: Directory holding the shards.
        split: Partition name.

    Yields:
        Each labelled function, exactly as written.

    Raises:
        DataError: If the shard does not exist.
    """
    path = shard_path(cache_dir, split)
    if not path.exists():
        raise DataError(
            f"Shard not found: {path}. Run scripts/build_cache.py.", path=str(path), split=split
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _decode(line)


@dataclass
class ShardStats:
    """Per-partition write counters.

    Attributes:
        split: Partition name.
        functions: Records written.
        line_labelled: Records carrying line-level supervision.
        flagged_lines: Total flagged lines written.
        issues: Derivation issues seen, by kind.
        digest: SHA-256 of the shard file.
        bytes_written: Size of the shard on disk.
    """

    split: str
    functions: int = 0
    line_labelled: int = 0
    flagged_lines: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    digest: str = ""
    bytes_written: int = 0

    def record(self, unit: LabelledFunction) -> None:
        """Fold one written record into the counters.

        Args:
            unit: The function just written.
        """
        self.functions += 1
        self.line_labelled += int(unit.has_line_labels)
        self.flagged_lines += len(unit.vulnerable_lines)
        for issue in unit.issues:
            self.issues[str(issue)] = self.issues.get(str(issue), 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary."""
        return {
            "split": self.split,
            "functions": self.functions,
            "line_labelled": self.line_labelled,
            "flagged_lines": self.flagged_lines,
            "issues": dict(sorted(self.issues.items())),
            "digest": self.digest,
            "bytes": self.bytes_written,
        }


def write_shards(
    units: Iterable[LabelledFunction],
    assignment: Mapping[str, str],
    cache_dir: Path | str,
    *,
    splits: Iterable[str] | None = None,
) -> dict[str, ShardStats]:
    """Write each labelled function to its partition's shard.

    Args:
        units: The corpus, already labelled.
        assignment: Mapping of ``unit_id`` to partition name, from
            ``artifacts/data/splits/assignment.csv``.
        cache_dir: Directory to write into.
        splits: Restrict to these partitions. ``None`` writes every partition present in
            the assignment, including the withheld ones — ``excluded`` and ``duplicate``
            are cheap and being able to inspect what was dropped is worth more than the
            disk they cost.

    Returns:
        Per-partition statistics, including each shard's digest.

    Raises:
        DataError: If a unit is missing from the assignment. That means the split and
            the corpus disagree, and silently skipping the unit would produce shards
            that quietly omit part of the data.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = set(splits) if splits is not None else None

    handles: dict[str, Any] = {}
    digests: dict[str, Any] = {}
    stats: dict[str, ShardStats] = {}
    missing = 0
    first_missing: str | None = None

    try:
        for unit in units:
            split = assignment.get(unit.unit_id)
            if split is None:
                missing += 1
                first_missing = first_missing or unit.unit_id
                continue
            if wanted is not None and split not in wanted:
                continue
            if split not in handles:
                handles[split] = shard_path(directory, split).open(
                    "w", encoding="utf-8", newline="\n"
                )
                digests[split] = hashlib.sha256()
                stats[split] = ShardStats(split)
            line = _encode(unit) + "\n"
            handles[split].write(line)
            digests[split].update(line.encode("utf-8"))
            stats[split].record(unit)
    finally:
        for handle in handles.values():
            handle.close()

    if missing:
        raise DataError(
            f"{missing} functions are absent from the split assignment (first: "
            f"{first_missing}). The corpus and the split disagree — rebuild the split "
            f"before caching, or the shards will silently omit data.",
            missing=missing,
            example=first_missing,
        )

    for split, stat in stats.items():
        stat.digest = digests[split].hexdigest()
        stat.bytes_written = shard_path(directory, split).stat().st_size
        _LOGGER.info("Shard written", extra=stat.as_dict())

    manifest = {
        "shards": {split: stat.as_dict() for split, stat in sorted(stats.items())},
        "total_functions": sum(stat.functions for stat in stats.values()),
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats
