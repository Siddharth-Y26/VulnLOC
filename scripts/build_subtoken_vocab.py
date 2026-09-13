"""Build the subtoken vocabulary — the second half of ADR-002 §D2.5's node features.

``scripts/build_node_vocab.py`` produced the grammar-symbol table. This produces the
identifier table, and the two are **not** built the same way, for a reason worth stating
before the code.

**This vocabulary is built from the TRAINING SPLIT ONLY.** The node-type vocabulary was
scanned corpus-wide and that was correct: grammar symbols are a property of the two
parsers, not of the data, so seeing ``binary_expression`` in `val` tells the model nothing
about `val`. Identifiers are the opposite. A vocabulary scanned over `val` and `test` would
give the model a dedicated embedding row for names that appear only in the evaluation
data — a small but real form of leakage, and exactly the kind this project has already
been burned by once (ADR-006). Names absent from `train` therefore map to the unknown
slot at evaluation time, which is the honest behaviour and the one the reported number
should be measured under.

**Frequency-thresholded, not exhaustive.** An identifier occurring twice in 149k functions
gets an embedding row that receives two gradient updates and stays effectively random. The
`min_count` floor drops that tail into the unknown slot instead, where it at least shares
statistics with the rest of the tail. The report records how much of the corpus each
threshold would keep, so the choice is made against a number.

Slot 0 is reserved for unknown, slot 1 for padding — a node has between 0 and
``MAX_SUBTOKENS`` subtokens and the short ones are padded.

Reads the preprocessed shards only; it never opens a Big-Vul CSV.

Usage::

    python scripts/build_subtoken_vocab.py
    python scripts/build_subtoken_vocab.py graph.report.limit=20000
    python scripts/build_subtoken_vocab.py +subtokens.min_count=10
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
from vulnloc.representation.subtokens import split_identifier
from vulnloc.representation.units import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

UNKNOWN_INDEX = 0
"""Subtokens unseen in `train`, and every subtoken below ``min_count``."""

PADDING_INDEX = 1
"""Fills the unused slots of a node with fewer than ``MAX_SUBTOKENS`` subtokens."""

FIRST_REAL_INDEX = 2

CANDIDATE_THRESHOLDS = (1, 2, 5, 10, 25, 50)
"""Thresholds the report costs out, so ``min_count`` is chosen against coverage."""


def _digest(vocab: dict[str, int]) -> str:
    """Fingerprint a vocabulary so a changed table is detectable.

    Args:
        vocab: The mapping.

    Returns:
        A hex SHA-256 over the sorted ``name:index`` pairs.
    """
    payload = "\n".join(f"{name}:{index}" for name, index in sorted(vocab.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(cfg: DictConfig) -> dict[str, Any]:
    """Scan the training split and write the subtoken vocabulary.

    Args:
        cfg: Composed Hydra config. Uses ``graph.builder``, ``graph.language`` and
            ``graph.report`` for paths and limits, and ``subtokens.min_count`` when set.

    Returns:
        The report, as written to disk.

    Raises:
        DataError: If the training shard is missing.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    limit = int(settings.limit)
    builder_name = str(cfg.graph.builder)
    policy = str(cfg.graph.language)
    min_count = int(OmegaConf.select(cfg, "subtokens.min_count") or 5)

    # Training split only. See the module docstring: identifiers from `val`/`test` in the
    # table would be a leak, unlike grammar symbols.
    split = "train"
    if not shard_path(cache_dir, split).exists():
        raise DataError(
            f"training shard not found in {cache_dir}",
            path=str(shard_path(cache_dir, split)),
            hint="Run scripts/build_cache.py first.",
        )

    parser = TreeSitterParser()
    builder = GRAPH_REGISTRY.get(builder_name)()

    counts: Counter[str] = Counter()
    units_seen = 0
    failures = 0
    nodes_with_text = 0
    subtoken_occurrences = 0

    with start_run(cfg, run_name="subtoken-vocab") as run_id:
        for labelled in iter_shard(cache_dir, split):
            if limit and units_seen >= limit:
                break
            units_seen += 1
            unit = CodeUnit.from_labelled_function(labelled)
            try:
                parsed = parser.parse_with_policy(
                    unit.source, policy=policy, declared=unit.language, unit_id=unit.unit_id
                )
                graph = builder.build(parsed)
            except Exception:  # noqa: BLE001 - counted and reported, never silent
                failures += 1
                continue
            for node in graph.nodes:
                pieces = split_identifier(node.text)
                if pieces:
                    nodes_with_text += 1
                    subtoken_occurrences += len(pieces)
                counts.update(pieces)
            if units_seen % 20000 == 0:
                _LOGGER.info(
                    "Scanning", extra={"units": units_seen, "distinct_subtokens": len(counts)}
                )

        total_occurrences = sum(counts.values())
        # Cost out every candidate threshold, so `min_count` is a decision with a number
        # beside it rather than a round number someone liked.
        coverage = {
            str(threshold): {
                "kept_types": sum(1 for c in counts.values() if c >= threshold),
                "kept_occurrence_pct": round(
                    100.0
                    * sum(c for c in counts.values() if c >= threshold)
                    / max(total_occurrences, 1),
                    3,
                ),
            }
            for threshold in CANDIDATE_THRESHOLDS
        }

        kept = sorted(name for name, count in counts.items() if count >= min_count)
        vocab = {name: index for index, name in enumerate(kept, start=FIRST_REAL_INDEX)}

        payload = {
            "run_id": run_id,
            "split_scanned": split,
            "scanned_from_train_only": True,
            "why_train_only": (
                "Identifiers from val/test in the table would give the model embeddings "
                "for names that only appear in evaluation data. Grammar symbols are a "
                "property of the parser and were correctly scanned corpus-wide; these are "
                "not."
            ),
            "builder": builder_name,
            "language_policy": policy,
            "units_scanned": units_seen,
            "graph_build_failures": failures,
            "nodes_with_subtokens": nodes_with_text,
            "subtoken_occurrences": subtoken_occurrences,
            "distinct_subtokens_seen": len(counts),
            "min_count": min_count,
            "size": len(vocab) + FIRST_REAL_INDEX,
            "unknown_index": UNKNOWN_INDEX,
            "padding_index": PADDING_INDEX,
            "digest": _digest(vocab),
            "coverage_by_threshold": coverage,
            "most_common": dict(counts.most_common(200)),
            "vocab": vocab,
        }

        output_path = Path(settings.output_path).with_name(f"subtoken_vocab_{builder_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output_path))
        mlflow.log_params({"builder": builder_name, "min_count": min_count, "split_scanned": split})
        mlflow.log_metrics(
            {
                "vocab_size": float(payload["size"]),
                "distinct_subtokens_seen": float(len(counts)),
                "units_scanned": float(units_seen),
                "graph_build_failures": float(failures),
                "kept_occurrence_pct": (
                    float(coverage[str(min_count)]["kept_occurrence_pct"])
                    if str(min_count) in coverage
                    else 0.0
                ),
            }
        )
        _LOGGER.info(
            "Subtoken vocabulary written",
            extra={
                "path": str(output_path),
                "size": payload["size"],
                "digest": payload["digest"][:12],
                "distinct_seen": len(counts),
                "min_count": min_count,
            },
        )

    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
