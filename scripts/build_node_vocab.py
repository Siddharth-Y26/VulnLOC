"""Build the node-type vocabulary the structural branch embeds — Phase 4's prerequisite.

ADR-002 §D2.5 fixes node features as a node-type embedding plus a subtoken embedding.
This script produces the first half's lookup table, and it produces it **by scanning the
corpus** rather than by reading a grammar's symbol list.

**Why that distinction is the whole point.** ADR-005 parses each unit with whichever of
the two grammars parses it best, so the node types occurring across the corpus are a
*union* over `tree-sitter-c` and `tree-sitter-cpp`. Either grammar's own symbol list is
therefore both incomplete (it misses the other's symbols) and wrong (it contains symbols
that never occur). `Memory.md` §7 records this as a landmine precisely because an
embedding table sized from one grammar looks completely healthy: unseen types silently
fold into slot 0 and the model simply never learns them.

Slot 0 is reserved for unknown types, so a symbol first seen at inference degrades one
node instead of failing a batch.

The vocabulary is an **artefact with a digest**, not something recomputed per run. An
embedding table whose indices shift between runs makes two checkpoints incomparable
without anything raising, which is the same failure mode the split fingerprint exists to
catch one layer down.

Reads the preprocessed shards only; it never opens a Big-Vul CSV.

Usage::

    python scripts/build_node_vocab.py
    python scripts/build_node_vocab.py graph.builder=cpg
    python scripts/build_node_vocab.py 'graph.report.splits=[train]' graph.report.limit=500
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
from vulnloc.representation.units import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

UNKNOWN_INDEX = 0
"""Reserved slot for grammar symbols unseen at vocabulary-build time."""


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
    """Scan the corpus and write the node-type vocabulary.

    Args:
        cfg: Composed Hydra config. Uses ``graph.builder``, ``graph.language`` and
            ``graph.report`` for paths and limits.

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
    limit = int(settings.limit)
    builder_name = str(cfg.graph.builder)
    policy = str(cfg.graph.language)

    requested = [str(s) for s in settings.splits]
    splits = sorted(requested) if requested else sorted(p.stem for p in cache_dir.glob("*.jsonl"))
    if not splits:
        raise DataError(
            f"No shards in {cache_dir}. Run scripts/build_cache.py first.",
            path=str(cache_dir),
        )

    parser = TreeSitterParser()
    builder = GRAPH_REGISTRY.get(builder_name)()

    # Counted, not just collected. A type occurring twice in 188,636 functions and one
    # occurring four million times both get an embedding row, and knowing which is which
    # is what tells Phase 4 whether the tail is worth keeping.
    counts: Counter[str] = Counter()
    per_split: dict[str, Any] = {}
    failures = 0
    units_seen = 0

    with start_run(cfg, run_name="node-vocab") as run_id:
        for split in splits:
            path = shard_path(cache_dir, split)
            if not path.exists():
                raise DataError(f"Shard not found: {path}", path=str(path))

            split_types: set[str] = set()
            split_units = 0
            for labelled in iter_shard(cache_dir, split):
                if limit and split_units >= limit:
                    break
                split_units += 1
                units_seen += 1
                unit = CodeUnit.from_labelled_function(labelled)
                try:
                    parsed = parser.parse_with_policy(
                        unit.source,
                        policy=policy,
                        declared=unit.language,
                        unit_id=unit.unit_id,
                    )
                    graph = builder.build(parsed)
                except Exception:  # noqa: BLE001 - counted and reported, never silent
                    # A unit that will not build a graph contributes no symbols. It is
                    # counted rather than ignored: if this number is large the vocabulary
                    # is missing whatever those units contain.
                    failures += 1
                    continue
                for node in graph.nodes:
                    counts[node.node_type] += 1
                    split_types.add(node.node_type)

            per_split[split] = {"units": split_units, "distinct_types": len(split_types)}
            _LOGGER.info(
                "Split scanned",
                extra={
                    "split": split,
                    "units": split_units,
                    "distinct_types": len(split_types),
                    "running_total_types": len(counts),
                },
            )

        vocab = {name: index for index, name in enumerate(sorted(counts), start=1)}
        singletons = [name for name, count in counts.items() if count == 1]

        payload = {
            "run_id": run_id,
            "builder": builder_name,
            "language_policy": policy,
            "units_scanned": units_seen,
            "graph_build_failures": failures,
            "size": len(vocab) + 1,
            "unknown_index": UNKNOWN_INDEX,
            "digest": _digest(vocab),
            "vocab": vocab,
            "counts": dict(counts.most_common()),
            "singleton_types": sorted(singletons),
            "per_split": per_split,
        }

        output_path = Path(settings.output_path).with_name(f"node_vocab_{builder_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output_path))
        mlflow.log_params({"builder": builder_name, "language_policy": policy})
        mlflow.log_metrics(
            {
                "vocab_size": float(len(vocab) + 1),
                "units_scanned": float(units_seen),
                "graph_build_failures": float(failures),
                "singleton_types": float(len(singletons)),
            }
        )
        _LOGGER.info(
            "Vocabulary written",
            extra={
                "path": str(output_path),
                "size": len(vocab) + 1,
                "digest": payload["digest"][:12],
                "failures": failures,
                "singletons": len(singletons),
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
