"""Preprocessed-shard entrypoint — Phase 1.

Reads the raw corpus once, joins it to the split assignment, and writes one JSONL shard
per partition. After this, nothing downstream needs to open an 11 GB CSV.

Requires an existing assignment: ``python scripts/build_splits.py`` first. The two are
kept as separate steps deliberately — the split is a decision with a fingerprint, the
cache is a derived artefact of it, and rebuilding the cache must never be able to
quietly change the split.

Usage::

    python scripts/build_cache.py
    python scripts/build_cache.py 'data.cache.splits=[train,val]'
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.cache import MANIFEST_FILENAME, write_shards
from vulnloc.data.pipeline import ASSIGNMENT_FILENAME
from vulnloc.data.registry import DATASET_REGISTRY
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def load_assignment(path: Path) -> dict[str, str]:
    """Read ``unit_id,split`` pairs written by ``scripts/build_splits.py``.

    Args:
        path: Path to ``assignment.csv``.

    Returns:
        Mapping of unit id to partition name.

    Raises:
        DataError: If the file is missing or is not an assignment.
    """
    if not path.exists():
        raise DataError(
            f"Split assignment not found: {path}. Run scripts/build_splits.py first.",
            path=str(path),
        )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if tuple(header or ()) != ("unit_id", "split"):
            raise DataError(f"{path.name} is not a split assignment.", path=str(path))
        return {row[0]: row[1] for row in reader if row}


def build(cfg: DictConfig) -> dict[str, Any]:
    """Write the preprocessed shards.

    Args:
        cfg: Composed Hydra config. Requires ``data.raw_dir`` and ``data.cache``.

    Returns:
        The shard manifest.

    Raises:
        DataError: If the assignment is missing or disagrees with the corpus.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.data.cache
    assignment = load_assignment(Path(cfg.data.splits.output_dir) / ASSIGNMENT_FILENAME)
    splits = [str(s) for s in settings.splits] or None

    loader = DATASET_REGISTRY.get(str(settings.dataset))(
        cfg.data.raw_dir,
        splits=tuple(str(s) for s in settings.source_splits),
        normalize_whitespace=bool(cfg.data.normalize_whitespace),
    )

    with start_run(cfg, run_name="cache") as run_id:
        stats = write_shards(
            loader.iter_functions(), assignment, settings.output_dir, splits=splits
        )
        manifest = json.loads(
            (Path(settings.output_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        manifest["run_id"] = run_id
        (Path(settings.output_dir) / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_metrics(
            {f"cache/{split}/functions": float(stat.functions) for split, stat in stats.items()}
        )
        mlflow.log_artifact(
            str(Path(settings.output_dir) / MANIFEST_FILENAME), artifact_path="data"
        )

    _LOGGER.info("Cache built", extra={"shards": len(stats), "run_id": run_id})
    return manifest


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
