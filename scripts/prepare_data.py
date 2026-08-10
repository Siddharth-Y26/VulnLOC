"""Raw corpus acquisition entrypoint (ADR-001).

Phase 1 scope: fetch and verify the raw corpora. Labelling, splitting and leakage
analysis are separate steps and are not performed here.

Usage::

    python scripts/prepare_data.py                       # fetch everything
    python scripts/prepare_data.py data.sources=[primevul]
    python scripts/prepare_data.py data.force=true       # re-download
    python scripts/prepare_data.py --cfg job             # print config, fetch nothing
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from vulnloc.data.download import download_all
from vulnloc.data.sources import SOURCES
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def prepare(cfg: DictConfig) -> dict[str, Path]:
    """Fetch the configured corpora into the raw-data directory.

    Args:
        cfg: Composed Hydra config. Requires a ``data`` section with ``raw_dir``,
            optional ``sources`` (defaults to all) and optional ``force``.

    Returns:
        Mapping of source name to the downloaded path.

    Raises:
        DataError: If any source fails; the message names every failure.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )

    root = Path(cfg.data.raw_dir)
    requested = cfg.data.get("sources")
    names = None if not requested else [str(name) for name in requested]

    unknown = sorted(set(names or []) - set(SOURCES))
    if unknown:
        raise DataError(
            f"Unknown data sources requested: {unknown}. Available: {sorted(SOURCES)}.",
            unknown=unknown,
        )

    _LOGGER.info(
        "Preparing corpora",
        extra={"root": str(root), "sources": names or sorted(SOURCES)},
    )
    fetched = download_all(root, names=names, force=bool(cfg.data.get("force", False)))

    for name, path in sorted(fetched.items()):
        source = SOURCES[name]
        _LOGGER.info(
            "Ready",
            extra={
                "source": name,
                "role": str(source.role),
                "license": source.license,
                "path": str(path),
            },
        )
    return fetched


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    prepare(cfg)


if __name__ == "__main__":
    main()
