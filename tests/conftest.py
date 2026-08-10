"""Shared pytest fixtures.

Phase 0 fixtures cover only what the scaffolding needs: composing the real Hydra
configs, and redirecting every side effect (MLflow store, log files) into a temporary
directory so tests never touch the developer's ``artifacts/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """Absolute path to the repository's Hydra config root."""
    return CONFIG_DIR


@pytest.fixture
def compose_config() -> Iterator[object]:
    """Return a callable that composes ``configs/config.yaml`` with overrides.

    Yields:
        ``compose_config(*overrides) -> DictConfig``, e.g.
        ``compose_config("train=debug", "seed=7")``.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):

        def _compose(*overrides: str) -> DictConfig:
            return compose(config_name="config", overrides=list(overrides))

        yield _compose


@pytest.fixture
def isolated_cfg(compose_config, tmp_path: Path):  # type: ignore[no-untyped-def]
    """A composed config whose MLflow store and log directory live under ``tmp_path``.

    Returns:
        ``isolated_cfg(*overrides) -> DictConfig`` with ``mlflow.tracking_uri``,
        ``mlflow.experiment_name`` and ``logging.log_dir`` rewritten.
    """

    def _isolated(*overrides: str) -> DictConfig:
        cfg = compose_config(*overrides)
        with open_dict(cfg):
            cfg.mlflow.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
            cfg.mlflow.artifact_location = (tmp_path / "mlartifacts").as_uri()
            cfg.mlflow.experiment_name = "pytest"
            cfg.logging.log_dir = str(tmp_path / "logs")
        return cfg

    return _isolated


@pytest.fixture(autouse=True)
def _reset_vulnloc_logger() -> Iterator[None]:
    """Detach handlers added by ``setup_logging`` so tests cannot leak file handles."""
    yield
    logger = logging.getLogger("vulnloc")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
