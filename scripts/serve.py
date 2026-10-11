r"""Serve one bundle over HTTP (``docs/Phases.md`` Phase 9).

The bundle — built and checked by ``scripts/export_serving_bundle.py`` — is named by
``+serve.bundle=<directory>`` or the ``VULNLOC_BUNDLE`` environment variable. The service
listens on ``127.0.0.1`` unless told otherwise: it has no authentication, so exposing it
beyond the machine is a decision, made with ``+serve.host=0.0.0.0`` (as the compose file
does inside its container).

Usage::

    python scripts/serve.py +serve.bundle=./artifacts/serving/ba7dcc5f
    # then: http://127.0.0.1:8080/docs

    curl -s http://127.0.0.1:8080/analyze -H "Content-Type: application/json" \
        -d '{"code": "void f(char *s) {\n  char b[8];\n  strcpy(b, s);\n}\n"}'
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import uvicorn
from omegaconf import DictConfig, OmegaConf

from vulnloc.serving.app import BUNDLE_ENV, create_app
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def serve(cfg: DictConfig) -> None:
    """Load the bundle and run the server until it is stopped.

    Args:
        cfg: Composed config. Reads ``serve.bundle``, ``serve.host`` and ``serve.port``.

    Raises:
        ConfigError: If no bundle is named, or the directory does not exist.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    bundle = OmegaConf.select(cfg, "serve.bundle") or os.environ.get(BUNDLE_ENV)
    if not bundle:
        raise ConfigError(
            "no bundle to serve",
            hint=f"Pass '+serve.bundle=<directory>' or set {BUNDLE_ENV}.",
        )
    if not Path(str(bundle)).is_dir():
        raise ConfigError("the bundle directory does not exist", path=str(bundle))
    host = str(OmegaConf.select(cfg, "serve.host") or DEFAULT_HOST)
    port = int(OmegaConf.select(cfg, "serve.port") or DEFAULT_PORT)
    _LOGGER.info("Serving", extra={"bundle": str(bundle), "host": host, "port": port})
    uvicorn.run(create_app(bundle_dir=str(bundle)), host=host, port=port, log_level="info")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    serve(cfg)


if __name__ == "__main__":
    main()
