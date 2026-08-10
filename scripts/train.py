"""Training entrypoint (``docs/Architecture.md`` §5).

Phase 0 scope: compose the Hydra config, seed the run, open an MLflow run, and execute
a **no-op loop** that logs a dummy metric. This exists to prove the plumbing — config
composition, determinism, logging, tracking — before any model code is written.

The real training loop lands in Phase 3 (LLM branch) and Phase 6 (full training).

Usage::

    python scripts/train.py train=debug
    python scripts/train.py train=debug seed=7
    python scripts/train.py train=debug --cfg job     # print resolved config, run nothing
"""

from __future__ import annotations

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from vulnloc.training.device import log_accelerator, resolve_device
from vulnloc.training.seeding import seed_everything
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import provenance
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def run_training(cfg: DictConfig) -> str:
    """Execute the Phase 0 no-op training loop under a tracked MLflow run.

    Args:
        cfg: Composed Hydra config. Requires ``seed``, ``device``, ``logging``,
            ``mlflow`` and a ``train`` group providing ``name`` and ``max_steps``.

    Returns:
        The MLflow run id, so callers (and tests) can trace the run.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    seed_everything(int(cfg.seed), deterministic=True)
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))

    with start_run(cfg, run_name=str(cfg.train.name)) as run_id:
        _LOGGER.info("Provenance", extra=provenance(cfg))
        mlflow.set_tags(log_accelerator(device))
        max_steps = int(cfg.train.max_steps)

        # Phase 0: no model, no data, no optimizer. The loop proves two things only —
        # that a metric reaches the tracking backend with a step index attached, and
        # that a tensor round-trips through the selected device. The second is what
        # turns "CUDA is available" into "CUDA actually computes".
        for step in range(max_steps):
            probe = torch.ones(8, 8, device=device)
            loss = float((probe @ probe).mean().item()) / (step + 1)
            mlflow.log_metric("noop/loss", loss, step=step)
            _LOGGER.info("step", extra={"step": step, "of": max_steps, "device": str(device)})

        mlflow.log_metric("noop/completed_steps", float(max_steps))
        _LOGGER.info(
            "No-op training complete",
            extra={"steps": max_steps, "run_id": run_id, "device": str(device)},
        )
        return run_id


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    run_training(cfg)


if __name__ == "__main__":
    main()
