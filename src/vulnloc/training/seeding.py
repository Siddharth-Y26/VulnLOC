"""Determinism controls (``docs/Architecture.md`` §8).

Every run records its seed in the run config, and every reported comparison is a
mean ± std over at least three seeds (``Rules.md`` §5.2). Reproducibility is therefore
infrastructure, not an afterthought: this module is the single place seeds are set.

Note:
    Full determinism costs throughput and, for a few CUDA kernels, is unavailable.
    :func:`seed_everything` with ``deterministic=True`` requests deterministic
    algorithms and reports what it could not guarantee rather than pretending.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger

__all__ = ["seed_everything", "seed_worker"]

_LOGGER = get_logger(__name__)

# Required by cuBLAS for deterministic GEMMs; must be set before CUDA initializes.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MAX_SEED = 2**32 - 1


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) and optionally force determinism.

    Args:
        seed: Seed value; must fit in an unsigned 32-bit integer.
        deterministic: Request deterministic algorithms and disable cuDNN
            benchmarking. Set False to trade reproducibility for speed.

    Returns:
        The seed that was applied, for convenient logging.

    Raises:
        ConfigError: If ``seed`` is outside ``[0, 2**32 - 1]``.
    """
    if not 0 <= seed <= _MAX_SEED:
        raise ConfigError(f"Seed must be in [0, {_MAX_SEED}], got {seed}.", seed=seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some ops (e.g. scatter reductions used by the line aligner) have
        # no deterministic kernel. Warn loudly instead of crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)

    _LOGGER.info(
        "Seeded run",
        extra={"seed": seed, "deterministic": deterministic, "cuda": torch.cuda.is_available()},
    )
    return seed


def seed_worker(worker_id: int) -> None:
    """Seed a ``torch.utils.data.DataLoader`` worker process.

    Pass as ``DataLoader(worker_init_fn=seed_worker)``. Without this, each worker
    inherits the same NumPy/Python seed and any random augmentation or negative
    sampling is duplicated across workers.

    Args:
        worker_id: Worker index supplied by the DataLoader.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (_MAX_SEED + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
