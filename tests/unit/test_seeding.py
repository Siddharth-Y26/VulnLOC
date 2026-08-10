"""Guards determinism (Architecture.md §8, Rules.md §5.2).

Reproducibility claims in the paper rest on this module, so the tests check that the
same seed really does reproduce the same draws across all three RNGs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from vulnloc.training.seeding import seed_everything, seed_worker
from vulnloc.utils.exceptions import ConfigError


def _draw() -> tuple[float, float, float]:
    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


def test_same_seed_reproduces_all_three_rngs():
    seed_everything(42)
    first = _draw()

    seed_everything(42)
    second = _draw()

    assert first == second


def test_different_seeds_diverge():
    seed_everything(42)
    first = _draw()

    seed_everything(43)
    second = _draw()

    assert first != second


def test_returns_the_applied_seed():
    assert seed_everything(7) == 7


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_out_of_range_seed_raises(seed):
    with pytest.raises(ConfigError, match="Seed must be in"):
        seed_everything(seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_boundary_seeds_are_accepted(seed):
    assert seed_everything(seed) == seed


def test_deterministic_flag_configures_cudnn():
    seed_everything(42, deterministic=True)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_non_deterministic_mode_still_seeds():
    seed_everything(42, deterministic=False)
    first = _draw()

    seed_everything(42, deterministic=False)

    assert _draw() == first


def test_worker_seeds_differ_across_workers():
    seed_everything(42)

    seed_worker(0)
    worker_0 = (random.random(), float(np.random.rand()))
    seed_worker(1)
    worker_1 = (random.random(), float(np.random.rand()))

    assert worker_0 != worker_1


def test_worker_seeding_is_reproducible():
    seed_everything(42)
    seed_worker(3)
    first = (random.random(), float(np.random.rand()))

    seed_everything(42)
    seed_worker(3)

    assert (random.random(), float(np.random.rand())) == first
