"""Statistics over seeds and over units (Phase 8; ``docs/EvaluationProtocol.md`` §6).

Three seeds an arm is what this project has, and three numbers support very little. What
lives here is the little they do support, stated the same way every time:

* an arm is its **mean and sample standard deviation** over seeds;
* a difference between two arms is given **paired by seed** and in units of the **pooled
  standard deviation**, and one under :data:`RESOLVED_AT` pooled sd is "not resolved" —
  the rule every comparison in ``Memory.md`` has been written under since two one-seed
  findings were retracted (§7 #37);
* a **bootstrap interval over units** says how far one checkpoint's number would move on
  another sample of functions — a different source of noise from the seed, and usually
  the larger one on a partition of a few hundred units.

No significance test is offered. With n = 3 a p-value would dress the same three numbers
in more authority than they have.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnloc.utils.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["RESOLVED_AT", "bootstrap_mean_interval", "compare_arms", "pooled_sd", "spread"]

RESOLVED_AT = 2.0
"""A difference smaller than this many pooled standard deviations is "not resolved"."""

_BOOTSTRAP_CHUNK = 1000
"""Resamples drawn at a time. Fixed, because it is part of what the seed reproduces."""


def spread(values: Sequence[float]) -> dict[str, Any]:
    """Mean and sample standard deviation.

    Args:
        values: One number per seed.

    Returns:
        ``{"mean", "std", "values"}``. ``std`` is ``None`` for a single value: one seed has
        no spread, and a 0.0 there would read as a measured one.

    Raises:
        DataError: If there are no values.
    """
    numbers = [float(value) for value in values]
    if not numbers:
        raise DataError("no values to summarise")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "values": numbers,
    }


def pooled_sd(std_a: float, std_b: float) -> float:
    """Pooled standard deviation of two arms with the same number of seeds.

    Args:
        std_a: Sample standard deviation of the first arm.
        std_b: Sample standard deviation of the second.

    Returns:
        ``sqrt((std_a**2 + std_b**2) / 2)``.
    """
    return math.sqrt((std_a**2 + std_b**2) / 2.0)


def compare_arms(first: Mapping[int, float], second: Mapping[int, float]) -> dict[str, Any]:
    """``first - second`` for one metric, paired by seed.

    Args:
        first: ``seed -> value`` for the first arm.
        second: ``seed -> value`` for the second arm.

    Returns:
        ``{"difference", "pooled_sd", "difference_in_pooled_sd", "resolved", "by_seed",
        "same_sign_on_every_seed", "n_seeds"}``. ``difference_in_pooled_sd`` is ``None``
        when there is one seed or neither arm varies; ``resolved`` is then ``False`` — a
        difference that cannot be scaled has not been shown to exceed the noise.

    Raises:
        DataError: If the arms were not run on the same seeds. A comparison over
            different seeds is not paired, and averaging over the seeds they share would
            quietly compare fewer runs than the tables report.
    """
    if set(first) != set(second) or not first:
        raise DataError(
            "arms can only be compared on the same, non-empty set of seeds",
            first=sorted(first),
            second=sorted(second),
        )
    seeds = sorted(first)
    by_seed = {seed: float(first[seed]) - float(second[seed]) for seed in seeds}
    a = spread([first[seed] for seed in seeds])
    b = spread([second[seed] for seed in seeds])
    difference = a["mean"] - b["mean"]
    pooled = None if a["std"] is None or b["std"] is None else pooled_sd(a["std"], b["std"])
    scaled = difference / pooled if pooled else None
    signs = {(value > 0) - (value < 0) for value in by_seed.values()}
    return {
        "difference": difference,
        "pooled_sd": pooled,
        "difference_in_pooled_sd": scaled,
        "resolved": scaled is not None and abs(scaled) >= RESOLVED_AT,
        "by_seed": {str(seed): value for seed, value in by_seed.items()},
        "same_sign_on_every_seed": len(signs) == 1 and 0 not in signs,
        "n_seeds": len(seeds),
    }


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-unit values.

    Units are resampled with replacement, never lines: the lines of one function are not
    independent draws.

    Args:
        values: ``[n_units]`` — one value per unit (a Top-k hit, an IFA count).
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.
        level: Coverage of the interval.

    Returns:
        ``(low, high)``.

    Raises:
        DataError: If there are no units, no resamples, or ``level`` is not in ``(0, 1)``.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise DataError("a bootstrap needs a flat, non-empty set of units", shape=data.shape)
    if resamples <= 0 or not 0.0 < level < 1.0:
        raise DataError("invalid bootstrap settings", resamples=resamples, level=level)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, _BOOTSTRAP_CHUNK):
        size = min(_BOOTSTRAP_CHUNK, resamples - start)
        draw = generator.integers(0, data.size, size=(size, data.size))
        means[start : start + size] = data[draw].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)
