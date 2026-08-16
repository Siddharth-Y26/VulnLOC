"""Deterministic, order-independent selection primitives.

Every "random" choice in this project that has to survive a rerun is made here, and
none of them uses an RNG. A seeded RNG reproduces a sequence only if it is consumed in
the same order, and corpus order is not something we control: pandas chunking, a
re-download, or a loader change all reorder the input. Content hashing has no such
dependency — a unit's fate is a function of its identity alone.

Used by dataset splitting (`Phases.md` Phase 1: *re-running reproduces byte-identical
splits*) and by label-audit sampling, which must draw the same functions for a second
reviewer as it did for the first.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

__all__ = ["stable_sample", "unit_interval"]

# 2**64 — the digest prefix is 8 bytes wide.
_SCALE = float(1 << 64)


def unit_interval(seed: int, key: str) -> float:
    """Map ``(seed, key)`` to a stable value in ``[0, 1)``.

    Uses SHA-256 rather than :func:`hash`, whose randomization differs between
    processes, so the value is identical across runs, platforms and Python versions.

    Args:
        seed: Salt. Changing it reshuffles every assignment.
        key: The value being bucketed — a unit id, a project name.

    Returns:
        A deterministic float in ``[0, 1)``.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _SCALE


def stable_sample(keys: Iterable[str], k: int, *, seed: int = 42) -> list[str]:
    """Draw ``k`` keys reproducibly, independent of input order.

    Takes the ``k`` keys with the smallest hash value — a standard consistent-sampling
    trick whose useful property here is that the sample is a pure function of the key
    set. Ties break on the key itself, so equal hashes cannot reintroduce an order
    dependence.

    Args:
        keys: The population. Duplicates are ignored.
        k: How many to draw. A ``k`` at or above the population size returns everything.
        seed: Salt.

    Returns:
        The drawn keys, sorted by their hash value.
    """
    ranked = sorted({str(key) for key in keys}, key=lambda key: (unit_interval(seed, key), key))
    return ranked[: max(k, 0)]
