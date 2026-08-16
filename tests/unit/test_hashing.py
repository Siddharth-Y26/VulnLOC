"""Guards the deterministic selection primitives.

These sit under both dataset splitting and label-audit sampling, so the property that
matters is not "reproducible with the same seed" — a seeded RNG has that — but
*reproducible regardless of input order*. Corpus order is not under our control.
"""

from __future__ import annotations

import random

from vulnloc.utils.hashing import stable_sample, unit_interval

KEYS = [f"bigvul-{i}" for i in range(500)]


def test_interval_is_in_range_and_stable():
    values = [unit_interval(42, key) for key in KEYS]

    assert all(0.0 <= v < 1.0 for v in values)
    assert values == [unit_interval(42, key) for key in KEYS]


def test_interval_is_not_pythons_hash():
    # str.__hash__ is salted per process; a split built on it would differ every run.
    assert unit_interval(42, "bigvul-1") == 0.4027377436665945


def test_seed_changes_the_assignment():
    assert unit_interval(1, "u") != unit_interval(2, "u")


def test_sample_ignores_input_order():
    shuffled = KEYS.copy()
    random.Random(0).shuffle(shuffled)

    assert stable_sample(KEYS, 20) == stable_sample(shuffled, 20)


def test_sample_is_a_prefix_as_k_grows():
    # Growing the sample must extend it, not redraw it — otherwise a reviewer asked for
    # twenty more functions would have to re-inspect the first twenty.
    assert stable_sample(KEYS, 30)[:10] == stable_sample(KEYS, 10)


def test_sample_deduplicates_and_caps_at_the_population():
    assert len(stable_sample(["a", "a", "b"], 10)) == 2


def test_sample_of_zero_is_empty():
    assert stable_sample(KEYS, 0) == []
