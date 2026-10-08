"""Guards the statistics over seeds and units (``docs/EvaluationProtocol.md`` §6).

The failure these guard against is a comparison that reads as stronger than three seeds
can make it: a spread reported for one run, arms compared over different seeds, a
difference called resolved because nothing scaled it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnloc.evaluation.statistics import (
    RESOLVED_AT,
    bootstrap_mean_interval,
    compare_arms,
    pooled_sd,
    spread,
)
from vulnloc.utils.exceptions import DataError

# `val` Top-5 of the retrained arms, as `run_queue.py --status` printed them on 2026-10-08:
# cross_attention ba7dcc5f / 8afbf7da / ff980345, concat 9ac7a146 / 7579bc33 / 10b43ce5.
CROSS_ATTENTION = {42: 0.5519, 1: 0.5373, 2: 0.5290}
CONCAT = {42: 0.5456, 1: 0.5622, 2: 0.5560}


def test_spread_is_the_mean_and_the_sample_standard_deviation():
    result = spread([1.0, 2.0, 3.0])

    assert result == {"mean": 2.0, "std": 1.0, "values": [1.0, 2.0, 3.0]}


def test_one_seed_has_no_standard_deviation():
    """``None``, not 0.0: a zero there would read as a measured absence of spread."""
    assert spread([0.5])["std"] is None


def test_nothing_to_summarise_is_an_error():
    with pytest.raises(DataError):
        spread([])


def test_pooled_sd_is_the_root_mean_of_the_two_variances():
    assert pooled_sd(3.0, 4.0) == pytest.approx((12.5) ** 0.5)
    assert pooled_sd(0.2, 0.2) == pytest.approx(0.2)


def test_a_logged_comparison_is_reproduced_and_is_not_resolved():
    """Cross-attention minus concat, retrained: −0.0152 Top-5, about 1.5 pooled sd."""
    result = compare_arms(CROSS_ATTENTION, CONCAT)

    assert result["difference"] == pytest.approx(-0.0152, abs=1e-4)
    assert result["difference_in_pooled_sd"] == pytest.approx(-1.50, abs=0.02)
    assert result["resolved"] is False
    assert result["n_seeds"] == 3
    assert result["by_seed"] == {
        "1": pytest.approx(-0.0249),
        "2": pytest.approx(-0.0270),
        "42": pytest.approx(0.0063),
    }
    assert result["same_sign_on_every_seed"] is False


def test_a_difference_of_two_pooled_sd_is_resolved():
    result = compare_arms({1: 0.70, 2: 0.72, 3: 0.71}, {1: 0.50, 2: 0.52, 3: 0.51})

    assert abs(result["difference_in_pooled_sd"]) >= RESOLVED_AT
    assert result["resolved"] is True
    assert result["same_sign_on_every_seed"] is True


def test_arms_run_on_different_seeds_are_not_compared():
    """Averaging over the seeds they share would compare fewer runs than the table shows."""
    with pytest.raises(DataError, match="same"):
        compare_arms({1: 0.5, 2: 0.6}, {1: 0.5, 3: 0.6})


def test_one_seed_gives_a_difference_and_nothing_to_scale_it_by():
    result = compare_arms({42: 0.6}, {42: 0.5})

    assert result["difference"] == pytest.approx(0.1)
    assert result["pooled_sd"] is None
    assert result["difference_in_pooled_sd"] is None
    assert result["resolved"] is False


def test_arms_that_do_not_vary_cannot_resolve_a_difference():
    result = compare_arms({1: 0.6, 2: 0.6}, {1: 0.5, 2: 0.5})

    assert result["pooled_sd"] == 0.0
    assert result["difference_in_pooled_sd"] is None
    assert result["resolved"] is False


def test_a_seed_on_which_the_arms_tie_breaks_the_same_sign_claim():
    result = compare_arms({1: 0.6, 2: 0.5}, {1: 0.5, 2: 0.5})

    assert result["same_sign_on_every_seed"] is False


def test_the_bootstrap_is_reproducible_for_a_seed_and_moves_with_it():
    values = np.random.default_rng(0).random(200)

    first = bootstrap_mean_interval(values, resamples=2000, seed=42)
    again = bootstrap_mean_interval(values, resamples=2000, seed=42)
    other = bootstrap_mean_interval(values, resamples=2000, seed=7)

    assert first == again
    assert first != other


def test_the_interval_contains_the_mean_it_is_an_interval_for():
    values = (np.random.default_rng(1).random(300) < 0.55).astype(float)

    low, high = bootstrap_mean_interval(values, resamples=3000)

    assert low < values.mean() < high
    assert high - low < 0.2


def test_units_that_all_agree_give_an_interval_of_no_width():
    assert bootstrap_mean_interval([1.0] * 50, resamples=500) == (1.0, 1.0)


def test_more_units_make_a_narrower_interval():
    generator = np.random.default_rng(2)
    few = bootstrap_mean_interval(generator.random(40), resamples=2000)
    many = bootstrap_mean_interval(generator.random(4000), resamples=2000)

    assert many[1] - many[0] < few[1] - few[0]


def test_a_resample_count_that_is_not_a_multiple_of_the_chunk_is_honoured():
    low, high = bootstrap_mean_interval(np.arange(10.0), resamples=1500)

    assert low < 4.5 < high


@pytest.mark.parametrize(
    "kwargs",
    [{"resamples": 0}, {"level": 0.0}, {"level": 1.0}],
)
def test_invalid_bootstrap_settings_are_refused(kwargs):
    with pytest.raises(DataError):
        bootstrap_mean_interval([1.0, 2.0], **kwargs)


def test_a_bootstrap_needs_units():
    with pytest.raises(DataError):
        bootstrap_mean_interval([])
