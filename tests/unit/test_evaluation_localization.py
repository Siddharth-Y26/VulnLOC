"""Guards the line-level localization metrics (Phase 8; ``ResearchPlan.md`` §4.2).

Every number in the paper's tables is one of these, so the tests are about the ways such a
number goes wrong without anything failing: a tie decided by whatever order ``argsort``
happened to return, a unit with nothing to find averaged in as a miss, a review budget
that rounds the wrong way, a precision-recall area that depends on how equal scores were
shuffled.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vulnloc.evaluation import METRIC_REGISTRY
from vulnloc.evaluation.metrics.localization import (
    TIE_ORDERS,
    average_precision,
    precision_recall_f1,
    ranked_labels,
    unit_ranking_metrics,
)
from vulnloc.training.metrics import initial_false_alarms, top_k_accuracy
from vulnloc.utils.exceptions import DataError

HIGHER_IS_BETTER = [
    "top_1",
    "top_3",
    "top_5",
    "top_10",
    "recall_at_1pct_loc",
    "recall_at_5pct_loc",
    "recall_at_20pct_loc",
]
LOWER_IS_BETTER = ["ifa", "first_rank", "average_rank"]

# Scores drawn from five values, so ties are the rule rather than the exception.
TIED_UNITS = st.lists(
    st.tuples(st.sampled_from([0.0, 0.5, 1.0, 1.5, 2.0]), st.booleans()), min_size=1, max_size=40
).filter(lambda lines: any(flagged for _, flagged in lines))


def metric(name: str):
    return METRIC_REGISTRY.get(name)


def flags(*positions: int, n: int) -> np.ndarray:
    ranked = np.zeros(n, dtype=bool)
    ranked[list(positions)] = True
    return ranked


# --- the ranking itself ---------------------------------------------------------------


def test_lines_are_ranked_most_suspicious_first():
    orders = ranked_labels(np.array([0.1, 0.9, 0.5]), np.array([0, 0, 1]))

    assert orders["source"].tolist() == [False, True, False]


def test_equal_scores_are_ordered_three_ways():
    """Source order, then every tie for the flagged line, then every tie against it."""
    orders = ranked_labels(np.array([1.0, 1.0, 1.0, 0.0]), np.array([0, 1, 0, 0]))

    assert orders["source"].tolist() == [False, True, False, False]
    assert orders["best"].tolist() == [True, False, False, False]
    assert orders["worst"].tolist() == [False, False, True, False]


def test_without_ties_the_three_orders_are_one():
    orders = ranked_labels(np.array([3.0, 1.0, 2.0]), np.array([0, 1, 0]))

    assert all(orders[tie].tolist() == [False, False, True] for tie in TIE_ORDERS)


def test_only_candidate_lines_are_ranked():
    orders = ranked_labels(
        np.array([5.0, 4.0, 3.0]), np.array([1, 0, 1]), np.array([False, True, True])
    )

    assert orders["source"].tolist() == [False, True]


def test_a_nan_score_is_refused_rather_than_ranked():
    with pytest.raises(DataError, match="NaN"):
        ranked_labels(np.array([1.0, np.nan]), np.array([1, 0]))


def test_a_nan_on_a_line_that_is_not_a_candidate_is_nobodys_business():
    orders = ranked_labels(np.array([1.0, np.nan]), np.array([1, 0]), np.array([True, False]))

    assert orders["source"].tolist() == [True]


@pytest.mark.parametrize(
    ("scores", "labels", "candidates"),
    [
        (np.zeros(3), np.zeros(2), None),
        (np.zeros((2, 2)), np.zeros((2, 2)), None),
        (np.zeros(3), np.zeros(3), np.ones(2, dtype=bool)),
    ],
)
def test_mismatched_shapes_are_refused(scores, labels, candidates):
    with pytest.raises(DataError):
        ranked_labels(scores, labels, candidates)


# --- the per-unit metrics --------------------------------------------------------------


def test_the_registry_holds_exactly_the_ranking_metrics_the_protocol_names():
    assert METRIC_REGISTRY.keys() == [
        "average_rank",
        "first_rank",
        "ifa",
        "recall_at_1pct_loc",
        "recall_at_20pct_loc",
        "recall_at_5pct_loc",
        "top_1",
        "top_10",
        "top_3",
        "top_5",
    ]


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (3, 0.0), (5, 1.0), (10, 1.0)])
def test_top_k_asks_whether_a_flagged_line_is_among_the_first_k(k, expected):
    assert metric(f"top_{k}")(flags(4, n=20)) == expected


def test_ifa_counts_the_clean_lines_read_before_the_first_flagged_one():
    assert metric("ifa")(flags(3, 7, n=10)) == 3.0
    assert metric("ifa")(flags(0, n=10)) == 0.0


def test_average_rank_is_pulled_down_by_every_flagged_line():
    ranked = flags(1, 3, n=4)

    assert metric("first_rank")(ranked) == 2.0
    assert metric("average_rank")(ranked) == 3.0


@pytest.mark.parametrize(
    ("percent", "n", "positions", "expected"),
    [
        (1, 250, (2, 3), 0.5),  # ceil(2.5) = 3 lines, not 2
        (1, 31, (0, 1), 0.5),  # ceil(0.31) = 1: a reviewer reads at least one line
        (20, 7, (1, 2), 0.5),  # ceil(1.4) = 2
        (5, 100, (4, 5), 0.5),  # exactly 5 lines, not 6: no float rounding up
        (20, 5, (0,), 1.0),
    ],
)
def test_the_review_budget_is_a_ceiling_of_the_units_own_length(percent, n, positions, expected):
    assert metric(f"recall_at_{percent}pct_loc")(flags(*positions, n=n)) == expected


def test_a_unit_with_no_flagged_candidate_has_no_metrics():
    """Undefined, not zero: averaging it in would call "nothing to find" a miss."""
    assert unit_ranking_metrics(np.array([1.0, 2.0]), np.array([0, 0])) is None
    assert (
        unit_ranking_metrics(np.array([1.0, 2.0]), np.array([1, 0]), np.array([False, True]))
        is None
    )


def test_every_registered_metric_is_reported_under_every_tie_order():
    values = unit_ranking_metrics(np.array([1.0, 1.0]), np.array([0, 1]))

    assert values is not None
    assert sorted(values) == METRIC_REGISTRY.keys()
    assert values["top_1"] == {"source": 0.0, "best": 1.0, "worst": 0.0}
    assert values["ifa"] == {"source": 1.0, "best": 0.0, "worst": 1.0}


@given(TIED_UNITS)
def test_the_tie_bounds_bracket_the_source_order_value(lines):
    scores = np.array([score for score, _ in lines])
    labels = np.array([flagged for _, flagged in lines])

    values = unit_ranking_metrics(scores, labels)

    assert values is not None
    for name in HIGHER_IS_BETTER:
        assert values[name]["worst"] <= values[name]["source"] <= values[name]["best"]
    for name in LOWER_IS_BETTER:
        assert values[name]["best"] <= values[name]["source"] <= values[name]["worst"]
    for tie in TIE_ORDERS:
        assert values["first_rank"][tie] == values["ifa"][tie] + 1.0


@given(st.integers(min_value=0, max_value=10_000))
def test_without_ties_it_agrees_with_what_the_training_loop_logs(seed):
    """Same question, same answer as ``training/metrics.py`` — the numbers every run logged."""
    generator = np.random.default_rng(seed)
    n = int(generator.integers(2, 60))
    scores = generator.permutation(n).astype(np.float64)  # distinct by construction
    labels = np.zeros(n, dtype=np.int64)
    labels[generator.choice(n, size=int(generator.integers(1, n)), replace=False)] = 1

    values = unit_ranking_metrics(scores, labels)

    assert values is not None
    torch_scores, torch_labels = torch.as_tensor(scores), torch.as_tensor(labels)
    assert values["top_1"]["source"] == top_k_accuracy(torch_scores, torch_labels, 1)
    assert values["top_5"]["source"] == top_k_accuracy(torch_scores, torch_labels, 5)
    assert values["ifa"]["source"] == initial_false_alarms(torch_scores, torch_labels)
    assert values["ifa"]["best"] == values["ifa"]["worst"]


# --- pooled metrics --------------------------------------------------------------------


def test_average_precision_on_a_ranking_worked_by_hand():
    # Thresholds admit the lines one at a time: P = 1, 1/2, 2/3, 1/2 and R = 1/2, 1/2, 1, 1.
    # Recall rises at the first and third: 1/2 * 1 + 1/2 * 2/3.
    value = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]))

    assert value == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_average_precision_does_not_depend_on_the_order_of_equal_scores():
    scores = np.array([1.0, 1.0, 1.0, 0.0])

    first = average_precision(scores, np.array([1, 0, 0, 0]))
    second = average_precision(scores, np.array([0, 0, 1, 0]))

    assert first == second == pytest.approx(1 / 3)


def test_a_perfect_ranking_has_average_precision_one():
    assert average_precision(np.array([3.0, 2.0, 1.0]), np.array([1, 1, 0])) == 1.0


def test_average_precision_is_undefined_without_a_flagged_line():
    with pytest.raises(DataError, match="undefined"):
        average_precision(np.array([1.0, 2.0]), np.array([0, 0]))


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_average_precision_matches_scikit_learn(seed):
    # Imported here, not at the top: scikit-learn is the reference, not a dependency of
    # what is under test. Its first import takes seconds on this disk, hence no deadline.
    metrics = pytest.importorskip("sklearn.metrics")
    generator = np.random.default_rng(seed)
    n = int(generator.integers(2, 200))
    scores = generator.integers(0, 12, size=n) / 4.0  # many ties
    labels = (generator.random(n) < 0.2).astype(np.int64)
    labels[int(generator.integers(0, n))] = 1

    ours = average_precision(scores, labels)

    assert ours == pytest.approx(metrics.average_precision_score(labels, scores), abs=1e-12)


def test_precision_recall_f1_at_a_threshold():
    result = precision_recall_f1(np.array([0.9, 0.6, 0.4, 0.1]), np.array([1, 0, 1, 0]), 0.5)

    assert (result["tp"], result["fp"], result["fn"]) == (1.0, 1.0, 1.0)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_the_threshold_is_inclusive():
    """``>=``, as ``best_f1_threshold`` chose it: the threshold is itself a line's score."""
    result = precision_recall_f1(np.array([0.5, 0.4]), np.array([1, 0]), 0.5)

    assert result["tp"] == 1.0 and result["fp"] == 0.0


def test_an_empty_denominator_is_zero_and_the_counts_say_so():
    result = precision_recall_f1(np.array([0.1, 0.2]), np.array([1, 0]), 0.9)

    assert result["precision"] == result["recall"] == result["f1"] == 0.0
    assert (result["tp"], result["fp"], result["fn"]) == (0.0, 0.0, 1.0)
