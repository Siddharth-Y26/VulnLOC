"""Guards dataset splitting (ADR-001 §D1.3, ADR-003, Phases.md Phase 1).

Two exit criteria are enforced here: *re-running reproduces byte-identical splits*, and
*leakage checks are run*. The determinism tests deliberately include input reordering —
a seed alone reproduces a shuffle, but not a shuffle of differently-ordered input, and
the corpus order is not something we control.
"""

from __future__ import annotations

import pytest

from vulnloc.data.splits import (
    CLEAN_EVAL,
    EXCLUDED,
    SPLIT_REGISTRY,
    TEST,
    TRAIN,
    VAL,
    SplitSpec,
    Unit,
    assign_splits,
    check_leakage,
)
from vulnloc.utils.exceptions import ConfigError, DataError, RegistryError

STRATEGIES = ["random", "project", "temporal"]


def make_units(n=300, projects=("a", "b", "c", "d"), start_year=2010):
    return [
        Unit(
            unit_id=f"u{i}",
            project=projects[i % len(projects)],
            func_hash=i,
            date=f"{start_year + i % 8:04d}-01-01",
            target=i % 10 == 0,
            has_line_labels=i % 20 == 0,
        )
        for i in range(n)
    ]


# --- SplitSpec validation ----------------------------------------------------------


def test_spec_defaults_sum_to_one():
    spec = SplitSpec()
    assert spec.train + spec.val + spec.test == pytest.approx(1.0)


@pytest.mark.parametrize(
    "ratios", [(0.8, 0.1, 0.2), (0.5, 0.2, 0.2), (1.0, 0.1, 0.0), (0.0, 0.0, 0.0)]
)
def test_spec_rejects_ratios_that_do_not_sum_to_one(ratios):
    with pytest.raises(ConfigError, match="sum to 1.0"):
        SplitSpec(*ratios)


def test_spec_rejects_negative_ratios():
    with pytest.raises(ConfigError, match="non-negative"):
        SplitSpec(train=1.2, val=-0.2, test=0.0)


def test_spec_allows_an_empty_split():
    spec = SplitSpec(train=0.9, val=0.1, test=0.0)
    assert spec.test == 0.0


# --- determinism (Phase 1 exit criterion) ------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_assignment_is_reproducible(strategy):
    units = make_units()

    first = assign_splits(units, strategy=strategy)
    second = assign_splits(units, strategy=strategy)

    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_assignment_is_independent_of_input_order(strategy):
    # A seeded shuffle reproduces only if the input order is identical. Corpus order is
    # not under our control, so the assignment must be content-derived.
    units = make_units()

    forward = assign_splits(units, strategy=strategy)
    backward = assign_splits(list(reversed(units)), strategy=strategy)

    assert forward.assignment == backward.assignment


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_changing_the_seed_changes_the_assignment(strategy):
    units = make_units()

    a = assign_splits(units, strategy=strategy, spec=SplitSpec(seed=1))
    b = assign_splits(units, strategy=strategy, spec=SplitSpec(seed=2))

    if strategy == "temporal":
        # Temporal ordering is a property of the data, not the seed.
        assert a.assignment == b.assignment
    else:
        assert a.assignment != b.assignment


def test_fingerprint_detects_a_single_changed_assignment():
    units = make_units(50)
    result = assign_splits(units, strategy="random")
    baseline = result.fingerprint()

    changed_id = next(uid for uid, s in result.assignment.items() if s != TEST)
    result.assignment[changed_id] = TEST

    assert result.fingerprint() != baseline


# --- coverage and proportions ------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_unit_is_assigned_exactly_once(strategy):
    units = make_units()

    result = assign_splits(units, strategy=strategy)

    assert set(result.assignment) == {u.unit_id for u in units}
    assert sum(result.counts().values()) == len(units)


def test_random_split_approximates_the_requested_ratios():
    units = make_units(4000)

    counts = assign_splits(units, strategy="random").counts()

    assert counts[TRAIN] / 4000 == pytest.approx(0.8, abs=0.03)
    assert counts[VAL] / 4000 == pytest.approx(0.1, abs=0.02)
    assert counts[TEST] / 4000 == pytest.approx(0.1, abs=0.02)


def test_unknown_strategy_raises():
    with pytest.raises(RegistryError, match="Unknown split_strategy"):
        assign_splits(make_units(10), strategy="chronological")


def test_duplicate_unit_ids_are_rejected():
    units = [Unit("dup", "a", 1), Unit("dup", "b", 2)]

    with pytest.raises(DataError, match="unit_id must be unique"):
        assign_splits(units, strategy="random")


def test_registry_exposes_the_three_strategies():
    assert set(SPLIT_REGISTRY.keys()) == set(STRATEGIES)


# --- project strategy: the Big-Vul concentration problem ---------------------------


def test_project_split_keeps_projects_intact():
    units = make_units(600, projects=tuple(f"p{i}" for i in range(12)))

    result = assign_splits(units, strategy="project")

    assert result.leakage.shared_projects == set()


def test_project_split_balances_despite_a_dominant_project():
    # Big-Vul's real shape: Chrome ~41%, linux ~24%, then a long tail. A hash-based
    # project assignment routinely puts both giants in one split; greedy packing must not.
    units = (
        [Unit(f"c{i}", "Chrome", i) for i in range(410)]
        + [Unit(f"l{i}", "linux", 10_000 + i) for i in range(240)]
        + [
            Unit(f"t{i}_{j}", f"tail{i}", 100_000 + i * 100 + j)
            for i in range(35)
            for j in range(10)
        ]
    )

    result = assign_splits(units, strategy="project")
    counts = result.counts()

    assert result.leakage.shared_projects == set()
    # Two projects hold 65% of the corpus, so exact quotas are impossible; what matters
    # is that val and test are non-empty and the tail is not all dumped into train.
    assert counts[VAL] > 0
    assert counts[TEST] > 0
    assert counts[TRAIN] >= counts[VAL] + counts[TEST]


def test_project_split_never_fills_a_zero_quota_split():
    units = make_units(400, projects=tuple(f"p{i}" for i in range(20)))

    counts = assign_splits(
        units, strategy="project", spec=SplitSpec(train=0.9, val=0.1, test=0.0)
    ).counts()

    assert counts[TEST] == 0


def test_project_split_handles_a_single_project():
    units = [Unit(f"u{i}", "only", i) for i in range(50)]

    counts = assign_splits(units, strategy="project").counts()

    assert counts[TRAIN] == 50


# --- temporal strategy -------------------------------------------------------------


def test_temporal_split_orders_the_partitions_in_time():
    units = [Unit(f"u{i}", "p", i, date=f"20{i // 10:02d}-01-01") for i in range(200)]

    result = assign_splits(units, strategy="temporal")
    dates = {TRAIN: [], VAL: [], TEST: []}
    for unit in units:
        dates[result.assignment[unit.unit_id]].append(str(unit.date))

    assert max(dates[TRAIN]) <= min(dates[VAL])
    assert max(dates[VAL]) <= min(dates[TEST])


def test_temporal_split_never_straddles_a_date():
    # Commits sharing a date must not land on both sides of a boundary.
    units = [Unit(f"u{i}", "p", i, date=f"2020-{1 + i % 12:02d}-01") for i in range(240)]

    result = assign_splits(units, strategy="temporal")
    per_date: dict[str, set[str]] = {}
    for unit in units:
        per_date.setdefault(str(unit.date), set()).add(result.assignment[unit.unit_id])

    assert all(len(splits) == 1 for splits in per_date.values())


def test_temporal_split_requires_dates():
    units = [Unit(f"u{i}", "p", i, date=None) for i in range(5)]

    with pytest.raises(DataError, match="requires a date"):
        assign_splits(units, strategy="temporal")


# --- ADR-003 hold-out and exclusion ------------------------------------------------


def test_excluded_hashes_are_dropped_and_never_split():
    units = make_units(100)

    result = assign_splits(units, strategy="random", excluded_hashes={1, 2, 3})

    assert result.counts()[EXCLUDED] == 3
    for unit in units:
        if unit.func_hash in {1, 2, 3}:
            assert result.assignment[unit.unit_id] == EXCLUDED


def test_clean_eval_hashes_are_held_out_of_training():
    units = make_units(100)
    reserved = {5, 15, 25, 35}

    result = assign_splits(units, strategy="random", clean_eval_hashes=reserved)

    assert result.counts()[CLEAN_EVAL] == len(reserved)
    assert result.leakage.clean_eval_in_train == set()


def test_exclusion_takes_precedence_over_clean_eval():
    # A contradicted label must be dropped even if it is also nominated as clean.
    units = make_units(20)

    result = assign_splits(units, strategy="random", clean_eval_hashes={7}, excluded_hashes={7})

    assigned = {result.assignment[u.unit_id] for u in units if u.func_hash == 7}
    assert assigned == {EXCLUDED}


def test_clean_eval_duplicate_in_training_is_detected():
    # Two units, same content hash: one reserved as clean-eval, one free to be trained
    # on. That is a leak, and the checker must catch it.
    units = [Unit("a", "p", 99), Unit("b", "p", 99)]
    assignment = {"a": CLEAN_EVAL, "b": TRAIN}

    report = check_leakage(units, assignment, clean_eval_hashes={99})

    assert report.clean_eval_in_train == {99}
    assert not report.is_clean


# --- leakage checks ----------------------------------------------------------------


def test_leakage_detects_a_duplicate_across_splits():
    units = [Unit("a", "p", 1), Unit("b", "q", 1)]

    report = check_leakage(units, {"a": TRAIN, "b": TEST})

    assert report.duplicate_hashes == {1}
    assert report.duplicated_units == 2
    assert not report.is_clean


def test_leakage_detects_a_shared_project():
    units = [Unit("a", "shared", 1), Unit("b", "shared", 2)]

    report = check_leakage(units, {"a": TRAIN, "b": TEST})

    assert report.shared_projects == {"shared"}


def test_duplicates_within_one_split_are_counted_but_not_leakage():
    units = [Unit("a", "p", 1), Unit("b", "p", 1)]

    report = check_leakage(units, {"a": TRAIN, "b": TRAIN})

    assert report.intra_split_duplicates == 1
    assert report.duplicate_hashes == set()
    assert report.is_clean


def test_excluded_units_are_ignored_by_leakage_checks():
    units = [Unit("a", "p", 1), Unit("b", "q", 1)]

    report = check_leakage(units, {"a": TRAIN, "b": EXCLUDED})

    assert report.is_clean


def test_a_clean_project_split_reports_no_leakage():
    units = make_units(600, projects=tuple(f"p{i}" for i in range(15)))

    result = assign_splits(units, strategy="project")

    assert result.leakage.is_clean, result.leakage.as_dict()
