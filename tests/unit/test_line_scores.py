"""The score file is what every later number is computed from, so it must come back whole.

A dump that returns the right lines for the wrong unit, or drops a unit's last line, would
make calibration, the threshold and every Phase 8 metric wrong together and consistently —
which is the kind of error that survives review.
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnloc.training.line_scores import UnitScores, compare_metrics, load_scores, save_scores
from vulnloc.utils.exceptions import DataError


def _unit(unit_id: str, logits: list[float], labels: list[int], scoreable: list[bool]):
    return UnitScores(
        unit_id=unit_id,
        logits=np.array(logits, dtype=np.float32),
        labels=np.array(labels, dtype=np.int8),
        scoreable=np.array(scoreable, dtype=np.bool_),
    )


def test_units_of_different_lengths_come_back_as_they_went_in(tmp_path):
    units = [
        _unit("bigvul-1", [0.5, -1.25, 3.0], [0, 1, 0], [True, True, False]),
        _unit("bigvul-2", [-10000.0], [1], [False]),
        _unit("bigvul-3", [0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0, 0, 1], [True] * 5),
    ]
    path = tmp_path / "nested" / "scores.npz"

    save_scores(path, units)
    restored = load_scores(path)

    assert [unit.unit_id for unit in restored] == ["bigvul-1", "bigvul-2", "bigvul-3"]
    for original, back in zip(units, restored, strict=True):
        assert np.array_equal(original.logits, back.logits)
        assert np.array_equal(original.labels, back.labels)
        assert np.array_equal(original.scoreable, back.scoreable)
        assert back.logits.dtype == np.float32
        assert back.scoreable.dtype == np.bool_


def test_a_repeated_unit_id_is_refused(tmp_path):
    """Two functions answering for one would corrupt every later join silently."""
    units = [_unit("same", [0.0], [0], [True]), _unit("same", [1.0], [1], [True])]

    with pytest.raises(DataError, match="duplicate unit_id"):
        save_scores(tmp_path / "scores.npz", units)


def test_nothing_to_save_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(DataError, match="no units to save"):
        save_scores(tmp_path / "scores.npz", [])
    assert not (tmp_path / "scores.npz").exists()


def test_a_units_three_arrays_must_be_one_length():
    with pytest.raises(DataError, match="must have one length"):
        _unit("bad", [0.0, 1.0], [0], [True, True])


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(DataError, match="no score file"):
        load_scores(tmp_path / "absent.npz")


def test_a_file_whose_offsets_disagree_with_its_arrays_is_refused(tmp_path):
    path = tmp_path / "scores.npz"
    np.savez_compressed(
        path,
        unit_ids=np.array(["a", "b"], dtype=np.str_),
        offsets=np.array([0, 2, 5], dtype=np.int64),
        logits=np.zeros(4, dtype=np.float32),
        labels=np.zeros(4, dtype=np.int8),
        scoreable=np.ones(4, dtype=np.bool_),
    )

    with pytest.raises(DataError, match="inconsistent"):
        load_scores(path)


# --- does a re-scoring reproduce its run ------------------------------------------------

KEYS = ("line/top_5_accuracy", "line/top_1_accuracy")


def test_agreement_within_float_noise_is_agreement():
    computed = {"line/top_5_accuracy": 0.6224066, "line/top_1_accuracy": 0.3402490}
    logged = {"line/top_5_accuracy": 0.6224066390, "line/top_1_accuracy": 0.3402489627}

    assert compare_metrics(computed, logged, KEYS, tolerance=1e-6) == []


def test_one_function_of_difference_is_named():
    """1 / 482 is the smallest change Top-5 can make on val, and it must not pass."""
    computed = {"line/top_5_accuracy": 0.6224066 + 1 / 482, "line/top_1_accuracy": 0.3402490}
    logged = {"line/top_5_accuracy": 0.6224066, "line/top_1_accuracy": 0.3402490}

    problems = compare_metrics(computed, logged, KEYS, tolerance=1e-4)

    assert len(problems) == 1
    assert problems[0].startswith("line/top_5_accuracy: re-scored 0.624481, logged 0.622407")


def test_a_metric_missing_from_either_side_is_a_disagreement():
    """An absent number is not a matching one."""
    problems = compare_metrics(
        {"line/top_5_accuracy": 0.6}, {"line/top_1_accuracy": 0.3}, KEYS, tolerance=1.0
    )

    assert problems == [
        "line/top_5_accuracy: missing from the run's record",
        "line/top_1_accuracy: missing from the re-scoring",
    ]
