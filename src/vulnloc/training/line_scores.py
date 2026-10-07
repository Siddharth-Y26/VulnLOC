"""Per-line scores on disk: the interface between a checkpoint and everything measured later.

Once a model has scored a partition, nothing downstream needs the model again. Calibration,
the threshold, every ranking metric Phase 8 adds and the examples Phase 7 explains are all
functions of *(unit, line) -> (logit, label, was it scoreable)*. Keeping that table means a
gigabyte checkpoint is loaded once per partition instead of once per question, and — more
to the point — that every later number is computed from the same scores rather than from a
re-run that may not reproduce them (``Memory.md`` §7 #50: CUDA does not promise it will).

The file is a NumPy ``.npz`` of flat arrays plus offsets, because units have different
lengths and a ragged structure would need ``pickle`` (``Rules.md`` §2). Lines are stored in
source order, one entry per line that exists in the unit — the line index is the position.

What is *not* stored is as deliberate as what is: no probabilities, since they depend on a
calibrator chosen later, and no decisions, since they depend on a threshold chosen later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vulnloc.utils.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["UnitScores", "compare_metrics", "load_scores", "save_scores"]


@dataclass(frozen=True)
class UnitScores:
    """One unit's line scores.

    Attributes:
        unit_id: Stable corpus identifier.
        logits: ``[n_lines]`` float32 line logits, in source order. A line the model could
            not score holds the head's masked sentinel, not a prediction.
        labels: ``[n_lines]`` int8, 1 where the line is flagged.
        scoreable: ``[n_lines]`` bool, ``True`` where the model had features for the line.
    """

    unit_id: str
    logits: np.ndarray
    labels: np.ndarray
    scoreable: np.ndarray

    def __post_init__(self) -> None:
        if not (self.logits.shape == self.labels.shape == self.scoreable.shape):
            raise DataError(
                "a unit's logits, labels and scoreable mask must have one length",
                unit_id=self.unit_id,
                logits=self.logits.shape,
                labels=self.labels.shape,
                scoreable=self.scoreable.shape,
            )
        if self.logits.ndim != 1:
            raise DataError("line scores are one-dimensional per unit", unit_id=self.unit_id)


def save_scores(path: Path, units: Sequence[UnitScores]) -> None:
    """Write units to ``path`` as flat arrays plus offsets.

    Args:
        path: Destination ``.npz``. Parent directories are created.
        units: The units, in the order they should be read back.

    Raises:
        DataError: If there is nothing to write or a ``unit_id`` repeats — a repeated id
            would let two different functions answer for one in every later join.
    """
    if not units:
        raise DataError("no units to save", path=str(path))
    ids = [unit.unit_id for unit in units]
    if len(set(ids)) != len(ids):
        raise DataError("duplicate unit_id in the scores being saved", path=str(path))
    lengths = np.array([unit.logits.shape[0] for unit in units], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_ids=np.array(ids, dtype=np.str_),
        offsets=offsets,
        logits=np.concatenate([unit.logits.astype(np.float32) for unit in units]),
        labels=np.concatenate([unit.labels.astype(np.int8) for unit in units]),
        scoreable=np.concatenate([unit.scoreable.astype(np.bool_) for unit in units]),
    )


def load_scores(path: Path) -> list[UnitScores]:
    """Read back what :func:`save_scores` wrote.

    Args:
        path: The ``.npz`` file.

    Returns:
        The units, in the order they were saved.

    Raises:
        DataError: If the file is missing or its arrays do not agree with its offsets.
    """
    if not path.is_file():
        raise DataError("no score file at this path", path=str(path))
    with np.load(path, allow_pickle=False) as archive:
        ids = [str(unit_id) for unit_id in archive["unit_ids"]]
        offsets = archive["offsets"]
        logits, labels, scoreable = archive["logits"], archive["labels"], archive["scoreable"]
    if offsets.shape[0] != len(ids) + 1 or int(offsets[-1]) != logits.shape[0]:
        raise DataError(
            "score file is inconsistent: offsets do not match its arrays",
            path=str(path),
            units=len(ids),
            lines=int(logits.shape[0]),
        )
    return [
        UnitScores(
            unit_id=unit_id,
            logits=logits[offsets[index] : offsets[index + 1]],
            labels=labels[offsets[index] : offsets[index + 1]],
            scoreable=scoreable[offsets[index] : offsets[index + 1]],
        )
        for index, unit_id in enumerate(ids)
    ]


def compare_metrics(
    computed: dict[str, float],
    logged: dict[str, float],
    keys: Sequence[str],
    *,
    tolerance: float,
) -> list[str]:
    """Name the metrics on which a re-scoring disagrees with what its run logged.

    A score dump is only worth keeping if it is the scores the run reported. This is the
    check: same checkpoint, same partition, same metric code — so anything beyond float
    noise means the wrong checkpoint, the wrong config or a changed dataset.

    Args:
        computed: Metrics from the re-scoring.
        logged: Metrics the run recorded for its best epoch.
        keys: Which metrics to compare. One missing from either side counts as a
            disagreement: an absent number is not a matching one.
        tolerance: Largest absolute difference that still counts as agreement.

    Returns:
        One line per disagreement, empty when everything agrees.
    """
    problems: list[str] = []
    for key in keys:
        if key not in computed or key not in logged:
            side = "the re-scoring" if key not in computed else "the run's record"
            problems.append(f"{key}: missing from {side}")
            continue
        difference = abs(float(computed[key]) - float(logged[key]))
        if difference > tolerance:
            problems.append(
                f"{key}: re-scored {float(computed[key]):.6f}, "
                f"logged {float(logged[key]):.6f} (|diff| {difference:.6f})"
            )
    return problems
