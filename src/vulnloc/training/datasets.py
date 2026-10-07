"""Shards → line-labelled window batches (L3, so it may read both L0 and L1).

Two decisions here shape everything the training loop can do.

**Batching is in windows, not units.** A function is 1 to 287 windows
(``Memory.md`` §7 #28), so a batch of 16 *units* varies in cost by two orders of magnitude
and, on a 4 GiB card, is bounded by its worst member. :class:`WindowBatchSampler` fills a
batch to a **window budget** instead, which makes memory a property of the sampler rather
than a gamble. A unit is never split across batches — its windows are scored together or
not at all, because the line pooling averages across them.

**Labels live on lines, and the line axis is capped.** ``n_lines`` truncates the label
vector exactly as it truncates the features, and the number of flagged lines pushed past
the cap is **counted and reported**, never quietly dropped. That count is the same class
of quantity as the truncation loss in ``Memory.md`` §1.8, and it would be dishonest to let
it disappear into a padding routine.

Tokenization is done once per unit and cached in memory as plain tuples; the tensors are
built at collation time. Encoding 149k functions costs a few minutes and is done once per
process rather than once per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, Sampler

from vulnloc.data.cache import iter_shard
from vulnloc.representation.tokenization import LineTokenizer, TokenWindow
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "EncodedUnit",
    "LineLabelledDataset",
    "WindowBatchSampler",
    "collate_units",
    "line_label_coverage",
]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EncodedUnit:
    """One tokenized, line-labelled function.

    Attributes:
        unit_id: Stable corpus identifier.
        windows: The unit's token windows.
        line_labels: ``[n_lines]`` of 0/1, 1-indexed line ``i`` at position ``i - 1``.
        n_lines: Source lines in the function, before the line cap.
        target: 1 if the function is vulnerable.
        flagged_beyond_cap: Flagged lines that fall past the line cap and can therefore
            never be predicted. Reported, not hidden.
    """

    unit_id: str
    windows: tuple[TokenWindow, ...]
    line_labels: tuple[int, ...]
    n_lines: int
    target: int
    flagged_beyond_cap: int

    @property
    def n_windows(self) -> int:
        """How many windows this unit needs."""
        return len(self.windows)


class LineLabelledDataset(Dataset[EncodedUnit]):
    """Units from one shard, tokenized once and held in memory.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: Shard name — ``train``, ``val``, ``test``, ``clean_eval``.
        tokenizer: The line-preserving tokenizer.
        n_lines: Line-axis cap, ``L``.
        limit: Stop after this many units. ``0`` means the whole shard.
        line_labelled_only: Keep only functions that carry line labels. Required for the
            line-level task: 24.4% of Big-Vul's positives have no line labels at all
            (``Memory.md`` §7 #10), and training a line head against an all-zero label
            vector for a function known to be vulnerable teaches it the wrong thing.
        normalize_whitespace: Strip edge whitespace from every source line before
            tokenizing. **On by default: leaving it off leaks the label.** Big-Vul's
            flagged lines keep their original indentation while every other line is
            stripped, so "line starts with a space" recovers the label at precision
            1.0000 (ADR-006, ``Memory.md`` §1.10). ``False`` reproduces pre-ADR-006 runs.
        strip_comments: Remove every comment before anything sees the text, keeping the
            line count (``Memory.md`` §1.25). **Off by default**: every number logged
            before 2026-10-07 was measured with comments in place.

    Raises:
        DataError: If the shard yields no units.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        split: str,
        tokenizer: LineTokenizer,
        *,
        n_lines: int = 512,
        limit: int = 0,
        line_labelled_only: bool = False,
        normalize_whitespace: bool = True,
        strip_comments: bool = False,
    ) -> None:
        self.split = split
        self.n_lines = n_lines
        self.normalize_whitespace = normalize_whitespace
        self.strip_comments = strip_comments
        self.units: list[EncodedUnit] = []
        flagged_lost = 0

        for index, labelled in enumerate(iter_shard(Path(cache_dir), split)):
            if limit and len(self.units) >= limit:
                break
            del index
            if line_labelled_only and not labelled.vulnerable_lines:
                continue
            unit = CodeUnit.from_labelled_function(
                labelled,
                normalize_whitespace=normalize_whitespace,
                strip_comments=strip_comments,
            )
            encoded = tokenizer.encode(unit)
            labels = [0] * n_lines
            beyond = 0
            for line in unit.vulnerable_lines:
                if 1 <= line <= n_lines:
                    labels[line - 1] = 1
                else:
                    beyond += 1
            flagged_lost += beyond
            self.units.append(
                EncodedUnit(
                    unit_id=unit.unit_id,
                    windows=encoded.windows,
                    line_labels=tuple(labels),
                    n_lines=unit.n_lines,
                    target=unit.target,
                    flagged_beyond_cap=beyond,
                )
            )

        if not self.units:
            raise DataError(
                "shard produced no units",
                split=split,
                cache_dir=str(cache_dir),
                line_labelled_only=line_labelled_only,
            )

        self.flagged_beyond_cap = flagged_lost
        total_flagged = sum(sum(u.line_labels) for u in self.units) + flagged_lost
        _LOGGER.info(
            "Dataset encoded",
            extra={
                "split": split,
                "units": len(self.units),
                "windows": sum(u.n_windows for u in self.units),
                "positives": sum(u.target for u in self.units),
                "flagged_lines": total_flagged,
                "flagged_beyond_line_cap": flagged_lost,
                "n_lines_cap": n_lines,
                "normalize_whitespace": normalize_whitespace,
                "strip_comments": strip_comments,
            },
        )

    def __len__(self) -> int:
        """Number of units."""
        return len(self.units)

    def __getitem__(self, index: int) -> EncodedUnit:
        """Return one encoded unit."""
        return self.units[index]

    def positive_line_weight(self) -> float:
        """Return ``negatives / positives`` over scoreable lines, for a weighted loss.

        Computed over the lines the model can actually predict — the label cap and the
        per-unit line count both bound it — so the weight matches the loss's denominator
        rather than a corpus-wide statistic.

        Returns:
            The ratio, or ``1.0`` when there are no positive lines.
        """
        positives = 0
        scoreable = 0
        for unit in self.units:
            usable = min(unit.n_lines, self.n_lines)
            scoreable += usable
            positives += sum(unit.line_labels[:usable])
        if positives == 0:
            return 1.0
        return (scoreable - positives) / positives


def collate_units(units: Sequence[EncodedUnit]) -> dict[str, Any]:
    """Flatten a list of units into one window batch.

    Args:
        units: The units in this batch.

    Returns:
        A batch carrying ``input_ids`` ``[W, T]``, ``attention_mask`` ``[W, T]``,
        ``token_to_line`` ``[W, T]``, ``window_to_unit`` ``[W]``, ``n_units``,
        ``line_labels`` ``[B, L]``, ``line_valid`` ``[B, L]``, ``targets`` ``[B]`` and
        ``unit_ids``.

    Raises:
        DataError: If ``units`` is empty.
    """
    if not units:
        raise DataError("cannot collate an empty batch")

    input_ids: list[tuple[int, ...]] = []
    attention: list[tuple[int, ...]] = []
    token_lines: list[tuple[int, ...]] = []
    window_to_unit: list[int] = []
    for unit_index, unit in enumerate(units):
        for window in unit.windows:
            input_ids.append(window.input_ids)
            attention.append(window.attention_mask)
            token_lines.append(window.token_to_line)
            window_to_unit.append(unit_index)

    n_lines = len(units[0].line_labels)
    # `line_valid` is the *label* mask: lines this unit actually has. It is not the
    # feature mask the model produces — a line can exist and still carry no tokens, and
    # conflating the two would silently excuse the model from scoring lines it skipped.
    line_valid = torch.zeros((len(units), n_lines), dtype=torch.bool)
    for index, unit in enumerate(units):
        line_valid[index, : min(unit.n_lines, n_lines)] = True

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "token_to_line": torch.tensor(token_lines, dtype=torch.long),
        "window_to_unit": torch.tensor(window_to_unit, dtype=torch.long),
        "n_units": len(units),
        "line_labels": torch.tensor([u.line_labels for u in units], dtype=torch.float32),
        "line_valid": line_valid,
        "targets": torch.tensor([u.target for u in units], dtype=torch.float32),
        "unit_ids": [u.unit_id for u in units],
    }


class WindowBatchSampler(Sampler[list[int]]):
    """Groups units into batches by a **window** budget, never splitting a unit.

    Sizing a batch in units makes its cost unpredictable, because a unit is 1 to 287
    windows. Sizing it in windows makes peak memory roughly constant, which is what turns
    ``Memory.md`` §1.9's "batch 16 windows" into a setting that means something.

    A unit larger than the budget on its own still gets its own batch. Dropping it would
    silently discard the longest functions — which are exactly the ones carrying the most
    flagged lines (§1.8) — so the budget is a target, not a guarantee.

    Args:
        dataset: The dataset to sample from.
        max_windows: Window budget per batch.
        shuffle: Shuffle unit order each epoch.
        seed: Base seed; the epoch number is mixed in so each epoch differs reproducibly.
        max_units: Optional cap on units per batch, regardless of window count.

    Raises:
        DataError: If ``max_windows`` is not positive.
    """

    def __init__(
        self,
        dataset: LineLabelledDataset,
        *,
        max_windows: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        max_units: int = 0,
    ) -> None:
        if max_windows <= 0:
            raise DataError("max_windows must be positive", max_windows=max_windows)
        self.dataset = dataset
        self.max_windows = max_windows
        self.shuffle = shuffle
        self.seed = seed
        self.max_units = max_units
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the shuffle for the next epoch, reproducibly.

        Args:
            epoch: The epoch about to start.
        """
        self.epoch = epoch

    def _order(self) -> list[int]:
        """Return this epoch's unit order."""
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        """Yield lists of unit indices, each within the window budget where possible."""
        batch: list[int] = []
        windows = 0
        for index in self._order():
            needed = self.dataset[index].n_windows
            over_budget = batch and windows + needed > self.max_windows
            over_units = self.max_units and len(batch) >= self.max_units
            if over_budget or over_units:
                yield batch
                batch, windows = [], 0
            batch.append(index)
            windows += needed
        if batch:
            yield batch

    def __len__(self) -> int:
        """Number of batches this epoch. Counted by walking, since it depends on order."""
        return sum(1 for _ in self.__iter__())


def line_label_coverage(batch: dict[str, Any], line_mask: torch.Tensor) -> dict[str, int]:
    """Count flagged lines the model had no features for.

    The per-batch version of ``Memory.md`` §7 #21: a flagged line with no token on it
    cannot be scored, so it is a ceiling on recall rather than a model failure. Counting
    it during training keeps the number visible instead of letting it be discovered at
    evaluation time.

    Args:
        batch: A collated batch.
        line_mask: ``[B, L]`` mask of lines that received features.

    Returns:
        ``{"flagged": n, "flagged_with_features": n}``.
    """
    flagged = batch["line_labels"] > 0
    return {
        "flagged": int(flagged.sum()),
        "flagged_with_features": int((flagged & line_mask.to(flagged.device)).sum()),
    }
