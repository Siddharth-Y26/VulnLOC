"""Line-label derivation from vulnerability-fixing commits (L0, ADR-001 §D1.2).

`labels_v1` — the only variant Phase 1 ships — takes the lines **deleted or modified by
the fix** as positive. That is the field convention (LineVul, LineVD, IVDetect) and is
what makes our numbers comparable to theirs.

`labels_v2` — lines control- or data-dependent on *added* lines — needs a DFG and lands
in Phase 2 (ADR-002 §D2.4). Added lines cannot be labelled directly: they do not exist
in the vulnerable function.

Two facts about Big-Vul's encoding, both established empirically rather than assumed,
because getting either wrong corrupts every label in the project silently:

* ``flaw_line_index`` is **0-indexed**. Verified on 400 sampled rows: 400 matched
  0-indexed, 0 matched 1-indexed. `Architecture.md` §1.1 requires 1-indexed lines
  internally, so :func:`derive_line_labels` converts.
* ``flaw_line`` holds the flagged source lines joined by ``/~/``. Same verification.

`Rules.md` §4.6 requires data-quality failures to be counted and reported, never
hidden, so every label is checked against its recorded text and mismatches are surfaced
rather than silently dropped.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vulnloc.utils.exceptions import DataError
from vulnloc.utils.hashing import unit_interval
from vulnloc.utils.logging import get_logger

__all__ = [
    "FLAW_LINE_SEPARATOR",
    "LabelIssue",
    "LabelledFunction",
    "LabellingStats",
    "derive_line_labels",
    "sample_for_review",
    "split_source_lines",
]

_LOGGER = get_logger(__name__)

# Big-Vul joins the flagged source lines with this marker inside `flaw_line`.
FLAW_LINE_SEPARATOR = "/~/"

# Cell values that mean "absent". This module takes raw `object` values so it stays
# pandas-free, which means it has to recognise pandas' stringified missing markers
# itself: "nan" from inferred float columns, "<na>" from nullable dtypes.
_MISSING_TOKENS = frozenset({"", "nan", "<na>", "none", "null"})


class LabelIssue(StrEnum):
    """Why a derived label could not be fully trusted.

    These are *counted and reported*, not raised: a corpus with some unusable labels is
    normal, a run that silently drops 30% of its samples is a corrupted experiment.
    """

    NONE = "none"
    NO_LINE_LABELS = "no_line_labels"
    """Positive function with an empty ``flaw_line_index`` — 24.4% of Big-Vul positives."""

    OUT_OF_RANGE = "out_of_range"
    """An index points past the end of the function body."""

    TEXT_MISMATCH = "text_mismatch"
    """The indexed line does not match the recorded ``flaw_line`` text."""

    COUNT_MISMATCH = "count_mismatch"
    """``flaw_line`` and ``flaw_line_index`` disagree on how many lines are flagged."""

    LABELS_ON_NEGATIVE = "labels_on_negative"
    """A non-vulnerable function carries line labels. Should never happen."""


@dataclass(frozen=True)
class LabelledFunction:
    """One corpus function with derived line-level supervision.

    This is the L0 output type. It is deliberately *not*
    :class:`~vulnloc.representation.CodeUnit`: L0 must not import L1
    (`Architecture.md` §3), and this record carries no parse, graph or token state.

    Args:
        unit_id: Stable identifier, unique within the corpus.
        source: Raw function text.
        language: Source language.
        target: 1 if the function is vulnerable.
        vulnerable_lines: **1-indexed** line numbers, converted from Big-Vul's
            0-indexed encoding.
        n_lines: Number of source lines, so callers can validate without re-splitting.
        issues: Data-quality findings for this function.
        metadata: Provenance — project, commit, CVE, CWE, date.
    """

    unit_id: str
    source: str
    language: str
    target: int
    vulnerable_lines: frozenset[int]
    n_lines: int
    issues: tuple[LabelIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_line_labels(self) -> bool:
        """Whether this function carries usable line-level supervision."""
        return bool(self.vulnerable_lines)

    @property
    def is_trustworthy(self) -> bool:
        """Whether the derived labels passed every consistency check."""
        return not self.issues

    def line_label_vector(self) -> list[int]:
        """Return a per-line 0/1 vector aligned to the source.

        Returns:
            A list of length ``n_lines`` where index ``i`` corresponds to line
            ``i + 1`` (`Architecture.md` §1.1 — lines are 1-indexed).
        """
        vector = [0] * self.n_lines
        for line_no in self.vulnerable_lines:
            vector[line_no - 1] = 1
        return vector


@dataclass
class LabellingStats:
    """Aggregate data-quality counters over a corpus pass."""

    total: int = 0
    positive: int = 0
    positive_with_lines: int = 0
    flagged_lines: int = 0
    issues: dict[str, int] = field(default_factory=dict)

    def record(self, unit: LabelledFunction) -> None:
        """Fold one labelled function into the running counters.

        Args:
            unit: The function just labelled.
        """
        self.total += 1
        if unit.target == 1:
            self.positive += 1
            if unit.has_line_labels:
                self.positive_with_lines += 1
        self.flagged_lines += len(unit.vulnerable_lines)
        for issue in unit.issues:
            self.issues[issue] = self.issues.get(issue, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary."""
        return {
            "total": self.total,
            "positive": self.positive,
            "positive_with_lines": self.positive_with_lines,
            "flagged_lines": self.flagged_lines,
            "issues": dict(sorted(self.issues.items())),
        }


def split_source_lines(source: str) -> list[str]:
    r"""Split a function body into lines, normalizing line endings.

    CRLF and lone CR are normalized first: Big-Vul mixes them, and splitting on ``\n``
    alone leaves a trailing ``\r`` that breaks text comparison and, later, the
    Tree-sitter byte-offset to line mapping.

    Args:
        source: Raw function text.

    Returns:
        The lines, without terminators.
    """
    return source.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _parse_index_token(token: str) -> int:
    """Parse one line index, tolerating pandas' float coercion.

    A column containing NaN is inferred as float64, so an index written ``2`` in the
    CSV arrives as ``2.0``. Integral floats are accepted; genuinely fractional ones are
    not, because a line number of 2.5 means the schema is not what we think.

    Args:
        token: One comma-separated index.

    Returns:
        The index as an integer.

    Raises:
        ValueError: If the token is not an integer or an integral float.
    """
    value = float(token)
    if not value.is_integer():
        raise ValueError(f"line index is not integral: {token!r}")
    return int(value)


def _parse_indices(raw: object) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text.lower() in _MISSING_TOKENS:
        return []
    try:
        return [_parse_index_token(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise DataError(f"Malformed flaw_line_index: {text!r}") from exc


def _parse_flaw_text(raw: object) -> list[str]:
    if raw is None:
        return []
    text = str(raw)
    if text.strip().lower() in _MISSING_TOKENS:
        return []
    return [part.strip() for part in text.split(FLAW_LINE_SEPARATOR)]


def derive_line_labels(
    *,
    unit_id: str,
    source: str,
    target: int,
    flaw_line_index: object,
    flaw_line: object = None,
    language: str = "C",
    metadata: Mapping[str, Any] | None = None,
) -> LabelledFunction:
    """Derive `labels_v1` line supervision for one function.

    Args:
        unit_id: Stable identifier.
        source: Function body.
        target: 1 if vulnerable.
        flaw_line_index: Big-Vul's comma-separated **0-indexed** line numbers.
        flaw_line: Big-Vul's ``/~/``-joined flagged source text, used to verify the
            indices. Optional — verification is skipped when absent.
        language: Source language.
        metadata: Provenance to attach.

    Returns:
        The labelled function, with 1-indexed line numbers and any quality issues.

    Raises:
        DataError: If ``flaw_line_index`` is malformed, which means the schema is not
            what the loader assumes and every downstream label would be suspect.
    """
    lines = split_source_lines(source)
    indices = _parse_indices(flaw_line_index)
    issues: list[LabelIssue] = []

    if indices and target != 1:
        issues.append(LabelIssue.LABELS_ON_NEGATIVE)

    in_range = [i for i in indices if 0 <= i < len(lines)]
    if len(in_range) != len(indices):
        issues.append(LabelIssue.OUT_OF_RANGE)

    texts = _parse_flaw_text(flaw_line)
    if texts and len(texts) != len(indices):
        issues.append(LabelIssue.COUNT_MISMATCH)
    elif texts and any(
        lines[i].strip() != expected for i, expected in zip(in_range, texts, strict=False)
    ):
        issues.append(LabelIssue.TEXT_MISMATCH)

    if target == 1 and not in_range:
        issues.append(LabelIssue.NO_LINE_LABELS)

    # 0-indexed on disk, 1-indexed everywhere in this codebase (Architecture.md §1.1).
    vulnerable_lines = frozenset(index + 1 for index in in_range)

    return LabelledFunction(
        unit_id=unit_id,
        source=source,
        language=language,
        target=int(target),
        vulnerable_lines=vulnerable_lines,
        n_lines=len(lines),
        issues=tuple(issues),
        metadata=dict(metadata or {}),
    )


def sample_for_review(
    units: Iterable[LabelledFunction], *, k: int = 20, seed: int = 42
) -> list[LabelledFunction]:
    """Draw ``k`` line-labelled functions for manual inspection.

    `Phases.md` Phase 1 requires a human to inspect sampled line labels and judge them
    correct; that verdict is only meaningful if the sample was not chosen to flatter the
    labels. Selection is therefore by content hash — the ``k`` functions with the
    smallest ``sha256(seed, unit_id)`` — which is reproducible, order-independent, and
    fixed before anyone looks at the data. A second reviewer gets the same twenty.

    Streaming and bounded: the corpus does not fit in memory as
    :class:`LabelledFunction` objects, so only ``k`` are retained at a time.

    Args:
        units: Labelled functions to sample from. Those without line labels are
            skipped — there is nothing for a reviewer to judge.
        k: Sample size.
        seed: Salt for the selection hash.

    Returns:
        The sampled functions, ordered by their selection hash.
    """
    # A max-heap of the k smallest keys: push, then evict the largest once over size.
    heap: list[tuple[float, str, LabelledFunction]] = []
    for unit in units:
        if not unit.has_line_labels:
            continue
        entry = (-unit_interval(seed, unit.unit_id), unit.unit_id, unit)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif k > 0 and entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    return [entry[2] for entry in sorted(heap, key=lambda e: (-e[0], e[1]))]


def summarize(units: Sequence[LabelledFunction]) -> LabellingStats:
    """Aggregate quality counters over labelled functions.

    Args:
        units: Labelled functions.

    Returns:
        The aggregate statistics.
    """
    stats = LabellingStats()
    for unit in units:
        stats.record(unit)
    _LOGGER.info("Labelling summary", extra=stats.as_dict())
    return stats
