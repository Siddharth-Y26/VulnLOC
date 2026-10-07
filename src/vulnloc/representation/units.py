"""The L1 view of one function (``docs/Architecture.md`` §4.1).

L0 emits :class:`~vulnloc.data.labeling.LabelledFunction`, which is deliberately inert:
raw text, labels, provenance, and nothing that has been parsed. :class:`CodeUnit` is the
same function once L1 owns it — the text split into lines, with a
:class:`~vulnloc.representation.line_map.LineMap` attached so every offset produced
downstream lands on a line without anyone re-deriving the mapping.

The split exists because the layer rule forbids L0 from importing L1
(``Architecture.md`` §3), so the conversion has to live on this side of the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vulnloc.representation.line_map import LineMap, split_lines
from vulnloc.utils.text import normalize_source_whitespace
from vulnloc.utils.text import strip_comments as strip_source_comments

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vulnloc.data.labeling import LabelledFunction

__all__ = ["CodeUnit"]


@dataclass(frozen=True)
class CodeUnit:
    """One function, parsed-ready, with its line map already built.

    Attributes:
        unit_id: Stable corpus identifier.
        source: Raw function text.
        language: Language string as the corpus recorded it — ``"C"``, ``"CPP"``,
            ``"C++"``. Kept verbatim rather than normalised, because *which grammar to
            actually parse it with* is an open decision (``Memory.md`` §5 Q8) and this
            field is the evidence for it.
        lines: 1-indexed via ``lines[i - 1]``.
        line_map: Offset to line map for :attr:`source`.
        target: 1 if the function is vulnerable, else 0.
        vulnerable_lines: 1-indexed line numbers carrying the flaw, empty when the
            corpus gave none (24.4% of Big-Vul positives).
        metadata: Provenance — project, commit, CVE, CWE, date.
    """

    unit_id: str
    source: str
    language: str
    lines: tuple[str, ...]
    line_map: LineMap = field(repr=False)
    target: int = 0
    vulnerable_lines: frozenset[int] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_lines(self) -> int:
        """Number of lines, counting a trailing empty line if the source ends with one."""
        return len(self.lines)

    @classmethod
    def from_source(
        cls,
        unit_id: str,
        source: str,
        *,
        language: str,
        target: int = 0,
        vulnerable_lines: frozenset[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        normalize_whitespace: bool = True,
        strip_comments: bool = False,
    ) -> CodeUnit:
        """Build a unit from raw text.

        Args:
            unit_id: Stable identifier.
            source: Function text.
            language: Corpus language string.
            target: 1 if vulnerable.
            vulnerable_lines: 1-indexed flagged lines.
            metadata: Provenance.
            normalize_whitespace: Strip edge whitespace from every line before anything
                sees the text. **On by default because leaving it off leaks the label** —
                Big-Vul's flagged lines keep their original indentation while every other
                line has been stripped, which recovers the label at precision 1.0
                (ADR-006). ``False`` reproduces every pre-ADR-006 number.
            strip_comments: Remove every comment, keeping the line count — the
                label-blind fix for the second formatting leak (``Memory.md`` §1.25).
                **Off by default: every number logged before 2026-10-07 was measured with
                comments in place**, and a run made with this on is a different
                experiment from one made with it off.

        Returns:
            The unit, with its line map built.
        """
        if strip_comments:
            source = strip_source_comments(source)
        if normalize_whitespace:
            source = normalize_source_whitespace(source)
        return cls(
            unit_id=unit_id,
            source=source,
            language=language,
            lines=tuple(split_lines(source)),
            line_map=LineMap.from_source(source, unit_id=unit_id),
            target=target,
            vulnerable_lines=vulnerable_lines or frozenset(),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_labelled_function(
        cls,
        unit: LabelledFunction,
        *,
        normalize_whitespace: bool = True,
        strip_comments: bool = False,
    ) -> CodeUnit:
        """Lift an L0 record into L1.

        Args:
            unit: The record as the shards store it.
            normalize_whitespace: See :meth:`from_source`. Defaults to ``True``; this is
                the boundary where the ADR-006 leak is closed for every consumer, since
                the shards store the corpus text verbatim.
            strip_comments: See :meth:`from_source`. Defaults to ``False``.

        Returns:
            The corresponding :class:`CodeUnit`.
        """
        return cls.from_source(
            unit.unit_id,
            unit.source,
            language=unit.language,
            target=unit.target,
            vulnerable_lines=unit.vulnerable_lines,
            metadata=unit.metadata,
            normalize_whitespace=normalize_whitespace,
            strip_comments=strip_comments,
        )
