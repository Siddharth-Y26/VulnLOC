"""Offset to line mapping. The single most safety-critical module in the repo.

The line index is the atomic unit of this project (``docs/Architecture.md`` §1.1), and
every label, metric and explanation is expressed in it. An off-by-one here does not
raise — it silently relabels the corpus, and every downstream number stays plausible.
So this module is deliberately paranoid: it validates its inputs, it refuses offsets it
cannot place, and it exposes exactly one definition of "line".

Three facts drive the design, each verified against our own files rather than assumed:

1. **Tree-sitter reports byte offsets; HuggingFace fast tokenizers report character
   offsets.** They agree only while the source is pure ASCII. Big-Vul contains
   non-ASCII comments and string literals, so both maps are built and kept separate.
   Mixing them shifts lines only in the files that contain unicode — the worst possible
   failure mode, because it looks fine in testing.

2. **A line feed ends a line, and nothing else does.** :meth:`str.splitlines` also
   breaks on carriage return, vertical tab, form feed, U+001C to U+001E, U+0085, U+2028
   and U+2029. Tree-sitter does not, and neither does Big-Vul's own ``flaw_line_index``.
   Measured on our corpus: exactly **1 function of 188,636** (``bigvul-76919``, which
   contains a form feed) is split differently by the two rules, so the trap is real,
   rare, and silent. Use :func:`split_lines`; never call :meth:`str.splitlines` on
   source code.

3. **Lines are 1-indexed**, matching the rest of the codebase and Big-Vul's converted
   labels.

**Never read ``node.start_point`` / ``node.end_point``.** Tree-sitter also reports a
0-indexed ``Point.row``, which looks like a free line number. It is not usable: on
``tree-sitter==0.26.0`` (the latest release) with ``tree-sitter-c==0.24.2``, walking a
tree and reading ``start_point`` **segfaults the interpreter** on real corpus input —
reproduced deterministically on ``bigvul-188127``, minimal trigger its first 258 lines
(7,810 bytes), while the identical walk reading ``start_byte`` completes over all 3,061
nodes. A native crash cannot be caught, so this is not a trade-off: byte offsets plus
this module are the only sanctioned route from a node to a line. See
``tests/unit/test_parser.py::test_walking_the_known_segfault_fixture_is_safe``.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from vulnloc.utils.exceptions import AlignmentError

__all__ = ["LINE_SEPARATOR", "LineMap", "split_lines"]

LINE_SEPARATOR = "\n"
"""The only character this project treats as a line break. See module docstring."""


def split_lines(source: str) -> list[str]:
    """Split source into lines the way the rest of the project counts them.

    The canonical splitter. ``lines[i - 1]`` is line ``i``. A trailing separator yields
    a final empty element, which is intentional: it keeps the line count equal to the
    number of separators plus one for every input, so line numbering never depends on
    whether a file happens to end with a newline.

    Args:
        source: Raw source text.

    Returns:
        The lines, without their separators, 0-indexed in the list.
    """
    return source.split(LINE_SEPARATOR)


@dataclass(frozen=True)
class LineMap:
    """Maps byte and character offsets of one source unit onto 1-indexed lines.

    Build with :meth:`from_source`; the constructor takes precomputed tables and does
    not validate them.

    Attributes:
        n_lines: Number of lines, counting a trailing empty line if the source ends
            with a separator.
        n_bytes: Length of the UTF-8 encoding of the source.
        n_chars: Length of the source in characters.
        unit_id: Identifier carried into error payloads.
    """

    n_lines: int
    n_bytes: int
    n_chars: int
    unit_id: str | None
    _byte_starts: tuple[int, ...]
    _char_starts: tuple[int, ...]

    @classmethod
    def from_source(cls, source: str, *, unit_id: str | None = None) -> LineMap:
        """Build the map for one source unit.

        Args:
            source: Raw source text.
            unit_id: Identifier to attach to any error this map later raises.

        Returns:
            A :class:`LineMap` for ``source``.
        """
        byte_starts = [0]
        char_starts = [0]
        byte_cursor = 0
        char_cursor = 0
        for line in split_lines(source)[:-1]:
            # +1 for the separator itself, which belongs to the line it terminates.
            byte_cursor += len(line.encode("utf-8")) + 1
            char_cursor += len(line) + 1
            byte_starts.append(byte_cursor)
            char_starts.append(char_cursor)
        return cls(
            n_lines=len(byte_starts),
            n_bytes=len(source.encode("utf-8")),
            n_chars=len(source),
            unit_id=unit_id,
            _byte_starts=tuple(byte_starts),
            _char_starts=tuple(char_starts),
        )

    def byte_to_line(self, offset: int) -> int:
        """Map a UTF-8 byte offset to its 1-indexed line.

        This is the Tree-sitter side of the bridge: ``node.start_byte`` and
        ``node.end_byte`` are byte offsets.

        Args:
            offset: Byte offset into the UTF-8 encoding of the source. The
                end-of-source offset is accepted so that half-open spans map cleanly.

        Returns:
            The 1-indexed line containing ``offset``.

        Raises:
            AlignmentError: If ``offset`` falls outside the source.
        """
        return self._lookup(offset, self._byte_starts, self.n_bytes, "byte")

    def char_to_line(self, offset: int) -> int:
        """Map a character offset to its 1-indexed line.

        This is the tokenizer side of the bridge: HuggingFace fast tokenizers report
        ``offset_mapping`` in characters, not bytes.

        Args:
            offset: Character offset into the source. The end-of-source offset is
                accepted so that half-open spans map cleanly.

        Returns:
            The 1-indexed line containing ``offset``.

        Raises:
            AlignmentError: If ``offset`` falls outside the source.
        """
        return self._lookup(offset, self._char_starts, self.n_chars, "char")

    def line_byte_span(self, line: int) -> tuple[int, int]:
        """Return the half-open byte span ``[start, end)`` of a line.

        The span excludes the line's own separator, so the slice is the line's text.

        Args:
            line: 1-indexed line number.

        Returns:
            ``(start, end)`` byte offsets.

        Raises:
            AlignmentError: If ``line`` is out of range.
        """
        self._check_line(line)
        start = self._byte_starts[line - 1]
        if line == self.n_lines:
            return start, self.n_bytes
        # -1 drops the separator that terminates this line.
        return start, self._byte_starts[line] - 1

    def _lookup(self, offset: int, starts: tuple[int, ...], total: int, kind: str) -> int:
        if offset < 0 or offset > total:
            raise AlignmentError(
                f"{kind} offset is outside the source; refusing to guess a line number",
                unit_id=self.unit_id,
                offset=offset,
                source_length=total,
                kind=kind,
            )
        return bisect_right(starts, offset)

    def _check_line(self, line: int) -> None:
        if line < 1 or line > self.n_lines:
            raise AlignmentError(
                "line number is outside the source",
                unit_id=self.unit_id,
                line=line,
                n_lines=self.n_lines,
            )
