"""Tree-sitter wrapper for C/C++ (ADR-002 §D2.1).

Parsing is where a corpus of real code stops cooperating. Big-Vul functions were cut
out of their translation units, so they arrive without their headers, typedefs and
macro definitions; a standalone parse of such a fragment is often *partially* wrong
rather than cleanly broken. Tree-sitter is chosen precisely because it recovers from
that instead of refusing — but recovery is not success, and pretending otherwise would
inflate every downstream number.

So this module never raises on malformed input. It returns a :class:`ParseResult` that
carries the damage as data — ``error_nodes``, ``missing_nodes``, ``first_error_line`` —
and lets the caller decide. ``Phases.md`` Phase 2 requires the corpus-wide parse success
rate to be *reported and categorised, not hidden* (``Rules.md`` §4.6), which is only
possible if failures survive as values.

The one thing that *is* fatal is an unknown language: guessing a grammar would silently
produce a plausible tree for the wrong syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Node, Parser, Tree

from vulnloc.representation.line_map import LINE_SEPARATOR, LineMap, split_lines
from vulnloc.utils.exceptions import AlignmentError, ParseError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "LANGUAGE_ALIASES",
    "LANGUAGE_POLICIES",
    "ParseResult",
    "TreeSitterParser",
    "iter_nodes",
    "node_to_line",
    "resolve_language",
    "verify_line_alignment",
]

_LOGGER = get_logger(__name__)

LANGUAGE_ALIASES: dict[str, str] = {
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
}
"""Dataset language strings → grammar key.

Big-Vul's own ``lang`` column uses ``C``, ``CPP`` and ``C++`` (measured: 185,997 /
2,636 / 3 functions), so all three must resolve. Matching is case-insensitive.
"""

STATIC_LANGUAGE_POLICIES: tuple[str, ...] = ("cpp", "c", "declared")
"""Policies that name a grammar without looking at the source."""

LANGUAGE_POLICIES: tuple[str, ...] = ("best_of_two", *STATIC_LANGUAGE_POLICIES)
"""The grammar-selection policies, in the order ADR-005 ranks them.

``best_of_two`` is the decision of record: parse as C++, and only if that leaves damage,
parse as C as well and keep the less damaged tree. The three static policies remain
selectable so the alternatives can be re-measured rather than re-argued — ``declared``
reproduces every number taken before ADR-005.
"""


def _damage(result: ParseResult) -> tuple[int, int]:
    """Return a comparable damage score for a parse.

    Args:
        result: A parse result.

    Returns:
        ``(error_nodes + missing_nodes, error_nodes)``. The total ranks first because a
        synthesised semicolon and a wrongly-shaped subtree both cost the graph a node;
        ERROR count breaks ties because a MISSING node is a repair and an ERROR node is
        an abandonment.
    """
    return (result.error_nodes + result.missing_nodes, result.error_nodes)


def resolve_language(policy: str, declared: str, *, unit_id: str) -> str:
    """Return the grammar a unit is parsed with under a **static** ``policy``.

    The corpus's ``lang`` column is an input to this function, not an authority over it:
    ADR-005 rejects it because following the declared language parses 37.0% of Big-Vul
    cleanly against 88.2% for C++.

    ``best_of_two`` — ADR-005's actual decision — cannot be answered here, because it is
    a function of the source rather than of the language string. Use
    :meth:`TreeSitterParser.parse_with_policy` instead.

    Args:
        policy: ``"cpp"``, ``"c"`` or ``"declared"``.
        declared: The unit's own language string, used only under ``"declared"``.
        unit_id: Identifier carried into any error payload.

    Returns:
        A grammar key — ``"c"`` or ``"cpp"``.

    Raises:
        ParseError: If ``policy`` is not a static policy, or if it is ``"declared"`` and
            ``declared`` names no grammar we have.
    """
    normalised = policy.strip().lower()
    if normalised not in STATIC_LANGUAGE_POLICIES:
        raise ParseError(
            "policy does not name a grammar on its own; parse under it instead",
            unit_id=unit_id,
            policy=policy,
            supported=list(STATIC_LANGUAGE_POLICIES),
            hint="TreeSitterParser.parse_with_policy handles every policy",
        )
    if normalised != "declared":
        return normalised

    key = LANGUAGE_ALIASES.get(declared.strip().lower())
    if key is None:
        raise ParseError(
            "no Tree-sitter grammar for this language; refusing to guess one",
            unit_id=unit_id,
            language=declared,
            supported=sorted(set(LANGUAGE_ALIASES)),
        )
    return key


@dataclass(frozen=True)
class ParseResult:
    """One parsed source unit, with its damage recorded rather than discarded.

    Attributes:
        unit_id: Identifier of the unit that was parsed.
        language: Resolved grammar key — ``"c"`` or ``"cpp"``.
        source: The exact text that was parsed.
        tree: The Tree-sitter tree. Holds the memory its nodes point into, so keep it
            alive for as long as any node is used.
        line_map: Offset → line map for :attr:`source`.
        error_nodes: Count of ``ERROR`` nodes the grammar inserted while recovering.
        missing_nodes: Count of ``MISSING`` nodes — tokens the grammar expected and
            synthesised, typically a dropped semicolon or brace.
        first_error_line: 1-indexed line of the earliest ERROR/MISSING node, or
            ``None`` when the parse is clean.
    """

    unit_id: str
    language: str
    source: str
    tree: Tree = field(repr=False)
    line_map: LineMap = field(repr=False)
    error_nodes: int
    missing_nodes: int
    first_error_line: int | None

    @property
    def root(self) -> Node:
        """The root node of the tree."""
        return self.tree.root_node

    @property
    def is_clean(self) -> bool:
        """Whether the parse produced no ERROR and no MISSING nodes."""
        return self.error_nodes == 0 and self.missing_nodes == 0

    def summary(self) -> dict[str, object]:
        """Return the per-unit record used to build the corpus parse report.

        Returns:
            A JSON-serializable mapping.
        """
        return {
            "unit_id": self.unit_id,
            "language": self.language,
            "n_lines": self.line_map.n_lines,
            "error_nodes": self.error_nodes,
            "missing_nodes": self.missing_nodes,
            "first_error_line": self.first_error_line,
            "is_clean": self.is_clean,
        }


@cache
def _language(key: str) -> Language:
    """Return the compiled grammar for ``key``, built once per process."""
    if key == "c":
        return Language(tree_sitter_c.language())
    return Language(tree_sitter_cpp.language())


def iter_nodes(node: Node) -> Iterator[Node]:
    """Walk a tree depth-first, yielding every node including the root.

    Uses an explicit stack rather than recursion: C functions in this corpus nest
    deeply enough to matter, and the recursion limit is not a parse property worth
    inheriting.

    Args:
        node: Node to start from.

    Yields:
        Every node in the subtree, parents before children.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def node_to_line(node: Node, line_map: LineMap) -> int:
    """Return the 1-indexed line a node starts on.

    Derived from ``node.start_byte`` through ``line_map``. Tree-sitter's own
    ``start_point.row`` is never read — see :mod:`vulnloc.representation.line_map` for
    the segfault that rules it out.

    Args:
        node: Any Tree-sitter node.
        line_map: The map for the source this node came from.

    Returns:
        The 1-indexed start line.

    Raises:
        AlignmentError: If the node's offset falls outside the source.
    """
    return line_map.byte_to_line(node.start_byte)


def verify_line_alignment(result: ParseResult) -> None:
    """Check every leaf node's text against the line it was assigned.

    An independent check rather than a restatement: the line number comes from bisecting
    the offset table, and is then confirmed by looking up that line's *text* and finding
    the node's own source there. An off-by-one fails immediately, because line ``n - 1``
    almost never contains line ``n``'s token.

    Only single-line leaves are checked. Interior nodes and multi-line leaves (block
    comments, string literals with escapes) span lines by construction, so "the text
    appears on the start line" is not a property they have.

    Args:
        result: A parsed unit.

    Raises:
        AlignmentError: On the first leaf whose text is absent from its assigned line.
    """
    lines = split_lines(result.source)
    source_bytes = result.source.encode("utf-8")
    for node in iter_nodes(result.root):
        if node.child_count:
            continue
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        if not text or LINE_SEPARATOR in text:
            continue
        line_no = node_to_line(node, result.line_map)
        if text not in lines[line_no - 1]:
            raise AlignmentError(
                "node text does not appear on the line it was mapped to",
                unit_id=result.unit_id,
                node_type=node.type,
                start_byte=node.start_byte,
                line=line_no,
                node_text=text[:60],
                line_text=lines[line_no - 1][:60],
            )


class TreeSitterParser:
    """Parses C/C++ source into Tree-sitter trees paired with a line map.

    One instance is reusable and holds one :class:`~tree_sitter.Parser` per grammar.
    Instances are not thread-safe; give each worker its own.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def parse_with_policy(
        self, source: str, *, policy: str, declared: str, unit_id: str
    ) -> ParseResult:
        """Parse one unit under a grammar-selection policy (ADR-005).

        Under ``best_of_two`` the grammar becomes a function of the source, which is the
        price ADR-005 accepted for the extra 1,712 clean parses. Two properties keep that
        reproducible: C++ is always tried first and always wins ties, so identical source
        always yields an identical grammar choice, and the second parse only happens when
        the first left damage — 11.8% of the corpus, measured.

        Args:
            source: The source text.
            policy: One of :data:`LANGUAGE_POLICIES`.
            declared: The unit's own language string, consulted only under ``declared``.
            unit_id: Identifier carried into the result and any error payload.

        Returns:
            The chosen parse. :attr:`ParseResult.language` records which grammar won.

        Raises:
            ParseError: If ``policy`` is not a known policy.
        """
        normalised = policy.strip().lower()
        if normalised != "best_of_two":
            grammar = resolve_language(normalised, declared, unit_id=unit_id)
            return self.parse(source, language=grammar, unit_id=unit_id)

        primary = self.parse(source, language="cpp", unit_id=unit_id)
        if primary.is_clean:
            return primary
        alternate = self.parse(source, language="c", unit_id=unit_id)
        return alternate if _damage(alternate) < _damage(primary) else primary

    def parse(self, source: str, *, language: str, unit_id: str) -> ParseResult:
        """Parse one source unit.

        Args:
            source: The source text. Encoded as UTF-8 for Tree-sitter, which works in
                bytes; :class:`LineMap` keeps both offset spaces straight.
            language: Dataset language string — ``"C"``, ``"CPP"`` and ``"C++"`` all
                resolve (see :data:`LANGUAGE_ALIASES`).
            unit_id: Identifier carried into the result and into any error payload.

        Returns:
            The parse result, including any recovery damage. Syntax errors do **not**
            raise; they are counted.

        Raises:
            ParseError: If ``language`` names no grammar we have.
        """
        key = LANGUAGE_ALIASES.get(language.strip().lower())
        if key is None:
            raise ParseError(
                "no Tree-sitter grammar for this language; refusing to guess one",
                unit_id=unit_id,
                language=language,
                supported=sorted(set(LANGUAGE_ALIASES)),
            )

        parser = self._parsers.get(key)
        if parser is None:
            parser = Parser(_language(key))
            self._parsers[key] = parser

        tree = parser.parse(source.encode("utf-8"))
        line_map = LineMap.from_source(source, unit_id=unit_id)
        error_nodes, missing_nodes, first_error_line = _scan_damage(tree, line_map)

        if error_nodes or missing_nodes:
            _LOGGER.debug(
                "parse recovered from syntax errors",
                extra={
                    "unit_id": unit_id,
                    "language": key,
                    "error_nodes": error_nodes,
                    "missing_nodes": missing_nodes,
                    "first_error_line": first_error_line,
                },
            )

        return ParseResult(
            unit_id=unit_id,
            language=key,
            source=source,
            tree=tree,
            line_map=line_map,
            error_nodes=error_nodes,
            missing_nodes=missing_nodes,
            first_error_line=first_error_line,
        )


def _scan_damage(tree: Tree, line_map: LineMap) -> tuple[int, int, int | None]:
    """Count ERROR and MISSING nodes and locate the earliest one.

    Returns:
        ``(error_nodes, missing_nodes, first_error_line)``.
    """
    if not tree.root_node.has_error:
        return 0, 0, None

    errors = 0
    missing = 0
    first_line: int | None = None
    for node in iter_nodes(tree.root_node):
        if node.is_missing:
            missing += 1
        elif node.type == "ERROR":
            errors += 1
        else:
            continue
        line = line_map.byte_to_line(node.start_byte)
        if first_line is None or line < first_line:
            first_line = line
    return errors, missing, first_line
