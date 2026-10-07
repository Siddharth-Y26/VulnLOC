"""Evidence extraction: what an explanation of a flagged line is allowed to cite.

Nothing here runs a model. The caller has already scored the unit and, if the fusion has
attention, kept its two maps from that same forward pass
(:meth:`vulnloc.models.fused.FusedLineModel.explain`); this module turns those arrays, the
unit's graphs and its source text into an :class:`~vulnloc.explanation.types.Evidence`.

Each part has a plain rule and no tunable judgement beyond the limits in
:class:`EvidenceSettings` (``configs/explainer/template.yaml``):

* **attended lines** — the row of each attention map that belongs to the flagged line,
  averaged over heads, heaviest first. The flagged line itself is left out: a line's
  weight on its own other-branch state is not *context*.
* **data-flow path** — walk the name-based DFG backwards from the flagged line, each time
  to the nearest earlier line that defines something the current line reads.
* **graph nodes** — named AST nodes that start on the flagged line, the kinds a memory
  flaw usually turns on first.
* **tokens** — identifiers on the flagged line, then on the attended lines in weight order.

Evidence is extracted **only for a flagged line**. Asking for any other line raises, so an
explanation of an unflagged line cannot be assembled by accident (FR-15).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vulnloc.explanation.types import (
    SEM_FROM_STR,
    STR_FROM_SEM,
    AttendedLine,
    Evidence,
    Prediction,
)
from vulnloc.utils.exceptions import ExplanationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vulnloc.representation.graphs.base import CodeGraph
    from vulnloc.representation.units import CodeUnit

__all__ = [
    "EvidenceSettings",
    "attended_lines",
    "dataflow_path",
    "extract_evidence",
    "identifiers",
    "line_nodes",
    "top_tokens",
]

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

_KEYWORDS = frozenset(
    {
        *("auto", "bool", "break", "case", "catch", "char", "class", "const", "constexpr"),
        *("continue", "default", "delete", "do", "double", "else", "enum", "explicit", "extern"),
        *("false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "namespace"),
        *("new", "nullptr", "operator", "private", "protected", "public", "register", "return"),
        *("short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw"),
        *("true", "try", "typedef", "typename", "union", "unsigned", "using", "virtual", "void"),
        *("volatile", "while"),
    }
)
"""C and C++ keywords: never an identifier worth citing."""

_NODE_PRIORITY = (
    "call_expression",
    "subscript_expression",
    "pointer_expression",
    "assignment_expression",
    "update_expression",
    "binary_expression",
    "cast_expression",
    "field_expression",
    "init_declarator",
    "declaration",
    "return_statement",
    "if_statement",
    "for_statement",
    "while_statement",
)
"""Node kinds listed first among a line's nodes — a selection rule, not an importance."""


@dataclass(frozen=True)
class EvidenceSettings:
    """Limits on how much evidence one explanation carries.

    Attributes:
        top_k_attended: Attended lines kept per attention direction.
        min_attention_weight: Attended lines below this head-averaged weight are dropped.
        max_dataflow_hops: Longest definition-to-use chain followed back from the line.
        max_graph_nodes: AST nodes kept for the flagged line.
        max_tokens: Identifiers kept.

    Raises:
        ExplanationError: If a limit is negative or the weight floor is outside ``[0, 1]``.
    """

    top_k_attended: int
    min_attention_weight: float
    max_dataflow_hops: int
    max_graph_nodes: int
    max_tokens: int

    def __post_init__(self) -> None:
        limits = {
            "top_k_attended": self.top_k_attended,
            "max_dataflow_hops": self.max_dataflow_hops,
            "max_graph_nodes": self.max_graph_nodes,
            "max_tokens": self.max_tokens,
        }
        negative = {name: value for name, value in limits.items() if value < 0}
        if negative:
            raise ExplanationError("evidence limits cannot be negative", limits=negative)
        if not 0.0 <= self.min_attention_weight <= 1.0:
            raise ExplanationError(
                "an attention weight floor must lie in [0, 1]",
                min_attention_weight=self.min_attention_weight,
            )


def attended_lines(
    attention: np.ndarray,
    line: int,
    *,
    n_lines: int,
    direction: str,
    top_k: int,
    min_weight: float = 0.0,
) -> tuple[AttendedLine, ...]:
    """The lines one line's cross-attention weighted most.

    Args:
        attention: ``[H, L, L]`` per-head weights or ``[L, L]`` already averaged, for one
            unit. Entry ``[..., i - 1, j - 1]`` is the weight line ``i`` put on line ``j``.
            ``L`` may be shorter than the unit — the model trims its line axis to the batch.
        line: The 1-indexed query line.
        n_lines: Lines in the unit. Key positions past it are padding and are ignored.
        direction: ``"sem_from_str"`` or ``"str_from_sem"``.
        top_k: How many to keep.
        min_weight: Drop lines weighted below this.

    Returns:
        Up to ``top_k`` other lines, heaviest first, ties to the earlier line. Empty when
        the query line lies beyond the map or put no weight anywhere.

    Raises:
        ExplanationError: If the map is not square over its last two axes, or the query
            row holds a value that is not a finite, non-negative number.
    """
    if attention.ndim not in (2, 3) or attention.shape[-1] != attention.shape[-2]:
        raise ExplanationError(
            "an attention map is [H, L, L] or [L, L]", shape=list(attention.shape), line=line
        )
    if line < 1:
        raise ExplanationError("lines are 1-indexed", line=line)
    axis = attention.shape[-1]
    if line > axis or top_k == 0:
        return ()
    rows = attention[..., line - 1, :].astype(np.float64)  # [H, L] or [L]
    row = rows.mean(axis=0) if rows.ndim == 2 else rows  # [L]
    row = row[: min(n_lines, axis)]
    if not np.all(np.isfinite(row)) or np.any(row < 0.0):
        raise ExplanationError(
            "an attention row holds a value that is not a finite, non-negative weight",
            line=line,
            direction=direction,
        )
    candidates = [
        (float(weight), index + 1)
        for index, weight in enumerate(row)
        if index + 1 != line and weight > 0.0 and weight >= min_weight
    ]
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    return tuple(
        AttendedLine(line=other, weight=weight, direction=direction)
        for weight, other in candidates[:top_k]
    )


def dataflow_path(
    dfg: CodeGraph, line: int, *, max_hops: int
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """A definition-to-use chain that ends at ``line``.

    From the current line, step to the **nearest earlier** line holding a definition the
    current line reads, and repeat. Later lines are never stepped to — a loop-carried
    definition is real data flow, but a path that runs backwards through the source does
    not help a reader, and the rule has to be one a test can state.

    The graph is name-based: no aliasing, no scoping, weak updates through ``*p`` and
    ``a[i]`` (``representation/graphs/dfg.py``). A path is a place to look, not a proof.

    Args:
        dfg: A graph carrying ``dfg`` edges (a DFG or a CPG) for the unit.
        line: The 1-indexed line the chain must end at.
        max_hops: Longest chain, in edges.

    Returns:
        ``(path, symbols)`` — ascending lines ending at ``line``, and the name carried
        over each hop. Both empty when nothing earlier flows into the line.
    """
    incoming: dict[int, dict[int, set[str]]] = {}
    for edge in dfg.edges:
        if edge.edge_type != "dfg":
            continue
        source, target = dfg.nodes[edge.source], dfg.nodes[edge.target]
        if source.line_no >= target.line_no:
            continue
        symbol = target.text or source.text
        if not symbol:
            continue
        incoming.setdefault(target.line_no, {}).setdefault(source.line_no, set()).add(symbol)

    path = [line]
    symbols: list[str] = []
    for _ in range(max_hops):
        definitions = incoming.get(path[0])
        if not definitions:
            break
        nearest = max(definitions)
        path.insert(0, nearest)
        symbols.insert(0, min(definitions[nearest]))
    if len(path) == 1:
        return (), ()
    return tuple(path), tuple(symbols)


def line_nodes(graph: CodeGraph, line: int, *, limit: int) -> tuple[str, ...]:
    """Named AST nodes that start on ``line``, as ``"n_<node_id>"``.

    Ordered by kind — calls, subscripts, dereferences and assignments first — then by
    position in the tree. A line the grammar gave no node to returns nothing; 7.84% of the
    corpus's flagged lines are like that (``Memory.md`` §7 #17).

    Args:
        graph: Any graph of the unit; all four types share one node set.
        line: The 1-indexed line.
        limit: How many to keep.

    Returns:
        Up to ``limit`` node identifiers.
    """
    priority = {kind: index for index, kind in enumerate(_NODE_PRIORITY)}
    on_line = [node for node in graph.nodes if node.line_no == line and node.is_named]
    on_line.sort(key=lambda node: (priority.get(node.node_type, len(priority)), node.node_id))
    return tuple(f"n_{node.node_id}" for node in on_line[:limit])


def identifiers(text: str) -> tuple[str, ...]:
    """Identifiers in a line of source, in order, without repeats or keywords.

    Args:
        text: One source line.

    Returns:
        The identifiers.
    """
    seen: dict[str, None] = {}
    for match in _IDENTIFIER.finditer(text):
        word = match.group(0)
        if word not in _KEYWORDS:
            seen.setdefault(word, None)
    return tuple(seen)


def top_tokens(
    unit: CodeUnit, line: int, attended: Sequence[AttendedLine], *, limit: int
) -> tuple[str, ...]:
    """Identifiers on the flagged line, then on the lines it attended to, heaviest first.

    Args:
        unit: The unit.
        line: The flagged line.
        attended: Its attended lines, in any order.
        limit: How many identifiers to keep.

    Returns:
        Up to ``limit`` identifiers, each of which occurs on ``line`` or on an attended line.
    """
    order = [line]
    for entry in sorted(attended, key=lambda item: (-item.weight, item.line)):
        if entry.line not in order and 1 <= entry.line <= unit.n_lines:
            order.append(entry.line)
    tokens: dict[str, None] = {}
    for number in order:
        for word in identifiers(unit.lines[number - 1]):
            tokens.setdefault(word, None)
    return tuple(tokens)[:limit]


def extract_evidence(
    unit: CodeUnit,
    prediction: Prediction,
    line: int,
    *,
    settings: EvidenceSettings,
    ast: CodeGraph | None = None,
    dfg: CodeGraph | None = None,
    attn_sem_from_str: np.ndarray | None = None,
    attn_str_from_sem: np.ndarray | None = None,
) -> Evidence:
    """Assemble the evidence for one flagged line.

    Every source is optional because every source can honestly be absent: a fusion without
    attention has no maps, and a unit whose graph will not build has no nodes and no data
    flow. Absent evidence is an empty field, never a guessed one.

    Args:
        unit: The unit, as the model saw it (ADR-006 normalized).
        prediction: The unit's calibrated scores.
        line: The line to gather evidence for. Must be flagged.
        settings: The limits.
        ast: The unit's AST, for :func:`line_nodes`.
        dfg: The unit's DFG, for :func:`dataflow_path`.
        attn_sem_from_str: ``[H, L, L]`` — semantic lines querying structural lines.
        attn_str_from_sem: ``[H, L, L]`` — structural lines querying semantic lines.

    Returns:
        The evidence.

    Raises:
        ExplanationError: If ``line`` is not flagged, or the prediction and the unit
            disagree about which unit this is.
    """
    if prediction.unit_id != unit.unit_id:
        raise ExplanationError(
            "the prediction belongs to another unit",
            unit_id=unit.unit_id,
            prediction_unit=prediction.unit_id,
        )
    if not prediction.is_flagged(line):
        raise ExplanationError(
            "evidence is gathered only for a line scored above the threshold (FR-15)",
            unit_id=unit.unit_id,
            line=line,
            threshold=prediction.threshold,
        )

    attended: list[AttendedLine] = []
    for attention, direction in (
        (attn_sem_from_str, SEM_FROM_STR),
        (attn_str_from_sem, STR_FROM_SEM),
    ):
        if attention is not None:
            attended.extend(
                attended_lines(
                    attention,
                    line,
                    n_lines=unit.n_lines,
                    direction=direction,
                    top_k=settings.top_k_attended,
                    min_weight=settings.min_attention_weight,
                )
            )
    attended.sort(key=lambda entry: (-entry.weight, entry.line, entry.direction))

    path: tuple[int, ...] = ()
    symbols: tuple[str, ...] = ()
    if dfg is not None and settings.max_dataflow_hops:
        path, symbols = dataflow_path(dfg, line, max_hops=settings.max_dataflow_hops)

    return Evidence(
        unit_id=unit.unit_id,
        line=line,
        attended=tuple(attended),
        dataflow_path=path,
        dataflow_symbols=symbols,
        graph_nodes=(
            line_nodes(ast, line, limit=settings.max_graph_nodes) if ast is not None else ()
        ),
        top_tokens=top_tokens(unit, line, attended, limit=settings.max_tokens),
    )
