"""AST builder — first of the four graphs in ADR-002 §D2.2's build order.

The AST is the one graph Tree-sitter hands over directly, so it is the cheapest route
to a working structural branch: Phase 4 may start as soon as this lands, without waiting
for the CFG or the DFG. Everything here is a transcription of the parse tree, not an
analysis of it — no control flow, no data flow, no inference. The compiler work arrives
in ``cfg.py`` and ``dfg.py``.

Two choices worth stating, because both are visible in the graph statistics:

* **Anonymous nodes are kept.** Tree-sitter marks punctuation and operators as
  unnamed, and a "clean" AST usually drops them. We do not: ``+``, ``==``, ``[`` and
  ``&`` are precisely the tokens an out-of-bounds or integer-overflow flaw turns on.
  They roughly double the node count, which is a real cost against 4 GiB of VRAM
  (``Memory.md`` §7 #7) — so it is a decision, recorded here, not an accident.
* **Traversal is iterative.** Some corpus functions nest deeply enough that a recursive
  walk would hit Python's recursion limit, turning a data property into a crash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulnloc.representation.graphs.base import CodeGraph, GraphEdge, GraphNode
from vulnloc.representation.registry import register_graph

if TYPE_CHECKING:
    from tree_sitter import Node

    from vulnloc.representation.parser import ParseResult

__all__ = ["AstGraphBuilder", "build_ast_graph"]

GRAPH_TYPE = "ast"


def build_ast_graph(result: ParseResult) -> CodeGraph:
    """Transcribe a parse tree into a :class:`CodeGraph`.

    Nodes are numbered in pre-order, so node 0 is the root and a node's children always
    carry higher ids than it does. Edges run parent to child.

    Line numbers come from ``start_byte`` through the unit's
    :class:`~vulnloc.representation.line_map.LineMap` — never from Tree-sitter's
    ``Point``, which segfaults (``Memory.md`` §7 #19).

    Args:
        result: A parsed unit.

    Returns:
        The AST as a graph, already validated.

    Raises:
        GraphBuildError: If the resulting graph violates a structural invariant.
        AlignmentError: If a node's offset falls outside the source.
    """
    line_map = result.line_map
    source_bytes = result.source.encode("utf-8")

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # (tree-sitter node, id of its parent in `nodes`); None for the root.
    stack: list[tuple[Node, int | None]] = [(result.root, None)]
    while stack:
        ts_node, parent_id = stack.pop()
        node_id = len(nodes)
        is_leaf = ts_node.child_count == 0
        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type=ts_node.type,
                line_no=line_map.byte_to_line(ts_node.start_byte),
                start_byte=ts_node.start_byte,
                end_byte=ts_node.end_byte,
                is_named=ts_node.is_named,
                text=(
                    source_bytes[ts_node.start_byte : ts_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    if is_leaf
                    else ""
                ),
            )
        )
        if parent_id is not None:
            edges.append(GraphEdge(source=parent_id, target=node_id, edge_type="ast"))
        # Reversed, so that popping yields children left to right and ids follow
        # source order.
        stack.extend((child, node_id) for child in reversed(ts_node.children))

    graph = CodeGraph(
        unit_id=result.unit_id,
        graph_type=GRAPH_TYPE,
        nodes=tuple(nodes),
        edges=tuple(edges),
        n_lines=line_map.n_lines,
    )
    graph.validate()
    return graph


@register_graph(GRAPH_TYPE)
class AstGraphBuilder:
    """Registry entry for the AST graph (``Architecture.md`` §9, extension point 1)."""

    graph_type = GRAPH_TYPE

    def build(self, result: ParseResult) -> CodeGraph:
        """Build the AST graph for one parsed unit.

        Args:
            result: A parsed unit.

        Returns:
            The AST as a graph.
        """
        return build_ast_graph(result)
