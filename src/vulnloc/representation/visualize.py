"""Render a graph with its line annotations (``Phases.md`` Phase 2 deliverable).

For debugging and for paper figures. The point of the rendering is not the tree shape —
it is that **every node is shown against the line it claims**, so a line-mapping bug is
visible to a person rather than only to an assertion. When a golden test fails, this is
what you look at.

Two renderers, no plotting dependency: a text tree for a terminal or a test failure
message, and Graphviz DOT for a figure. DOT is emitted as a string, so producing an
image stays an explicit, offline step rather than a runtime dependency.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vulnloc.representation.graphs.base import CodeGraph
    from vulnloc.representation.units import CodeUnit

__all__ = ["to_dot", "render_tree"]

_MAX_LABEL = 24


def _label(node_type: str, text: str) -> str:
    """Build a short node label. Not escaped for any particular output format."""
    if not text:
        return node_type
    flat = text.replace("\n", "\\n")
    if len(flat) > _MAX_LABEL:
        flat = flat[:_MAX_LABEL] + "…"
    return f"{node_type} '{flat}'"


def _dot_escape(label: str) -> str:
    """Escape a label for a DOT string literal.

    Backslash first, or the escapes we add would themselves be escaped. C source is
    full of both characters — a string literal node's text is often a bare ``"`` — so
    an unescaped label produces DOT that Graphviz refuses to parse.

    Args:
        label: Raw label text.

    Returns:
        The label, safe to place inside double quotes.
    """
    return label.replace("\\", "\\\\").replace('"', '\\"')


def render_tree(
    graph: CodeGraph,
    *,
    unit: CodeUnit | None = None,
    highlight: frozenset[int] | None = None,
    max_nodes: int = 200,
) -> str:
    """Render the graph as an indented text tree, annotated by line.

    Args:
        graph: The graph to render.
        unit: The source unit. When given, each node's line text is shown beside it,
            which is what makes a misalignment obvious.
        highlight: 1-indexed lines to mark with ``>>`` — pass the unit's
            ``vulnerable_lines`` to see whether the flagged lines carry nodes at all.
        max_nodes: Stop after this many nodes; real graphs run to thousands.

    Returns:
        The rendered tree.
    """
    children: dict[int, list[int]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type == "ast":
            children[edge.source].append(edge.target)

    marked = highlight or frozenset()
    lines: list[str] = [f"# {graph.unit_id} [{graph.graph_type}] {len(graph.nodes)} nodes"]
    shown = 0
    stack: list[tuple[int, int]] = [(0, 0)] if graph.nodes else []
    while stack and shown < max_nodes:
        node_id, depth = stack.pop()
        node = graph.nodes[node_id]
        marker = ">>" if node.line_no in marked else "  "
        source_text = ""
        if unit is not None and 1 <= node.line_no <= unit.n_lines:
            source_text = f"  | {unit.lines[node.line_no - 1].strip()[:60]}"
        lines.append(
            f"{marker} {node.line_no:4} {'  ' * depth}{_label(node.node_type, node.text)}"
            f"{source_text}"
        )
        shown += 1
        stack.extend((child, depth + 1) for child in reversed(children[node_id]))

    if shown < len(graph.nodes):
        lines.append(f"   ... {len(graph.nodes) - shown} more nodes")
    return "\n".join(lines)


def to_dot(graph: CodeGraph, *, highlight: frozenset[int] | None = None) -> str:
    """Render the graph as Graphviz DOT.

    Nodes are grouped into one cluster per line, so the figure reads as "which lines
    own which structure" rather than as an undifferentiated tree.

    Args:
        graph: The graph to render.
        highlight: 1-indexed lines to fill, typically the flagged lines.

    Returns:
        DOT source. Render with ``dot -Tpng``.
    """
    marked = highlight or frozenset()
    by_line: dict[int, list[int]] = defaultdict(list)
    for node in graph.nodes:
        by_line[node.line_no].append(node.node_id)

    out: list[str] = [
        f"digraph {graph.graph_type} {{",
        f'  label="{graph.unit_id} [{graph.graph_type}]";',
        "  node [shape=box, fontname=monospace, fontsize=9];",
    ]
    for line_no in sorted(by_line):
        fill = ', style=filled, fillcolor="#ffdddd"' if line_no in marked else ""
        out.append(f"  subgraph cluster_line_{line_no} {{")
        out.append(f'    label="line {line_no}"{fill};')
        for node_id in by_line[line_no]:
            node = graph.nodes[node_id]
            label = _dot_escape(_label(node.node_type, node.text))
            out.append(f'    n{node_id} [label="{label}"];')
        out.append("  }")
    for edge in graph.edges:
        style = "" if edge.edge_type == "ast" else f' [label="{edge.edge_type}"]'
        out.append(f"  n{edge.source} -> n{edge.target}{style};")
    out.append("}")
    return "\n".join(out)
