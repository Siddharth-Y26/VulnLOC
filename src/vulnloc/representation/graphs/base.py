"""Shared graph types (``docs/Architecture.md`` §4.1).

Every builder registered under ``@register_graph`` returns a :class:`CodeGraph`, and
ablation A3 (AST · CFG · DFG · AST+CFG · CPG) works by selecting a different builder
from config — so the container has to be the same shape for all of them, with the edge
*type* carrying the difference rather than the class.

The invariant the rest of the project leans on: **every node carries a line number**.
:meth:`CodeGraph.validate` enforces it, and the CPG assembly in ADR-002 §D2.2 is only
sound because a union of graphs over the same node set keeps the same line mapping.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from vulnloc.utils.exceptions import GraphBuildError

__all__ = ["EDGE_TYPES", "CodeGraph", "EdgeType", "GraphEdge", "GraphNode"]

EdgeType = Literal["ast", "cfg", "dfg", "call"]

EDGE_TYPES: tuple[EdgeType, ...] = ("ast", "cfg", "dfg", "call")
"""The closed set of edge types. A CPG is a union over these, not a new type."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One node, always attributable to a line.

    Attributes:
        node_id: Index into :attr:`CodeGraph.nodes`; edges refer to these.
        node_type: Grammar symbol, e.g. ``"binary_expression"``. The categorical
            feature ADR-002 §D2.5 embeds.
        line_no: 1-indexed line the node starts on.
        start_byte: Start offset in the UTF-8 encoding of the source.
        end_byte: End offset, exclusive.
        is_named: False for anonymous grammar tokens — punctuation and operators.
            Kept rather than dropped: ``+``, ``==`` and ``[`` are exactly the tokens a
            memory-safety flaw turns on.
        text: Source text, populated for leaves only. Interior nodes span their
            children's text, so storing it again would multiply the graph's memory by
            the tree depth for no added signal.
    """

    node_id: int
    node_type: str
    line_no: int
    start_byte: int
    end_byte: int
    is_named: bool
    text: str = ""


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One typed, directed edge.

    Attributes:
        source: Origin node id.
        target: Destination node id.
        edge_type: Which relation this edge encodes.
    """

    source: int
    target: int
    edge_type: EdgeType


@dataclass(frozen=True)
class CodeGraph:
    """A typed graph over one :class:`~vulnloc.representation.units.CodeUnit`.

    Attributes:
        unit_id: The unit this graph was built from.
        graph_type: ``"ast"``, ``"cfg"``, ``"dfg"`` or ``"cpg"``.
        nodes: Nodes, positionally indexed by :attr:`GraphNode.node_id`.
        edges: Typed edges.
        n_lines: Line count of the source, so line coverage can be computed without
            the source being carried along.
    """

    unit_id: str
    graph_type: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    n_lines: int = 0

    @property
    def node_to_line(self) -> dict[int, int]:
        """Node id to 1-indexed line — the structural half of the line-alignment bridge.

        Returns:
            A mapping covering every node.
        """
        return {node.node_id: node.line_no for node in self.nodes}

    @property
    def lines_covered(self) -> frozenset[int]:
        """Lines that at least one node starts on.

        Blank lines and lines holding only a comment the grammar discards will be
        absent, which is why line coverage is reported rather than assumed.

        Returns:
            The set of covered 1-indexed lines.
        """
        return frozenset(node.line_no for node in self.nodes)

    def edge_type_counts(self) -> dict[str, int]:
        """Count edges by type, for the graph statistics Phase 2 must report.

        Returns:
            A mapping from edge type to count.
        """
        return dict(Counter(edge.edge_type for edge in self.edges))

    def stats(self) -> dict[str, object]:
        """Return the per-graph record used to build corpus-wide graph statistics.

        Returns:
            A JSON-serializable mapping.
        """
        return {
            "unit_id": self.unit_id,
            "graph_type": self.graph_type,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_lines": self.n_lines,
            "n_lines_covered": len(self.lines_covered),
            "edge_types": self.edge_type_counts(),
        }

    def validate(self) -> None:
        """Check the invariants no downstream layer re-checks.

        Node ids must be dense and positional, every edge must land on a real node, and
        every node must carry a line inside the source. A graph violating any of these
        would fail much later, inside a scatter over the line axis, as a shape error
        with no way back to the unit that caused it.

        Raises:
            GraphBuildError: On the first violation found.
        """
        for index, node in enumerate(self.nodes):
            if node.node_id != index:
                raise GraphBuildError(
                    "node ids are not dense and positional",
                    unit_id=self.unit_id,
                    expected=index,
                    found=node.node_id,
                )
            if node.line_no < 1 or (self.n_lines and node.line_no > self.n_lines):
                raise GraphBuildError(
                    "node line is outside the source",
                    unit_id=self.unit_id,
                    node_id=node.node_id,
                    line_no=node.line_no,
                    n_lines=self.n_lines,
                )

        n_nodes = len(self.nodes)
        for edge in self.edges:
            if not (0 <= edge.source < n_nodes and 0 <= edge.target < n_nodes):
                raise GraphBuildError(
                    "edge refers to a node that does not exist",
                    unit_id=self.unit_id,
                    source=edge.source,
                    target=edge.target,
                    n_nodes=n_nodes,
                )
