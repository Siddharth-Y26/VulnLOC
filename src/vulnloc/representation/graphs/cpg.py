"""Code property graph — the union of AST, CFG and DFG (ADR-002 §D2.2, fourth).

There is no extraction here, and that is the point. ADR-002 fixes the CPG as "a typed
union of the above; no new extraction, only edge-type composition", which is only sound
because every builder numbers its nodes through
:func:`~vulnloc.representation.graphs.ast.transcribe`. Node *n* is the same node, on the
same line, in all four graph types, so a model can be given syntax, control flow and
data flow at once without any risk that the three disagree about where line *n* is.

Ablation A3 (AST · CFG · DFG · AST+CFG · CPG) is therefore a choice of which edge types
to keep, made in config, over one construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulnloc.representation.graphs.ast import transcribe
from vulnloc.representation.graphs.base import EDGE_TYPES, CodeGraph, EdgeType, GraphEdge
from vulnloc.representation.graphs.cfg import build_cfg_edges
from vulnloc.representation.graphs.dfg import build_dfg_edges
from vulnloc.representation.registry import register_graph
from vulnloc.utils.exceptions import GraphBuildError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vulnloc.representation.parser import ParseResult

__all__ = ["CpgGraphBuilder", "build_cpg_graph"]

GRAPH_TYPE = "cpg"

DEFAULT_EDGE_TYPES: tuple[EdgeType, ...] = ("ast", "cfg", "dfg")
"""What a CPG contains. ``call`` is reserved for the interprocedural work that does not
exist: Big-Vul units are single functions, so there is nothing to call into."""


def build_cpg_graph(
    result: ParseResult,
    edge_types: Iterable[EdgeType] = DEFAULT_EDGE_TYPES,
) -> CodeGraph:
    """Build the code property graph for a parsed unit.

    Args:
        result: A parsed unit.
        edge_types: Which relations to include. Restricting this is how ablation A3
            selects AST+CFG, or any other combination, without a second construction.

    Returns:
        The CPG over the AST's node set.

    Raises:
        GraphBuildError: If an unknown edge type is requested, or if the resulting
            graph violates a structural invariant.
    """
    wanted = tuple(edge_types)
    unknown = [name for name in wanted if name not in EDGE_TYPES]
    if unknown:
        raise GraphBuildError(
            "unknown edge type requested for a CPG",
            unit_id=result.unit_id,
            unknown=unknown,
            known=list(EDGE_TYPES),
        )

    parts = transcribe(result)
    edges: list[GraphEdge] = []

    # The DFG needs the CFG, so build control flow whenever either is asked for.
    cfg_edges: tuple[GraphEdge, ...] = ()
    if "cfg" in wanted or "dfg" in wanted:
        cfg_edges = build_cfg_edges(result, parts.node_ids)

    if "ast" in wanted:
        edges.extend(parts.ast_edges)
    if "cfg" in wanted:
        edges.extend(cfg_edges)
    if "dfg" in wanted:
        edges.extend(build_dfg_edges(result, parts.node_ids, cfg_edges))

    graph = CodeGraph(
        unit_id=result.unit_id,
        graph_type=GRAPH_TYPE,
        nodes=parts.nodes,
        edges=tuple(edges),
        n_lines=parts.n_lines,
    )
    graph.validate()
    return graph


@register_graph(GRAPH_TYPE)
class CpgGraphBuilder:
    """Registry entry for the code property graph."""

    graph_type = GRAPH_TYPE

    def build(self, result: ParseResult) -> CodeGraph:
        """Build the CPG for one parsed unit.

        Args:
            result: A parsed unit.

        Returns:
            The code property graph.
        """
        return build_cpg_graph(result)
