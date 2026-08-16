"""AST / CFG / DFG / CPG builders, each registered with ``@register_graph``.

Importing a builder module is what puts it in the registry, so every implementation is
imported here explicitly. A builder that is merely present on disk is not selectable
from config, and the failure looks like a missing key rather than a missing import
(``Architecture.md`` §9).

All four share one node set, produced by :func:`~vulnloc.representation.graphs.ast.transcribe`
and differing only in which edge types they add — which is what makes the CPG a union
rather than a fifth extraction (ADR-002 §D2.2).
"""

from vulnloc.representation.graphs.ast import (
    AstGraphBuilder,
    Transcription,
    build_ast_graph,
    transcribe,
)
from vulnloc.representation.graphs.base import (
    EDGE_TYPES,
    CodeGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
)
from vulnloc.representation.graphs.cfg import CfgGraphBuilder, build_cfg_graph, terminal_nodes
from vulnloc.representation.graphs.cpg import CpgGraphBuilder, build_cpg_graph
from vulnloc.representation.graphs.dfg import DfgGraphBuilder, build_dfg_graph

__all__ = [
    "EDGE_TYPES",
    "AstGraphBuilder",
    "CfgGraphBuilder",
    "CodeGraph",
    "CpgGraphBuilder",
    "DfgGraphBuilder",
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "Transcription",
    "build_ast_graph",
    "build_cfg_graph",
    "build_cpg_graph",
    "build_dfg_graph",
    "terminal_nodes",
    "transcribe",
]
