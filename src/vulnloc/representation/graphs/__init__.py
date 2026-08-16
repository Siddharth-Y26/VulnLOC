"""AST / CFG / DFG / CPG builders, each registered with ``@register_graph``.

Importing a builder module is what puts it in the registry, so every implementation is
imported here explicitly. A builder that is merely present on disk is not selectable
from config, and the failure looks like a missing key rather than a missing import
(``Architecture.md`` §9).

CFG, DFG and CPG are not built yet — ADR-002 §D2.2 fixes the order AST → CFG → DFG →
CPG, and Phase 4 is unblocked by the AST alone.
"""

from vulnloc.representation.graphs.ast import AstGraphBuilder, build_ast_graph
from vulnloc.representation.graphs.base import (
    EDGE_TYPES,
    CodeGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
)

__all__ = [
    "EDGE_TYPES",
    "AstGraphBuilder",
    "CodeGraph",
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "build_ast_graph",
]
