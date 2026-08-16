"""L1 — Representation: parsing, graph construction, tokenization, line maps.

Every artefact produced here carries a line index; anything that breaks the line
mapping is a bug, not a trade-off (``docs/Architecture.md`` §1.1).
Language and graph representation are fixed by ADR-002: C/C++, built incrementally
AST → CFG → DFG → CPG, with in-house passes rather than an external CPG generator.
"""

from vulnloc.representation.line_map import LINE_SEPARATOR, LineMap, split_lines
from vulnloc.representation.parser import (
    LANGUAGE_ALIASES,
    ParseResult,
    TreeSitterParser,
    iter_nodes,
    node_to_line,
    verify_line_alignment,
)
from vulnloc.representation.registry import GRAPH_REGISTRY, register_graph

__all__ = [
    "GRAPH_REGISTRY",
    "LANGUAGE_ALIASES",
    "LINE_SEPARATOR",
    "LineMap",
    "ParseResult",
    "TreeSitterParser",
    "iter_nodes",
    "node_to_line",
    "register_graph",
    "split_lines",
    "verify_line_alignment",
]
