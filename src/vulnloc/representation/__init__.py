"""L1 — Representation: parsing, graph construction, tokenization, line maps.

Every artefact produced here carries a line index; anything that breaks the line
mapping is a bug, not a trade-off (``docs/Architecture.md`` §1.1).
Blocked on ADR-002 (language Q2, graph type Q3, CPG tooling Q6) until Phase 2.
"""

from vulnloc.representation.registry import GRAPH_REGISTRY, register_graph

__all__ = ["GRAPH_REGISTRY", "register_graph"]
