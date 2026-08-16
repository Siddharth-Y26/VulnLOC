"""CodeGraph → NetworkX and → PyTorch Geometric (``Architecture.md`` §5).

The structural branch consumes ``Data`` objects; the debugging and paper figures
consume NetworkX. Both conversions live here so the tensor layout is defined exactly
once, and so nothing downstream has to guess which column of ``edge_attr`` holds the
edge type.

The field that matters is **``line``**. It is the structural half of the line-alignment
bridge (``Architecture.md`` §4.3): ``[N, D]`` node states get scatter-mean'd over it to
reach ``[B, L, D_str]``. It is carried as a tensor on the graph, 1-indexed like
everywhere else, so batching cannot silently lose it.

Node *features* are deliberately not built here. ADR-002 §D2.5 fixes them as a node-type
embedding plus a subtoken embedding, and both are model parameters — they belong to
Phase 4, not to the representation. What this module emits is the vocabulary index the
embedding will look up.

``torch``, ``torch_geometric`` and ``networkx`` are imported inside the functions, and
this module is not re-exported from the package ``__init__``. Importing
``vulnloc.representation`` therefore does not drag in torch — which keeps parsing, line
mapping and the corpus report usable in a plain CPU process, and keeps the test suite
from paying a multi-second import for tests that never touch a tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vulnloc.representation.graphs.base import EDGE_TYPES, CodeGraph
from vulnloc.utils.exceptions import GraphBuildError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["EDGE_TYPE_TO_INDEX", "build_node_type_vocab", "to_networkx", "to_pyg"]

EDGE_TYPE_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(EDGE_TYPES)}
"""Edge type to the integer stored in ``edge_type``. Fixed order, so a model trained
against one release reads the same channel in the next."""

_UNKNOWN_NODE_TYPE = 0
"""Vocabulary slot 0 is reserved for grammar symbols unseen at vocabulary-build time."""


def build_node_type_vocab(graphs: list[CodeGraph]) -> dict[str, int]:
    """Build a node-type vocabulary from a corpus sample.

    Sorted rather than insertion-ordered, so the same sample yields the same indices
    regardless of iteration order — the same reasoning as ``utils/hashing.py``.

    Args:
        graphs: Graphs to collect grammar symbols from.

    Returns:
        A mapping from node type to index, with index 0 reserved for unknown types.
    """
    types = sorted({node.node_type for graph in graphs for node in graph.nodes})
    return {name: index for index, name in enumerate(types, start=1)}


def to_networkx(graph: CodeGraph) -> Any:
    """Convert to a directed NetworkX graph, attributes intact.

    Used for debugging and for the line-annotated figures Phase 2 owes the paper; not
    on the training path.

    Args:
        graph: The graph to convert.

    Returns:
        A ``networkx.DiGraph`` whose nodes carry ``node_type``, ``line_no``,
        ``is_named`` and ``text``, and whose edges carry ``edge_type``.
    """
    import networkx as nx

    converted = nx.DiGraph(unit_id=graph.unit_id, graph_type=graph.graph_type)
    for node in graph.nodes:
        converted.add_node(
            node.node_id,
            node_type=node.node_type,
            line_no=node.line_no,
            is_named=node.is_named,
            text=node.text,
        )
    for edge in graph.edges:
        converted.add_edge(edge.source, edge.target, edge_type=edge.edge_type)
    return converted


def to_pyg(graph: CodeGraph, node_type_vocab: Mapping[str, int] | None = None) -> Any:
    """Convert to a PyTorch Geometric ``Data`` object.

    Args:
        graph: The graph to convert.
        node_type_vocab: Node type to index. Types absent from it map to 0, so an
            unseen grammar symbol degrades a single node rather than failing a batch.
            When omitted, a vocabulary is built from this graph alone — useful in tests,
            wrong for training, since indices would then differ per sample.

    Returns:
        A ``Data`` with ``node_type`` ``[N]``, ``line`` ``[N]`` (1-indexed),
        ``edge_index`` ``[2, E]``, ``edge_type`` ``[E]``, plus ``num_lines`` and
        ``unit_id``.

    Raises:
        GraphBuildError: If the graph has no nodes; an empty graph cannot be scattered
            onto a line axis and would surface much later as a shape error.
    """
    import torch
    from torch_geometric.data import Data

    if not graph.nodes:
        raise GraphBuildError(
            "cannot convert an empty graph",
            unit_id=graph.unit_id,
            graph_type=graph.graph_type,
        )

    vocab = node_type_vocab if node_type_vocab is not None else build_node_type_vocab([graph])

    node_type = torch.tensor(
        [vocab.get(node.node_type, _UNKNOWN_NODE_TYPE) for node in graph.nodes],
        dtype=torch.long,
    )
    line = torch.tensor([node.line_no for node in graph.nodes], dtype=torch.long)

    if graph.edges:
        edge_index = torch.tensor(
            [[edge.source for edge in graph.edges], [edge.target for edge in graph.edges]],
            dtype=torch.long,
        )
        edge_type = torch.tensor(
            [EDGE_TYPE_TO_INDEX[edge.edge_type] for edge in graph.edges],
            dtype=torch.long,
        )
    else:
        # A single-node function is legal; an empty [2, 0] keeps the shape contract.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return Data(
        node_type=node_type,
        line=line,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=len(graph.nodes),
        num_lines=graph.n_lines,
        unit_id=graph.unit_id,
    )
