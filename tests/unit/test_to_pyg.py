"""Guards the NetworkX and PyTorch Geometric conversions.

The field under test is ``line``. It is the structural half of the line-alignment
bridge, and a conversion that drops it, reorders it, or quietly makes it 0-indexed
would surface far downstream as a fusion shape error with no route back to the unit
that caused it.
"""

from __future__ import annotations

import pytest

from vulnloc.representation import CodeGraph, GraphNode, TreeSitterParser, build_ast_graph
from vulnloc.representation.to_pyg import (
    EDGE_TYPE_TO_INDEX,
    build_node_type_vocab,
    to_networkx,
    to_pyg,
)
from vulnloc.utils.exceptions import GraphBuildError

SIMPLE_C = "int f(int a)\n{\n    return a + 1;\n}\n"


@pytest.fixture
def graph():
    parser = TreeSitterParser()
    return build_ast_graph(parser.parse(SIMPLE_C, language="C", unit_id="u1"))


def test_edge_type_indices_are_fixed():
    # A model trained on one release must read the same channel in the next.
    assert EDGE_TYPE_TO_INDEX == {"ast": 0, "cfg": 1, "dfg": 2, "call": 3}


def test_vocab_is_order_independent(graph):
    types = sorted({node.node_type for node in graph.nodes})
    vocab = build_node_type_vocab([graph])

    assert list(vocab) == types
    assert 0 not in vocab.values(), "index 0 is reserved for unknown types"


def test_to_pyg_carries_the_line_axis(graph):
    data = to_pyg(graph)

    assert data.num_nodes == len(graph.nodes)
    assert data.line.tolist() == [node.line_no for node in graph.nodes]
    assert int(data.line.min()) >= 1, "lines are 1-indexed"
    assert int(data.num_lines) == graph.n_lines


def test_to_pyg_edges_match_the_graph(graph):
    data = to_pyg(graph)

    assert data.edge_index.shape == (2, len(graph.edges))
    assert data.edge_index[0].tolist() == [edge.source for edge in graph.edges]
    assert data.edge_index[1].tolist() == [edge.target for edge in graph.edges]
    assert set(data.edge_type.tolist()) == {EDGE_TYPE_TO_INDEX["ast"]}


def test_unknown_node_types_degrade_one_node(graph):
    # An unseen grammar symbol must cost one node, not a whole batch.
    data = to_pyg(graph, node_type_vocab={"translation_unit": 1})

    assert int(data.node_type[0]) == 1
    assert int(data.node_type[1:].max()) == 0


def test_edgeless_graph_keeps_the_shape_contract():
    lone = CodeGraph(
        unit_id="u1",
        graph_type="ast",
        nodes=(GraphNode(0, "translation_unit", 1, 0, 0, True),),
        n_lines=1,
    )

    data = to_pyg(lone)

    assert data.edge_index.shape == (2, 0)
    assert data.edge_type.shape == (0,)


def test_empty_graph_is_refused():
    empty = CodeGraph(unit_id="u1", graph_type="ast", nodes=(), n_lines=1)

    with pytest.raises(GraphBuildError):
        to_pyg(empty)


def test_to_networkx_preserves_attributes(graph):
    converted = to_networkx(graph)

    assert converted.number_of_nodes() == len(graph.nodes)
    assert converted.number_of_edges() == len(graph.edges)
    assert converted.graph["unit_id"] == "u1"
    assert converted.nodes[0]["line_no"] == graph.nodes[0].line_no
    assert all(data["edge_type"] == "ast" for _, _, data in converted.edges(data=True))
