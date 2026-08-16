"""Guards the code property graph.

The CPG carries no analysis of its own, so what is tested is the property that makes it
legitimate: it is a *union*. If AST, CFG and DFG ever stopped sharing a node set, the
union would silently misattribute edges to the wrong nodes — and therefore to the wrong
lines — which is the one failure mode this project cannot absorb.
"""

from __future__ import annotations

import pytest

from vulnloc.representation import (
    GRAPH_REGISTRY,
    CpgGraphBuilder,
    TreeSitterParser,
    build_ast_graph,
    build_cfg_graph,
    build_cpg_graph,
    build_dfg_graph,
)
from vulnloc.utils.exceptions import GraphBuildError

SOURCE = (
    "int sum(const int *values, int n)\n"
    "{\n"
    "    int total = 0;\n"
    "    for (int i = 0; i < n; i++) {\n"
    "        total += values[i];\n"
    "    }\n"
    "    return total;\n"
    "}\n"
)


@pytest.fixture
def parser():
    return TreeSitterParser()


@pytest.fixture
def result(parser):
    return parser.parse(SOURCE, language="C", unit_id="u1")


def test_every_builder_is_registered():
    assert GRAPH_REGISTRY.keys() == ["ast", "cfg", "cpg", "dfg"]
    assert GRAPH_REGISTRY.get("cpg") is CpgGraphBuilder


def test_cpg_is_exactly_the_union(result):
    ast = build_ast_graph(result)
    cfg = build_cfg_graph(result)
    dfg = build_dfg_graph(result)

    cpg = build_cpg_graph(result)

    assert set(cpg.edges) == set(ast.edges) | set(cfg.edges) | set(dfg.edges)
    assert cpg.edge_type_counts() == {
        "ast": len(ast.edges),
        "cfg": len(cfg.edges),
        "dfg": len(dfg.edges),
    }


def test_all_graph_types_share_one_node_set(result):
    graphs = [
        build_ast_graph(result),
        build_cfg_graph(result),
        build_dfg_graph(result),
        build_cpg_graph(result),
    ]

    # Identical nodes, therefore identical lines — the property the union rests on.
    assert all(graph.nodes == graphs[0].nodes for graph in graphs)
    assert all(graph.node_to_line == graphs[0].node_to_line for graph in graphs)


def test_edge_types_can_be_restricted_for_ablation(result):
    # Ablation A3 is a config choice over one construction, not a second extraction.
    combined = build_cpg_graph(result, edge_types=("ast", "cfg"))

    assert set(combined.edge_type_counts()) == {"ast", "cfg"}
    assert combined.nodes == build_cpg_graph(result).nodes


def test_dfg_only_still_builds_control_flow_internally(result):
    # The data-flow analysis walks the CFG, so asking for dfg alone must still work.
    dfg_only = build_cpg_graph(result, edge_types=("dfg",))

    assert set(dfg_only.edge_type_counts()) == {"dfg"}
    assert dfg_only.edges == build_dfg_graph(result).edges


def test_unknown_edge_type_is_refused(result):
    with pytest.raises(GraphBuildError) as excinfo:
        build_cpg_graph(result, edge_types=("ast", "telepathy"))

    assert "telepathy" in str(excinfo.value)


def test_call_edges_are_not_emitted(result):
    # `call` is reserved for interprocedural work that does not exist; Big-Vul units are
    # single functions, so there is nothing to call into.
    assert "call" not in build_cpg_graph(result).edge_type_counts()


def test_every_node_still_carries_a_line(result):
    cpg = build_cpg_graph(result)

    cpg.validate()
    assert all(1 <= node.line_no <= cpg.n_lines for node in cpg.nodes)


def test_builder_class_matches_the_function(result):
    assert CpgGraphBuilder().build(result).stats() == build_cpg_graph(result).stats()


def test_cpg_over_a_damaged_parse_is_still_valid(parser):
    damaged = parser.parse("int f(void) {\n  if (x) {\n    y = 1;\n", language="C", unit_id="u1")

    build_cpg_graph(damaged).validate()
