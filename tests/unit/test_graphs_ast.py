"""Guards the AST builder and the graph container.

The golden files are five hand-written functions covering the constructs Phase 2's exit
criteria name — loops, branches, pointers and aliasing — plus a goto/cleanup ladder and
a preprocessor case, because macro continuations and ``#ifdef`` are where the line map
is most likely to slip. They are checked as whole records rather than as node counts:
if a grammar upgrade moves a node to a different line, that must be a visible diff and
a deliberate re-approval, not a silent shift in what the model is trained on.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vulnloc.representation import (
    GRAPH_REGISTRY,
    AstGraphBuilder,
    CodeGraph,
    GraphEdge,
    GraphNode,
    TreeSitterParser,
    build_ast_graph,
    split_lines,
)
from vulnloc.utils.exceptions import GraphBuildError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "graphs"
GOLDEN = json.loads((FIXTURES / "golden_ast.json").read_text(encoding="utf-8"))

SIMPLE_C = "int f(int a)\n{\n    return a + 1;\n}\n"


@pytest.fixture
def parser():
    return TreeSitterParser()


def build(parser, source, unit_id="u1", language="C"):
    return build_ast_graph(parser.parse(source, language=language, unit_id=unit_id))


# --- the registry ------------------------------------------------------------------


def test_ast_builder_is_registered():
    # A builder that is only on disk is not selectable from config.
    assert GRAPH_REGISTRY.get("ast") is AstGraphBuilder


def test_builder_class_matches_the_function(parser):
    result = parser.parse(SIMPLE_C, language="C", unit_id="u1")

    assert AstGraphBuilder().build(result).stats() == build_ast_graph(result).stats()


# --- golden files ------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_ast(parser, name):
    source = (FIXTURES / name).read_text(encoding="utf-8")
    expected = GOLDEN[name]

    graph = build(parser, source, unit_id=name)
    per_line = Counter(node.line_no for node in graph.nodes)

    assert len(graph.nodes) == expected["n_nodes"]
    assert len(graph.edges) == expected["n_edges"]
    assert graph.n_lines == expected["n_lines"]
    assert sorted(graph.lines_covered) == expected["lines_covered"]
    assert {str(k): per_line[k] for k in sorted(per_line)} == expected["nodes_per_line"]
    assert sum(1 for node in graph.nodes if node.is_named) == expected["named_nodes"]
    assert graph.nodes[0].node_type == expected["root_type"]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_fixtures_parse_without_recovery(parser, name):
    # If a fixture starts erroring, the golden record above is describing damage rather
    # than a grammar.
    source = (FIXTURES / name).read_text(encoding="utf-8")

    result = parser.parse(source, language="C", unit_id=name)

    assert result.is_clean, f"{name}: {result.error_nodes} errors, {result.missing_nodes} missing"


def test_macro_continuation_keeps_both_lines(parser):
    # The #define spans lines 2-3 via a backslash continuation; both must be covered,
    # and line 4 (blank) must not be.
    source = (FIXTURES / "preprocessor.c").read_text(encoding="utf-8")

    covered = build(parser, source).lines_covered

    assert {2, 3} <= covered
    assert 4 not in covered


# --- structure ---------------------------------------------------------------------


def test_tree_shape(parser):
    graph = build(parser, SIMPLE_C)

    # A transcribed tree: one edge per node except the root, all of type "ast".
    assert len(graph.edges) == len(graph.nodes) - 1
    assert graph.edge_type_counts() == {"ast": len(graph.edges)}
    assert graph.graph_type == "ast"


def test_node_ids_are_dense_and_preorder(parser):
    graph = build(parser, SIMPLE_C)

    assert [node.node_id for node in graph.nodes] == list(range(len(graph.nodes)))
    # Pre-order: a parent is always numbered before its children.
    assert all(edge.source < edge.target for edge in graph.edges)


def test_anonymous_nodes_are_kept(parser):
    # Deliberate: "+", "[" and "==" are the tokens memory-safety flaws turn on.
    graph = build(parser, SIMPLE_C)

    assert any(not node.is_named for node in graph.nodes)
    assert any(node.text == "+" for node in graph.nodes)


def test_text_is_stored_on_leaves_only(parser):
    graph = build(parser, SIMPLE_C)
    children = {edge.source for edge in graph.edges}

    for node in graph.nodes:
        if node.node_id in children:
            assert node.text == ""
        else:
            assert node.text != ""


def test_node_to_line_covers_every_node(parser):
    graph = build(parser, SIMPLE_C)

    assert set(graph.node_to_line) == {node.node_id for node in graph.nodes}


def test_deeply_nested_input_does_not_recurse(parser):
    # An iterative walk is not a style choice: recursion here would turn a property of
    # the corpus into a RecursionError.
    source = "int f(void){ return " + " + ".join(["1"] * 3000) + "; }\n"

    graph = build(parser, source)

    assert len(graph.nodes) > 6000


@pytest.mark.parametrize(
    ("name", "source"),
    [("empty", ""), ("only_newline", "\n"), ("whitespace", "   \n\t\n")],
)
def test_degenerate_sources_still_produce_a_valid_graph(parser, name, source):
    # Tree-sitter still emits a root for these, and it sits at end-of-source — which
    # means the last line. A graph with no nodes would break to_pyg downstream.
    graph = build(parser, source, unit_id=name)

    graph.validate()
    assert len(graph.nodes) == 1
    assert graph.nodes[0].line_no == graph.n_lines


def test_unparsable_input_still_yields_a_graph(parser):
    # Fragments are the norm in Big-Vul; a graph over a damaged tree is still a graph.
    graph = build(parser, "int broken(void) { return")

    assert graph.nodes
    graph.validate()


# --- validation --------------------------------------------------------------------


def test_validate_rejects_non_positional_ids():
    graph = CodeGraph(
        unit_id="u1",
        graph_type="ast",
        nodes=(GraphNode(5, "x", 1, 0, 1, True),),
        n_lines=1,
    )

    with pytest.raises(GraphBuildError):
        graph.validate()


def test_validate_rejects_a_line_outside_the_source():
    graph = CodeGraph(
        unit_id="u1",
        graph_type="ast",
        nodes=(GraphNode(0, "x", 9, 0, 1, True),),
        n_lines=3,
    )

    with pytest.raises(GraphBuildError):
        graph.validate()


def test_validate_rejects_a_dangling_edge():
    graph = CodeGraph(
        unit_id="u1",
        graph_type="ast",
        nodes=(GraphNode(0, "x", 1, 0, 1, True),),
        edges=(GraphEdge(0, 7, "ast"),),
        n_lines=1,
    )

    with pytest.raises(GraphBuildError):
        graph.validate()


# --- property tests (Phases.md Phase 2: "every node maps to a valid line") ----------

# Fragments of real C syntax, assembled at random. Pure random text would mostly
# produce ERROR nodes; this keeps the grammar engaged while still being unpredictable.
FRAGMENTS = st.sampled_from(
    [
        "int x = 0;",
        "x += p[i];",
        "if (a > b) { return a; }",
        "while (n--) { *d++ = *s++; }",
        "for (i = 0; i < n; i++) {}",
        "switch (k) { case 1: break; default: break; }",
        "char buf[64];",
        "goto out;",
        "/* c */",
        "#define Q(a) ((a) + 1)",
        "\t",
        "",
        "// é",
        "return;",
        "}",
        "{",
    ]
)


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(FRAGMENTS, min_size=1, max_size=25))
def test_every_node_maps_to_a_line_in_the_source(parser, fragments):
    source = "\n".join(fragments)

    graph = build(parser, source, unit_id="prop")
    n_lines = len(split_lines(source))

    assert graph.nodes
    for node in graph.nodes:
        assert 1 <= node.line_no <= n_lines


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(FRAGMENTS, min_size=1, max_size=25))
def test_single_line_leaf_text_appears_on_its_line(parser, fragments):
    # The independent check: the line number is confirmed against the line's own text,
    # so an off-by-one cannot pass.
    source = "\n".join(fragments)
    lines = split_lines(source)

    graph = build(parser, source, unit_id="prop")

    children = {edge.source for edge in graph.edges}
    for node in graph.nodes:
        if node.node_id in children or not node.text or "\n" in node.text:
            continue
        assert node.text in lines[node.line_no - 1]


@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(FRAGMENTS, min_size=1, max_size=25))
def test_graph_invariants_hold_on_random_input(parser, fragments):
    graph = build(parser, "\n".join(fragments), unit_id="prop")

    graph.validate()
    assert len(graph.edges) == len(graph.nodes) - 1
