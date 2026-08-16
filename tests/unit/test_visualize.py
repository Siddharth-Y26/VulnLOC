"""Guards the graph renderers.

These are debugging tools, so the tests check the one thing a debugging tool must not
get wrong: the line numbers and source text it shows have to be the graph's own, or it
would hide exactly the bug it exists to reveal.
"""

from __future__ import annotations

import pytest

from vulnloc.representation import CodeUnit, TreeSitterParser, build_ast_graph
from vulnloc.representation.visualize import render_tree, to_dot

SOURCE = "int f(int a)\n{\n    return a + 1;\n}\n"


@pytest.fixture
def unit():
    return CodeUnit.from_source("u1", SOURCE, language="C", vulnerable_lines=frozenset({3}))


@pytest.fixture
def graph(unit):
    parser = TreeSitterParser()
    return build_ast_graph(parser.parse(unit.source, language="C", unit_id=unit.unit_id))


def line_of(row):
    """Read the line number a rendered row claims, past its 2-char marker."""
    return int(row[2:].split()[0])


def test_render_tree_shows_the_graphs_own_lines(graph, unit):
    text = render_tree(graph, unit=unit)

    assert "u1" in text
    # "return" lives on line 3 of the source, so every row naming it must say 3.
    return_rows = [row for row in text.splitlines() if "'return'" in row]
    assert return_rows
    assert all(line_of(row) == 3 for row in return_rows)


def test_render_tree_shows_the_source_line_beside_the_node(graph, unit):
    # The whole point of the renderer: a node claiming line 3 is shown next to what
    # line 3 actually says, so a misalignment is visible rather than merely asserted.
    rows = [row for row in render_tree(graph, unit=unit).splitlines() if "|" in row]

    assert rows
    for row in rows:
        assert row.split("|", 1)[1].strip() in unit.lines[line_of(row) - 1]


def test_render_tree_marks_highlighted_lines(graph, unit):
    text = render_tree(graph, unit=unit, highlight=unit.vulnerable_lines)

    marked = [row for row in text.splitlines() if row.startswith(">>")]
    assert marked
    assert all(line_of(row) == 3 for row in marked)


def test_render_tree_truncates(graph, unit):
    text = render_tree(graph, unit=unit, max_nodes=3)

    assert "more nodes" in text


def test_render_tree_without_a_unit_omits_source_text(graph):
    text = render_tree(graph)

    assert "|" not in text


def test_to_dot_clusters_by_line(graph, unit):
    dot = to_dot(graph, highlight=unit.vulnerable_lines)

    assert dot.startswith("digraph ast {")
    assert dot.rstrip().endswith("}")
    for line_no in sorted(graph.lines_covered):
        assert f"cluster_line_{line_no}" in dot
    assert "fillcolor" in dot, "the flagged line should be filled"
    assert dot.count("->") == len(graph.edges)


def test_to_dot_escapes_quotes_in_source_text():
    parser = TreeSitterParser()
    source = 'const char *s = "he said \\"hi\\"";\n'
    graph = build_ast_graph(parser.parse(source, language="C", unit_id="q"))

    dot = to_dot(graph)

    # An unescaped double quote would end the DOT string early and Graphviz would
    # refuse the file. Escaped ones are fine, so drop those before looking.
    rows = [row for row in dot.splitlines() if "label=" in row]
    assert any("escape_sequence" in row for row in rows), "fixture must exercise quotes"
    for row in rows:
        body = row.split('label="', 1)[1].rsplit('"', 1)[0]
        assert '"' not in body.replace("\\\\", "").replace('\\"', "")
