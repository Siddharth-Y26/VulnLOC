"""Guards the control-flow builder.

CFG construction is the part of Phase 2 that `Phases.md` warns is real compiler work,
and its failures are quiet: a missing back edge or a `break` bound to the wrong loop
still yields a plausible graph. So these tests assert the *shape* of control flow for
each construct — which statement reaches which — rather than counting edges.

Edges are asserted as (line, line) pairs, because the line is what the rest of the
project consumes and it is readable in a failure message.
"""

from __future__ import annotations

import pytest

from vulnloc.representation import TreeSitterParser, build_cfg_graph, terminal_nodes, transcribe
from vulnloc.representation.graphs.cfg import MAX_NESTING_DEPTH, build_cfg_edges
from vulnloc.utils.exceptions import GraphBuildError


@pytest.fixture
def parser():
    return TreeSitterParser()


def flow(parser, source, unit_id="u1"):
    """Return the CFG as a set of (from_line, to_line) pairs, plus the graph."""
    graph = build_cfg_graph(parser.parse(source, language="C", unit_id=unit_id))
    by_id = {node.node_id: node for node in graph.nodes}
    pairs = {(by_id[e.source].line_no, by_id[e.target].line_no) for e in graph.edges}
    return pairs, graph


def test_statements_chain_in_order(parser):
    pairs, _ = flow(parser, "void f(void)\n{\n  a();\n  b();\n  c();\n}\n")

    assert (1, 3) in pairs, "the function entry flows to its first statement"
    assert (3, 4) in pairs
    assert (4, 5) in pairs


def test_if_else_branches_and_joins(parser):
    source = "void f(int c)\n{\n  if (c) {\n    a();\n  } else {\n    b();\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 4) in pairs, "true branch"
    assert (3, 6) in pairs, "false branch"
    assert (4, 8) in pairs and (6, 8) in pairs, "both branches join"
    assert (3, 8) not in pairs, "with an else, the test cannot fall straight through"


def test_if_without_else_falls_through(parser):
    source = "void f(int c)\n{\n  if (c) {\n    a();\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 4) in pairs
    assert (3, 6) in pairs, "the false branch skips the body"
    assert (4, 6) in pairs


def test_else_if_chains(parser):
    source = (
        "void f(int c)\n{\n  if (c) {\n    a();\n  } else if (c > 1) {\n    b();\n  }\n  d();\n}\n"
    )

    pairs, _ = flow(parser, source)

    assert (3, 5) in pairs, "the first test falls to the second"
    assert (5, 6) in pairs
    assert (5, 8) in pairs


def test_while_has_a_back_edge_and_an_exit(parser):
    source = "void f(int n)\n{\n  while (n) {\n    a();\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 4) in pairs, "test enters the body"
    assert (4, 3) in pairs, "body returns to the test"
    assert (3, 6) in pairs, "test exits the loop"


def test_for_has_a_back_edge_and_an_exit(parser):
    source = "void f(int n)\n{\n  for (int i = 0; i < n; i++) {\n    a();\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 4) in pairs
    assert (4, 3) in pairs
    assert (3, 6) in pairs


def test_do_while_runs_the_body_before_the_test(parser):
    source = "void f(int n)\n{\n  do {\n    a();\n  } while (n);\n  d();\n}\n"

    pairs, graph = flow(parser, source)

    assert (2, 4) not in pairs  # there is no line 2 statement; guard against drift
    assert (4, 3) in pairs, "body flows to the test"
    assert (3, 4) in pairs, "the test loops back into the body"
    assert (3, 6) in pairs, "the test exits"
    # The loop's entry is the body, not the test: control reaches line 4 first.
    entries = {e.target for e in graph.edges}
    by_id = {n.node_id: n for n in graph.nodes}
    assert 4 in {by_id[t].line_no for t in entries}


def test_break_leaves_the_innermost_loop(parser):
    source = (
        "void f(int n)\n{\n  while (n) {\n    for (;;) {\n      break;\n    }\n"
        "    m();\n  }\n  d();\n}\n"
    )

    pairs, _ = flow(parser, source)

    assert (5, 7) in pairs, "break exits the for, landing after it"
    assert (5, 9) not in pairs, "break must not escape the outer while"


def test_continue_returns_to_the_loop_header(parser):
    source = "void f(int n)\n{\n  for (int i = 0; i < n; i++) {\n    continue;\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (4, 3) in pairs


def test_return_has_no_successor(parser):
    source = "int f(int c)\n{\n  if (c) {\n    return 1;\n  }\n  return 2;\n}\n"

    pairs, graph = flow(parser, source)
    by_id = {node.node_id: node for node in graph.nodes}

    assert not any(source_line == 4 for source_line, _ in pairs), "return flows nowhere"
    assert {by_id[n].line_no for n in terminal_nodes(graph)} == {4, 6}


def test_goto_reaches_its_label(parser):
    source = "void f(int c)\n{\n  if (c) goto out;\n  a();\nout:\n  b();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 5) in pairs, "goto jumps to the labelled statement"
    assert (5, 6) in pairs, "the label falls through to its statement"


def test_goto_to_a_missing_label_is_left_dangling(parser):
    # Big-Vul functions are fragments; the target may genuinely have been cut away.
    # Guessing a destination would invent control flow that does not exist.
    pairs, graph = flow(parser, "void f(void)\n{\n  goto nowhere;\n  a();\n}\n")

    assert not any(source_line == 3 for source_line, _ in pairs)
    graph.validate()


def test_switch_reaches_every_case_and_break_exits(parser):
    source = (
        "void f(int n)\n{\n  switch (n) {\n  case 1:\n    a();\n    break;\n"
        "  default:\n    b();\n  }\n  d();\n}\n"
    )

    pairs, _ = flow(parser, source)

    assert (3, 4) in pairs and (3, 7) in pairs, "the switch reaches both labels"
    assert (6, 10) in pairs, "break leaves the switch"
    assert (3, 10) not in pairs, "a switch with a default cannot fall past every case"


def test_switch_without_default_can_fall_past_every_case(parser):
    source = "void f(int n)\n{\n  switch (n) {\n  case 1:\n    a();\n    break;\n  }\n  d();\n}\n"

    pairs, _ = flow(parser, source)

    assert (3, 8) in pairs


def test_switch_cases_fall_through_without_break(parser):
    source = (
        "void f(int n)\n{\n  switch (n) {\n  case 1:\n    a();\n  case 2:\n"
        "    b();\n  }\n  d();\n}\n"
    )

    pairs, _ = flow(parser, source)

    assert (5, 6) in pairs, "case 1 falls into case 2"


def test_node_set_is_the_asts(parser):
    from vulnloc.representation import build_ast_graph

    result = parser.parse("void f(int n)\n{\n  if (n) a();\n}\n", language="C", unit_id="u1")

    assert build_cfg_graph(result).nodes == build_ast_graph(result).nodes


def test_only_cfg_edges_are_emitted(parser):
    _, graph = flow(parser, "void f(int n)\n{\n  if (n) a();\n}\n")

    assert set(graph.edge_type_counts()) == {"cfg"}


def test_edges_are_deduplicated_and_ordered(parser):
    # An `if` whose branches both fall through can reach the same join twice.
    result = parser.parse(
        "void f(int c)\n{\n  if (c) {\n    a();\n  }\n  d();\n}\n", language="C", unit_id="u1"
    )
    parts_edges = build_cfg_graph(result).edges

    pairs = [(e.source, e.target) for e in parts_edges]
    assert len(pairs) == len(set(pairs))
    assert pairs == sorted(pairs)


def test_damaged_parse_still_produces_a_valid_graph(parser):
    result = parser.parse("void f(int n) {\n  if (n) {\n    a();\n", language="C", unit_id="u1")

    graph = build_cfg_graph(result)

    graph.validate()


def test_multiple_functions_do_not_share_labels(parser):
    source = "void f(void)\n{\nout:\n  a();\n}\nvoid g(void)\n{\n  goto out;\n}\n"

    pairs, _ = flow(parser, source)

    assert (8, 3) not in pairs, "a goto must not reach a label in a different function"


def test_deep_nesting_raises_as_ourselves_not_as_a_recursion_error(parser):
    # The deepest statement nesting measured over 20,031 real functions is 40, so this
    # input is pathological by construction. What matters is that it fails as a
    # GraphBuildError naming the unit, never as a RecursionError from inside the walk.
    depth = MAX_NESTING_DEPTH + 50
    source = "void f(int n)\n{\n" + "if (n) {\n" * depth + "a();\n" + "}\n" * depth + "}\n"
    result = parser.parse(source, language="C", unit_id="deep")
    index = transcribe(result).node_ids

    with pytest.raises(GraphBuildError) as excinfo:
        build_cfg_edges(result, index)

    assert "nesting" in str(excinfo.value) or "stack" in str(excinfo.value)


def test_realistic_nesting_is_well_inside_the_guard(parser):
    # 40 is the measured worst case in the corpus; it must build without complaint.
    depth = 40
    source = "void f(int n)\n{\n" + "if (n) {\n" * depth + "a();\n" + "}\n" * depth + "}\n"

    graph = build_cfg_graph(parser.parse(source, language="C", unit_id="deep40"))

    graph.validate()
    assert graph.edges
