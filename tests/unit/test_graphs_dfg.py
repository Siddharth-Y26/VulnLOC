"""Guards the data-flow builder.

Reaching definitions is easy to get *almost* right, and almost-right is invisible: a
missing kill silently adds a stale definition, a missing declarator form silently drops
every pointer parameter. Both happened while this was being written, so the tests below
pin the behaviours that caught them.

The documented approximations — no aliasing, no scoping, weak updates through pointers
and subscripts — are asserted too. They are limits we chose, so they should fail loudly
if someone "fixes" one without updating the module docstring and ADR-002.
"""

from __future__ import annotations

import pytest

import vulnloc.representation.graphs.dfg as dfg
from vulnloc.representation import TreeSitterParser, build_dfg_graph
from vulnloc.utils.exceptions import GraphBuildError


@pytest.fixture
def parser():
    return TreeSitterParser()


def chains(parser, source, unit_id="u1"):
    """Return def-use edges as {(name, def_line, use_line)}."""
    graph = build_dfg_graph(parser.parse(source, language="C", unit_id=unit_id))
    by_id = {node.node_id: node for node in graph.nodes}
    return {
        (by_id[e.source].text, by_id[e.source].line_no, by_id[e.target].line_no)
        for e in graph.edges
    }


def test_definition_reaches_its_use(parser):
    source = "int f(void)\n{\n  int x = 1;\n  return x;\n}\n"

    assert ("x", 3, 4) in chains(parser, source)


def test_redefinition_kills_the_earlier_definition(parser):
    source = "int f(void)\n{\n  int x = 1;\n  x = 2;\n  return x;\n}\n"

    links = chains(parser, source)

    assert ("x", 4, 5) in links
    assert ("x", 3, 5) not in links, "the killed definition must not reach the use"


def test_both_branches_reach_a_later_use(parser):
    source = (
        "int f(int c)\n{\n  int x;\n  if (c) {\n    x = 1;\n  } else {\n"
        "    x = 2;\n  }\n  return x;\n}\n"
    )

    links = chains(parser, source)

    assert ("x", 5, 9) in links
    assert ("x", 7, 9) in links
    assert ("x", 3, 9) not in links, "both paths redefine x, so the declaration is killed"


def test_loop_carried_dependence(parser):
    source = (
        "void f(int n)\n{\n  int s = 0;\n  for (int i = 0; i < n; i++) {\n"
        "    s = s + 1;\n  }\n  use(s);\n}\n"
    )

    links = chains(parser, source)

    assert ("s", 3, 5) in links, "first iteration reads the initial value"
    assert ("s", 5, 5) in links, "later iterations read the previous iteration's value"
    assert ("s", 3, 7) in links, "zero iterations leaves the initial value live"
    assert ("s", 5, 7) in links, "one or more iterations leaves the loop's value live"


def test_parameters_define_their_names(parser):
    source = "int f(int a)\n{\n  return a;\n}\n"

    assert ("a", 1, 3) in chains(parser, source)


def test_pointer_parameters_define_their_names(parser):
    # Regression: without pointer_declarator in the traversal, every `int *p` parameter
    # went undefined and all of its uses had no reaching definition at all.
    source = "int f(int *p)\n{\n  return *p;\n}\n"

    assert ("p", 1, 3) in chains(parser, source)


def test_array_declarations_define_their_names(parser):
    # Regression: `int a[10];` is an array_declarator, not a bare identifier.
    source = "void f(void)\n{\n  int a[10];\n  use(a);\n}\n"

    assert ("a", 3, 4) in chains(parser, source)


def test_uninitialised_declaration_is_still_a_definition(parser):
    # It is the state a use-of-uninitialised flaw reads, so it must be visible.
    source = "int f(void)\n{\n  int x;\n  return x;\n}\n"

    assert ("x", 3, 4) in chains(parser, source)


def test_compound_assignment_reads_what_it_writes(parser):
    source = "int f(void)\n{\n  int x = 1;\n  x += 2;\n  return x;\n}\n"

    links = chains(parser, source)

    assert ("x", 3, 4) in links, "+= reads the previous value"
    assert ("x", 4, 5) in links


def test_increment_reads_what_it_writes(parser):
    source = "int f(void)\n{\n  int i = 0;\n  i++;\n  return i;\n}\n"

    links = chains(parser, source)

    assert ("i", 3, 4) in links
    assert ("i", 4, 5) in links


def test_store_through_a_subscript_is_a_weak_update(parser):
    # a[i] = v writes *an element*, so the previous definition of `a` stays reachable.
    # Killing it would claim knowledge a name-based analysis does not have.
    source = "void f(int i)\n{\n  int a[4];\n  a[i] = 1;\n  use(a);\n}\n"

    links = chains(parser, source)

    assert ("a", 3, 5) in links, "the declaration still reaches"
    assert ("a", 4, 5) in links, "so does the store"


def test_store_through_a_pointer_is_a_weak_update(parser):
    source = "void f(int *p)\n{\n  *p = 1;\n  use(p);\n}\n"

    links = chains(parser, source)

    assert ("p", 1, 3) in links
    assert ("p", 3, 4) in links


def test_aliasing_is_not_modelled(parser):
    # Documented limitation (ADR-002 §D2.3): analysis is by name. If this ever starts
    # passing, the module docstring and the ADR are out of date.
    source = "int f(void)\n{\n  int x = 1;\n  int *p = &x;\n  *p = 2;\n  return x;\n}\n"

    links = chains(parser, source)

    assert ("x", 3, 6) in links, "the name-based definition reaches"
    assert not any(name == "p" and use == 6 for name, _, use in links)


def test_scoping_is_flat(parser):
    # Documented over-approximation: two different `i`s are treated as one variable.
    source = "void f(void)\n{\n  { int i = 1; use(i); }\n  { int i = 2; use(i); }\n}\n"

    links = chains(parser, source)

    assert ("i", 3, 3) in links
    assert ("i", 4, 4) in links


def test_only_dfg_edges_are_emitted(parser):
    graph = build_dfg_graph(
        parser.parse("int f(void){ int x = 1; return x; }", language="C", unit_id="u1")
    )

    assert set(graph.edge_type_counts()) <= {"dfg"}


def test_node_set_is_the_asts(parser):
    from vulnloc.representation import build_ast_graph

    result = parser.parse(
        "int f(void)\n{\n  int x = 1;\n  return x;\n}\n", language="C", unit_id="u1"
    )

    assert build_dfg_graph(result).nodes == build_ast_graph(result).nodes


def test_edges_are_deterministic(parser):
    source = (
        "void f(int n)\n{\n  int s = 0;\n  for (int i = 0; i < n; i++) { s += i; }\n  use(s);\n}\n"
    )
    result = parser.parse(source, language="C", unit_id="u1")

    first = build_dfg_graph(result).edges
    second = build_dfg_graph(result).edges

    assert first == second
    assert list(first) == sorted(first, key=lambda e: (e.source, e.target))


def test_damaged_parse_still_produces_a_valid_graph(parser):
    result = parser.parse("int f(void) {\n  int x = 1;\n  return x", language="C", unit_id="u1")

    build_dfg_graph(result).validate()


def test_definitions_propagate_the_length_of_a_loop(parser):
    # A long-range dependence: defined at the top, read 800 statements later and again
    # after the loop. Cheap to run and it fails outright if propagation stops early.
    body = "".join(f"  t{i} = seed + {i};\n" for i in range(800))
    source = (
        "int f(int n)\n{\n  int seed = 1;\n  while (n--) {\n" + body + "  }\n  return seed;\n}\n"
    )

    links = chains(parser, source)

    assert ("seed", 3, 5) in links, "reaches the first loop statement"
    assert ("seed", 3, 804) in links, "and the last one, 800 statements later"
    assert ("seed", 3, 806) in links, "and the return after the loop"


def test_exhausted_budget_raises_rather_than_truncating(parser, monkeypatch):
    """Regression: a partial fixpoint must never be returned as if it were complete.

    This is the bug that shipped and had to be caught by hand: a flat 10,000-iteration
    cap stopped the analysis early on 25 of 400 real functions and **halved** the
    corpus's data-flow edges — one unit returned 1,093 of its 13,534 — with no error
    raised and a perfectly plausible-looking graph.

    Reproducing the magnitude needs ~900 branchy statements and 13 seconds, which does
    not belong in a unit test. What is worth pinning is the mechanism: whatever the
    budget, exhausting it is an error, not a smaller answer.
    """
    monkeypatch.setattr(dfg, "FIXPOINT_ROUNDS_PER_NODE", 0)
    monkeypatch.setattr(dfg, "FIXPOINT_ROUNDS_FLOOR", 2)
    source = "int f(int n)\n{\n  int x = 1;\n  while (n) {\n    x = x + 1;\n  }\n  return x;\n}\n"
    result = parser.parse(source, language="C", unit_id="tiny-budget")

    with pytest.raises(GraphBuildError) as excinfo:
        dfg.build_dfg_graph(result)

    assert "converge" in str(excinfo.value)


def test_fixpoint_budget_scales_with_the_function():
    # Convergence measured on real code needs ~14 iterations per statement, so a
    # constant budget is wrong for any large function. Scaling is the fix.
    assert dfg.FIXPOINT_ROUNDS_PER_NODE > 14, "must leave headroom over the measured ratio"
    assert dfg.FIXPOINT_ROUNDS_FLOOR >= 1000


def test_function_with_no_body_yields_no_edges(parser):
    result = parser.parse("int f(int a);\n", language="C", unit_id="u1")

    assert build_dfg_graph(result).edges == ()
