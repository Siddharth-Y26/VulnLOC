"""Guards the Tree-sitter wrapper.

Two things are being protected here. The first is the contract that malformed input
produces a *result*, not an exception — Big-Vul functions are fragments cut out of
their translation units and many do not parse standalone, and that rate is a reportable
number rather than a crash. The second is the ban on ``Point``: reading
``node.start_point`` segfaults the interpreter on real corpus input, which no
``pytest.raises`` can catch, so the guard is a fixture that would take the whole suite
down if anyone reintroduced it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vulnloc.representation import (
    LANGUAGE_POLICIES,
    LineMap,
    ParseResult,
    TreeSitterParser,
    iter_nodes,
    node_to_line,
    resolve_language,
    verify_line_alignment,
)
from vulnloc.utils.exceptions import AlignmentError, ParseError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "parser"

CLEAN_C = """\
int add(int a, int b)
{
    return a + b;
}
"""

# Covers the constructs ADR-002 flags as line-map hazards: a preprocessor directive, a
# macro with a line continuation, a multi-line comment, a tab-indented block, and a
# non-ASCII literal.
HAZARDS_C = """\
#include <stdio.h>
#define MAX(a, b) \\
    ((a) > (b) ? (a) : (b))
/* a comment
   spanning lines */
int pick(int a, int b)
{
\tconst char *note = "café";
\tif (a > b) {
\t\treturn MAX(a, b);
\t}
\treturn b;
}
"""


@pytest.fixture
def parser():
    return TreeSitterParser()


def test_parses_clean_c(parser):
    result = parser.parse(CLEAN_C, language="C", unit_id="u1")

    assert isinstance(result, ParseResult)
    assert result.language == "c"
    assert result.is_clean
    assert result.error_nodes == 0
    assert result.missing_nodes == 0
    assert result.first_error_line is None


@pytest.mark.parametrize("declared", ["C", "c", "CPP", "C++", "cc", " cxx "])
def test_dataset_language_strings_resolve(parser, declared):
    # Big-Vul's own lang column uses C, CPP and C++ — all three must resolve.
    result = parser.parse(CLEAN_C, language=declared, unit_id="u1")

    assert result.language in {"c", "cpp"}


def test_unknown_language_raises_rather_than_guessing(parser):
    with pytest.raises(ParseError) as excinfo:
        parser.parse(CLEAN_C, language="rust", unit_id="u1")

    assert excinfo.value.unit_id == "u1"
    assert excinfo.value.details["language"] == "rust"


def test_syntax_errors_are_recorded_not_raised(parser):
    # The contract that makes a corpus-wide parse rate measurable at all.
    result = parser.parse("int broken(void) { return", language="C", unit_id="u2")

    assert not result.is_clean
    assert result.error_nodes or result.missing_nodes
    assert result.first_error_line == 1


def test_first_error_line_points_at_the_damage(parser):
    source = "int ok(void) { return 1; }\nint bad(void) { return\n"

    result = parser.parse(source, language="C", unit_id="u3")

    assert not result.is_clean
    assert result.first_error_line == 2


def test_summary_is_serializable(parser):
    result = parser.parse(CLEAN_C, language="C", unit_id="u1")

    summary = result.summary()

    assert summary["unit_id"] == "u1"
    assert summary["is_clean"] is True
    assert summary["n_lines"] == 5


def test_hazardous_constructs_keep_their_lines(parser):
    result = parser.parse(HAZARDS_C, language="C", unit_id="u4")

    verify_line_alignment(result)

    # Spot-check the constructs by name rather than trusting the sweep alone.
    lines = {
        node_to_line(node, result.line_map)
        for node in iter_nodes(result.root)
        if node.type == "string_literal"
    }
    assert lines == {8}


def test_node_lines_come_from_the_offset_table(parser):
    result = parser.parse(CLEAN_C, language="C", unit_id="u1")
    root = result.root

    assert node_to_line(root, result.line_map) == 1
    assert node_to_line(root, result.line_map) == result.line_map.byte_to_line(root.start_byte)


def test_verify_line_alignment_detects_a_shifted_map(parser):
    # Feed the result a map built from different text; the node text can then no longer
    # be found on its claimed line, which is exactly the corruption we cannot detect by
    # eye later.
    result = parser.parse(CLEAN_C, language="C", unit_id="u1")
    shifted = ParseResult(
        unit_id=result.unit_id,
        language=result.language,
        source=result.source,
        tree=result.tree,
        line_map=LineMap.from_source("\n\n" + result.source, unit_id="u1"),
        error_nodes=result.error_nodes,
        missing_nodes=result.missing_nodes,
        first_error_line=result.first_error_line,
    )

    with pytest.raises(AlignmentError):
        verify_line_alignment(shifted)


def test_walking_the_known_segfault_fixture_is_safe(parser):
    """Regression guard for the ``start_point`` crash.

    ``tree-sitter==0.26.0`` with ``tree-sitter-c==0.24.2`` segfaults when a walk of this
    file reads ``node.start_point``; reading ``start_byte`` over the same 1,349 nodes is
    fine. The fixture is the minimal crashing prefix of ``bigvul-188127``. If someone
    reintroduces ``Point``, this test does not fail — it takes the interpreter down,
    which is the loudest signal available for a native crash.
    """
    source = (FIXTURES / "segfault_start_point.c").read_text(encoding="utf-8")

    result = parser.parse(source, language="C", unit_id="bigvul-188127")
    verify_line_alignment(result)

    assert sum(1 for _ in iter_nodes(result.root)) > 1000


def test_iter_nodes_yields_root_first_and_covers_the_tree(parser):
    result = parser.parse(CLEAN_C, language="C", unit_id="u1")

    nodes = list(iter_nodes(result.root))

    assert nodes[0] == result.root
    assert any(node.type == "function_definition" for node in nodes)
    # Counted independently, against the tree's own child counts: every node is
    # somebody's child exactly once, except the root.
    assert len(nodes) == 1 + sum(node.child_count for node in nodes)


def test_parser_instance_is_reusable_across_languages(parser):
    first = parser.parse(CLEAN_C, language="C", unit_id="u1")
    second = parser.parse(CLEAN_C, language="CPP", unit_id="u2")
    third = parser.parse(CLEAN_C, language="C", unit_id="u3")

    assert first.is_clean and second.is_clean and third.is_clean
    assert first.language == third.language == "c"
    assert second.language == "cpp"


# --- ADR-005: grammar selection -------------------------------------------------------
#
# `best_of_two` makes the grammar a function of the source, which is the one property
# that could quietly break reproducibility. These tests pin the tie-break rules that
# stop it from doing so.

# `class` and `new` are ordinary identifiers in C and reserved words in C++, so this
# compiles as C and cannot parse as C++. It is the whole reason `cpp`-everywhere loses
# 1,712 units that `best_of_two` keeps.
C_ONLY = """int scan(void)
{
    int class = 1;
    int new = 2;
    return class + new;
}
"""

# Scope resolution and a reference parameter: C++ that Big-Vul labels `C` for 98.6% of
# its rows. 100,735 corpus functions are clean only under this grammar.
CPP_ONLY = """void AudioContext::createConvolver(const Buffer& source)
{
    m_buffer = source;
}
"""


def test_best_of_two_falls_back_to_c_when_cpp_cannot_parse(parser):
    result = parser.parse_with_policy(C_ONLY, policy="best_of_two", declared="C", unit_id="u1")

    assert result.language == "c"
    assert result.is_clean


def test_best_of_two_keeps_cpp_for_cpp_source_labelled_c(parser):
    result = parser.parse_with_policy(CPP_ONLY, policy="best_of_two", declared="C", unit_id="u1")

    assert result.language == "cpp"
    assert result.is_clean


def test_best_of_two_prefers_cpp_on_a_tie(parser):
    """C++ is tried first and wins ties, so the choice never depends on iteration order.

    Neither grammar can parse this, and both fail identically. A rule that broke ties any
    other way would make the grammar depend on something other than the source.
    """
    source = "int f(void) { @@@ }\n"

    result = parser.parse_with_policy(source, policy="best_of_two", declared="C", unit_id="u1")
    as_c = parser.parse(source, language="c", unit_id="u1")
    as_cpp = parser.parse(source, language="cpp", unit_id="u1")

    assert (as_c.error_nodes, as_c.missing_nodes) == (as_cpp.error_nodes, as_cpp.missing_nodes)
    assert result.language == "cpp"


def test_best_of_two_is_deterministic_across_calls(parser):
    """Same source, same grammar — every time, and independent of the declared label."""
    for declared in ("C", "CPP", "C++"):
        chosen = {
            parser.parse_with_policy(
                C_ONLY, policy="best_of_two", declared=declared, unit_id="u1"
            ).language
            for _ in range(3)
        }
        assert chosen == {"c"}


def test_static_policies_ignore_the_source(parser):
    assert (
        parser.parse_with_policy(C_ONLY, policy="cpp", declared="C", unit_id="u1").language == "cpp"
    )
    assert (
        parser.parse_with_policy(CPP_ONLY, policy="c", declared="CPP", unit_id="u1").language == "c"
    )
    assert (
        parser.parse_with_policy(CPP_ONLY, policy="declared", declared="CPP", unit_id="u1").language
        == "cpp"
    )


def test_unknown_policy_raises_rather_than_defaulting(parser):
    with pytest.raises(ParseError) as excinfo:
        parser.parse_with_policy(CLEAN_C, policy="auto", declared="C", unit_id="u1")

    assert excinfo.value.details["policy"] == "auto"


def test_resolve_language_refuses_the_source_dependent_policy():
    """`best_of_two` has no answer without the source; saying so beats guessing one."""
    with pytest.raises(ParseError):
        resolve_language("best_of_two", "C", unit_id="u1")


def test_every_declared_policy_is_usable(parser):
    for policy in LANGUAGE_POLICIES:
        result = parser.parse_with_policy(CLEAN_C, policy=policy, declared="C", unit_id="u1")
        assert result.is_clean
