"""ADR-006: the flaw-line label must not be recoverable from formatting.

The leak these tests pin is not hypothetical. On the shipped corpus a one-line rule —
"does this line begin with a space or tab?" — recovers Big-Vul's line labels at precision
1.0000 with zero false positives on all four partitions, because the LineVul release
strips indentation from every line *except* the flagged ones it pastes back verbatim.
"""

from __future__ import annotations

import pytest

from vulnloc.data.labeling import LabelledFunction
from vulnloc.representation.line_map import split_lines
from vulnloc.representation.normalization import normalize_source_whitespace
from vulnloc.representation.units import CodeUnit

# Shaped exactly like a Big-Vul `processed_func`: every line flush left except the
# flagged ones, which kept their original indentation.
LEAKY_SOURCE = (
    "int f(char *p) {\n"
    "int n = 0;\n"
    "if (p) {\n"
    "        n = strlen(p);   \n"  # flagged: leading AND trailing whitespace
    "\t\tmemcpy(dst, p, n);\n"  # flagged: leading tabs
    "}\n"
    "return n;\n"
    "}\n"
)
FLAGGED = frozenset({4, 5})


def _leading_ws_predictor(source: str) -> frozenset[int]:
    """The attack: guess the label from indentation alone."""
    return frozenset(
        i for i, line in enumerate(split_lines(source), start=1) if line[:1] in (" ", "\t")
    )


def test_the_leak_is_real_before_normalization() -> None:
    """Without the fix, indentation alone recovers the labels exactly."""
    assert _leading_ws_predictor(LEAKY_SOURCE) == FLAGGED


def test_normalization_destroys_the_leak() -> None:
    """After the fix, the predictor fires on nothing."""
    assert _leading_ws_predictor(normalize_source_whitespace(LEAKY_SOURCE)) == frozenset()


def test_line_count_is_preserved_exactly() -> None:
    """The whole safety argument: labels stay attached to the lines they labelled."""
    before = split_lines(LEAKY_SOURCE)
    after = split_lines(normalize_source_whitespace(LEAKY_SOURCE))
    assert len(before) == len(after)
    # And each line still corresponds, modulo the stripped edges.
    assert [line.strip() for line in before] == after


def test_normalization_strips_both_ends() -> None:
    """Trailing whitespace is a second tell at precision 1.0 — it must go too."""
    normalized = split_lines(normalize_source_whitespace(LEAKY_SOURCE))
    assert normalized[3] == "n = strlen(p);"
    assert normalized[4] == "memcpy(dst, p, n);"


def test_empty_and_blank_lines_survive() -> None:
    """A blank line must stay a line, or every label after it shifts."""
    assert normalize_source_whitespace("a\n   \n\nb") == "a\n\n\nb"


def test_content_without_edge_whitespace_is_unchanged() -> None:
    source = "int f(void) {\nreturn 0;\n}\n"
    assert normalize_source_whitespace(source) == source


def test_inner_whitespace_is_left_alone() -> None:
    """Only the edges carry the leak; collapsing inner spacing would change tokens."""
    assert normalize_source_whitespace("  a  =  b  ") == "a  =  b"


@pytest.mark.parametrize("normalize", [True, False])
def test_code_unit_honours_the_flag(normalize: bool) -> None:
    """The knob must actually be consumed.

    ``Memory.md`` §7 #22: an unread knob looks exactly like a working one, so this test
    fails if :class:`CodeUnit` ever ignores the flag.
    """
    unit = CodeUnit.from_source(
        "u1",
        LEAKY_SOURCE,
        language="C",
        target=1,
        vulnerable_lines=FLAGGED,
        normalize_whitespace=normalize,
    )
    leaked = _leading_ws_predictor(unit.source)
    assert leaked == (frozenset() if normalize else FLAGGED)
    # Either way the line axis is intact.
    assert unit.n_lines == len(split_lines(LEAKY_SOURCE))
    assert unit.vulnerable_lines == FLAGGED


def test_code_unit_defaults_to_normalizing() -> None:
    """The safe behaviour is the default one."""
    unit = CodeUnit.from_source("u1", LEAKY_SOURCE, language="C")
    assert _leading_ws_predictor(unit.source) == frozenset()


def test_from_labelled_function_closes_the_leak() -> None:
    """The L0 -> L1 boundary is where the shipped corpus text gets fixed."""
    labelled = LabelledFunction(
        unit_id="u1",
        source=LEAKY_SOURCE,
        language="C",
        target=1,
        vulnerable_lines=FLAGGED,
        n_lines=len(split_lines(LEAKY_SOURCE)),
    )
    assert _leading_ws_predictor(CodeUnit.from_labelled_function(labelled).source) == frozenset()
    assert (
        _leading_ws_predictor(
            CodeUnit.from_labelled_function(labelled, normalize_whitespace=False).source
        )
        == FLAGGED
    )


def test_line_map_agrees_with_normalized_source() -> None:
    """The line map is rebuilt from the normalized text, not the original."""
    unit = CodeUnit.from_source("u1", LEAKY_SOURCE, language="C")
    assert unit.line_map.n_lines == unit.n_lines
    for index, line in enumerate(unit.lines, start=1):
        del line
        assert 1 <= index <= unit.line_map.n_lines
