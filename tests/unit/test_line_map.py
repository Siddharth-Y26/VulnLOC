"""Guards the offset → line mapping.

This module is where an off-by-one would do the most damage and raise the least noise,
so the tests are written against the failure modes rather than the happy path: the
wrong line-splitting rule, the two offset spaces that agree only on ASCII, and offsets
that fall outside the source.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vulnloc.representation.line_map import LineMap, split_lines
from vulnloc.utils.exceptions import AlignmentError

# Characters str.splitlines() breaks on and this project does not.
NON_SEPARATORS = ["\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]


def test_only_newline_ends_a_line():
    for char in NON_SEPARATORS:
        source = f"int a;{char}int b;"

        assert len(split_lines(source)) == 1
        assert LineMap.from_source(source).n_lines == 1


def test_splitlines_would_disagree():
    # The bug this module exists to prevent. Measured on our corpus: one function of
    # 188,636 (bigvul-76919, a form feed) is split differently by the two rules.
    source = "int a;\fint b;"

    assert len(source.splitlines()) == 2
    assert len(split_lines(source)) == 1


def test_trailing_newline_yields_a_final_empty_line():
    # Keeps len(lines) == source.count("\n") + 1 for every input, so numbering never
    # depends on whether a file ends with a newline.
    assert split_lines("a\nb\n") == ["a", "b", ""]
    assert LineMap.from_source("a\nb\n").n_lines == 3


def test_lines_are_one_indexed():
    line_map = LineMap.from_source("aaa\nbbb\nccc")

    assert line_map.byte_to_line(0) == 1
    assert line_map.byte_to_line(4) == 2
    assert line_map.byte_to_line(8) == 3


def test_separator_belongs_to_the_line_it_terminates():
    line_map = LineMap.from_source("ab\ncd")

    assert line_map.byte_to_line(2) == 1  # the "\n" itself
    assert line_map.byte_to_line(3) == 2  # "c"


def test_byte_and_char_offsets_diverge_on_non_ascii():
    # The reason both maps exist: Tree-sitter reports bytes, HF tokenizers report
    # characters. On this input a byte offset of 4 is line 1 but a char offset of 4 is
    # line 2 — reading one with the other's number silently shifts the label.
    source = "//é\nint x;"
    line_map = LineMap.from_source(source)

    assert line_map.n_bytes == 11
    assert line_map.n_chars == 10
    assert line_map.byte_to_line(4) == 1  # the "\n", since "é" occupies bytes 2-3
    assert line_map.char_to_line(4) == 2  # the "i" of "int"


def test_end_of_source_offset_is_accepted():
    # Half-open spans end here; refusing it would make every last node an error.
    line_map = LineMap.from_source("abc")

    assert line_map.byte_to_line(3) == 1
    assert line_map.char_to_line(3) == 1


@pytest.mark.parametrize("offset", [-1, 99])
def test_offsets_outside_the_source_are_refused(offset):
    line_map = LineMap.from_source("abc", unit_id="u1")

    with pytest.raises(AlignmentError) as excinfo:
        line_map.byte_to_line(offset)

    assert excinfo.value.unit_id == "u1"


def test_line_byte_span_excludes_the_separator():
    source = "abc\ndefg\nhi"
    line_map = LineMap.from_source(source)
    raw = source.encode("utf-8")

    assert raw[slice(*line_map.line_byte_span(1))] == b"abc"
    assert raw[slice(*line_map.line_byte_span(2))] == b"defg"
    assert raw[slice(*line_map.line_byte_span(3))] == b"hi"


@pytest.mark.parametrize("line", [0, 4])
def test_line_span_outside_the_source_is_refused(line):
    with pytest.raises(AlignmentError):
        LineMap.from_source("a\nb\nc").line_byte_span(line)


# Property tests (Rules.md §6): the invariants must hold on arbitrary text, not just
# on the C-like samples above.

SOURCES = st.text(
    alphabet=st.characters(min_codepoint=1, max_codepoint=0x2FFF),
    max_size=300,
)


@given(SOURCES)
def test_every_byte_offset_maps_into_range(source):
    line_map = LineMap.from_source(source)

    for offset in range(line_map.n_bytes + 1):
        assert 1 <= line_map.byte_to_line(offset) <= line_map.n_lines


@given(SOURCES)
def test_byte_map_agrees_with_counting_separators(source):
    # Independent derivation: the line of an offset is one more than the number of
    # separators before it. Slow by design, correct by construction.
    line_map = LineMap.from_source(source)
    raw = source.encode("utf-8")

    for offset in range(len(raw) + 1):
        assert line_map.byte_to_line(offset) == raw[:offset].count(b"\n") + 1


@given(SOURCES)
def test_char_map_agrees_with_counting_separators(source):
    line_map = LineMap.from_source(source)

    for offset in range(len(source) + 1):
        assert line_map.char_to_line(offset) == source[:offset].count("\n") + 1


@given(SOURCES)
def test_line_spans_reconstruct_the_source(source):
    # The strongest statement available: the spans partition the source exactly, so no
    # byte is unreachable and none is claimed twice.
    line_map = LineMap.from_source(source)
    raw = source.encode("utf-8")

    rebuilt = b"\n".join(
        raw[slice(*line_map.line_byte_span(i))] for i in range(1, line_map.n_lines + 1)
    )

    assert rebuilt == raw
