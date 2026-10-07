"""The second formatting leak: a flagged line's comment terminator lost its slash.

``Memory.md`` §1.25. On the shipped corpus "this line ends a comment with a bare ``*``" is
flagged at precision 1.0000 on ``val``, and "this line ends with ``*/``" is never flagged
anywhere. :func:`~vulnloc.utils.text.strip_comments` removes comments from every line,
label-blind. What is pinned here is that it kills both tells, that it never moves a line —
the line index is the unit every label is attached to — and the three cases a naive
comment stripper gets wrong on real corpus text.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vulnloc.representation.line_map import split_lines
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.text import (
    ends_comment_without_slash,
    normalize_source_whitespace,
    strip_comments,
)

# Shaped like the corpus: the flagged lines (3, 6, 7) ended comments and lost the slash.
LEAKY_SOURCE = (
    "int f(struct ctx *c) {\n"  # 1
    "int n = 0;  /* count */\n"  # 2  unflagged: terminator intact
    "qio_channel_wait(c->ioc);  /* needed by nbd_reply_ready.  *\n"  # 3  flagged
    "if (n) {\n"  # 4
    "/* XXX figure out if we can create a variant\n"  # 5  comment opens
    "* that works with AIO contexts\n"  # 6  flagged, inside the comment
    "* and consider using that in this branch *\n"  # 7  flagged, comment ends here
    "qemu_coroutine_yield();\n"  # 8  code — must survive
    "}\n"  # 9
    "return n;  // done\n"  # 10
    "}"  # 11
)
FLAGGED = frozenset({3, 6, 7})


def lost_slash_lines(source: str) -> frozenset[int]:
    """The attack: read the label off a comment terminator that lost its slash."""
    return frozenset(
        number
        for number, line in enumerate(split_lines(source), start=1)
        if ends_comment_without_slash(line)
    )


def intact_lines(source: str) -> frozenset[int]:
    """The mirror attack: a line ending in ``*/`` is readable as *not* flagged."""
    return frozenset(
        number
        for number, line in enumerate(split_lines(source), start=1)
        if line.rstrip().endswith("*/")
    )


def test_the_leak_is_real_before_stripping():
    assert lost_slash_lines(LEAKY_SOURCE) == {3, 7}
    assert lost_slash_lines(LEAKY_SOURCE) <= FLAGGED
    assert intact_lines(LEAKY_SOURCE) == {2}
    assert not intact_lines(LEAKY_SOURCE) & FLAGGED


def test_stripping_destroys_both_tells():
    stripped = strip_comments(LEAKY_SOURCE)

    assert lost_slash_lines(stripped) == frozenset()
    assert intact_lines(stripped) == frozenset()


def test_the_line_count_is_preserved_and_code_lines_stay_where_they_were():
    stripped = split_lines(strip_comments(LEAKY_SOURCE))

    assert len(stripped) == len(split_lines(LEAKY_SOURCE))
    assert stripped == [
        "int f(struct ctx *c) {",
        "int n = 0;",
        "qio_channel_wait(c->ioc);",
        "if (n) {",
        "",
        "",
        "",
        "qemu_coroutine_yield();",
        "}",
        "return n;",
        "}",
    ]


def test_a_terminator_that_lost_its_slash_still_ends_the_comment():
    # Without this rule the open comment swallows `free(p);`, and the flagged line's own
    # formatting has destroyed the code after it.
    source = "x = 1;  /* note *\nfree(p);\ny = 2;  /* fine */"

    assert split_lines(strip_comments(source)) == ["x = 1;", "free(p);", "y = 2;"]


def test_a_doc_comment_opener_does_not_end_its_own_comment():
    source = "/**\n * Frees the buffer.\n */\nvoid f(void);"

    assert split_lines(strip_comments(source)) == ["", "", "", "void f(void);"]


def test_a_continuation_line_ending_in_several_stars_ends_the_comment():
    source = "/* banner\n* text **\ncode();"

    assert split_lines(strip_comments(source)) == ["", "", "code();"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('memcmp(ptr, "/* XPM */", 9);', 'memcmp(ptr, "/* XPM */", 9);'),
        ('puts("http://example");', 'puts("http://example");'),
        ("c = '\"';  /* a quote */", "c = '\"';"),
        ('s = "a\\"b";  // tail', 's = "a\\"b";'),
        ("int/**/x;", "int x;"),
        ("foo(/* arg */1);", "foo(1);"),
        ("a = b /* why */ + c;", "a = b + c;"),
        ("sin->sin_port = 0 /* skb->h.uh->source */;", "sin->sin_port = 0;"),
        ("x = a / b;", "x = a / b;"),
        ("x = a * b;", "x = a * b;"),
        ("*p = 0;", "*p = 0;"),
        ("// only a comment", ""),
        ("/* only a comment */", ""),
        ("", ""),
    ],
)
def test_single_lines(line, expected):
    assert strip_comments(line) == expected


def test_an_apostrophe_that_opens_no_literal_cannot_swallow_the_next_line():
    source = "#error don't /* keep */\nx = 1;  /* gone */"

    assert split_lines(strip_comments(source)) == ["#error don't /* keep */", "x = 1;"]


def test_the_result_is_whitespace_normalized_with_no_doubled_spaces_left_behind():
    stripped = strip_comments("   a = 1;   /* one */   b = 2;  /* two */  ")

    assert stripped == "a = 1; b = 2;"
    assert "  " not in stripped


def test_stripping_is_idempotent():
    once = strip_comments(LEAKY_SOURCE)

    assert strip_comments(once) == once


def test_a_code_unit_strips_only_when_asked():
    plain = CodeUnit.from_source("u1", LEAKY_SOURCE, language="C")
    stripped = CodeUnit.from_source("u1", LEAKY_SOURCE, language="C", strip_comments=True)

    assert plain.lines[2].endswith("*"), "every number logged before the fix depends on this"
    assert stripped.lines[2] == "qio_channel_wait(c->ioc);"
    assert stripped.n_lines == plain.n_lines


def test_a_code_units_offsets_are_those_of_the_stripped_text():
    # Landmine #31: anything deriving offsets must use `unit.source`, which is the
    # transformed text.
    unit = CodeUnit.from_source("u1", LEAKY_SOURCE, language="C", strip_comments=True)

    assert unit.source == strip_comments(LEAKY_SOURCE)
    assert "\n".join(unit.lines) == unit.source


_CODE_ALPHABET = st.sampled_from(list("abcxyz019 _;(){}=+-<>,.&|!\t"))
_CODE_LINE = st.text(_CODE_ALPHABET, max_size=30)


@settings(max_examples=200, deadline=None)
@given(st.lists(_CODE_LINE, min_size=1, max_size=12))
def test_text_without_comment_markers_or_quotes_is_only_whitespace_normalized(lines):
    source = "\n".join(lines)

    assert strip_comments(source) == normalize_source_whitespace(source)


@settings(max_examples=300, deadline=None)
@given(st.text(st.sampled_from(list("ab1 /*\"'\\\n;")), max_size=120))
def test_any_text_keeps_its_line_count_and_is_stable(source):
    stripped = strip_comments(source)

    assert len(split_lines(stripped)) == len(split_lines(source))
    assert strip_comments(stripped) == stripped
