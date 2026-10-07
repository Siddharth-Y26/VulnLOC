"""Guards the explanation template library (Phase 7a).

A template turns a line's text into a sentence a reader will take as the reason the line
was flagged, so the failures that matter are the quiet ones: a pattern that fires on safe
code and states something false about it, a sentence that quotes a symbol the line does
not contain, and wording that drifts from *why this is risky* to *how to use it*
(``Rules.md`` §7.2, §7.3). The last is the automatic half of Phase 7's "reviewed for
absence of attack instructions"; the other half is a person reading the sentences.
"""

from __future__ import annotations

import re

import pytest

from vulnloc.explanation.templates import (
    FLOW_TEMPLATES,
    LINE_TEMPLATES,
    NO_PATTERN,
    all_templates,
    match_lines,
)
from vulnloc.explanation.verifier import BANNED_PHRASES


def match_one(text: str):
    return match_lines([text]).get(1)


@pytest.mark.parametrize(
    ("text", "template_id", "symbols"),
    [
        ("gets(line);", "gets", ("gets",)),
        ("strcpy(dst, src);", "unbounded-string-copy", ("strcpy",)),
        ('sprintf(buf, "%s/%s", dir, name);', "unbounded-format-write", ("sprintf",)),
        ("memcpy(buf, src, n);", "memcpy-length", ("memcpy",)),
        ('scanf("%s", name);', "scanf-string", ("scanf",)),
        ("printf(msg);", "format-string-printf", ("printf", "msg")),
        ("fprintf(stderr, msg);", "format-string-stream", ("fprintf", "msg")),
        ("snprintf(buf, size, fmt);", "format-string-snprintf", ("snprintf", "fmt")),
        ("p = malloc(count * elem);", "allocation-size-arithmetic", ("malloc",)),
        ("buf = realloc(buf, n);", "realloc-same-pointer", ("buf", "realloc")),
        ("strncpy(dst, src, sizeof dst);", "bounded-copy-termination", ("strncpy",)),
        ("int n = strlen(src);", "signed-length", ("int", "strlen")),
        ("system(cmd);", "command-injection", ("system",)),
        ("x = *(p + off);", "pointer-offset-dereference", ("p", "off")),
        ("table[idx] = value;", "array-index-write", ("table", "idx")),
        ("ctx->slots[i + 1] |= flag;", "array-index-write", ("ctx->slots", "i + 1")),
        ("return table[idx];", "array-index-read", ("table", "idx")),
    ],
)
def test_each_template_matches_its_pattern_and_quotes_only_what_is_on_the_line(
    text, template_id, symbols
):
    found = match_one(text)

    assert found is not None, f"nothing matched {text!r}"
    assert found.template.id == template_id
    assert found.symbols == symbols
    assert all(symbol in text for symbol in found.symbols)
    assert "{" not in found.reason, "an unfilled placeholder would reach the reader"


@pytest.mark.parametrize(
    "text",
    [
        "memcpy(dst, src, sizeof dst);",  # length derived from the destination
        'printf("%s", msg);',  # literal format
        'scanf("%31s", name);',  # bounded conversion
        "p = malloc(sizeof *p);",  # `*` is a dereference, not a product
        "p = malloc(sizeof(*p));",
        "tmp = realloc(buf, n);",  # the original pointer survives
        "size_t n = strlen(src);",  # unsigned length
        "area = width * (height + 1);",  # a product, not a dereference
        "char buf[SIZE];",  # declares an array, reads nothing
        "static const struct entry table[COUNT];",
        "if (table[0] == 1)",  # constant index
        "// strcpy(dst, src);",  # a comment
        "#define COPY(d, s) strcpy(d, s)",  # preprocessor
        "",
    ],
)
def test_safe_or_irrelevant_lines_match_nothing(text):
    # The second line is the NULL check a real function would carry; without it the two
    # allocations here would rightly be reported as unchecked.
    found = match_lines([text, "if (!p || !tmp) return;"]).get(1)

    assert found is None, f"{text!r} matched {found}"


def test_a_comparison_is_not_an_array_write():
    found = match_one("if (table[idx] == limit)")

    assert found is not None
    assert found.template.id == "array-index-read"


def test_use_after_free_names_the_line_of_the_free():
    lines = ["free(ctx->buf);", "len = 0;", "ctx->buf[0] = 0;"]

    found = match_lines(lines)[3]

    assert found.template.id == "use-after-free"
    assert found.related_line == 1
    assert found.symbols == ("ctx->buf",)
    assert "freed on line 1" in found.reason


def test_a_reassignment_between_free_and_use_clears_it():
    lines = ["free(p);", "p = next;", "use(p);"]

    assert 3 not in match_lines(lines)


def test_double_free_is_reported_on_the_second_free():
    lines = ["free(p);", "count--;", "free(p);"]

    found = match_lines(lines)[3]

    assert found.template.id == "double-free"
    assert found.related_line == 1


def test_an_allocation_never_checked_against_null_is_reported_on_its_own_line():
    lines = ["p = malloc(n);", "p[0] = 0;"]

    found = match_lines(lines)[1]

    assert found.template.id == "unchecked-allocation"
    assert found.related_line is None
    assert found.symbols == ("malloc", "p")


@pytest.mark.parametrize("guard", ["if (!p)", "if (p == NULL)", "if (NULL == p)", "if (p)"])
def test_a_null_check_anywhere_later_clears_the_unchecked_allocation(guard):
    lines = ["p = calloc(n, m);", guard, "return -1;"]

    assert 1 not in match_lines(lines) or match_lines(lines)[1].template.id != (
        "unchecked-allocation"
    )


def test_a_flow_template_outranks_a_single_line_one_on_the_same_line():
    lines = ["free(p);", "memcpy(p, src, n);"]

    assert match_lines(lines)[2].template.id == "use-after-free"


def test_an_unchecked_allocation_ranks_below_a_construct_that_is_on_the_line():
    # The absence of a NULL check is weaker evidence than arithmetic in the size itself.
    assert match_one("p = malloc(count * elem);").template.id == "allocation-size-arithmetic"


def test_matching_is_keyed_by_one_indexed_line():
    lines = ["int x = 0;", "strcpy(dst, src);"]

    assert set(match_lines(lines)) == {2}


def test_template_ids_are_unique_and_none_is_the_fallback():
    ids = [template.id for template in all_templates()]

    assert len(ids) == len(set(ids))
    assert NO_PATTERN not in ids


def test_every_template_carries_a_cwe_identifier_and_a_label():
    for template in all_templates():
        assert re.fullmatch(r"CWE-\d+", template.cwe), template.id
        assert template.label, template.id
        assert template.reason.endswith("."), template.id


def test_single_line_templates_have_a_pattern_and_flow_templates_do_not():
    assert all(template.pattern is not None for template in LINE_TEMPLATES)
    assert all(template.pattern is None for template in FLOW_TEMPLATES.values())


def test_no_template_sentence_contains_attack_vocabulary():
    # Rules.md §7.2 and §7.3: an explanation says why a construct is risky, never how to
    # use it. Checked here on the library, and again on every finished text by the verifier.
    for template in all_templates():
        lowered = template.reason.lower()
        hits = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
        assert not hits, f"{template.id}: {hits}"


def test_every_placeholder_in_a_single_line_sentence_is_a_group_of_its_pattern():
    for template in LINE_TEMPLATES:
        assert template.pattern is not None
        wanted = set(re.findall(r"\{(\w+)\}", template.reason))
        assert wanted <= set(template.pattern.groupindex), template.id
