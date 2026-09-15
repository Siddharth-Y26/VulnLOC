"""The inspector's fix templates — pattern matches, never presented as the model's reasoning.

A template that fires on safe code trains a developer to ignore the tab; one that misses
the textbook case makes the tab useless. Both directions are pinned here, including the
cases a regex is known to get wrong, so a change that "improves" one pattern cannot quietly
break another.
"""

from __future__ import annotations

import pytest
from frontend.fix_templates import match_templates


def ids(source: str) -> dict[int, set[str]]:
    lines = source.strip("\n").split("\n")
    return {line: {m["id"] for m in found} for line, found in match_templates(lines).items()}


@pytest.mark.parametrize(
    ("line", "template"),
    [
        ("gets(buf);", "gets"),
        ("strcpy(dst, buf);", "unbounded-string-copy"),
        ("strcat(path, name);", "unbounded-string-copy"),
        ('sprintf(out, "%d", n);', "unbounded-format-write"),
        ("memcpy(buf, src, n);", "memcpy-length"),
        ('scanf("%s", name);', "scanf-string"),
        ("printf(msg);", "format-string"),
        ("fprintf(stderr, user_text);", "format-string"),
        ("p = malloc(count * size);", "allocation-size-arithmetic"),
        ("buf = realloc(buf, n);", "realloc-same-pointer"),
        ("strncpy(dst, src, 16);", "bounded-copy-termination"),
        ("int n = strlen(src);", "signed-length"),
        ("system(cmd);", "command-injection"),
    ],
)
def test_the_textbook_case_matches(line: str, template: str) -> None:
    assert template in ids(line).get(1, set())


@pytest.mark.parametrize(
    "line",
    [
        "memcpy(buf, src, sizeof buf);",
        'printf("%s", msg);',
        'fprintf(stderr, "%s\\n", user_text);',
        'scanf("%31s", name);',
        "size_t n = strlen(src);",
        "char *tmp = realloc(buf, n); if (tmp == NULL) return;",
        "p = malloc(sizeof *p); if (!p) return;",
        "p = malloc(sizeof(*p)); if (!p) return;",
        "// strcpy(dst, src); in a comment",
        "fgets(line, sizeof line, stdin);",
    ],
)
def test_the_safe_form_does_not_match(line: str) -> None:
    assert ids(line) == {}


def test_use_after_free_points_at_the_use_and_names_the_free() -> None:
    found = match_templates(["free(p);", "n = 1;", "p->len = 0;"])

    assert [m["id"] for m in found[3]] == ["use-after-free"]
    assert found[3][0]["related_line"] == 1


def test_reassigning_the_pointer_clears_the_use_after_free() -> None:
    assert match_templates(["free(p);", "p = NULL;", "if (p) return;"]) == {}


def test_double_free_is_reported_on_the_second_free() -> None:
    found = match_templates(["free(p);", "x = 1;", "free(p);"])

    assert [m["id"] for m in found[3]] == ["double-free"]


def test_an_allocation_checked_later_is_not_flagged() -> None:
    assert "unchecked-allocation" not in {
        m["id"] for m in match_templates(["p = malloc(n);", "if (p == NULL)", "return;"]).get(1, [])
    }


def test_an_allocation_never_checked_is_flagged() -> None:
    assert "unchecked-allocation" in {
        m["id"] for m in match_templates(["p = malloc(n);", "p[0] = 0;"])[1]
    }


def test_the_inspector_sample_matches_where_a_reviewer_would_look() -> None:
    sample = """
void copy_name(char *dst, const char *src)
{
char buf[32];
int n = strlen(src);
if (dst == NULL)
return;
memcpy(buf, src, n);
strcpy(dst, buf);
}
"""
    assert ids(sample) == {
        4: {"signed-length"},
        7: {"memcpy-length"},
        8: {"unbounded-string-copy"},
    }
