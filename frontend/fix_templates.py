"""Fix guidance from deterministic templates — not from the model.

**Decided by the project owner on 2026-09-15** (open decision Q4's recommendation: a
template baseline first, generative explanation only with a reference set to judge it).
Phase 7 — evidence-grounded explanation built from the model's own attention — is not
built, so nothing here claims to say *why the model* ranked a line. It says what a
well-known C/C++ weakness pattern on that line looks like and how it is usually fixed.

The two sources are kept visibly separate in the interface, and must stay separate:

- **The model** decides which lines rank highest. It gives no reason.
- **A template** matches the text of a line. It knows nothing about the model, the corpus
  or the data flow, and it matches safe code too — ``memcpy(dst, src, sizeof dst)`` is fine
  and a regex cannot always tell.

A line the model ranks highly and no template matches is reported as exactly that. It is
never filled with a generic "check your bounds" that would read like an explanation.

Every CWE identifier below is MITRE's; the titles are shortened, not paraphrased into
different claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["NO_ADVICE", "TEMPLATES", "advice_for", "match_templates"]


@dataclass(frozen=True)
class Template:
    """One weakness pattern and its usual fix.

    Attributes:
        id: Stable key.
        cwe: MITRE CWE identifier.
        title: Short CWE title.
        pattern: Regex over one normalized source line.
        problem: What to check, phrased as a question about this line.
        fix: The usual remedy.
        example: A before/after pair, generic rather than derived from the paste.
    """

    id: str
    cwe: str
    title: str
    pattern: re.Pattern[str]
    problem: str
    fix: str
    example: tuple[str, str]


def _re(expression: str) -> re.Pattern[str]:
    return re.compile(expression)


TEMPLATES: tuple[Template, ...] = (
    Template(
        id="gets",
        cwe="CWE-242",
        title="Use of inherently dangerous function",
        pattern=_re(r"\bgets\s*\("),
        problem="gets() cannot be told the size of its buffer; any long enough input overflows it.",
        fix="Use fgets() with the buffer size, and strip the trailing newline.",
        example=("gets(line);", "if (fgets(line, sizeof line, stdin) == NULL) return -1;"),
    ),
    Template(
        id="unbounded-string-copy",
        cwe="CWE-120",
        title="Buffer copy without checking size of input",
        pattern=_re(r"\b(strcpy|strcat|wcscpy|wcscat)\s*\("),
        problem=(
            "The destination's size is never consulted. Is the source guaranteed to fit, including"
            " its terminator?"
        ),
        fix=(
            "Check the length against the destination size first, or use a bounded copy (snprintf,"
            " strlcpy where available) and handle truncation."
        ),
        example=(
            "strcpy(dst, src);",
            "if (strlen(src) >= dst_size) return -EINVAL;\nmemcpy(dst, src, strlen(src) + 1);",
        ),
    ),
    Template(
        id="unbounded-format-write",
        cwe="CWE-120",
        title="Buffer copy without checking size of input",
        pattern=_re(r"\b(sprintf|vsprintf)\s*\("),
        problem="sprintf() writes as much as the format produces, regardless of the buffer.",
        fix="Use snprintf() with the buffer size and check the return value for truncation.",
        example=(
            'sprintf(buf, "%s/%s", dir, name);',
            'int n = snprintf(buf, sizeof buf, "%s/%s", dir, name);\n'
            "if (n < 0 || (size_t)n >= sizeof buf) return -1;",
        ),
    ),
    Template(
        id="memcpy-length",
        cwe="CWE-787",
        title="Out-of-bounds write",
        # A length argument that is not a sizeof expression is the case worth a look.
        pattern=_re(r"\b(memcpy|memmove|memset)\s*\((?![^;]*,\s*sizeof\b)[^;]*\)"),
        problem=(
            "The length is not derived from the destination. Can it exceed the destination's size,"
            " or be negative before conversion to size_t?"
        ),
        fix=(
            "Validate the length against the destination size immediately before the call, using "
            "an unsigned type."
        ),
        example=(
            "memcpy(buf, src, n);",
            "if (n > sizeof buf) return -EINVAL;\nmemcpy(buf, src, n);",
        ),
    ),
    Template(
        id="scanf-string",
        cwe="CWE-120",
        title="Buffer copy without checking size of input",
        pattern=_re(r"\b(f|s)?scanf\s*\([^;]*\"[^\"]*%s"),
        problem="%s with no field width reads an unbounded token into a fixed buffer.",
        fix="Give every %s a width one less than the buffer size, e.g. %31s for char[32].",
        example=('scanf("%s", name);', 'scanf("%31s", name);'),
    ),
    Template(
        id="format-string",
        cwe="CWE-134",
        title="Use of externally-controlled format string",
        pattern=_re(
            r"\b(printf|syslog\s*\(\s*\w+\s*,|fprintf\s*\(\s*\w+\s*,|snprintf\s*\([^,]+,[^,]+,)\s*\(?\s*[A-Za-z_][\w\->.\[\]]*\s*\)"
        ),
        problem=(
            "The format argument is a variable. If its contents can come from outside, %n and %s "
            "in it read or write memory."
        ),
        fix='Pass a literal format and the data as an argument: printf("%s", msg).',
        example=("printf(msg);", 'printf("%s", msg);'),
    ),
    Template(
        id="allocation-size-arithmetic",
        cwe="CWE-190",
        title="Integer overflow or wraparound",
        # `*` after `sizeof` or `(` is a dereference (`sizeof *p`, `sizeof(*p)`), not a product.
        pattern=_re(
            r"\b(malloc|realloc|alloca)\s*\([^;]*(?<!sizeof)(?<!sizeof )(?<!\()[*+][^;]*\)"
        ),
        problem=(
            "The allocation size is computed. If the multiplication or addition wraps, the buffer "
            "is smaller than the code later assumes."
        ),
        fix=(
            "Check for overflow before computing the size (e.g. count > SIZE_MAX / elem), or use "
            "calloc(count, elem), which checks."
        ),
        example=(
            "p = malloc(count * sizeof *p);",
            "if (count > SIZE_MAX / sizeof *p) return NULL;\np = malloc(count * sizeof *p);",
        ),
    ),
    Template(
        id="realloc-same-pointer",
        cwe="CWE-401",
        title="Missing release of memory after effective lifetime",
        pattern=_re(r"\b(\w+)\s*=\s*(\([^)]*\)\s*)?realloc\s*\(\s*\1\s*,"),
        problem=(
            "If realloc() fails it returns NULL and the original block is still allocated — but "
            "its only pointer was just overwritten."
        ),
        fix="Assign to a temporary, check it, then replace the original.",
        example=(
            "buf = realloc(buf, n);",
            "char *tmp = realloc(buf, n);\nif (tmp == NULL) { free(buf); return -1; }\nbuf = tmp;",
        ),
    ),
    Template(
        id="bounded-copy-termination",
        cwe="CWE-170",
        title="Improper null termination",
        pattern=_re(r"\bstrncpy\s*\("),
        problem=(
            "strncpy() does not terminate the destination when the source is as long as the limit."
        ),
        fix="Terminate explicitly after the call, or use snprintf().",
        example=(
            "strncpy(dst, src, sizeof dst);",
            "strncpy(dst, src, sizeof dst - 1);\ndst[sizeof dst - 1] = '\\0';",
        ),
    ),
    Template(
        id="signed-length",
        cwe="CWE-681",
        title="Incorrect conversion between numeric types",
        pattern=_re(r"\b(int|short|long|char)\s+\w+\s*=\s*(strlen|sizeof|wcslen)\b"),
        problem=(
            "A size_t length is stored in a signed type. Large values turn negative, and a later "
            "size check can pass when it should fail."
        ),
        fix="Keep lengths in size_t end to end.",
        example=("int n = strlen(src);", "size_t n = strlen(src);"),
    ),
    Template(
        id="command-injection",
        cwe="CWE-78",
        title="OS command injection",
        pattern=_re(r"\b(system|popen)\s*\("),
        problem="The string is run by a shell. Any part built from input can add commands.",
        fix=(
            "Use an exec-family call with an argument vector, and never pass input through a "
            "shell."
        ),
        example=("system(cmd);", "execvp(argv[0], argv);  /* after fork(), no shell involved */"),
    ),
)


_FREE = re.compile(r"\bfree\s*\(\s*([A-Za-z_]\w*(?:->\w+|\.\w+)*)\s*\)")
_ALLOC = re.compile(
    r"\b([A-Za-z_]\w*(?:->\w+|\.\w+)*)\s*=\s*(\([^)]*\)\s*)?(malloc|calloc|realloc)\s*\("
)


def _flow_findings(lines: list[str]) -> list[dict[str, Any]]:
    """Patterns that need more than one line: use after free, double free, unchecked alloc.

    Deliberately crude — names only, no aliasing, no control flow — the same caveat the
    project's own DFG carries (``Memory.md`` §7 #19). A match is a place to look.

    Args:
        lines: Normalized source lines.

    Returns:
        ``[{"line", "template", "related_line"}]``.
    """
    found: list[dict[str, Any]] = []
    for number, text in enumerate(lines, start=1):
        for freed in _FREE.findall(text):
            use = re.compile(rf"(?<![\w>.]){re.escape(freed)}(?![\w])")
            reassign = re.compile(rf"(?<![\w>.]){re.escape(freed)}\s*=(?!=)")
            for later, later_text in enumerate(lines[number:], start=number + 1):
                if reassign.search(later_text):
                    break
                if _FREE.search(later_text) and freed in _FREE.findall(later_text):
                    found.append({"line": later, "template": "double-free", "related_line": number})
                    break
                if use.search(later_text):
                    found.append(
                        {"line": later, "template": "use-after-free", "related_line": number}
                    )
                    break
        for match in _ALLOC.finditer(text):
            target = match.group(1)
            guard = re.compile(
                rf"(!\s*{re.escape(target)}\b|\b{re.escape(target)}\s*[!=]=\s*NULL|NULL\s*[!=]=\s*{re.escape(target)}\b|if\s*\(\s*{re.escape(target)}\s*\))"
            )
            if not any(guard.search(other) for other in lines[number - 1 :]):
                found.append(
                    {"line": number, "template": "unchecked-allocation", "related_line": None}
                )
    return found


FLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "use-after-free": {
        "cwe": "CWE-416",
        "title": "Use after free",
        "problem": (
            "This line uses a pointer that was freed earlier (line {related}) and not reassigned "
            "in between."
        ),
        "fix": (
            "Set the pointer to NULL immediately after free(), and make sure no path reaches a use"
            " afterwards."
        ),
        "example": ("free(p);\n...\np->len = 0;", "free(p);\np = NULL;"),
    },
    "double-free": {
        "cwe": "CWE-415",
        "title": "Double free",
        "problem": (
            "This pointer was already freed on line {related} and not reassigned in between."
        ),
        "fix": (
            "Free each allocation on exactly one path; set the pointer to NULL after freeing it."
        ),
        "example": (
            "free(p);\n...\nfree(p);",
            "free(p);\np = NULL;\n...\nfree(p);  /* now a no-op */",
        ),
    },
    "unchecked-allocation": {
        "cwe": "CWE-476",
        "title": "NULL pointer dereference",
        "problem": (
            "The result of this allocation is never compared against NULL in the rest of the "
            "function."
        ),
        "fix": "Check the result before its first use and handle the failure.",
        "example": (
            "p = malloc(n);\np[0] = 0;",
            "p = malloc(n);\nif (p == NULL) return -ENOMEM;\np[0] = 0;",
        ),
    },
}


_ADVICE_ALIASES = {
    "format-string-printf": "format-string",
    "format-string-stream": "format-string",
    "format-string-snprintf": "format-string",
}
"""The released explainer splits one weakness — a variable used as the format — by the call
it sits in; the fix is the same for all three."""

NO_ADVICE = frozenset({"pointer-offset-dereference", "array-index-write", "array-index-read"})
"""Patterns of the released explainer this library has no fix for. An index or an offset
is bounded somewhere else in the function, in a way a one-line template cannot write; a
card for one of these carries the explanation and no fix, rather than a generic one."""


def advice_for(pattern_id: str) -> dict[str, Any] | None:
    """The usual fix for a pattern the released explainer matched.

    The explainer (``vulnloc.explanation.templates``) says why a line matches; this
    library says how such a line is usually fixed. They are joined on the pattern's id, so
    the text of the line is matched once, by the explainer.

    Args:
        pattern_id: The explainer's template id.

    Returns:
        ``{fix, example: {before, after}}``, or ``None`` for a pattern in :data:`NO_ADVICE`
        or one this library does not know.
    """
    key = _ADVICE_ALIASES.get(pattern_id, pattern_id)
    for template in TEMPLATES:
        if template.id == key:
            return {
                "fix": template.fix,
                "example": {"before": template.example[0], "after": template.example[1]},
            }
    spec = FLOW_TEMPLATES.get(key)
    if spec is not None:
        return {
            "fix": spec["fix"],
            "example": {"before": spec["example"][0], "after": spec["example"][1]},
        }
    return None


def match_templates(lines: list[str]) -> dict[int, list[dict[str, Any]]]:
    """Every template match in a function, keyed by 1-indexed line.

    Args:
        lines: Normalized source lines.

    Returns:
        ``line -> [{id, cwe, title, problem, fix, example: {before, after}}]``.
    """
    matches: dict[int, list[dict[str, Any]]] = {}
    for number, text in enumerate(lines, start=1):
        stripped = text.strip()
        if stripped.startswith(("//", "/*", "*", "#")):
            continue
        for template in TEMPLATES:
            if template.pattern.search(text):
                matches.setdefault(number, []).append(
                    {
                        "id": template.id,
                        "cwe": template.cwe,
                        "title": template.title,
                        "problem": template.problem,
                        "fix": template.fix,
                        "example": {"before": template.example[0], "after": template.example[1]},
                    }
                )
    for finding in _flow_findings(lines):
        spec = FLOW_TEMPLATES[finding["template"]]
        matches.setdefault(finding["line"], []).append(
            {
                "id": finding["template"],
                "cwe": spec["cwe"],
                "title": spec["title"],
                "problem": spec["problem"].format(related=finding["related_line"]),
                "fix": spec["fix"],
                "example": {"before": spec["example"][0], "after": spec["example"][1]},
                "related_line": finding["related_line"],
            }
        )
    return matches
