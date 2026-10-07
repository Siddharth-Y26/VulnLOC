"""The template library: weakness patterns over a line's text, and the sentence each yields.

**A template is not the model.** The model decides which lines are flagged and gives no
reason; a template looks at the text of a flagged line and, if it recognises a well-known
C/C++ weakness pattern, names it. The category an explanation carries is therefore
*template-derived* — the project owner deferred a learned type classifier on 2026-10-07
(open decision Q5) — and an explanation says so by opening with "Matches the pattern".

Three rules the library keeps, each because the alternative reads like an explanation and
is not one:

* **A sentence states something checkable about the line itself** — "the length is not
  derived from the destination's size on this line" — never a conclusion the pattern
  cannot support ("this overflows").
* **No pattern, no cause.** A flagged line nothing matches is reported as exactly that
  (:data:`NO_PATTERN`); it is never filled with generic advice.
* **No attack instructions** (``Rules.md`` §7.2, §7.3). A sentence explains why a
  construct is risky. None says how to trigger or use it, and
  ``tests/unit/test_explanation_templates.py`` scans every sentence for that vocabulary.

Every CWE identifier is MITRE's and is the entry the pattern is *conventionally filed
under* — it is not a claim about the CVE the function was labelled for.

The single-line patterns started as ``frontend/fix_templates.py`` (2026-09-15), which
remains the Findings tab's own copy with fix advice attached; this module carries no fix
advice, because an explanation says why a line was flagged, not how to change it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FLOW_TEMPLATES",
    "LINE_TEMPLATES",
    "NO_PATTERN",
    "ExplanationTemplate",
    "TemplateMatch",
    "all_templates",
    "match_lines",
]

NO_PATTERN = "no-pattern"
"""Template id of the fallback: the line is flagged and no pattern names a cause."""

_NAME = r"[A-Za-z_]\w*(?:(?:->|\.)\w+)*"
_VARIABLE = r"[A-Za-z_][\w\->.\[\]]*"


@dataclass(frozen=True)
class ExplanationTemplate:
    """One weakness pattern and the sentence it yields.

    Attributes:
        id: Stable key.
        label: Vulnerability category shown to the reader.
        cwe: MITRE CWE identifier the pattern is conventionally filed under.
        reason: The sentence, with ``{group}`` placeholders filled from the match and
            ``{related}`` from a flow template's other line.
        pattern: Regex over one normalized source line. ``None`` for a flow template,
            which is matched by :func:`_flow_matches` over the whole function.
    """

    id: str
    label: str
    cwe: str
    reason: str
    pattern: re.Pattern[str] | None = None


@dataclass(frozen=True)
class TemplateMatch:
    """A template that matched one line.

    Attributes:
        template: The template.
        line: The 1-indexed line it matched.
        reason: The rendered sentence.
        symbols: Source fragments the sentence quotes; each occurs on ``line`` or on
            ``related_line``.
        related_line: A second line the sentence refers to (flow templates), or ``None``.
    """

    template: ExplanationTemplate
    line: int
    reason: str
    symbols: tuple[str, ...]
    related_line: int | None = None


def _template(key: str, label: str, cwe: str, pattern: str, reason: str) -> ExplanationTemplate:
    return ExplanationTemplate(
        id=key, label=label, cwe=cwe, reason=reason, pattern=re.compile(pattern)
    )


LINE_TEMPLATES: tuple[ExplanationTemplate, ...] = (
    _template(
        "gets",
        "Buffer overflow",
        "CWE-242",
        r"\b(?P<call>gets)\s*\(",
        "`{call}()` cannot be told the size of its buffer, so input longer than the buffer is "
        "written past its end.",
    ),
    _template(
        "unbounded-string-copy",
        "Buffer overflow",
        "CWE-120",
        r"\b(?P<call>strcpy|strcat|wcscpy|wcscat)\s*\(",
        "`{call}()` copies up to the source's terminator and never consults the size of the "
        "destination.",
    ),
    _template(
        "unbounded-format-write",
        "Buffer overflow",
        "CWE-120",
        r"\b(?P<call>sprintf|vsprintf)\s*\(",
        "`{call}()` writes as much as the format produces, whatever the size of the destination "
        "buffer.",
    ),
    _template(
        "memcpy-length",
        "Out-of-bounds write",
        "CWE-787",
        # A length that is not a sizeof expression is the case worth a look.
        r"\b(?P<call>memcpy|memmove|memset)\s*\((?![^;]*,\s*sizeof\b)[^;]*\)",
        "The length passed to `{call}()` is not derived from the destination's size on this "
        "line, so nothing here bounds the write.",
    ),
    _template(
        "scanf-string",
        "Buffer overflow",
        "CWE-120",
        r"\b(?P<call>f?scanf|sscanf)\s*\([^;]*\"[^\"]*%s",
        "`{call}()` is given a `%s` conversion with no field width, which reads a token of any "
        "length into a fixed buffer.",
    ),
    _template(
        "format-string-printf",
        "Format string",
        "CWE-134",
        rf"\b(?P<call>printf)\s*\(\s*(?P<format>{_VARIABLE})\s*\)",
        "The format argument of `{call}()` is the variable `{format}` rather than a literal, so "
        "its contents decide how the remaining arguments are read.",
    ),
    _template(
        "format-string-stream",
        "Format string",
        "CWE-134",
        rf"\b(?P<call>fprintf|syslog|dprintf)\s*\(\s*[^,()]+,\s*(?P<format>{_VARIABLE})\s*\)",
        "The format argument of `{call}()` is the variable `{format}` rather than a literal, so "
        "its contents decide how the remaining arguments are read.",
    ),
    _template(
        "format-string-snprintf",
        "Format string",
        "CWE-134",
        rf"\b(?P<call>snprintf)\s*\([^,()]+,[^,()]+,\s*(?P<format>{_VARIABLE})\s*\)",
        "The format argument of `{call}()` is the variable `{format}` rather than a literal, so "
        "its contents decide how the remaining arguments are read.",
    ),
    _template(
        "allocation-size-arithmetic",
        "Integer overflow",
        "CWE-190",
        # `*` after `sizeof` or `(` is a dereference (`sizeof *p`, `sizeof(*p)`), not a product.
        r"\b(?P<call>malloc|realloc|alloca)\s*\([^;]*(?<!sizeof)(?<!sizeof )(?<!\()[*+][^;]*\)",
        "The size passed to `{call}()` is computed by arithmetic on this line; if it wraps, the "
        "allocation is smaller than later code assumes.",
    ),
    _template(
        "realloc-same-pointer",
        "Memory leak",
        "CWE-401",
        r"\b(?P<ptr>\w+)\s*=\s*(?:\([^)]*\)\s*)?(?P<call>realloc)\s*\(\s*(?P=ptr)\s*,",
        "If `{call}()` fails it returns NULL while the original block stays allocated, and its "
        "only pointer `{ptr}` is overwritten here.",
    ),
    _template(
        "bounded-copy-termination",
        "Improper null termination",
        "CWE-170",
        r"\b(?P<call>strncpy)\s*\(",
        "`{call}()` leaves the destination unterminated when the source is at least as long as "
        "the limit.",
    ),
    _template(
        "signed-length",
        "Numeric conversion error",
        "CWE-681",
        r"\b(?P<type>int|short|long|char)\s+\w+\s*=\s*(?P<call>strlen|sizeof|wcslen)\b",
        "A size_t length from `{call}` is stored in a signed `{type}`; a large value turns "
        "negative and can pass a later size check.",
    ),
    _template(
        "command-injection",
        "Command injection",
        "CWE-78",
        r"\b(?P<call>system|popen)\s*\(",
        "`{call}()` hands its string to a shell, so any part of it built from input is read as "
        "shell syntax.",
    ),
    _template(
        "pointer-offset-dereference",
        "Out-of-bounds access",
        "CWE-119",
        # The `*` must be a dereference, not a product: `y * (a + b)` has an operand before it.
        r"(?:^|[=(,;!&|<>?:{}\[+\-/%]|\breturn\b)\s*"
        r"\*\s*\(\s*(?P<ptr>[A-Za-z_]\w*)\s*[+-]\s*(?P<offset>[^()]{1,40})\)",
        "`{ptr}` is dereferenced at an offset computed from `{offset}`, and nothing on this "
        "line bounds that offset.",
    ),
    _template(
        "array-index-write",
        "Out-of-bounds write",
        "CWE-787",
        rf"(?P<base>{_NAME})\s*\[\s*(?P<index>[^\[\]]*[A-Za-z_][^\[\]]{{0,40}})\]\s*"
        r"(?:[-+*/|&^]|<<|>>)?=(?!=)",
        "`{base}` is written at an index computed from `{index}`, and nothing on this line "
        "bounds that index.",
    ),
    _template(
        "array-index-read",
        "Out-of-bounds read",
        "CWE-125",
        rf"(?P<base>{_NAME})\s*\[\s*(?P<index>[^\[\]]*[A-Za-z_][^\[\]]{{0,40}})\]",
        "`{base}` is read at an index computed from `{index}`, and nothing on this line bounds "
        "that index.",
    ),
)
"""Single-line templates, most specific first: the first that matches a line wins."""

FLOW_TEMPLATES: dict[str, ExplanationTemplate] = {
    "use-after-free": ExplanationTemplate(
        id="use-after-free",
        label="Use after free",
        cwe="CWE-416",
        reason=(
            "`{ptr}` is used here after being freed on line {related}, and no line in between "
            "reassigns it."
        ),
    ),
    "double-free": ExplanationTemplate(
        id="double-free",
        label="Double free",
        cwe="CWE-415",
        reason=(
            "`{ptr}` is freed here after already being freed on line {related}, and no line in "
            "between reassigns it."
        ),
    ),
    "unchecked-allocation": ExplanationTemplate(
        id="unchecked-allocation",
        label="NULL pointer dereference",
        cwe="CWE-476",
        reason=(
            "The result of `{call}()` assigned to `{ptr}` is never compared against NULL in the "
            "rest of the function."
        ),
    ),
}
"""Templates that need more than one line. Crude by design: names only, no aliasing, no
control flow — the caveat the project's own DFG carries (``Memory.md`` §7 #19)."""

_FREE = re.compile(rf"\bfree\s*\(\s*({_NAME})\s*\)")
_ALLOC = re.compile(rf"\b({_NAME})\s*=\s*(?:\([^)]*\)\s*)?(malloc|calloc|realloc)\s*\(")

# `T name[n]` declares an array; it does not read one. `return a[i]` is not a declaration.
_ARRAY_DECLARATION = re.compile(
    r"^(?!(?:return|else|case|goto|sizeof|delete|new|throw)\b)"
    r"(?:(?:const|static|unsigned|signed|struct|volatile|extern)\s+)*"
    r"[A-Za-z_]\w*[\s*]+\**[A-Za-z_]\w*\s*\["
)

_NOT_CODE = ("//", "/*", "*", "#")


def all_templates() -> tuple[ExplanationTemplate, ...]:
    """Every template in the library, single-line ones first.

    Returns:
        The templates, for review and for tests.
    """
    return (*LINE_TEMPLATES, *FLOW_TEMPLATES.values())


def _is_code(text: str) -> bool:
    return bool(text.strip()) and not text.strip().startswith(_NOT_CODE)


def _line_match(text: str, line: int) -> TemplateMatch | None:
    """The first single-line template that matches ``text``.

    Args:
        text: One normalized source line.
        line: Its 1-indexed number.

    Returns:
        The match, or ``None``.
    """
    for template in LINE_TEMPLATES:
        assert template.pattern is not None
        if template.id == "array-index-read" and _ARRAY_DECLARATION.search(text):
            continue
        found = template.pattern.search(text)
        if found is None:
            continue
        groups = {name: value.strip() for name, value in found.groupdict().items()}
        return TemplateMatch(
            template=template,
            line=line,
            reason=template.reason.format(**groups),
            symbols=tuple(dict.fromkeys(groups.values())),
        )
    return None


def _flow_matches(lines: Sequence[str]) -> dict[int, TemplateMatch]:
    """Use after free, double free and unchecked allocation, keyed by the line they land on.

    Args:
        lines: Normalized source lines.

    Returns:
        ``line -> match``. A line that qualifies twice keeps the first found.
    """
    found: dict[int, TemplateMatch] = {}

    def record(kind: str, line: int, related: int | None, **groups: str) -> None:
        template = FLOW_TEMPLATES[kind]
        found.setdefault(
            line,
            TemplateMatch(
                template=template,
                line=line,
                reason=template.reason.format(related=related, **groups),
                symbols=tuple(groups.values()),
                related_line=related,
            ),
        )

    for number, text in enumerate(lines, start=1):
        if not _is_code(text):
            continue
        for freed in _FREE.findall(text):
            use = re.compile(rf"(?<![\w>.]){re.escape(freed)}(?![\w])")
            reassign = re.compile(rf"(?<![\w>.]){re.escape(freed)}\s*=(?!=)")
            for later, later_text in enumerate(lines[number:], start=number + 1):
                if reassign.search(later_text):
                    break
                if freed in _FREE.findall(later_text):
                    record("double-free", later, number, ptr=freed)
                    break
                if use.search(later_text):
                    record("use-after-free", later, number, ptr=freed)
                    break
        for match in _ALLOC.finditer(text):
            target, call = match.group(1), match.group(2)
            escaped = re.escape(target)
            guard = re.compile(
                rf"(!\s*{escaped}\b|\b{escaped}\s*[!=]=\s*NULL|NULL\s*[!=]=\s*{escaped}\b"
                rf"|if\s*\(\s*{escaped}\s*\))"
            )
            if not any(guard.search(other) for other in lines[number - 1 :]):
                record("unchecked-allocation", number, None, call=call, ptr=target)
    return found


def match_lines(lines: Sequence[str]) -> dict[int, TemplateMatch]:
    """The template that explains each line of a function, where one does.

    Use after free and double free outrank a single-line template on the same line: each
    needed a second line to agree before it matched, so it says more. An unchecked
    allocation ranks *below* the single-line templates — it rests on the absence of a
    check, which is weaker than a construct that is actually on the line.

    Args:
        lines: The unit's normalized source lines, line ``i`` at position ``i - 1``.

    Returns:
        ``line -> match`` for every line some template matched. Comment and preprocessor
        lines never match.
    """
    flow = _flow_matches(lines)
    matches = {
        number: found
        for number, found in flow.items()
        if found.template.id != "unchecked-allocation"
    }
    for number, text in enumerate(lines, start=1):
        if number in matches or not _is_code(text):
            continue
        found = _line_match(text, number) or flow.get(number)
        if found is not None:
            matches[number] = found
    return matches
