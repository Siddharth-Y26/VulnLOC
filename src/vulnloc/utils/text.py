"""Label-blind normalization of corpus source text (ADR-006; comment stripping 2026-10-07).

**This exists to remove label leaks, not to tidy the code.**

The LineVul release of Big-Vul strips leading whitespace from the lines of
``processed_func`` but re-inserts the flagged lines verbatim from its ``flaw_line``
column, indentation intact. The label is therefore encoded in the formatting of the
input, and a one-line rule — *"does this line begin with a space or tab?"* — recovers it
with **precision 1.0000 and zero false positives** on all six partitions
(recall 0.827–0.986). Trailing whitespace is a second tell at precision 1.0000.

A model trained on the raw text learns the formatting, not the vulnerability. That is
exactly what happened in run ``1843412a...``: it matched the indentation rule to three
decimal places and its held-out metrics were flat from the first epoch. See ADR-006 and
``Memory.md`` §1.10.

**Why this lives in ``utils`` rather than ``representation``.** The fix is applied at
**L0**, in the loader, so that ``source`` never carries the defect anywhere — not into
the function hashes, not into the shards, not into the split. ``Architecture.md`` §3
forbids L0 from importing L1, and ``utils`` sits below every layer, so this is the only
home from which both the loader and :class:`~vulnloc.representation.units.CodeUnit` can
reach it.

:func:`normalize_source_whitespace` strips both ends of every line. It is a **line-count
preserving** transform by construction — the split and the join use the same separator —
so every 1-indexed line label stays attached to the line it labelled (``Rules.md``: never
break the line mapping). It is also **idempotent**, which is what lets L1 keep applying it
as defence in depth over text L0 has already cleaned.

After the transform no surface feature we measured carries signal above the base rate:
tabs, doubled inner spaces and line length all land at or below the fraction of lines
that are flagged. Verified corpus-wide by ``scripts/audit_label_leakage.py``.

**The second leak, and :func:`strip_comments`.** Whitespace was not the only thing the
pasted-back flagged lines carry. A flagged line that ended a block comment has lost the
slash of its terminator — ``foo();  /* note *`` — so "ends a comment with a bare ``*``" is
flagged at precision 0.97–1.00 on ``train``, ``val`` and ``test``, and "ends with ``*/``"
is never flagged anywhere (``Memory.md`` §1.25). On ``clean_eval`` the same defect takes
another form: almost no flagged line carries a comment at all. Both are about comments, so
the label-blind fix is to remove comments from every line, flagged or not.
:func:`strip_comments` does that, preserves the line count, and is **off unless asked
for**: every number logged before it existed was measured with comments in place, and
turning it on silently would make those numbers unreproducible.
"""

from __future__ import annotations

import re

__all__ = ["ends_comment_without_slash", "normalize_source_whitespace", "strip_comments"]

# The canonical separator. Kept local rather than imported from
# `representation.line_map`, which is L1 and therefore off-limits to L0's callers.
# Both split on "\n" and only "\n"; no corpus unit contains a CR at all (measured across
# all 188,636 functions), so there is no line-ending case to reconcile here.
_LINE_SEPARATOR = "\n"

# The characters the leak is carried in. Deliberately not `str.strip()`'s default set:
# stripping form feeds or vertical tabs as *content* would change what Tree-sitter sees
# for reasons unrelated to the leak, and `Memory.md` §7 records how expensive surprising
# whitespace handling has already been here. They are included as edge characters only.
_EDGE_WHITESPACE = " \t\r\v\f"

# Text, then exactly one `*`, at the end of a line. `/**` and `****` do not match: a
# doc-comment opener and a banner end in `*` in any corpus, flagged or not.
_BARE_STAR_END = re.compile(r"[^\s*/]\s*\*$")

# Inside a block comment the same thing with any number of stars: `* note **` is `* note **/`
# without its slash. A line of stars alone stays ambiguous and does not close the comment.
_OPEN_BLOCK_LOST_END = re.compile(r"[^\s*/]\s*\*+$")

_QUOTES = "\"'"


def normalize_source_whitespace(source: str) -> str:
    """Strip leading and trailing whitespace from every line of ``source``.

    The line count is preserved exactly, so labels indexed against the original text
    remain valid against the result. The function is idempotent.

    Args:
        source: Raw function text.

    Returns:
        The text with each line stripped of edge whitespace.
    """
    return _LINE_SEPARATOR.join(
        line.strip(_EDGE_WHITESPACE) for line in source.split(_LINE_SEPARATOR)
    )


def ends_comment_without_slash(line: str) -> bool:
    """Whether a line ends a block comment with ``*`` where ``*/`` belongs.

    This is the tell of ``Memory.md`` §1.25, as a predicate on one line: it carries a
    comment — it contains ``/*`` or starts with a continuation ``*`` — and ends with text
    followed by exactly one bare ``*``.

    Args:
        line: One source line.

    Returns:
        ``True`` for ``x = 1;  /* note *`` and for a continuation line ``* note *``.
    """
    stripped = line.strip(_EDGE_WHITESPACE)
    return bool(_BARE_STAR_END.search(stripped)) and ("/*" in stripped or stripped[:1] == "*")


def _is_word(char: str) -> bool:
    return char.isalnum() or char == "_"


def _ends_in_space(kept: list[str]) -> bool:
    return bool(kept) and kept[-1][-1:] in _EDGE_WHITESPACE and kept[-1] != ""


def _strip_line(line: str, *, in_block: bool) -> tuple[str, bool]:
    """Remove the comments from one line.

    Args:
        line: One source line.
        in_block: Whether a block comment opened on an earlier line is still open.

    Returns:
        ``(code, in_block)`` — the line without its comments, edge whitespace stripped, and
        whether a block comment is open at the end of it.
    """
    kept: list[str] = []
    quote = ""
    # `gap`: a comment was just removed and the next code character has not arrived yet.
    # `space_before` / `space_after`: whether whitespace stood on each side of it.
    gap = space_before = space_after = False
    index, length = 0, len(line)
    while index < length:
        char = line[index]
        if in_block:
            end = line.find("*/", index)
            if end == -1:
                # A terminator that lost its slash still ends the comment; otherwise the
                # code that follows on later lines would be swallowed with it.
                in_block = not _OPEN_BLOCK_LOST_END.search(line.rstrip(_EDGE_WHITESPACE))
                break
            in_block = False
            if not gap:
                gap, space_before, space_after = True, _ends_in_space(kept), False
            index = end + 2
            continue
        if quote:
            kept.append(char)
            if char == "\\" and index + 1 < length:
                kept.append(line[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        pair = line[index : index + 2]
        if pair == "//":
            break
        if pair == "/*":
            in_block = True
            if not gap:
                gap, space_before, space_after = True, _ends_in_space(kept), False
            index += 2
            continue
        if gap:
            if char in _EDGE_WHITESPACE:
                space_after = True
                index += 1
                continue
            # Exactly one space where the comment sat between spaced tokens, or where it
            # was the only thing keeping two identifiers apart — and never two.
            tail = "".join(kept).rstrip(_EDGE_WHITESPACE)
            kept = [tail]
            glued = bool(tail) and _is_word(tail[-1]) and _is_word(char)
            if tail and ((space_before and space_after) or glued):
                kept.append(" ")
            gap = False
        if char in _QUOTES:
            quote = char
        kept.append(char)
        index += 1
    # A string or character literal does not span lines; `quote` simply dies here.
    return "".join(kept).strip(_EDGE_WHITESPACE), in_block


def strip_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments from every line, keeping the line count.

    Label-blind by construction: it reads only the text. A line that held nothing but a
    comment becomes empty and **stays in place**, so every 1-indexed line label remains
    attached to the line it labelled. The result is also whitespace-normalized, and the
    function is idempotent.

    Three cases a naive pass gets wrong, each of which occurs in the corpus:

    * comment markers inside a string or character literal are text, not comments
      (``memcmp(ptr, "/* XPM */", 9)``);
    * a block comment whose terminator lost its slash (``/* note *``, ``Memory.md`` §1.25)
      still ends on that line — without this rule it would swallow the code after it;
    * removing a comment that was the only thing between two identifiers leaves one space
      (``int/**/x`` becomes ``int x``), and never leaves doubled spaces behind, which would
      be a new formatting tell.

    Args:
        source: Function text.

    Returns:
        The text with comments removed and the same number of lines.
    """
    lines: list[str] = []
    in_block = False
    for line in source.split(_LINE_SEPARATOR):
        code, in_block = _strip_line(line, in_block=in_block)
        lines.append(code)
    return _LINE_SEPARATOR.join(lines)
