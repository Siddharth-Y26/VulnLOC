"""Per-line whitespace normalization for the L1 source text (ADR-006).

**This exists to remove a label leak, not to tidy the code.**

The LineVul release of Big-Vul strips leading whitespace from the lines of
``processed_func`` but re-inserts the flagged lines verbatim from its ``flaw_line``
column, indentation intact. The label is therefore encoded in the formatting of the
input, and a one-line rule — *"does this line begin with a space or tab?"* — recovers it
with **precision 1.0000 and zero false positives** on all four partitions
(recall 0.930–0.986). Trailing whitespace is a second, smaller tell at precision 1.0000.

A model trained on the raw text learns the formatting, not the vulnerability. That is
exactly what happened in run ``1843412a…``: it matched the indentation rule to three
decimal places and its held-out metrics were flat from the first epoch. See ADR-006 and
``Memory.md`` §1.10.

:func:`normalize_source_whitespace` strips both ends of every line. It is a **line-count
preserving** transform by construction — the split and the join use the same separator —
so every 1-indexed line label stays attached to the line it labelled. That property is
the reason this is safe to apply at L1 rather than being a data-cleaning pass that would
invalidate the split fingerprint (``Rules.md``: never break the line mapping).

After the transform no surface feature we measured carries signal above the base rate:
tabs, doubled inner spaces and line length all land at or below the fraction of lines
that are flagged.
"""

from __future__ import annotations

from vulnloc.representation.line_map import LINE_SEPARATOR, split_lines

__all__ = ["normalize_source_whitespace"]

# The characters the leak is carried in. Deliberately not `str.strip()`'s default set:
# stripping form feeds or vertical tabs would change what Tree-sitter sees for reasons
# unrelated to the leak, and `Memory.md` §7 records how expensive surprising whitespace
# handling has already been here.
_EDGE_WHITESPACE = " \t\r\v\f"


def normalize_source_whitespace(source: str) -> str:
    """Strip leading and trailing whitespace from every line of ``source``.

    The line count is preserved exactly, so labels indexed against the original text
    remain valid against the result.

    Args:
        source: Raw function text.

    Returns:
        The text with each line stripped of edge whitespace.
    """
    return LINE_SEPARATOR.join(line.strip(_EDGE_WHITESPACE) for line in split_lines(source))
