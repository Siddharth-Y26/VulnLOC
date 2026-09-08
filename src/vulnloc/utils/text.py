"""Per-line whitespace normalization for corpus source text (ADR-006).

**This exists to remove a label leak, not to tidy the code.**

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
"""

from __future__ import annotations

__all__ = ["normalize_source_whitespace"]

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
