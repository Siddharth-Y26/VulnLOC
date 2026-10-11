"""The reader's view of one analysis: what was found, what was looked at, what was not.

``POST /analyze`` returns the ``PRD.md`` §5.4 object and nothing else, and that object has
one explanation and no room for scope. This module assembles what a person reading a
result needs beside it — and nothing the analysis did not already compute:

* a **summary** in plain words, which for an unflagged function says that nothing was
  flagged and that this is not a statement that the function is safe (``Rules.md`` §7.8);
* the **highest-ranked lines whether or not any was flagged** — the model ranks lines, and
  a ranking is still worth reading when no line reaches the threshold;
* every flagged line with its **verified** explanation and the evidence under it. A line
  whose explanation the verifier rejected is listed with no explanation, never with a
  softened one;
* the **weakness patterns that were tested**, and which of them matched. "No known pattern
  matched" then has a visible scope;
* **coverage**: the lines that were not examined, and why — past the line cap, no
  features, a parse the grammar had to repair, no data-flow graph;
* what the bundle says was **measured** for its checkpoint, when the bundle carries it.

Nothing here is a second model or a second opinion. The sentences are fixed wording
around counts; a template is never the model's reasoning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vulnloc.explanation.templates import NO_PATTERN, all_templates

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vulnloc.explanation.types import Evidence, Explanation, Prediction
    from vulnloc.representation.units import CodeUnit

__all__ = ["NOT_SAFE_SENTENCE", "TOP_RANKED", "build_report", "patterns_checked", "summarize"]

TOP_RANKED = 5
"""How many of the highest-ranked lines a report lists, flagged or not."""

NOT_SAFE_SENTENCE = "This is not a statement that the function is safe."


def patterns_checked() -> list[dict[str, str]]:
    """The weakness patterns every flagged line is tested against.

    Returns:
        One ``{id, label, cwe}`` per template, in the library's order.
    """
    return [
        {"id": template.id, "label": template.label, "cwe": template.cwe}
        for template in all_templates()
    ]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def summarize(pred: Prediction, *, matched: int, lines_scored: int) -> str:
    """The result in two or three plain sentences.

    Args:
        pred: The calibrated scores.
        matched: Flagged lines a weakness pattern matched.
        lines_scored: Lines the model was offered.

    Returns:
        Fixed wording around counts and line numbers — no claim the scores do not carry.
    """
    ranking = pred.ranking
    flagged = pred.flagged_lines
    if not ranking:
        return f"No line could be scored, so no line was flagged. {NOT_SAFE_SENTENCE}"
    if not flagged:
        top = ranking[0]
        return (
            f"No line was flagged above the threshold. {NOT_SAFE_SENTENCE} "
            f"The highest-ranked line is line {top}, with a score of {pred.score(top):.3f} "
            f"against a threshold of {pred.threshold:.3f}."
        )
    verb = "is" if len(flagged) == 1 else "are"
    pattern = (
        f"A known weakness pattern matches {matched} of them."
        if matched
        else "No known weakness pattern matches any of them, so no cause is named."
    )
    return (
        f"{len(flagged)} of the {_plural(lines_scored, 'scored line')} {verb} flagged. "
        f"The most suspicious is line {flagged[0]}. {pattern}"
    )


def _finding(
    unit: CodeUnit,
    pred: Prediction,
    line: int,
    explanation: Explanation | None,
    evidence: Evidence | None,
) -> dict[str, Any]:
    matched = explanation is not None and explanation.template_id != NO_PATTERN
    return {
        "line": line,
        "rank": pred.rank(line),
        "score": pred.score(line),
        "code": unit.lines[line - 1],
        "label": explanation.label if explanation is not None else "",
        "cwe": explanation.cwe if explanation is not None else "",
        "pattern": explanation.template_id if matched and explanation is not None else None,
        "explanation": explanation.text if explanation is not None else None,
        "attended_lines": [
            {
                "line": attended.line,
                "weight": attended.weight,
                "direction": attended.direction,
                "code": unit.lines[attended.line - 1],
            }
            for attended in (evidence.attended if evidence is not None else ())
        ],
        "dataflow_path": list(evidence.dataflow_path) if evidence is not None else [],
        "dataflow_symbols": list(evidence.dataflow_symbols) if evidence is not None else [],
        "graph_nodes": list(evidence.graph_nodes) if evidence is not None else [],
        "attention_top_tokens": list(evidence.top_tokens) if evidence is not None else [],
    }


def build_report(
    unit: CodeUnit,
    pred: Prediction,
    explanations: Mapping[int, Explanation],
    evidence: Mapping[int, Evidence],
    *,
    coverage: Mapping[str, Any],
    measured: Mapping[str, Any] | None = None,
    top_ranked: int = TOP_RANKED,
) -> dict[str, Any]:
    """Assemble the report for one analyzed unit.

    Args:
        unit: The unit, as the model saw it.
        pred: Its calibrated scores.
        explanations: ``line -> explanation``, verified ones only.
        evidence: ``line -> evidence`` for the same lines.
        coverage: What was and was not examined — passed through as given, with
            ``lines_scored`` read for the summary.
        measured: What the bundle records as measured for its checkpoint, or ``None``.
        top_ranked: How many of the highest-ranked lines to list.

    Returns:
        A JSON-serialisable mapping: ``summary``, ``threshold``, ``ranked_lines``,
        ``findings``, ``patterns``, ``coverage``, ``measured``.
    """
    flagged = pred.flagged_lines
    findings = [
        _finding(unit, pred, line, explanations.get(line), evidence.get(line)) for line in flagged
    ]
    matched = sorted({finding["pattern"] for finding in findings if finding["pattern"]})
    ranked = [
        {
            "line": line,
            "rank": position,
            "score": pred.score(line),
            "code": unit.lines[line - 1],
            "flagged": pred.is_flagged(line),
        }
        for position, line in enumerate(pred.ranking[:top_ranked], start=1)
    ]
    return {
        "summary": summarize(
            pred,
            matched=sum(1 for finding in findings if finding["pattern"]),
            lines_scored=int(coverage["lines_scored"]),
        ),
        "threshold": pred.threshold,
        "ranked_lines": ranked,
        "findings": findings,
        "patterns": {"checked": patterns_checked(), "matched": matched},
        "coverage": dict(coverage),
        "measured": dict(measured) if measured is not None else None,
    }
