"""The output contract of ``PRD.md`` §5.4, assembled from a prediction and its explanations.

One unit in, one JSON-serialisable mapping out, with **exactly** the keys the PRD lists —
:data:`CONTRACT_KEYS`, :data:`LINE_KEYS`, :data:`EVIDENCE_KEYS` — which is what Phase 7's
"output matches PRD §5.4 schema exactly" means and what a test pins. Phase 9's Pydantic
schema mirrors these constants rather than restating them.

Three fields need saying what they are here, because the PRD's example cannot:

* ``risk_score`` is the **highest calibrated line probability** in the unit. It is not a
  function-level score: unit-level aggregation was deferred in writing on 2026-10-07, and
  every model in this repository was trained on vulnerable functions only.
* ``is_vulnerable`` is "at least one line reached the threshold". ``False`` therefore
  means *no line was flagged* — **never that the function is safe** (``Rules.md`` §7.8),
  and the explanation text of such a unit says so.
* ``vulnerability_label`` is the template-derived category of the top flagged line
  (open decision Q5), ``"Unclassified"`` when no pattern matched it.

The unit's ``explanation`` and ``evidence`` are those of its top-ranked flagged line; the
other flagged lines are listed in ``vulnerable_lines`` with their scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vulnloc.utils.exceptions import ExplanationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vulnloc.explanation.types import Evidence, Explanation, Prediction
    from vulnloc.representation.units import CodeUnit

__all__ = ["CONTRACT_KEYS", "EVIDENCE_KEYS", "LINE_KEYS", "NOT_FLAGGED_TEXT", "output_contract"]

CONTRACT_KEYS = (
    "unit_id",
    "is_vulnerable",
    "risk_score",
    "vulnerability_label",
    "vulnerable_lines",
    "explanation",
    "evidence",
)
LINE_KEYS = ("line", "score", "code")
EVIDENCE_KEYS = ("graph_nodes", "dataflow_path", "attention_top_tokens")

NOT_FLAGGED_TEXT = (
    "No line was scored above the threshold. That is not evidence that the function is safe."
)


def output_contract(
    unit: CodeUnit,
    pred: Prediction,
    explanations: Mapping[int, Explanation],
    evidence: Mapping[int, Evidence],
) -> dict[str, Any]:
    """Build the §5.4 object for one unit.

    Args:
        unit: The unit.
        pred: Its calibrated scores.
        explanations: ``line -> explanation`` for its flagged lines. The top-ranked flagged
            line must be present; a flagged line whose explanation the verifier rejected is
            simply absent, and is still listed in ``vulnerable_lines``.
        evidence: ``line -> evidence``, for at least the top-ranked flagged line.

    Returns:
        A mapping with exactly :data:`CONTRACT_KEYS`.

    Raises:
        ExplanationError: If the unit has a flagged line but no explanation and evidence
            for its top-ranked one, or an explanation is offered for an unflagged line.
    """
    if pred.unit_id != unit.unit_id:
        raise ExplanationError(
            "the prediction belongs to another unit",
            unit_id=unit.unit_id,
            prediction_unit=pred.unit_id,
        )
    unflagged = sorted(line for line in explanations if not pred.is_flagged(line))
    if unflagged:
        raise ExplanationError(
            "an explanation was offered for a line that is not flagged (FR-15)",
            unit_id=unit.unit_id,
            lines=unflagged,
        )

    ranking = pred.ranking
    flagged = pred.flagged_lines
    risk = pred.score(ranking[0]) if ranking else 0.0
    lines = [
        {"line": line, "score": pred.score(line), "code": unit.lines[line - 1]} for line in flagged
    ]
    if not flagged:
        return {
            "unit_id": unit.unit_id,
            "is_vulnerable": False,
            "risk_score": risk,
            "vulnerability_label": "",
            "vulnerable_lines": lines,
            "explanation": NOT_FLAGGED_TEXT,
            "evidence": {"graph_nodes": [], "dataflow_path": [], "attention_top_tokens": []},
        }

    top = flagged[0]
    if top not in explanations or top not in evidence:
        raise ExplanationError(
            "the unit's top flagged line has no verified explanation to report",
            unit_id=unit.unit_id,
            line=top,
        )
    return {
        "unit_id": unit.unit_id,
        "is_vulnerable": True,
        "risk_score": risk,
        "vulnerability_label": explanations[top].label,
        "vulnerable_lines": lines,
        "explanation": explanations[top].text,
        "evidence": {
            "graph_nodes": list(evidence[top].graph_nodes),
            "dataflow_path": list(evidence[top].dataflow_path),
            "attention_top_tokens": list(evidence[top].top_tokens),
        },
    }
