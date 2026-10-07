"""The deterministic template explainer — Phase 7a's baseline (FR-13, FR-14, FR-16).

One flagged line in, a few fixed-form sentences out:

1. **What the line's text matches**, from the template library — or, when nothing matches,
   a plain statement that no cause is named.
2. **What the model said**: the line's rank and calibrated score.
3. **Where the model looked**: the other lines its cross-attention weighted most.
4. **What flows into the line**, from the static data-flow graph.

Sentences 2 and 3 are the model's; 1 and 4 are static analysis of the function and say so
in their wording ("Matches the pattern", "Name-based data flow"). Nothing is generated, so
the same inputs give the same text, and every number and line in it can be traced to a
field of the :class:`~vulnloc.explanation.types.Evidence` it was built from — which is
what lets ``explanation/verifier.py`` check the text instead of trusting it.

The explainer refuses an unflagged line outright (FR-15). The verifier checks again.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from vulnloc.explanation.registry import register_explainer
from vulnloc.explanation.templates import NO_PATTERN, TemplateMatch, match_lines
from vulnloc.explanation.types import Evidence, Explanation, Prediction
from vulnloc.utils.exceptions import ExplanationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vulnloc.representation.units import CodeUnit

__all__ = ["UNCLASSIFIED", "TemplateExplainer", "join_numbers"]

UNCLASSIFIED = "Unclassified"
"""The label of a flagged line no template matched."""

_MATCH_CACHE_UNITS = 8


@lru_cache(maxsize=_MATCH_CACHE_UNITS)
def _matches_for(lines: tuple[str, ...]) -> dict[int, TemplateMatch]:
    """Template matches for a unit, memoised: a unit is explained once per flagged line."""
    return match_lines(lines)


def join_numbers(numbers: Sequence[int]) -> str:
    """``[3]`` -> ``"3"``, ``[3, 7]`` -> ``"3 and 7"``, ``[3, 7, 9]`` -> ``"3, 7 and 9"``.

    Args:
        numbers: At least one number.

    Returns:
        The list as prose.
    """
    words = [str(number) for number in numbers]
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} and {words[-1]}"


@register_explainer("template")
class TemplateExplainer:
    """Template + evidence explanation of one flagged line.

    Args:
        max_context_lines: How many attended lines the text names.

    Raises:
        ExplanationError: If ``max_context_lines`` is negative.
    """

    def __init__(self, *, max_context_lines: int) -> None:
        if max_context_lines < 0:
            raise ExplanationError(
                "max_context_lines cannot be negative", max_context_lines=max_context_lines
            )
        self.max_context_lines = max_context_lines

    def explain(self, unit: CodeUnit, pred: Prediction, evidence: Evidence) -> Explanation:
        """Explain the line ``evidence`` is about.

        Args:
            unit: The unit, as the model saw it.
            pred: Its calibrated scores.
            evidence: Evidence for one flagged line of it.

        Returns:
            The explanation. Its ``line`` is ``evidence.line`` and is the only line it
            calls flagged.

        Raises:
            ExplanationError: If the three arguments do not describe the same unit, or the
                line is not flagged (FR-15).
        """
        if not unit.unit_id == pred.unit_id == evidence.unit_id:
            raise ExplanationError(
                "unit, prediction and evidence must describe one unit",
                unit_id=unit.unit_id,
                prediction_unit=pred.unit_id,
                evidence_unit=evidence.unit_id,
            )
        line = evidence.line
        if not pred.is_flagged(line):
            raise ExplanationError(
                "refusing to explain a line that was not scored above the threshold (FR-15)",
                unit_id=unit.unit_id,
                line=line,
                threshold=pred.threshold,
            )

        match = _matches_for(unit.lines).get(line)
        sentences: list[str] = []
        cited_lines: list[int] = []
        cited_symbols: list[str] = []

        if match is None:
            label, cwe, template_id = UNCLASSIFIED, "", NO_PATTERN
            sentences.append(
                "No known weakness pattern matches this line's text, so no cause is named."
            )
        else:
            label, cwe, template_id = match.template.label, match.template.cwe, match.template.id
            sentences.append(f"Matches the pattern '{label}' ({cwe}): {match.reason}")
            cited_symbols.extend(match.symbols)
            if match.related_line is not None:
                cited_lines.append(match.related_line)

        sentences.append(
            f"The model ranks line {line} at position {pred.rank(line)} of "
            f"{len(pred.ranking)} scored lines (calibrated score {pred.score(line):.2f}, "
            f"threshold {pred.threshold:.2f})."
        )

        context: list[int] = []
        for entry in evidence.attended:
            if entry.line != line and entry.line not in context:
                context.append(entry.line)
        context = context[: self.max_context_lines]
        if context:
            noun = "line" if len(context) == 1 else "lines"
            sentences.append(
                f"Its cross-attention puts most weight on {noun} {join_numbers(context)}."
            )
            cited_lines.extend(context)

        hops = [number for number in evidence.dataflow_path if number != line]
        if hops:
            noun = "line" if len(hops) == 1 else "lines"
            quoted = ", ".join(f"`{symbol}`" for symbol in dict.fromkeys(evidence.dataflow_symbols))
            sentences.append(
                f"Name-based data flow reaches it through {noun} {join_numbers(hops)} ({quoted})."
            )
            cited_lines.extend(hops)
            cited_symbols.extend(evidence.dataflow_symbols)

        return Explanation(
            unit_id=unit.unit_id,
            line=line,
            label=label,
            cwe=cwe,
            text=" ".join(sentences),
            template_id=template_id,
            cited_lines=tuple(dict.fromkeys(cited_lines)),
            cited_symbols=tuple(dict.fromkeys(cited_symbols)),
        )
