"""The deterministic template explainer — Phase 7a's baseline (FR-13, FR-14, FR-16).

One flagged line in, a few fixed-form sentences out:

1. **What the line's text matches**, from the template library — or, when nothing matches,
   a plain statement that no cause is named. A line that carries the corpus's
   comment-terminator tell (``Memory.md`` §1.25) is told so, pattern or no pattern.
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
from vulnloc.utils.text import ends_comment_without_slash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vulnloc.representation.units import CodeUnit

__all__ = [
    "FIXED_SENTENCES",
    "FORMATTING_ARTIFACT",
    "FORMATTING_ARTIFACT_LABEL",
    "UNCLASSIFIED",
    "TemplateExplainer",
    "join_numbers",
]

UNCLASSIFIED = "Unclassified"
"""The label of a flagged line no template matched."""

FORMATTING_ARTIFACT = "formatting-artifact"
"""Template id of a flagged line that matches no weakness pattern and carries the
comment-terminator tell of ``Memory.md`` §1.25."""

FORMATTING_ARTIFACT_LABEL = "Formatting artifact"

_ARTIFACT_SENTENCE = (
    "This line ends a comment with a bare * where */ belongs. That is a formatting artifact "
    "of the dataset which marks labelled lines, so this score may rest on the formatting "
    "rather than on the code."
)
"""Said whenever the tell is on the line, pattern or no pattern. It is the one case where
the explainer can state *why the model scored the line* with measured backing: the nine
D9.7 checkpoints rank such a line first in most units that have one."""

_NO_PATTERN_SENTENCE = "No known weakness pattern matches this line's text, so no cause is named."
_NO_PATTERN_ARTIFACT_SENTENCE = "No known weakness pattern matches this line's text."
_PATTERN_FORMAT = "Matches the pattern '{label}' ({cwe}): {reason}"
_RANK_FORMAT = (
    "The model ranks line {line} at position {rank} of {scored} scored lines "
    "(calibrated score {score:.2f}, threshold {threshold:.2f})."
)
_ATTENTION_FORMAT = "Its cross-attention puts most weight on {noun} {lines}."
_DATAFLOW_FORMAT = "Name-based data flow reaches it through {noun} {lines} ({symbols})."

FIXED_SENTENCES: dict[str, str] = {
    "pattern (wraps a template sentence)": _PATTERN_FORMAT,
    "no pattern": _NO_PATTERN_SENTENCE,
    "no pattern, on a line carrying the formatting tell": _NO_PATTERN_ARTIFACT_SENTENCE,
    "formatting tell (added whenever the tell is on the line)": _ARTIFACT_SENTENCE,
    "rank and score": _RANK_FORMAT,
    "attention": _ATTENTION_FORMAT,
    "data flow": _DATAFLOW_FORMAT,
}
"""Every sentence form the explainer emits besides the template library's own — the whole
of what a reviewer has to read (``Rules.md`` §7.3). ``{...}`` are filled from the
prediction and the evidence."""

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
        artifact = ends_comment_without_slash(unit.lines[line - 1])
        sentences: list[str] = []
        cited_lines: list[int] = []
        cited_symbols: list[str] = []

        if match is None and artifact:
            label, cwe, template_id = FORMATTING_ARTIFACT_LABEL, "", FORMATTING_ARTIFACT
            sentences.append(_NO_PATTERN_ARTIFACT_SENTENCE)
        elif match is None:
            label, cwe, template_id = UNCLASSIFIED, "", NO_PATTERN
            sentences.append(_NO_PATTERN_SENTENCE)
        else:
            label, cwe, template_id = match.template.label, match.template.cwe, match.template.id
            sentences.append(_PATTERN_FORMAT.format(label=label, cwe=cwe, reason=match.reason))
            cited_symbols.extend(match.symbols)
            if match.related_line is not None:
                cited_lines.append(match.related_line)
        if artifact:
            sentences.append(_ARTIFACT_SENTENCE)

        sentences.append(
            _RANK_FORMAT.format(
                line=line,
                rank=pred.rank(line),
                scored=len(pred.ranking),
                score=pred.score(line),
                threshold=pred.threshold,
            )
        )

        context: list[int] = []
        for entry in evidence.attended:
            if entry.line != line and entry.line not in context:
                context.append(entry.line)
        context = context[: self.max_context_lines]
        if context:
            noun = "line" if len(context) == 1 else "lines"
            sentences.append(_ATTENTION_FORMAT.format(noun=noun, lines=join_numbers(context)))
            cited_lines.extend(context)

        hops = [number for number in evidence.dataflow_path if number != line]
        if hops:
            noun = "line" if len(hops) == 1 else "lines"
            quoted = ", ".join(f"`{symbol}`" for symbol in dict.fromkeys(evidence.dataflow_symbols))
            sentences.append(
                _DATAFLOW_FORMAT.format(noun=noun, lines=join_numbers(hops), symbols=quoted)
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
