"""The grounding verifier: an explanation is checked against the prediction, not trusted.

FR-15: *an explanation must never claim a line is vulnerable that the localization head
did not score above threshold.* The template explainer cannot do that by construction —
which is exactly why it is checked here anyway, by code that does not share its logic. The
verifier reads the finished text, and it is the gate a generative explainer (7b, deferred)
would have to pass unchanged.

What is checked, in this order, and an explanation that fails any of it is **rejected,
not softened** — :func:`verify` raises, and the caller counts the rejection:

1. The line the explanation is about was scored at or above the threshold.
2. Every other line it cites is a real line of the unit and is backed by the evidence it
   was given — an attended line, a data-flow line — or, for a template's second line, by
   a quoted symbol that occurs on it.
3. The text mentions no line number beyond those.
4. No sentence that makes a claim about the model's verdict mentions any line but the one
   the explanation is about. A context line is cited; it is never called flagged.
5. Every source fragment the text quotes in backticks occurs on a line it cites.
6. The text contains no attack-instruction vocabulary (``Rules.md`` §7.2, §7.3).

Checks 2 and 5 together are the *grounding rate* of ``ResearchPlan.md`` §4.3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vulnloc.utils.exceptions import ExplanationError

if TYPE_CHECKING:
    from vulnloc.explanation.types import Evidence, Explanation, Prediction
    from vulnloc.representation.units import CodeUnit

__all__ = [
    "BANNED_PHRASES",
    "CLAIM_WORDS",
    "GroundingReport",
    "mentioned_lines",
    "quoted_fragments",
    "verify",
]

CLAIM_WORDS = ("vulnerable", "flagged", "ranks", "scored", "score")
"""Words that assert something about the model's verdict on a line."""

BANNED_PHRASES = (
    "exploit",
    "payload",
    "shellcode",
    "proof of concept",
    "proof-of-concept",
    "attacker",
    "to attack",
    "craft",
    "trigger the",
    "gain control",
    "execute arbitrary",
)
"""Vocabulary of *how to attack*, which an explanation of *why a line is risky* never needs."""

_QUOTED = re.compile(r"`([^`]*)`")
_LINE_MENTION = re.compile(r"\blines?\s+(\d+(?:(?:\s*,\s*|\s+and\s+)\d+)*)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class GroundingReport:
    """What a verified explanation was found to rest on.

    Attributes:
        unit_id: The unit.
        line: The flagged line the explanation is about.
        mentioned_lines: Every line number in the text, ascending.
        quoted: Every backtick-quoted fragment in the text, in order.
    """

    unit_id: str
    line: int
    mentioned_lines: tuple[int, ...]
    quoted: tuple[str, ...]


def quoted_fragments(text: str) -> tuple[str, ...]:
    """Source fragments an explanation quotes, in order of appearance.

    Args:
        text: The explanation.

    Returns:
        The contents of each backtick pair.
    """
    return tuple(_QUOTED.findall(text))


def mentioned_lines(text: str) -> tuple[int, ...]:
    """Line numbers an explanation mentions in prose.

    Quoted source is removed first: ``lines 2`` inside backticks is code, not a citation.

    Args:
        text: The explanation.

    Returns:
        The distinct line numbers after the word "line" or "lines", ascending.
    """
    prose = _QUOTED.sub(" ", text)
    found: set[int] = set()
    for match in _LINE_MENTION.finditer(prose):
        found.update(int(number) for number in re.findall(r"\d+", match.group(1)))
    return tuple(sorted(found))


def _reject(reason: str, explanation: Explanation, **details: object) -> ExplanationError:
    return ExplanationError(
        f"explanation rejected: {reason}",
        unit_id=explanation.unit_id,
        line=explanation.line,
        template=explanation.template_id,
        **details,
    )


def verify(
    explanation: Explanation, unit: CodeUnit, pred: Prediction, evidence: Evidence
) -> GroundingReport:
    """Check one explanation against the prediction and evidence it claims to rest on.

    Args:
        explanation: The explanation to check.
        unit: The unit it is about.
        pred: The unit's calibrated scores — the only source of "flagged".
        evidence: The evidence the explainer was given for this line.

    Returns:
        What the explanation was found to rest on.

    Raises:
        ExplanationError: On the first check that fails. The message names the check.
    """
    line = explanation.line
    if not explanation.unit_id == unit.unit_id == pred.unit_id == evidence.unit_id:
        raise _reject("it does not describe the unit it was checked against", explanation)
    if evidence.line != line:
        raise _reject(
            "its evidence is about another line", explanation, evidence_line=evidence.line
        )
    if not pred.is_flagged(line):
        raise _reject(
            "it names a line that was not scored above the threshold (FR-15)",
            explanation,
            threshold=pred.threshold,
        )

    quoted = quoted_fragments(explanation.text)
    for cited in explanation.cited_lines:
        if not 1 <= cited <= unit.n_lines or cited == line:
            raise _reject("it cites a line that is not a context line", explanation, cited=cited)
        if cited in evidence.context_lines:
            continue
        symbols = [fragment.removesuffix("()") for fragment in quoted]
        if not any(symbol and symbol in unit.lines[cited - 1] for symbol in symbols):
            raise _reject(
                "it cites a line no evidence and no quoted symbol supports",
                explanation,
                cited=cited,
            )

    mentioned = mentioned_lines(explanation.text)
    allowed = {line, *explanation.cited_lines}
    stray = [number for number in mentioned if number not in allowed]
    if stray:
        raise _reject("its text mentions a line it does not cite", explanation, stray=stray)
    if line not in mentioned:
        raise _reject("its text never names the line it is about", explanation)

    for sentence in _SENTENCE_END.split(_QUOTED.sub(" ", explanation.text)):
        lowered = sentence.lower()
        if any(re.search(rf"\b{word}\b", lowered) for word in CLAIM_WORDS):
            others = [number for number in mentioned_lines(sentence) if number != line]
            if others:
                raise _reject(
                    "a sentence about the model's verdict mentions another line",
                    explanation,
                    others=others,
                )

    source = [unit.lines[number - 1] for number in sorted(allowed)]
    for fragment in quoted:
        needle = fragment.removesuffix("()")
        if not needle or not any(needle in text for text in source):
            raise _reject(
                "it quotes a fragment that is not on any line it cites",
                explanation,
                fragment=fragment,
            )

    prose = _QUOTED.sub(" ", explanation.text).lower()
    banned = [phrase for phrase in BANNED_PHRASES if phrase in prose]
    if banned:
        raise _reject("it contains attack-instruction vocabulary", explanation, banned=banned)

    return GroundingReport(
        unit_id=explanation.unit_id, line=line, mentioned_lines=mentioned, quoted=quoted
    )
