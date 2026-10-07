"""The three values an explainer is handed and the one it returns (``Architecture.md`` §4.2).

``Explainer.explain(unit, pred, evidence) -> Explanation`` names four types and the
architecture document left three of them undefined until Phase 7. They are defined here,
as plain frozen dataclasses with no tensors in them: this layer re-runs no inference
(``Architecture.md`` §3), so by the time anything in it is called the model's numbers are
already Python floats and line indices.

**Lines are 1-indexed everywhere**, as in the rest of the codebase. A tuple of per-line
values holds line ``i`` at position ``i - 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vulnloc.utils.exceptions import ExplanationError

__all__ = ["AttendedLine", "Evidence", "Explanation", "Prediction"]

SEM_FROM_STR = "sem_from_str"
STR_FROM_SEM = "str_from_sem"
DIRECTIONS = (SEM_FROM_STR, STR_FROM_SEM)
"""The two cross-attention maps, named as ``models/fusion/cross_attention.py`` names them."""


@dataclass(frozen=True)
class Prediction:
    """One unit's calibrated line scores and the threshold that turns them into flags.

    This is the only definition of "flagged" the explanation layer has. FR-15 — no
    explanation may call a line vulnerable that the head did not score above threshold —
    is checked against :meth:`is_flagged` and nothing else.

    Attributes:
        unit_id: Stable corpus identifier.
        probabilities: ``[n_lines]`` calibrated probabilities, line ``i`` at ``i - 1``.
        scoreable: ``[n_lines]``; ``False`` where the model had no features for the line,
            in which case its probability is not a prediction and is never read.
        threshold: The decision threshold, chosen on validation (``Phases.md`` Phase 6).

    Raises:
        ExplanationError: If the two tuples differ in length, a scoreable probability is
            not a number in ``[0, 1]``, or the threshold is outside ``[0, 1]``.
    """

    unit_id: str
    probabilities: tuple[float, ...]
    scoreable: tuple[bool, ...]
    threshold: float

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(self.scoreable):
            raise ExplanationError(
                "a prediction's probabilities and scoreable mask must have one length",
                unit_id=self.unit_id,
                probabilities=len(self.probabilities),
                scoreable=len(self.scoreable),
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ExplanationError(
                "a threshold on probabilities must lie in [0, 1]",
                unit_id=self.unit_id,
                threshold=self.threshold,
            )
        for line, (value, usable) in enumerate(
            zip(self.probabilities, self.scoreable, strict=True), start=1
        ):
            if usable and not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ExplanationError(
                    "a scoreable line's probability must be a number in [0, 1]",
                    unit_id=self.unit_id,
                    line=line,
                    probability=value,
                )

    @property
    def n_lines(self) -> int:
        """How many lines this prediction covers."""
        return len(self.probabilities)

    @property
    def ranking(self) -> tuple[int, ...]:
        """Scoreable lines, most suspicious first; ties go to the earlier line."""
        lines = [line for line in range(1, self.n_lines + 1) if self.scoreable[line - 1]]
        return tuple(sorted(lines, key=lambda line: (-self.probabilities[line - 1], line)))

    @property
    def flagged_lines(self) -> tuple[int, ...]:
        """Lines at or above the threshold, most suspicious first."""
        return tuple(line for line in self.ranking if self.is_flagged(line))

    def is_flagged(self, line: int) -> bool:
        """Whether ``line`` was scored at or above the threshold.

        Args:
            line: 1-indexed line. A line outside the prediction is simply not flagged.

        Returns:
            ``True`` only for a scoreable line whose probability reaches the threshold.
        """
        if not 1 <= line <= self.n_lines or not self.scoreable[line - 1]:
            return False
        return self.probabilities[line - 1] >= self.threshold

    def score(self, line: int) -> float:
        """The calibrated probability of a scoreable line.

        Args:
            line: 1-indexed line.

        Returns:
            Its probability.

        Raises:
            ExplanationError: If the line is outside the prediction or was not scoreable.
        """
        if not 1 <= line <= self.n_lines or not self.scoreable[line - 1]:
            raise ExplanationError(
                "this line has no score to report", unit_id=self.unit_id, line=line
            )
        return self.probabilities[line - 1]

    def rank(self, line: int) -> int:
        """1-based position of a scoreable line in :attr:`ranking`.

        Args:
            line: 1-indexed line.

        Returns:
            ``1`` for the most suspicious line.

        Raises:
            ExplanationError: If the line was not scoreable.
        """
        ranking = self.ranking
        if line not in ranking:
            raise ExplanationError(
                "this line has no rank to report", unit_id=self.unit_id, line=line
            )
        return ranking.index(line) + 1


@dataclass(frozen=True)
class AttendedLine:
    """One line a flagged line's cross-attention put weight on.

    Attributes:
        line: 1-indexed line attended to.
        weight: Attention weight, averaged over heads; a row sums to 1 over the lines the
            attended branch covers.
        direction: ``"sem_from_str"`` — the flagged line's semantic state read this line's
            structural state — or ``"str_from_sem"``, the reverse.
    """

    line: int
    weight: float
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ExplanationError(
                "unknown attention direction", direction=self.direction, known=list(DIRECTIONS)
            )
        if self.line < 1:
            raise ExplanationError("lines are 1-indexed", line=self.line)


@dataclass(frozen=True)
class Evidence:
    """What an explanation of one flagged line may cite, and nothing else.

    Three of the four parts are **not** things the model computed, and the difference is
    kept visible because it is the difference between evidence and narration:

    * ``attended`` is the model's own cross-attention, read off the forward pass that
      produced the score.
    * ``dataflow_path`` comes from a static, name-based reaching-definitions analysis
      (``representation/graphs/dfg.py``, with every caveat listed there). **The full model
      was trained on the AST and never saw these edges.**
    * ``graph_nodes`` are AST nodes on the flagged line chosen by node type. The model has
      no node-importance output, so this is a selection rule, not a learned ranking.
    * ``top_tokens`` are identifiers on the flagged line and on the lines it attended to.
      The attention is between *lines* — the fusion operates on the line axis — so the
      order is by line weight, and within a line tokens keep source order.

    Attributes:
        unit_id: The unit.
        line: The flagged line this evidence is about.
        attended: Lines its cross-attention weighted most, heaviest first, both directions.
        dataflow_path: Ascending lines of a definition-to-use chain ending at ``line``;
            empty when nothing flows into the line.
        dataflow_symbols: ``dataflow_symbols[k]`` is the name carried from
            ``dataflow_path[k]`` to ``dataflow_path[k + 1]``.
        graph_nodes: AST node identifiers on ``line``, as ``"n_<node_id>"``.
        top_tokens: Identifiers, most relevant first.

    Raises:
        ExplanationError: If the path does not end at ``line`` or its symbols do not match
            its hops.
    """

    unit_id: str
    line: int
    attended: tuple[AttendedLine, ...] = ()
    dataflow_path: tuple[int, ...] = ()
    dataflow_symbols: tuple[str, ...] = ()
    graph_nodes: tuple[str, ...] = ()
    top_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dataflow_path:
            if self.dataflow_path[-1] != self.line:
                raise ExplanationError(
                    "a data-flow path is evidence about its last line",
                    unit_id=self.unit_id,
                    line=self.line,
                    path=list(self.dataflow_path),
                )
            if len(self.dataflow_symbols) != len(self.dataflow_path) - 1:
                raise ExplanationError(
                    "a data-flow path needs one symbol per hop",
                    unit_id=self.unit_id,
                    path=list(self.dataflow_path),
                    symbols=list(self.dataflow_symbols),
                )
        elif self.dataflow_symbols:
            raise ExplanationError(
                "data-flow symbols without a path", unit_id=self.unit_id, line=self.line
            )

    @property
    def context_lines(self) -> frozenset[int]:
        """Every line other than ``line`` that this evidence lets a text mention."""
        lines = {entry.line for entry in self.attended} | set(self.dataflow_path)
        lines.discard(self.line)
        return frozenset(lines)


@dataclass(frozen=True)
class Explanation:
    """One flagged line, explained.

    The structured fields exist so the verifier does not have to trust the text: what the
    explanation *claims* and what it merely *cites* are stated outright, and the text is
    then checked against them (``explanation/verifier.py``).

    Attributes:
        unit_id: The unit.
        line: The line said to be vulnerable. The only such line in this explanation.
        label: Vulnerability category. **Template-derived** — the name of the pattern the
            line's text matched, never a model output (open decision Q5, deferred by the
            owner on 2026-10-07). ``"Unclassified"`` when no pattern matched.
        cwe: The MITRE identifier that pattern is conventionally filed under, or ``""``.
        text: The explanation, as shown to a reader.
        template_id: Which template produced it; ``"no-pattern"`` for the fallback.
        cited_lines: Other lines the text mentions as context. Never called vulnerable.
        cited_symbols: Identifiers the text quotes.
    """

    unit_id: str
    line: int
    label: str
    cwe: str
    text: str
    template_id: str
    cited_lines: tuple[int, ...] = ()
    cited_symbols: tuple[str, ...] = ()
