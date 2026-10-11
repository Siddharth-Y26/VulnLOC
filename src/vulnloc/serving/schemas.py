"""Request and response schemas of the serving API (``PRD.md`` §5.4, §5.6).

:class:`AnalyzeResponse` is the §5.4 output contract. Its fields are the keys
``vulnloc.explanation.contract`` builds — ``CONTRACT_KEYS``, ``LINE_KEYS``,
``EVIDENCE_KEYS`` — and a test pins the two together, so the schema cannot drift from the
object the explanation layer assembles and Phase 7 verified.

Errors have one shape everywhere (``Rules.md`` §4.7)::

    {"error": {"type": "ParseError", "message": "...", "unit_id": "..."}}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_BATCH_ITEMS",
    "MAX_SOURCE_CHARS",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "BatchItem",
    "BatchRequest",
    "BatchResponse",
    "ErrorBody",
    "ErrorResponse",
    "EvidenceBlock",
    "HealthResponse",
    "VersionResponse",
    "VulnerableLine",
]

MAX_SOURCE_CHARS = 200_000
"""Longest source text one request may carry. The model scores a function's first 512
lines; this bound is about refusing a mistaken upload, not about what is analyzable."""

MAX_BATCH_ITEMS = 16
"""Functions one batch request may carry. They are analyzed one after another."""


class AnalyzeRequest(BaseModel):
    """One function to analyze."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        max_length=MAX_SOURCE_CHARS,
        description="Source text of one C or C++ function.",
    )
    language: Literal["c", "cpp"] = Field(
        default="c",
        description="The language you believe it is. A hint only: the grammar is chosen "
        "from the source itself.",
    )
    unit_id: str = Field(
        default="request",
        min_length=1,
        max_length=200,
        description="An identifier of your choosing, echoed in the result.",
    )


class VulnerableLine(BaseModel):
    """One flagged line."""

    line: int = Field(ge=1, description="1-indexed line number in the submitted text.")
    score: float = Field(ge=0.0, le=1.0, description="Calibrated probability.")
    code: str = Field(
        description="The line as the model saw it: comments and edge whitespace removed."
    )


class EvidenceBlock(BaseModel):
    """What the top flagged line's explanation rests on."""

    graph_nodes: list[str] = Field(description="AST nodes on the flagged line.")
    dataflow_path: list[int] = Field(
        description="Lines on a name-based definition-to-use chain ending at the flagged line."
    )
    attention_top_tokens: list[str] = Field(
        description="Identifiers on the lines the model attended to."
    )


class AnalyzeResponse(BaseModel):
    """The ``PRD.md`` §5.4 output contract."""

    unit_id: str
    is_vulnerable: bool = Field(
        description="True when at least one line reached the threshold. False means only "
        "that none did - it is NOT a statement that the function is safe."
    )
    risk_score: float = Field(
        ge=0.0, le=1.0, description="The highest line probability. Not a function-level score."
    )
    vulnerability_label: str = Field(
        description="Category of the matching explanation template, 'Unclassified' when "
        "no pattern matched. Derived from the template, not predicted by the model."
    )
    vulnerable_lines: list[VulnerableLine] = Field(
        description="Flagged lines, highest score first."
    )
    explanation: str
    evidence: EvidenceBlock


class ErrorBody(BaseModel):
    """The body of every error response."""

    type: str = Field(description="The error class, e.g. ParseError or ValidationError.")
    message: str
    unit_id: str | None = None


class ErrorResponse(BaseModel):
    """Every error response."""

    error: ErrorBody


class HealthResponse(BaseModel):
    """Liveness."""

    status: Literal["ok"] = "ok"
    model_loaded: bool


class VersionResponse(BaseModel):
    """Which model is answering (``PRD.md`` FR-23)."""

    vulnloc_version: str
    run_id: str = Field(description="The training run the weights came from.")
    git_sha: str = Field(description="Commit the run was trained at.")
    config_hash: str = Field(description="Hash of the run's resolved config.")
    seed: int
    split_fingerprint: str
    fusion: str = Field(description="Fusion strategy of the checkpoint.")
    threshold: float = Field(description="Calibrated probability at which a line is flagged.")
    calibrator: str
    strip_comments: bool
    line_cap: int = Field(description="Lines of a function that are scored; later lines are not.")
    exported_at: str
    max_logit_deviation: float | None = Field(
        description="Largest difference from the checkpoint's stored validation scores "
        "found when the bundle was exported."
    )


class BatchRequest(BaseModel):
    """Several functions in one request (``PRD.md`` FR-5)."""

    model_config = ConfigDict(extra="forbid")

    items: list[AnalyzeRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class BatchItem(BaseModel):
    """One function's outcome: a result or an error, never both."""

    result: AnalyzeResponse | None = None
    error: ErrorBody | None = None


class BatchResponse(BaseModel):
    """Outcomes in the order of the request's items."""

    results: list[BatchItem]
