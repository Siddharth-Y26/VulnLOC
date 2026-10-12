"""The API answers in the contract's shape, and fails in one shape (``Rules.md`` §4.7).

The model is replaced by a stand-in here — what the model computes is pinned in
``test_serving_inference.py``. These tests are about the boundary: that ``POST /analyze``
returns exactly the ``PRD.md`` §5.4 object, that every way a request can fail leaves with
the right status and the same three keys, and that a failure is never dressed as a result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from vulnloc.explanation.contract import CONTRACT_KEYS, EVIDENCE_KEYS, LINE_KEYS  # noqa: E402
from vulnloc.serving.app import BUNDLE_ENV, create_app, status_for  # noqa: E402
from vulnloc.serving.inference import Analysis  # noqa: E402
from vulnloc.serving.schemas import (  # noqa: E402
    MAX_BATCH_ITEMS,
    MAX_SOURCE_CHARS,
    AnalyzeResponse,
    EvidenceBlock,
    VulnerableLine,
)
from vulnloc.utils.exceptions import (  # noqa: E402
    AlignmentError,
    ConfigError,
    DataError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
    VulnLocError,
)

ERROR_KEYS = {"type", "message", "unit_id"}

MANIFEST = {
    "run_id": "a" * 32,
    "git_sha": "abc1234",
    "config_hash": "c0ffee",
    "seed": 42,
    "split_fingerprint": "f" * 64,
    "exported_at": "2026-10-10T00:00:00+00:00",
    "text": {"normalize_whitespace": True, "strip_comments": True},
    "llm": {"n_lines": 512},
    "fusion": {"kind": "cross_attention"},
    "calibration": {"kind": "platt", "threshold": 0.0925},
    "verification": {"max_logit_deviation": 0.01},
}


def contract(unit_id: str) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "is_vulnerable": True,
        "risk_score": 0.94,
        "vulnerability_label": "Buffer Overflow",
        "vulnerable_lines": [{"line": 4, "score": 0.94, "code": "strcpy(buf, n);"}],
        "explanation": "Line 4 copies with strcpy() and no bound is visible.",
        "evidence": {
            "graph_nodes": ["n_12"],
            "dataflow_path": [2, 4],
            "attention_top_tokens": ["strcpy", "buf"],
        },
    }


class StandInBundle:
    """The two things the application reads off a bundle."""

    run_id = MANIFEST["run_id"]
    manifest = MANIFEST


class StandIn:
    """Answers like an ``Analyzer``; raises what the request's first word names."""

    bundle = StandInBundle()
    failures: dict[str, Exception] = {
        "unparseable": ParseError("the source cannot be parsed"),
        "graphless": GraphBuildError("no graph could be built"),
        "ungrounded": ExplanationError("the top flagged line has no verified explanation"),
        "misaligned": AlignmentError("a token maps to no line"),
        "broken": ModelError("the model failed to run"),
        "crash": RuntimeError("something nobody anticipated"),
    }

    def __init__(self) -> None:
        self.languages: list[str] = []

    def analyze(self, code: str, *, language: str = "C", unit_id: str = "request") -> Analysis:
        self.languages.append(language)
        failure = self.failures.get(code.split()[0])
        if failure is not None:
            if isinstance(failure, VulnLocError):
                failure.unit_id = unit_id
            raise failure
        return Analysis(
            contract=contract(unit_id),
            diagnostics={
                "lines": 600,
                "lines_scored": 500,
                "lines_beyond_cap": 88,
                "lines_without_features": 12,
                "flagged": 1,
                "explanations_rejected": 0,
                "seconds": 0.25,
            },
            logits=np.zeros(1),
            scoreable=np.ones(1, dtype=bool),
        )


@pytest.fixture()
def stand_in() -> StandIn:
    return StandIn()


@pytest.fixture()
def client(stand_in: StandIn) -> Any:
    with TestClient(create_app(stand_in), raise_server_exceptions=False) as running:  # type: ignore[arg-type]
        yield running


def test_the_response_schema_is_the_contract_the_explanation_layer_builds() -> None:
    assert tuple(AnalyzeResponse.model_fields) == CONTRACT_KEYS
    assert tuple(VulnerableLine.model_fields) == LINE_KEYS
    assert tuple(EvidenceBlock.model_fields) == EVIDENCE_KEYS


def test_analyze_returns_the_contract_and_nothing_else_in_the_body(client: Any) -> None:
    response = client.post("/analyze", json={"code": "int f() { return 0; }", "unit_id": "a.c::f"})
    assert response.status_code == 200
    assert response.json() == contract("a.c::f")
    assert response.headers["X-VulnLoc-Run"] == "a" * 8
    assert response.headers["X-VulnLoc-Lines-Scored"] == "500"
    assert response.headers["X-VulnLoc-Lines-Beyond-Cap"] == "88"
    assert response.headers["X-VulnLoc-Lines-Without-Features"] == "12"


def test_the_declared_language_reaches_the_pipeline_as_the_corpus_spells_it(
    client: Any, stand_in: StandIn
) -> None:
    client.post("/analyze", json={"code": "x"})
    client.post("/analyze", json={"code": "x", "language": "cpp"})
    assert stand_in.languages == ["C", "CPP"]


@pytest.mark.parametrize(
    ("word", "status", "kind"),
    [
        ("unparseable", 400, "ParseError"),
        ("graphless", 400, "GraphBuildError"),
        ("ungrounded", 500, "ExplanationError"),
        ("misaligned", 500, "AlignmentError"),
        ("broken", 500, "ModelError"),
        ("crash", 500, "InternalError"),
    ],
)
def test_every_failure_leaves_in_one_shape_with_the_status_its_class_deserves(
    client: Any, word: str, status: int, kind: str
) -> None:
    response = client.post("/analyze", json={"code": f"{word} input", "unit_id": "u1"})
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == ERROR_KEYS
    assert body["error"]["type"] == kind
    assert body["error"]["message"]
    assert body["error"]["unit_id"] == (None if kind == "InternalError" else "u1")
    # An unexpected failure says nothing about its insides.
    assert "anticipated" not in body["error"]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"code": ""},
        {"code": "x", "language": "java"},
        {"code": "x", "surprise": 1},
        {"code": "x" * (MAX_SOURCE_CHARS + 1)},
        {"code": "x", "unit_id": ""},
    ],
)
def test_a_request_that_does_not_validate_is_a_422_in_the_same_shape(
    client: Any, payload: dict[str, Any]
) -> None:
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == ERROR_KEYS
    assert body["error"]["type"] == "ValidationError"


def test_a_batch_gives_each_function_its_own_result_or_its_own_error(client: Any) -> None:
    items = [
        {"code": "int f() { return 0; }", "unit_id": "ok"},
        {"code": "unparseable thing", "unit_id": "bad"},
        {"code": "int g() { return 1; }", "unit_id": "ok2"},
    ]
    response = client.post("/analyze/batch", json={"items": items})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [bool(entry["result"]) for entry in results] == [True, False, True]
    assert results[0]["result"] == contract("ok")
    assert results[0]["error"] is None
    assert results[1]["error"] == {
        "type": "ParseError",
        "message": "the source cannot be parsed",
        "unit_id": "bad",
    }


def test_a_batch_has_a_size_limit(client: Any) -> None:
    too_many = [{"code": "x"}] * (MAX_BATCH_ITEMS + 1)
    assert client.post("/analyze/batch", json={"items": too_many}).status_code == 422
    assert client.post("/analyze/batch", json={"items": []}).status_code == 422


def test_health_and_version_say_which_model_is_answering(client: Any) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    version = client.get("/version").json()
    assert version["run_id"] == "a" * 32
    assert (version["git_sha"], version["config_hash"]) == ("abc1234", "c0ffee")
    assert version["threshold"] == 0.0925
    assert version["fusion"] == "cross_attention"
    assert version["line_cap"] == 512
    assert version["max_logit_deviation"] == 0.01


def test_the_api_documentation_says_a_clean_result_is_not_proof_of_safety(client: Any) -> None:
    schema = client.get("/openapi.json").json()
    assert "not evidence that the function is safe" in " ".join(
        schema["info"]["description"].split()
    )
    described = schema["components"]["schemas"]["AnalyzeResponse"]["properties"]["is_vulnerable"]
    assert "NOT a statement that the function is safe" in described["description"]


def test_the_demo_page_is_self_contained_and_says_what_a_clean_result_is_not(client: Any) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    page = response.text
    assert "not evidence that the function is safe" in " ".join(page.split())
    # No CDN, no external font or script (Design.md §3): the container never calls out.
    assert "http://" not in page
    assert "https://" not in page
    # Pasted code is only ever written into the page as text.
    assert "innerHTML" not in page
    assert "/openapi.json" not in page
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_the_demo_page_reads_the_report_and_invents_no_measurement(client: Any) -> None:
    page = " ".join(client.get("/").text.split())
    # The page shows the reader's report, not only the contract's one explanation.
    assert 'fetch("/analyze/report"' in page
    # A template is never presented as the model's reasoning, and static analysis is named.
    assert "not the model's reasoning" in page
    assert "static analysis" in page
    # The measured sentence is built only from the bundle's own record, and says its limit.
    assert "if (!measured" in page
    assert "never measured on functions without a vulnerability" in page


def test_a_service_with_no_bundle_does_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    with pytest.raises(ConfigError, match="no serving bundle configured"), TestClient(create_app()):
        pass


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ParseError("x"), 400),
        (GraphBuildError("x"), 400),
        (DataError("x"), 400),
        (ConfigError("x"), 500),
        (ModelError("x"), 500),
        (ExplanationError("x"), 500),
    ],
)
def test_only_unreadable_input_is_the_clients_fault(error: VulnLocError, status: int) -> None:
    assert status_for(error) == status
