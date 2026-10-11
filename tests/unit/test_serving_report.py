"""The reader's report says what was found, what was tested and what was not looked at.

Its failure modes are the ones worth pinning: a function with nothing flagged must never
read as safe, a line the verifier rejected must never arrive with softened wording, and
"no known pattern matched" must carry its scope.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from vulnloc.explanation.templates import NO_PATTERN, all_templates  # noqa: E402
from vulnloc.explanation.types import (  # noqa: E402
    AttendedLine,
    Evidence,
    Explanation,
    Prediction,
)
from vulnloc.representation import CodeUnit  # noqa: E402
from vulnloc.serving.report import (  # noqa: E402
    NOT_SAFE_SENTENCE,
    TOP_RANKED,
    build_report,
    patterns_checked,
    summarize,
)
from vulnloc.serving.schemas import AnalysisReport  # noqa: E402

SOURCE = "\n".join(
    [
        "void copy(char *n) {",
        "char buf[8];",
        "",
        "strcpy(buf, n);",
        "puts(buf);",
        "return;",
        "}",
    ]
)
COVERAGE: dict[str, Any] = {
    "lines": 7,
    "lines_scored": 6,
    "lines_beyond_cap": 0,
    "lines_without_features": 0,
    "blank_lines": 1,
    "explanations_rejected": 0,
    "line_cap": 512,
    "parse_clean": True,
    "first_parse_error_line": None,
    "dataflow_available": True,
}


def unit() -> CodeUnit:
    return CodeUnit.from_source("u", SOURCE, language="C")


def prediction(scores: list[float], threshold: float = 0.5) -> Prediction:
    scoreable = tuple(index != 2 for index in range(len(scores)))  # line 3 is blank
    return Prediction(
        unit_id="u", probabilities=tuple(scores), scoreable=scoreable, threshold=threshold
    )


def explanation(line: int, template_id: str, label: str = "Buffer overflow") -> Explanation:
    return Explanation(
        unit_id="u",
        line=line,
        label=label,
        cwe="CWE-120" if template_id != NO_PATTERN else "",
        text=f"Line {line} is flagged.",
        template_id=template_id,
    )


def test_an_unflagged_function_is_never_reported_as_safe() -> None:
    pred = prediction([0.1, 0.2, 0.0, 0.3, 0.1, 0.05, 0.01])

    report = build_report(unit(), pred, {}, {}, coverage=COVERAGE)

    assert report["findings"] == []
    assert "No line was flagged above the threshold." in report["summary"]
    assert NOT_SAFE_SENTENCE in report["summary"]
    assert "safe" not in report["summary"].replace(NOT_SAFE_SENTENCE, "")


def test_the_highest_ranked_lines_are_listed_even_when_none_is_flagged() -> None:
    pred = prediction([0.1, 0.2, 0.0, 0.3, 0.1, 0.05, 0.01])

    ranked = build_report(unit(), pred, {}, {}, coverage=COVERAGE)["ranked_lines"]

    assert [entry["line"] for entry in ranked] == list(pred.ranking[:TOP_RANKED])
    assert [entry["rank"] for entry in ranked] == list(range(1, TOP_RANKED + 1))
    assert ranked[0] == {
        "line": 4,
        "rank": 1,
        "score": 0.3,
        "code": "strcpy(buf, n);",
        "flagged": False,
    }
    assert "line 4" in build_report(unit(), pred, {}, {}, coverage=COVERAGE)["summary"]


def test_a_blank_line_is_never_among_the_ranked_lines() -> None:
    pred = prediction([0.1, 0.2, 0.0, 0.3, 0.1, 0.05, 0.01])

    ranked = build_report(unit(), pred, {}, {}, coverage=COVERAGE, top_ranked=10)["ranked_lines"]

    assert 3 not in [entry["line"] for entry in ranked]
    assert len(ranked) == 6


def test_every_flagged_line_is_a_finding_with_its_evidence() -> None:
    pred = prediction([0.1, 0.6, 0.0, 0.9, 0.1, 0.05, 0.01])
    template = all_templates()[0]
    explanations = {
        4: explanation(4, template.id, template.label),
        2: explanation(2, NO_PATTERN, "Unclassified"),
    }
    evidence = {
        4: Evidence(
            unit_id="u",
            line=4,
            attended=(AttendedLine(line=2, weight=0.4, direction="sem_from_str"),),
            dataflow_path=(2, 4),
            dataflow_symbols=("buf",),
            graph_nodes=("n_7",),
            top_tokens=("strcpy", "buf"),
        ),
        2: Evidence(unit_id="u", line=2),
    }

    report = build_report(unit(), pred, explanations, evidence, coverage=COVERAGE)

    first, second = report["findings"]
    assert (first["line"], first["rank"], first["pattern"]) == (4, 1, template.id)
    assert first["explanation"] == "Line 4 is flagged."
    assert first["attended_lines"] == [
        {"line": 2, "weight": 0.4, "direction": "sem_from_str", "code": "char buf[8];"}
    ]
    assert first["dataflow_path"] == [2, 4]
    assert (second["line"], second["pattern"]) == (2, None)
    assert report["patterns"]["matched"] == [template.id]
    assert report["summary"].startswith("2 of the 6 scored lines are flagged.")
    assert "matches 1 of them" in report["summary"]


def test_a_rejected_explanation_leaves_the_line_listed_with_nothing_said() -> None:
    pred = prediction([0.1, 0.6, 0.0, 0.9, 0.1, 0.05, 0.01])
    explanations = {4: explanation(4, NO_PATTERN, "Unclassified")}
    evidence = {4: Evidence(unit_id="u", line=4)}

    report = build_report(unit(), pred, explanations, evidence, coverage=COVERAGE)

    rejected = report["findings"][1]
    assert rejected["line"] == 2
    assert rejected["explanation"] is None
    assert rejected["pattern"] is None
    assert rejected["attended_lines"] == []
    assert "No known weakness pattern matches any of them" in report["summary"]


def test_the_patterns_checked_are_the_whole_template_library() -> None:
    checked = patterns_checked()

    assert [entry["id"] for entry in checked] == [template.id for template in all_templates()]
    assert all(entry["label"] for entry in checked)
    assert NO_PATTERN not in [entry["id"] for entry in checked]


def test_coverage_and_the_measured_record_pass_through_unchanged() -> None:
    pred = prediction([0.1, 0.2, 0.0, 0.3, 0.1, 0.05, 0.01])
    coverage = {**COVERAGE, "lines_beyond_cap": 40, "parse_clean": False}
    measured = {"partition": "test", "top_5": 0.5, "run_id": "abc"}

    report = build_report(unit(), pred, {}, {}, coverage=coverage, measured=measured)

    assert report["coverage"] == coverage
    assert report["measured"] == measured
    assert build_report(unit(), pred, {}, {}, coverage=coverage)["measured"] is None


def test_a_unit_with_nothing_scoreable_says_so() -> None:
    pred = Prediction(unit_id="u", probabilities=(0.0,) * 7, scoreable=(False,) * 7, threshold=0.5)

    summary = summarize(pred, matched=0, lines_scored=0)

    assert summary.startswith("No line could be scored")
    assert NOT_SAFE_SENTENCE in summary


def test_the_report_fits_the_schema_the_api_publishes() -> None:
    pred = prediction([0.1, 0.6, 0.0, 0.9, 0.1, 0.05, 0.01])
    explanations = {4: explanation(4, NO_PATTERN, "Unclassified")}
    evidence = {4: Evidence(unit_id="u", line=4)}

    report = build_report(unit(), pred, explanations, evidence, coverage=COVERAGE)

    assert AnalysisReport.model_validate(report).model_dump() == report


def test_the_report_endpoint_returns_the_contract_and_the_report_beside_it() -> None:
    pytest.importorskip("httpx")
    import numpy as np
    from fastapi.testclient import TestClient

    from vulnloc.explanation.contract import CONTRACT_KEYS, output_contract
    from vulnloc.serving.app import create_app
    from vulnloc.serving.inference import Analysis

    pred = prediction([0.1, 0.2, 0.0, 0.3, 0.1, 0.05, 0.01])

    class Bundle:
        run_id = "a" * 32
        manifest: dict[str, Any] = {}

    class StandIn:
        bundle = Bundle()

        def analyze(self, code: str, *, language: str = "C", unit_id: str = "request") -> Analysis:
            return Analysis(
                contract=output_contract(unit(), pred, {}, {}),
                diagnostics={**COVERAGE, "seconds": 0.1},
                logits=np.zeros(1),
                scoreable=np.ones(1, dtype=bool),
                report=build_report(unit(), pred, {}, {}, coverage=COVERAGE),
            )

    with TestClient(create_app(StandIn())) as client:  # type: ignore[arg-type]
        response = client.post("/analyze/report", json={"code": SOURCE})
        plain = client.post("/analyze", json={"code": SOURCE})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"result", "report"}
    assert tuple(body["result"]) == CONTRACT_KEYS
    assert body["result"] == plain.json()
    assert NOT_SAFE_SENTENCE in body["report"]["summary"]
    assert response.headers["X-VulnLoc-Lines-Scored"] == "6"
