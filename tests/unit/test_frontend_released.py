"""The Findings tab over the released bundle — what reaches the page, with the model stubbed.

The model is pinned elsewhere (``test_serving_inference.py``); the arrangement is under test
here. The owner asked two things of it on 2026-10-12: no count of flagged lines as a
headline, and a reason only for a line a known weakness pattern matches. The stand-in below
builds its report with the service's own ``build_report``, so the shape these tests read is
the real one; one test at the end reads the real bundle when it is on disk.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import frontend.released as released  # noqa: E402
from frontend.fix_templates import NO_ADVICE, advice_for  # noqa: E402
from frontend.pipeline import sample_functions  # noqa: E402
from frontend.samples import GROUPS  # noqa: E402

from vulnloc.explanation.templates import NO_PATTERN, all_templates  # noqa: E402
from vulnloc.explanation.types import (  # noqa: E402
    AttendedLine,
    Evidence,
    Explanation,
    Prediction,
)
from vulnloc.models.heads.calibration import LogitCalibrator  # noqa: E402
from vulnloc.representation.units import CodeUnit  # noqa: E402
from vulnloc.serving.report import build_report  # noqa: E402

SOURCE = "void f(char *s, int i)\n{\nchar buf[8];\nstrcpy(buf, s);\nprintf(s);\nbuf[i] = 0;\n}\n"

MANIFEST = {
    "run_id": "a" * 32,
    "git_sha": "abc1234",
    "seed": 42,
    "exported_at": "2026-10-12T00:00:00+00:00",
    "text": {"normalize_whitespace": True, "strip_comments": True},
    "llm": {"n_lines": 512},
    "fusion": {"kind": "cross_attention"},
    "calibration": {"kind": "platt", "threshold": 0.5},
}

PATTERNS = {
    4: ("unbounded-string-copy", "Buffer overflow", "CWE-120"),
    5: ("format-string-printf", "Format string", "CWE-134"),
    6: ("array-index-write", "Out-of-bounds write", "CWE-787"),
}


class StandIn:
    """Answers like the service's ``Analyzer``; the report is the service's own assembly.

    Args:
        logits: ``line -> logit``; a line not named sits far below the cut-off.
        patterns: ``line -> (template id, label, cwe)`` for the lines a pattern matches.
    """

    normalize_whitespace = True
    strip_comments = True
    threshold = 0.5
    calibrator = LogitCalibrator(scale=1.0, shift=0.0, kind="platt")
    bundle = SimpleNamespace(manifest=MANIFEST)

    def __init__(self, logits: dict[int, float], patterns: dict[int, tuple[str, str, str]]) -> None:
        self.logits = logits
        self.patterns = patterns

    def analyze(self, code: str, *, language: str = "C", unit_id: str = "request") -> Any:
        unit = CodeUnit.from_source(
            unit_id, code, language=language, normalize_whitespace=True, strip_comments=True
        )
        logits = np.array(
            [self.logits.get(line, -9.0) for line in range(1, unit.n_lines + 1)], dtype=np.float32
        )
        scoreable = np.ones(unit.n_lines, dtype=bool)
        offered = [bool(text.strip()) for text in unit.lines]
        probabilities = self.calibrator.probabilities(torch.from_numpy(logits).double()).tolist()
        pred = Prediction(
            unit_id=unit_id,
            probabilities=tuple(
                p if ok else 0.0 for p, ok in zip(probabilities, offered, strict=True)
            ),
            scoreable=tuple(offered),
            threshold=self.threshold,
        )
        explanations, evidence = {}, {}
        for line in pred.flagged_lines:
            template, label, cwe = self.patterns.get(line, (NO_PATTERN, "Unclassified", ""))
            explanations[line] = Explanation(
                unit_id=unit_id,
                line=line,
                label=label,
                cwe=cwe,
                text=f"The service's sentence about line {line}.",
                template_id=template,
            )
            evidence[line] = Evidence(
                unit_id=unit_id,
                line=line,
                attended=(
                    AttendedLine(line=1, weight=0.5, direction="sem_from_str"),
                    AttendedLine(line=3, weight=0.001, direction="str_from_sem"),
                ),
                # One symbol per hop: a path over two hops names `buf` twice.
                dataflow_path=(3, 4, line) if line > 4 else (),
                dataflow_symbols=("buf", "buf") if line > 4 else (),
                top_tokens=("buf",),
            )
        coverage = {
            "lines": unit.n_lines,
            "lines_scored": sum(offered),
            "lines_beyond_cap": 0,
            "lines_without_features": 0,
            "blank_lines": unit.n_lines - sum(offered),
            "explanations_rejected": 0,
            "line_cap": 512,
            "parse_clean": True,
            "first_parse_error_line": None,
            "dataflow_available": True,
        }
        report = build_report(unit, pred, explanations, evidence, coverage=coverage, measured=None)
        return SimpleNamespace(
            report=report, logits=logits, scoreable=scoreable, diagnostics={"seconds": 0.01}
        )


def run(monkeypatch: pytest.MonkeyPatch, logits: dict[int, float], tmp_path: Any) -> dict[str, Any]:
    stand_in = StandIn(logits, PATTERNS)
    monkeypatch.setattr(released, "_load", lambda: stand_in)
    # Hermetic: the page's measured sentence about patterns is read from disk when present.
    monkeypatch.setattr(released, "PATTERN_RECORD", tmp_path / "absent.json")
    return released.released_findings(SOURCE)


def test_a_card_is_a_line_over_the_cut_off_that_a_pattern_matches(monkeypatch, tmp_path):
    # Lines 3, 4, 5 are over the cut-off; a pattern matches 4 and 5. Line 6 matches a
    # pattern too, and sits below the cut-off.
    report = run(monkeypatch, {3: 4.0, 5: 3.0, 4: 2.0, 6: -2.0}, tmp_path)

    assert [card["line"] for card in report["cards"]] == [5, 4]
    by_line = {entry["line"]: entry for entry in report["scores"]}
    assert by_line[3]["flagged"] and by_line[3]["rank"] == 1  # ranked first, and no card
    assert not by_line[6]["flagged"]


def test_no_count_of_flagged_lines_is_handed_to_the_page(monkeypatch, tmp_path):
    report = run(monkeypatch, {3: 4.0, 5: 3.0, 4: 2.0}, tmp_path)

    # The service's summary is the sentence "3 of the 6 scored lines are flagged…".
    assert "summary" not in report
    assert "flagged" not in report["coverage"]
    assert not [key for key in report if "count" in key or "flagged" in key]


def test_a_card_keeps_the_services_rank_and_its_sentence_whole(monkeypatch, tmp_path):
    report = run(monkeypatch, {3: 4.0, 5: 3.0, 4: 2.0}, tmp_path)

    assert [(card["line"], card["rank"]) for card in report["cards"]] == [(5, 2), (4, 3)]
    assert report["cards"][0]["explanation"] == "The service's sentence about line 5."
    assert report["cards"][0]["label"] == "Format string"
    assert report["cards"][0]["cwe"] == "CWE-134"


def test_the_models_ordering_is_kept_whether_or_not_a_pattern_matches(monkeypatch, tmp_path):
    report = run(monkeypatch, {3: 4.0, 5: 3.0, 4: 2.0}, tmp_path)

    assert [(entry["line"], entry["rank"]) for entry in report["top"]][:3] == [
        (3, 1),
        (5, 2),
        (4, 3),
    ]
    assert [entry["has_card"] for entry in report["top"]][:3] == [False, True, True]
    # Every source line is in the whole-function view; a blank one carries no score.
    assert [entry["line"] for entry in report["scores"]] == list(range(1, len(report["lines"]) + 1))
    blank = [entry for entry in report["scores"] if not report["lines"][entry["line"] - 1].strip()]
    assert blank
    assert all(entry["score"] is None and entry["rank"] is None for entry in blank)


def test_a_function_no_pattern_matches_has_no_card_and_still_has_its_ranking(monkeypatch, tmp_path):
    report = run(monkeypatch, {3: 4.0, 1: 3.0}, tmp_path)

    assert report["cards"] == []
    assert [entry["line"] for entry in report["top"]][:2] == [3, 1]


def test_a_fix_is_joined_on_the_patterns_id(monkeypatch, tmp_path):
    report = run(monkeypatch, {4: 4.0, 5: 3.0, 6: 2.0}, tmp_path)
    advice = {card["pattern"]: card["advice"] for card in report["cards"]}

    assert "bounded copy" in advice["unbounded-string-copy"]["fix"]
    # The explainer splits the format-string weakness by call; the fix is one.
    assert 'printf("%s", msg)' in advice["format-string-printf"]["fix"]
    # An index is bounded somewhere else in the function: no one-line fix is offered.
    assert advice["array-index-write"] is None


def test_every_explainer_pattern_has_a_fix_or_is_declared_without_one():
    for template in all_templates():
        assert (advice_for(template.id) is None) == (template.id in NO_ADVICE), template.id
    assert advice_for("a-pattern-nobody-wrote") is None


def test_the_cwe_659_reference_rides_along_and_names_only_what_the_cards_name(
    monkeypatch, tmp_path
):
    report = run(monkeypatch, {3: 4.0, 5: 3.0, 4: 2.0}, tmp_path)
    reference = report["cwe_659"]

    # Cards on lines 5 (CWE-134) and 4 (CWE-120); line 3 is ranked first and has none.
    assert reference["named_here"] == ["CWE-134", "CWE-120"]
    checked = {member["cwe"] for member in reference["members"] if member["checked"]}
    assert checked == {p["cwe"] for p in report["patterns"]["checked"]} & {
        member["cwe"] for member in reference["members"]
    }
    assert reference["note"].startswith("Reference list, not detection.")


def test_attention_that_rounds_to_nothing_is_not_listed(monkeypatch, tmp_path):
    report = run(monkeypatch, {5: 4.0}, tmp_path)

    assert [(a["line"], a["direction"]) for a in report["cards"][0]["attended"]] == [
        (1, "sem_from_str")
    ]
    assert report["cards"][0]["dataflow_path"] == [3, 4, 5]
    assert report["cards"][0]["dataflow_symbols"] == ["buf"]


def test_a_ranking_that_disagrees_with_the_services_is_refused(monkeypatch, tmp_path):
    stand_in = StandIn({3: 4.0, 5: 3.0, 4: 2.0}, PATTERNS)
    honest = stand_in.analyze

    def tampered(code: str, **kwargs: Any) -> Any:
        analysis = honest(code, **kwargs)
        analysis.report["ranked_lines"][0]["line"] = 4
        return analysis

    monkeypatch.setattr(stand_in, "analyze", tampered)
    monkeypatch.setattr(released, "_load", lambda: stand_in)
    monkeypatch.setattr(released, "PATTERN_RECORD", tmp_path / "absent.json")

    with pytest.raises(RuntimeError, match="not the service's ranking"):
        released.released_findings(SOURCE)


def test_nothing_to_analyse_is_an_input_error(monkeypatch, tmp_path):
    monkeypatch.setattr(released, "_load", lambda: StandIn({}, PATTERNS))

    with pytest.raises(ValueError, match="Paste a function"):
        released.released_findings("   \n")


def record(bundle: str) -> dict[str, Any]:
    kind = {"n": 10, "raised": 3, "rate": 0.3, "by_length": {}}
    return {
        "run_id": "r" * 32,
        "bundle": bundle,
        "held_out": False,
        "headlines": {"pattern": {"vulnerable": kind, "not_vulnerable": {**kind, "raised": 2}}},
    }


def test_the_pattern_record_is_quoted_only_for_the_bundle_it_was_measured_with(
    monkeypatch, tmp_path
):
    path = tmp_path / "verdicts_val.json"
    monkeypatch.setattr(released, "PATTERN_RECORD", path)
    monkeypatch.setenv(released.BUNDLE_ENV, str(tmp_path / "served"))

    assert released.pattern_record() is None  # not generated

    path.write_text(json.dumps(record("D:/elsewhere/another")), encoding="utf-8")
    assert released.pattern_record() is None  # another checkpoint's

    path.write_text(json.dumps(record("D:/elsewhere/served")), encoding="utf-8")
    quoted = released.pattern_record()
    assert quoted is not None
    assert quoted["vulnerable"] == {"n": 10, "raised": 3}
    assert quoted["not_vulnerable"] == {"n": 10, "raised": 2}
    assert quoted["run_id"] == "r" * 32


@pytest.mark.skipif(not released.PATTERN_RECORD.exists(), reason="no screening record on disk")
def test_the_record_on_disk_reads_in_the_shape_the_page_quotes(monkeypatch):
    # A fixture written from memory of a file's shape passes against itself (landmine #71).
    stored = json.loads(released.PATTERN_RECORD.read_text(encoding="utf-8"))
    monkeypatch.setenv(released.BUNDLE_ENV, str(stored["bundle"]))

    quoted = released.pattern_record()

    assert quoted is not None
    for kind in ("vulnerable", "not_vulnerable"):
        assert 0 <= quoted[kind]["raised"] <= quoted[kind]["n"]


def test_every_sample_names_lines_of_its_own_code():
    samples = sample_functions()

    # `?sample=0` has always opened the first pipeline sample.
    assert [sample["group"] for sample in samples[:4]] == ["pipeline"] * 4
    assert {sample["group"] for sample in samples} == set(GROUPS)
    for sample in samples:
        lines = sample["code"].split("\n")
        assert sample["name"]
        assert sample["note"]
        assert (sample["group"] == "weakness") == bool(sample["lines"]), sample["name"]
        for number in sample["lines"]:
            # 1-indexed, inside the function, and on a line with code.
            assert 1 <= number <= len(lines), sample["name"]
            assert lines[number - 1].strip(), sample["name"]
        if sample["group"] != "pipeline":
            # The released model strips comments: one would be a line the reader sees and
            # the model does not.
            assert "/*" not in sample["code"], sample["name"]
            assert "//" not in sample["code"], sample["name"]


@pytest.mark.slow
@pytest.mark.skipif(not released.BUNDLE.is_dir(), reason="the bundle of record is not on disk")
def test_the_page_shows_what_the_packaged_service_answers(monkeypatch):
    monkeypatch.delenv(released.BUNDLE_ENV, raising=False)
    code = next(s["code"] for s in sample_functions() if s["name"] == "Integer overflow")

    page = released.released_findings(code)
    served = released._load().analyze(code, language="C", unit_id="paste").report

    matched = [finding for finding in served["findings"] if finding["pattern"]]
    assert [(c["line"], c["rank"], c["pattern"]) for c in page["cards"]] == [
        (f["line"], f["rank"], f["pattern"]) for f in matched
    ]
    assert [c["explanation"] for c in page["cards"]] == [f["explanation"] for f in matched]
    assert [(e["line"], e["rank"]) for e in page["top"]] == [
        (e["line"], e["rank"]) for e in served["ranked_lines"]
    ]
    flagged = {entry["line"] for entry in page["scores"] if entry["flagged"]}
    assert flagged == {finding["line"] for finding in served["findings"]}
    assert page["model"]["run_id"].startswith("ba7dcc5f")
