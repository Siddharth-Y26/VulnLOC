"""The explanation script's own decisions — the parts no model is needed to check.

Three of them can each produce a clean-looking file from something that should have been
refused or counted: a calibrator borrowed from another checkpoint, a masked sentinel
turned into a probability, and an explanation the verifier rejected slipping into the
output anyway. The fourth is arithmetic the Phase 7 exit criteria are read from.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnloc.explanation import EvidenceSettings, Explanation, TemplateExplainer
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.representation import CodeUnit
from vulnloc.training.line_scores import UnitScores
from vulnloc.utils.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "explain_checkpoint.py"


def _load() -> Any:
    """Import ``scripts/explain_checkpoint.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_explain_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


explain_checkpoint = _load()

SETTINGS = EvidenceSettings(
    top_k_attended=2, min_attention_weight=0.0, max_dataflow_hops=2, max_graph_nodes=3, max_tokens=6
)
SOURCE = "void f(char *dst, char *src, int n)\n{\nmemcpy(dst, src, n);\nreturn;\n}"
UNIT = CodeUnit.from_source(
    "u1",
    SOURCE,
    language="C",
    target=1,
    vulnerable_lines=frozenset({3}),
    metadata={"CWE ID": "CWE-787"},
)


def report(tmp_path: Path, per_run: dict[str, Any]) -> Path:
    path = tmp_path / "calibration_report.json"
    path.write_text(json.dumps({"per_run": per_run}), encoding="utf-8")
    return path


def test_a_calibrator_is_read_for_the_checkpoint_it_was_fitted_on(tmp_path):
    path = report(
        tmp_path,
        {"run-a": {"calibrator": {"kind": "platt", "scale": 2.0, "shift": -1.0}, "threshold": 0.2}},
    )

    calibrator, threshold = explain_checkpoint.load_calibration(path, "run-a")

    assert (calibrator.scale, calibrator.shift, calibrator.kind) == (2.0, -1.0, "platt")
    assert threshold == 0.2


def test_a_calibrator_is_never_borrowed_from_another_checkpoint(tmp_path):
    path = report(
        tmp_path,
        {"run-a": {"calibrator": {"kind": "platt", "scale": 2.0, "shift": -1.0}, "threshold": 0.2}},
    )

    with pytest.raises(ConfigError, match="never borrowed") as refusal:
        explain_checkpoint.load_calibration(path, "run-b")

    assert refusal.value.details["covered"] == ["run-a"]


def test_a_missing_calibration_report_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no calibration report"):
        explain_checkpoint.load_calibration(tmp_path / "absent.json", "run-a")


def test_each_text_variant_reads_its_own_scores_and_its_own_calibrator():
    shipped = explain_checkpoint.default_paths("")
    stripped = explain_checkpoint.default_paths("_stripped")
    retrained = explain_checkpoint.default_paths("_retrained")

    # The two variants measured before the retraining keep the paths they were written to.
    assert shipped == {
        "scores_dir": "./artifacts/phase6/scores",
        "calibration_report": "./artifacts/phase6/calibration_report.json",
        "output_dir": "./artifacts/phase7/explanations",
    }
    assert stripped["calibration_report"] == "./artifacts/phase6/stripped/calibration_report.json"
    assert retrained == {
        "scores_dir": "./artifacts/phase6/scores_retrained",
        "calibration_report": "./artifacts/phase6/retrained/calibration_report.json",
        "output_dir": "./artifacts/phase7/explanations_retrained",
    }
    for key in shipped:
        assert len({shipped[key], stripped[key], retrained[key]}) == 3, key


def test_stored_logits_become_calibrated_probabilities_and_a_sentinel_never_does():
    scores = UnitScores(
        unit_id="u1",
        logits=np.array([0.0, -1.0e4, 2.0], dtype=np.float32),  # the middle one is masked
        labels=np.array([0, 0, 1], dtype=np.int8),
        scoreable=np.array([True, False, True]),
    )

    pred = explain_checkpoint.prediction_from_scores(
        scores, LogitCalibrator(scale=1.0, shift=0.0, kind="identity"), 0.6
    )

    assert pred.probabilities == pytest.approx((0.5, 0.0, 0.8808), abs=1e-4)
    assert pred.scoreable == (True, False, True)
    assert pred.flagged_lines == (3,)


def _prediction(threshold: float = 0.5):
    return explain_checkpoint.Prediction("u1", (0.01, 0.01, 0.9, 0.7, 0.01), (True,) * 5, threshold)


def _explain(explainer: Any):
    return explain_checkpoint.explain_unit(
        UNIT,
        _prediction(),
        explainer,
        settings=SETTINGS,
        ast=None,
        dfg=None,
        attn_sem_from_str=None,
        attn_str_from_sem=None,
    )


def test_every_flagged_line_is_explained_and_recorded_with_the_datasets_label():
    result = _explain(TemplateExplainer(max_context_lines=2))

    assert result.flagged == 2
    assert [record["line"] for record in result.details] == [3, 4]
    assert [record["is_labelled"] for record in result.details] == [True, False]
    assert result.details[0]["template_id"] == "memcpy-length"
    assert result.details[0]["unit_cwe"] == "CWE-787"
    assert result.rejections == []
    assert result.contract is not None and result.contract["is_vulnerable"] is True


class _Ungrounded:
    """An explainer that cites a line it has no evidence for, on the top line only."""

    def explain(self, unit, pred, evidence):
        text = f"The model ranks line {evidence.line} first."
        if evidence.line == 3:
            text += " See also line 1."
        return Explanation(unit.unit_id, evidence.line, "Unclassified", "", text, "no-pattern")


def test_a_rejected_explanation_is_counted_and_never_reaches_the_output():
    result = _explain(_Ungrounded())

    assert len(result.rejections) == 1
    assert [record["line"] for record in result.details] == [4]
    # The unit's top flagged line has no verified explanation, so no contract is reported.
    assert result.contract is None


def _record(template_id: str, *, rank: int, labelled: bool, cwe: str = "", unit_cwe: str = ""):
    return {
        "template_id": template_id,
        "rank": rank,
        "is_labelled": labelled,
        "cwe": cwe,
        "unit_cwe": unit_cwe,
        "attended": [],
        "dataflow_path": [3] if labelled else [],
        "graph_nodes": ["n_1"],
    }


def test_the_summary_counts_what_the_exit_criteria_are_read_from():
    unit_result = explain_checkpoint.UnitResult
    results = {
        "a": unit_result(
            contract={},
            flagged=2,
            details=[
                _record("memcpy-length", rank=1, labelled=True, cwe="CWE-787", unit_cwe="CWE-787"),
                _record("no-pattern", rank=2, labelled=False),
            ],
        ),
        "b": unit_result(
            contract={},
            flagged=2,
            details=[_record("formatting-artifact", rank=1, labelled=True, unit_cwe="CWE-20")],
            rejections=[{"message": "rejected"}],
        ),
        "c": unit_result(contract={}, flagged=0),
    }

    summary = explain_checkpoint.summarise(results)

    assert summary["units"] == 3
    assert summary["units_flagged"] == 2
    assert summary["flagged_lines"] == 4
    assert (summary["explanations_verified"], summary["explanations_rejected"]) == (3, 1)
    assert summary["explained_pct"] == 75.0
    assert summary["grounding_rate_pct"] == 75.0
    # The artifact notice names no cause, so it is not a matched pattern.
    assert summary["pattern_matched_lines"] == 1
    assert summary["formatting_artifact_lines"] == 1
    assert summary["top_line_pattern_matched_pct"] == 50.0
    assert summary["top_line_formatting_artifact_pct"] == 50.0
    assert summary["labelled_pct_pattern_matched"] == 100.0
    assert summary["labelled_pct_no_pattern"] == 0.0
    assert (summary["cwe_comparable_units"], summary["cwe_exact_agreement_pct"]) == (1, 100.0)
    assert summary["templates"]["memcpy-length"] == 1


def test_an_empty_partition_summarises_to_zeros_not_a_division_error():
    summary = explain_checkpoint.summarise({})

    assert summary["explained_pct"] == 0.0
    assert summary["grounding_rate_pct"] == 0.0
