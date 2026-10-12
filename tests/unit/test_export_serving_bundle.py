"""The measured record a bundle can carry (``scripts/export_serving_bundle.py``).

The record is what the service shows beside every result, so what is pinned here is that
it is a quotation: every figure is the evaluation file's own, the file is named with its
digest, and a file made under other conditions than the bundle's — another run, another
threshold, another text variant — is refused instead of quoted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from vulnloc.serving.bundle import file_digest
from vulnloc.utils.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "export_serving_bundle.py"

RUN = "a" * 32
THRESHOLD = 0.0925


def _load() -> Any:
    """Import ``scripts/export_serving_bundle.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_export_serving_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporting = _load()


def _metric(value: float) -> dict[str, float]:
    """A ranking metric as the evaluation stores it: one value per tie order."""
    return {"source": value, "best": value + 0.01, "worst": value - 0.01}


def _reading(units: int, top_5: float) -> dict[str, Any]:
    return {
        "units": units,
        "ifa_median": 5.0,
        "metrics": {
            "top_1": _metric(0.25),
            "top_3": _metric(0.4),
            "top_5": _metric(top_5),
            "top_10": _metric(0.6),
        },
    }


def _evaluation(tmp_path: Path, **changes: Any) -> Path:
    payload: dict[str, Any] = {
        "run_id": "e" * 32,
        "split": "test",
        "held_out": True,
        "strip_comments": True,
        "tie_order": "source",
        "runs": {
            RUN: {
                "ranking": {
                    "all_lines": _reading(541, 0.46),
                    "blank_lines_withheld": _reading(533, 0.47),
                },
                "source_order": {
                    "all_lines": _reading(541, 0.26),
                    "blank_lines_withheld": _reading(533, 0.28),
                },
                "pooled": {"threshold": THRESHOLD, "precision": 0.16, "recall": 0.3},
            }
        },
    }
    payload.update(changes)
    path = tmp_path / "evaluation_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_every_figure_is_the_evaluation_files_own(tmp_path: Path) -> None:
    path = _evaluation(tmp_path)

    block = exporting.measured_block(path, RUN, threshold=THRESHOLD, strip_comments=True)

    assert block["partition"] == "test"
    assert block["held_out"] is True
    assert block["evaluation_run"] == "e" * 32
    assert block["source"] == path.as_posix()
    assert block["source_sha256"] == file_digest(path)
    # The service never ranks a blank line, so it quotes the reading that withholds them.
    assert block["reading"] == "blank_lines_withheld"
    assert block["units"] == 533
    # ...and of the three tie orders, the one the evaluation itself reports.
    assert block["tie_order"] == "source"
    assert block["top_5"] == 0.47
    assert block["source_order_top_5"] == 0.28
    assert block["top_1"] == 0.25
    assert block["top_10"] == 0.6
    assert block["line_precision"] == 0.16
    assert block["line_recall"] == 0.3
    assert "not a rate of detecting vulnerable functions" in block["note"]


def test_a_file_with_no_record_of_the_run_is_refused(tmp_path: Path) -> None:
    path = _evaluation(tmp_path)

    with pytest.raises(ConfigError, match="no record of this run"):
        exporting.measured_block(path, "b" * 32, threshold=THRESHOLD, strip_comments=True)


def test_figures_taken_at_another_threshold_are_not_this_bundles(tmp_path: Path) -> None:
    path = _evaluation(tmp_path)

    with pytest.raises(ConfigError, match="another threshold"):
        exporting.measured_block(path, RUN, threshold=0.5, strip_comments=True)


def test_figures_taken_on_the_other_text_variant_are_refused(tmp_path: Path) -> None:
    path = _evaluation(tmp_path, strip_comments=False)

    with pytest.raises(ConfigError, match="another text variant"):
        exporting.measured_block(path, RUN, threshold=THRESHOLD, strip_comments=True)


def test_a_missing_evaluation_file_is_an_error_not_an_empty_record(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        exporting.measured_block(
            tmp_path / "absent.json", RUN, threshold=THRESHOLD, strip_comments=True
        )


def test_the_shape_assumed_here_is_the_shape_of_the_evaluation_of_record() -> None:
    """The fixture above is hand-made; this reads the real file when the tree has one."""
    real = REPO_ROOT / "artifacts" / "phase8" / "evaluation_test.json"
    if not real.is_file():
        pytest.skip("no held-out evaluation in this tree (a fresh clone has none)")
    payload = json.loads(real.read_text(encoding="utf-8"))
    run_id, entry = next(iter(payload["runs"].items()))

    block = exporting.measured_block(
        real,
        run_id,
        threshold=float(entry["pooled"]["threshold"]),
        strip_comments=bool(payload["strip_comments"]),
    )

    reading = entry["ranking"]["blank_lines_withheld"]
    assert block["top_5"] == reading["metrics"]["top_5"][payload["tie_order"]]
    assert block["units"] == reading["units"]
    assert 0.0 <= block["line_recall"] <= 1.0
