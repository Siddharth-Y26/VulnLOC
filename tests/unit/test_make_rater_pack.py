"""Guards the rater pack's rendering — what three people will actually read.

The sampling is ``utils.hashing.stable_sample`` and is tested there. What is pinned here
is the page: the flagged line is marked and only it, the explanation is shown verbatim,
and the reference a rater judges against comes from the dataset and says so.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from vulnloc.representation import CodeUnit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "make_rater_pack.py"


def _load() -> Any:
    """Import ``scripts/make_rater_pack.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_make_rater_pack", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pack = _load()

SOURCE = "\n".join(f"line_{number}();" for number in range(1, 61))
UNIT = CodeUnit.from_source(
    "bigvul-1",
    SOURCE,
    language="C",
    target=1,
    vulnerable_lines=frozenset({30, 31}),
    metadata={"CVE ID": "CVE-2000-0001", "CWE ID": "CWE-119", "project": "demo"},
)
RECORD = {
    "line": 30,
    "cited_lines": [2, 31],
    "text": "No known weakness pattern matches this line's text, so no cause is named.",
    "label": "Unclassified",
    "cwe": "",
}


def test_the_window_is_the_flagged_line_its_context_and_each_cited_line():
    lines = pack.shown_lines(60, 30, [2], context=3)

    assert lines == [1, 2, 3, 4, 27, 28, 29, 30, 31, 32, 33]


def test_the_window_stops_at_the_ends_of_the_function():
    assert pack.shown_lines(5, 1, [5], context=2) == [1, 2, 3, 4, 5]


def test_only_the_flagged_line_carries_the_flag_marker():
    page = pack.render_sample("S001", RECORD, UNIT)

    flagged = [row for row in page.splitlines() if row.startswith(">>")]
    assert flagged == [">>   30 | line_30();"]
    assert " ·   31 | line_31();" in page
    assert " ·    2 | line_2();" in page


def test_a_gap_between_shown_lines_is_marked():
    page = pack.render_sample("S001", RECORD, UNIT)

    assert "      ..." in page


def test_the_explanation_is_shown_verbatim_and_the_reference_is_attributed():
    page = pack.render_sample("S001", RECORD, UNIT)

    assert RECORD["text"] in page
    assert "CVE-2000-0001, CWE-119, project `demo`" in page
    assert "from the dataset, not from the tool" in page
    assert "The fix changed line(s) 30, 31 — the flagged line is one of them." in page


def test_a_flagged_line_the_fix_did_not_touch_is_not_called_one_of_them():
    page = pack.render_sample("S001", {**RECORD, "line": 10, "cited_lines": []}, UNIT)

    assert "is one of them" not in page


def test_the_readme_carries_all_four_dimensions_and_the_scale():
    source = {
        "source_run": "abc",
        "seed": 42,
        "fusion_kind": "cross_attention",
        "split": "val",
        "strip_comments": True,
    }

    readme = pack.render_readme(samples=100, raters=3, source=source)

    for name, _ in pack.DIMENSIONS:
        assert f"`{name}`" in readme
    assert "**1** Not at all" in readme and "**5** Completely" in readme
    assert "100 samples, 3 raters" in readme
