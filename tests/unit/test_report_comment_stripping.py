"""Guards the corpus report that is ADR-010's evidence.

Its headline — "the stripper agrees with the grammar on every clean unit" — is only worth
quoting if the comparison can fail. So the grammar-side stripper is pinned on its own, and
the per-partition counts are checked on a case small enough to count by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from vulnloc.representation import CodeUnit, TreeSitterParser
from vulnloc.utils.text import strip_comments

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_comment_stripping.py"


def _load() -> Any:
    """Import ``scripts/report_comment_stripping.py`` by path."""
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_report_comment_stripping", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load()

CLEAN = "int f(int a)\n{\nint b = a;  /* keep a copy */\n// nothing else\nreturn b;\n}"
LOST = "int g(int a)\n{\nint b = a;  /* keep a copy *\nreturn b;\n}"


def unit(unit_id: str, source: str, flagged: set[int]) -> CodeUnit:
    return CodeUnit.from_source(
        unit_id, source, language="C", target=1, vulnerable_lines=frozenset(flagged)
    )


def test_the_grammar_side_stripper_removes_exactly_the_comment_nodes():
    parsed = TreeSitterParser().parse(CLEAN, language="C", unit_id="u")

    stripped = report.strip_with_grammar(parsed)

    assert report.squeezed(stripped) == ["intf(inta)", "{", "intb=a;", "", "returnb;", "}"]
    assert len(stripped.split("\n")) == len(CLEAN.split("\n"))


def test_the_two_strippers_agree_on_a_clean_unit_and_the_comparison_can_fail():
    parsed = TreeSitterParser().parse(CLEAN, language="C", unit_id="u")
    by_grammar = report.squeezed(report.strip_with_grammar(parsed))

    assert report.squeezed(strip_comments(CLEAN)) == by_grammar
    assert report.squeezed(CLEAN) != by_grammar, "leaving the comments in must not pass"


def test_a_partition_is_counted_line_by_line():
    units = [unit("clean", CLEAN, {4}), unit("lost", LOST, {3})]

    section = report.report_split(units, TreeSitterParser(), policy="best_of_two")

    assert section["line_labelled_units"] == 2
    assert section["lines"] == 11
    assert section["flagged_lines"] == 2
    # CLEAN line 4 held only a comment: flagged, and blank once it is stripped.
    assert (section["flagged_blank_before"], section["flagged_blank_after"]) == (0, 1)
    assert section["units_whose_flagged_lines_are_all_blank_after"] == 1
    assert (section["lost_slash_before"], section["lost_slash_after"]) == (1, 0)
    assert (section["intact_end_before"], section["intact_end_after"]) == (1, 0)
    assert section["units_with_lost_slash"] == 1
    # Only the unit without a lost-slash line is compared with the grammar.
    assert (section["compared_with_grammar"], section["identical_to_grammar"]) == (1, 1)
    assert section["identical_to_grammar_pct"] == 100.0
    assert section["disagreements_with_grammar"] == []
    assert section["clean_after"] == 2


def test_markers_are_counted_separately_for_flagged_and_unflagged_lines():
    section = report.report_split(
        [unit("clean", CLEAN, {3})], TreeSitterParser(), policy="best_of_two"
    )

    assert section["lines_with_comment_marker"] == 2
    assert section["flagged_lines_with_comment_marker"] == 1
    assert section["flagged_lines_with_comment_marker_pct"] == 100.0
    assert section["unflagged_lines_with_comment_marker_pct"] == 20.0


def test_percentages_of_nothing_are_zero_not_an_error():
    assert report.pct(0, 0) == 0.0
    assert report.pct(1, 4) == 25.0
