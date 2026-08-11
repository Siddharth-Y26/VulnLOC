"""Guards line-label derivation (ADR-001 §D1.2, Architecture.md §1.1).

The line index is the atomic unit of this project, so the off-by-one is the single most
consequential thing in the data layer: Big-Vul stores 0-indexed lines and everything
downstream expects 1-indexed. That conversion gets an explicit regression test, because
getting it wrong corrupts every label without raising anything.
"""

from __future__ import annotations

import pytest

from vulnloc.data.labeling import (
    FLAW_LINE_SEPARATOR,
    LabelIssue,
    LabellingStats,
    derive_line_labels,
    split_source_lines,
    summarize,
)
from vulnloc.utils.exceptions import DataError

SOURCE = "int f(int n) {\n  char buf[8];\n  strcpy(buf, n);\n  return 0;\n}"


def label(**overrides):
    kwargs = {
        "unit_id": "u1",
        "source": SOURCE,
        "target": 1,
        "flaw_line_index": "2",
        "flaw_line": "strcpy(buf, n);",
    }
    kwargs.update(overrides)
    return derive_line_labels(**kwargs)


# --- the off-by-one, which is the whole ballgame -----------------------------------


def test_flaw_line_index_is_converted_from_zero_to_one_indexed():
    # Big-Vul index 2 is the third line, "strcpy(buf, n);", which is line 3 1-indexed.
    unit = label(flaw_line_index="2")

    assert unit.vulnerable_lines == {3}
    assert unit.source.split("\n")[3 - 1].strip() == "strcpy(buf, n);"


def test_first_line_maps_to_one_not_zero():
    unit = label(flaw_line_index="0", flaw_line="int f(int n) {")

    assert unit.vulnerable_lines == {1}
    assert 0 not in unit.vulnerable_lines


def test_multiple_indices_all_shift_by_one():
    unit = label(
        flaw_line_index="1,2,3",
        flaw_line=FLAW_LINE_SEPARATOR.join(["char buf[8];", "strcpy(buf, n);", "return 0;"]),
    )

    assert unit.vulnerable_lines == {2, 3, 4}


def test_line_label_vector_is_aligned_to_the_source():
    unit = label(flaw_line_index="2")

    vector = unit.line_label_vector()

    assert len(vector) == unit.n_lines == 5
    assert vector == [0, 0, 1, 0, 0]


def test_line_vector_marks_every_flagged_line():
    unit = label(
        flaw_line_index="0,4",
        flaw_line=FLAW_LINE_SEPARATOR.join(["int f(int n) {", "}"]),
    )

    assert unit.line_label_vector() == [1, 0, 0, 0, 1]


# --- line splitting ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a\nb", ["a", "b"]),
        ("a\r\nb", ["a", "b"]),
        ("a\rb", ["a", "b"]),
        ("a\r\nb\nc\rd", ["a", "b", "c", "d"]),
        ("", [""]),
    ],
)
def test_line_splitting_normalizes_endings(text, expected):
    # Big-Vul mixes CRLF and lone CR. A trailing \r breaks text comparison now and the
    # Tree-sitter byte-offset mapping later.
    assert split_source_lines(text) == expected


def test_crlf_source_still_yields_correct_line_numbers():
    unit = derive_line_labels(
        unit_id="u",
        source="int a;\r\nint b;\r\nint c;",
        target=1,
        flaw_line_index="1",
        flaw_line="int b;",
    )

    assert unit.vulnerable_lines == {2}
    assert unit.issues == ()


# --- empty and missing labels ------------------------------------------------------


@pytest.mark.parametrize("empty", ["", "   ", None, "nan", "NaN"])
def test_positive_without_line_labels_is_flagged_not_dropped(empty):
    # 24.4% of Big-Vul positives look like this. They stay usable for the unit-level
    # task, so they must be marked rather than discarded.
    unit = label(flaw_line_index=empty, flaw_line=None)

    assert unit.vulnerable_lines == frozenset()
    assert unit.has_line_labels is False
    assert LabelIssue.NO_LINE_LABELS in unit.issues
    assert unit.target == 1


def test_negative_without_labels_is_clean():
    unit = label(target=0, flaw_line_index="", flaw_line=None)

    assert unit.issues == ()
    assert unit.is_trustworthy


def test_labels_on_a_negative_function_are_flagged():
    unit = label(target=0, flaw_line_index="2")

    assert LabelIssue.LABELS_ON_NEGATIVE in unit.issues


# --- consistency checks ------------------------------------------------------------


def test_out_of_range_index_is_flagged_and_dropped():
    unit = label(flaw_line_index="2,99", flaw_line=None)

    assert LabelIssue.OUT_OF_RANGE in unit.issues
    assert unit.vulnerable_lines == {3}  # the valid one survives


def test_negative_index_is_out_of_range():
    unit = label(flaw_line_index="-1", flaw_line=None)

    assert LabelIssue.OUT_OF_RANGE in unit.issues
    assert unit.vulnerable_lines == frozenset()


def test_text_mismatch_is_detected():
    unit = label(flaw_line_index="2", flaw_line="this is not the line")

    assert LabelIssue.TEXT_MISMATCH in unit.issues


def test_count_mismatch_is_detected():
    unit = label(flaw_line_index="1,2", flaw_line="char buf[8];")

    assert LabelIssue.COUNT_MISMATCH in unit.issues


def test_matching_text_produces_no_issues():
    unit = label(flaw_line_index="2", flaw_line="  strcpy(buf, n);  ")

    assert unit.issues == ()
    assert unit.is_trustworthy


def test_verification_is_skipped_when_flaw_line_is_absent():
    unit = label(flaw_line_index="2", flaw_line=None)

    assert unit.issues == ()


def test_malformed_index_raises():
    with pytest.raises(DataError, match="Malformed flaw_line_index"):
        label(flaw_line_index="1,oops")


# --- metadata and aggregation ------------------------------------------------------


def test_metadata_is_carried_through():
    unit = label(metadata={"project": "Chrome", "CVE ID": "CVE-2019-1"})

    assert unit.metadata["project"] == "Chrome"


def test_metadata_defaults_to_empty_and_is_copied():
    source = {"project": "x"}
    unit = label(metadata=source)
    source["project"] = "mutated"

    assert unit.metadata["project"] == "x"


def test_summarize_counts_positives_and_issues():
    units = [
        label(unit_id="a", flaw_line_index="2"),
        label(unit_id="b", flaw_line_index="", flaw_line=None),
        label(unit_id="c", target=0, flaw_line_index="", flaw_line=None),
    ]

    stats = summarize(units).as_dict()

    assert stats["total"] == 3
    assert stats["positive"] == 2
    assert stats["positive_with_lines"] == 1
    assert stats["flagged_lines"] == 1
    assert stats["issues"][LabelIssue.NO_LINE_LABELS] == 1


def test_empty_summary_is_all_zeros():
    assert summarize([]).as_dict()["total"] == 0


def test_stats_accumulate_incrementally():
    stats = LabellingStats()
    stats.record(label(flaw_line_index="1,2", flaw_line=None))
    stats.record(label(flaw_line_index="3", flaw_line=None))

    assert stats.flagged_lines == 3
    assert stats.positive == 2
