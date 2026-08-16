"""Guards the split-building pipeline (Phases.md Phase 1, ADR-003).

The exit criterion this file exists for is *re-running the pipeline reproduces
byte-identical splits*. That is tested as byte-identity of the written file, not as
equality of the in-memory assignment: the assignment could match while a changed writer
produced different line endings or ordering, and downstream phases read the file.

The other half is refusal. A split that silently changes is worse than one that fails,
because every experiment after it would be comparing against a different partition.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vulnloc.data.pipeline import (
    ASSIGNMENT_FILENAME,
    REPORT_FILENAME,
    Adr003Membership,
    build_splits,
    clean_eval_overlap,
    compose_report,
    load_adr003_membership,
    load_units,
    project_breakdown,
    split_composition,
    write_split_artifacts,
)
from vulnloc.data.splits import CLEAN_EVAL, EXCLUDED, TRAIN, SplitSpec, Unit
from vulnloc.utils.exceptions import DataError

PROJECTS = ("chrome", "linux", "ffmpeg", "qemu", "php", "krb5")


def make_units(n: int = 240) -> list[Unit]:
    return [
        Unit(
            unit_id=f"u{i:04d}",
            project=PROJECTS[i % len(PROJECTS)],
            func_hash=1000 + i,
            date=f"{2010 + i % 9:04d}-01-01",
            target=int(i % 8 == 0),
            has_line_labels=i % 16 == 0,
            n_lines=10 + i % 5,
            n_flagged_lines=2 if i % 16 == 0 else 0,
        )
        for i in range(n)
    ]


def build(units, **kwargs):
    return build_splits(units, spec=SplitSpec(seed=7), **kwargs)


# --- reproducibility -------------------------------------------------------------


def test_rerunning_writes_a_byte_identical_assignment(tmp_path):
    units = make_units()
    for _ in range(2):
        result = build(units)
        report = write_split_artifacts(tmp_path, result, {"strategy": result.strategy})
    first = (tmp_path / ASSIGNMENT_FILENAME).read_bytes()

    # Reorder the corpus: content hashing must not care, a shuffle would.
    result = build(list(reversed(units)))
    write_split_artifacts(tmp_path, result, {"strategy": result.strategy})

    assert (tmp_path / ASSIGNMENT_FILENAME).read_bytes() == first
    assert report["reproduced"] is True


def test_first_run_reports_nothing_to_compare_against(tmp_path):
    result = build(make_units())

    report = write_split_artifacts(tmp_path, result, {})

    assert report["reproduced"] is None


def test_a_changed_split_fails_instead_of_overwriting(tmp_path):
    units = make_units()
    write_split_artifacts(tmp_path, build(units), {})
    before = (tmp_path / ASSIGNMENT_FILENAME).read_bytes()

    changed = build_splits(units, spec=SplitSpec(train=0.5, val=0.25, test=0.25, seed=7))
    with pytest.raises(DataError, match="assigned differently"):
        write_split_artifacts(tmp_path, changed, {})

    assert (tmp_path / ASSIGNMENT_FILENAME).read_bytes() == before


def test_a_deliberate_change_is_allowed_with_overwrite(tmp_path):
    units = make_units()
    write_split_artifacts(tmp_path, build(units), {})

    changed = build_splits(units, spec=SplitSpec(train=0.5, val=0.25, test=0.25, seed=7))
    report = write_split_artifacts(tmp_path, changed, {}, overwrite=True)

    assert report["reproduced"] is False


def test_assignment_file_uses_lf_endings(tmp_path):
    # Byte-identity has to hold across platforms, and csv defaults to CRLF on Windows.
    write_split_artifacts(tmp_path, build(make_units()), {})

    assert b"\r\n" not in (tmp_path / ASSIGNMENT_FILENAME).read_bytes()


def test_a_foreign_csv_is_not_mistaken_for_an_assignment(tmp_path):
    (tmp_path / ASSIGNMENT_FILENAME).write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(DataError, match="not a split assignment"):
        write_split_artifacts(tmp_path, build(make_units()), {})


# --- ADR-003 membership ----------------------------------------------------------


def test_membership_is_read_from_the_audit_report(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"clean_eval_hashes": [1, 2], "excluded_hashes": [3]}))

    membership = load_adr003_membership(path)

    assert membership.clean_eval == {1, 2}
    assert membership.excluded == {3}


def test_an_audit_without_the_overlap_is_refused(tmp_path):
    # skip_overlap=true produces a report with no membership. Treating that as "no
    # exclusions" would train on functions ADR-003 drops and evaluate on nothing.
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"totals": {"rows": 10}}))

    with pytest.raises(DataError, match="skip_overlap"):
        load_adr003_membership(path)


def test_a_missing_audit_report_is_refused(tmp_path):
    with pytest.raises(DataError, match="Run scripts/audit_data.py"):
        load_adr003_membership(tmp_path / "nope.json")


def test_held_out_and_excluded_units_leave_the_train_val_test_pool():
    units = make_units()
    membership = Adr003Membership(clean_eval=frozenset({1000, 1001}), excluded=frozenset({1002}))

    result = build(units, membership=membership)

    assert result.assignment["u0000"] == CLEAN_EVAL
    assert result.assignment["u0001"] == CLEAN_EVAL
    assert result.assignment["u0002"] == EXCLUDED
    assert result.leakage.clean_eval_in_train == set()


# --- statistics ------------------------------------------------------------------


def test_composition_keeps_the_two_denominators_apart():
    units = make_units()
    result = build(units)

    comp = split_composition(units, result.assignment)["all"].as_dict()

    assert comp["functions"] == len(units)
    assert comp["vulnerable"] == sum(u.target for u in units)
    assert comp["line_labelled_functions"] == sum(
        1 for u in units if u.target and u.has_line_labels
    )
    assert comp["line_labelled_functions"] < comp["vulnerable"]
    assert comp["mean_lines_per_function"] == pytest.approx(
        sum(u.n_lines for u in units) / len(units), abs=0.01
    )


def test_composition_covers_every_partition_including_the_held_out_ones():
    units = make_units()
    result = build(units, membership=Adr003Membership(clean_eval=frozenset({1000})))

    compositions = split_composition(units, result.assignment)

    assert CLEAN_EVAL in compositions
    assert sum(c.functions for n, c in compositions.items() if n != "all") == len(units)


def test_excluded_units_are_counted_but_never_in_a_reportable_partition():
    units = make_units()
    result = build(units, membership=Adr003Membership(excluded=frozenset({1000})))

    compositions = split_composition(units, result.assignment)

    assert compositions[EXCLUDED].functions == 1
    assert "u0000" not in project_breakdown(units, result.assignment).get(TRAIN, [])


def test_clean_eval_overlap_is_measured_not_assumed():
    # ADR-003 selects by label confidence, so the clean set shares projects with train.
    # The caveat in Memory.md §7 #13 needs a number behind it.
    units = make_units()
    result = build(units, membership=Adr003Membership(clean_eval=frozenset({1000, 1001, 1002})))

    overlap = clean_eval_overlap(units, result.assignment)

    assert overlap["clean_eval_functions"] == 3
    assert overlap["in_a_training_project"] <= overlap["clean_eval_functions"]
    assert 0 <= overlap["in_a_training_project_pct"] <= 100


def test_report_is_json_serializable_and_carries_the_fingerprint(tmp_path):
    units = make_units()
    result = build(units)

    report = compose_report(
        result,
        split_composition(units, result.assignment),
        dataset="bigvul",
        membership=Adr003Membership(),
    )
    written = write_split_artifacts(tmp_path, result, report)
    reloaded = json.loads((tmp_path / REPORT_FILENAME).read_text(encoding="utf-8"))

    assert reloaded["fingerprint"] == result.fingerprint()
    assert reloaded["leakage"]["is_clean"] is True
    assert reloaded == written


# --- loading ---------------------------------------------------------------------


def test_load_units_goes_through_the_registry(tmp_path):
    pd.DataFrame(
        [
            {
                "index": 1,
                "target": 1,
                "flaw_line": "x = 1;",
                "flaw_line_index": "1",
                "processed_func": "int f() {\n  x = 1;\n}",
                "project": "linux",
                "commit_id": "a",
                "CVE ID": "CVE-1",
                "CWE ID": "CWE-1",
                "Publish Date": "2019-01-01",
                "lang": "C",
            }
        ]
    ).to_csv(tmp_path / "bigvul_train.csv", index=False)

    units = load_units("bigvul", tmp_path, splits=("bigvul_train",))

    assert [u.unit_id for u in units] == ["bigvul-1"]
    assert units[0].n_lines == 3
    assert units[0].n_flagged_lines == 1


def test_an_empty_corpus_is_refused_rather_than_split(tmp_path):
    pd.DataFrame(
        columns=[
            "index",
            "target",
            "flaw_line",
            "flaw_line_index",
            "processed_func",
            "project",
            "commit_id",
            "CVE ID",
            "CWE ID",
            "Publish Date",
            "lang",
        ]
    ).to_csv(tmp_path / "bigvul_train.csv", index=False)

    with pytest.raises(DataError, match="no units"):
        load_units("bigvul", tmp_path, splits=("bigvul_train",))
