"""Guards the corpus audit (ADR-003).

The 19.2% label-noise rate is a *reported result*, so the code producing it is tested
like any other result-bearing code (`Rules.md` §1.3). The join key is the fragile part:
if PrimeVul's hash normalisation is changed, the intersection silently collapses to
near-zero rather than failing, so it gets an explicit regression test.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from vulnloc.data.audit import (
    OverlapReport,
    flaw_line_indices,
    function_hash,
    load_primevul_labels,
    overlap_with_primevul,
    scan_split,
)
from vulnloc.utils.exceptions import DataError


def test_function_hash_matches_primevul_definition():
    # Regression guard: PrimeVul's func_hash is md5 of the source with spaces, tabs,
    # newlines and carriage returns removed, as an int. Verified against 300 real
    # records. Changing this silently breaks the join.
    source = "int f(int x) {\n\treturn x;\r\n}"
    expected = int(hashlib.md5(b"intf(intx){returnx;}").hexdigest(), 16)

    assert function_hash(source) == expected


def test_function_hash_ignores_all_whitespace_differences():
    assert function_hash("int f() { return 1; }") == function_hash("int  f(){\n return 1;\t}")


def test_function_hash_is_sensitive_to_real_changes():
    assert function_hash("return x;") != function_hash("return y;")


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("1,2,3", [1, 2, 3]),
        ("7", [7]),
        (" 4 , 5 ", [4, 5]),
        ("1,2,", [1, 2]),
        ("", []),
        ("   ", []),
        (None, []),
        (float("nan"), []),
    ],
)
def test_flaw_line_indices_parsing(cell, expected):
    assert flaw_line_indices(cell) == expected


def test_flaw_line_indices_rejects_garbage():
    with pytest.raises(DataError, match="Malformed flaw_line_index"):
        flaw_line_indices("1,foo,3")


def _write_bigvul(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def _bigvul_row(*, target, func, flaw_index="", project="proj", cwe="CWE-119"):
    return {
        "target": target,
        "flaw_line": "",
        "flaw_line_index": flaw_index,
        "processed_func": func,
        "project": project,
        "CVE ID": "CVE-2020-0001",
        "CWE ID": cwe,
    }


def test_scan_split_counts_composition(tmp_path):
    path = _write_bigvul(
        tmp_path / "bigvul_mini.csv",
        [
            _bigvul_row(target=1, func="int a(){}", flaw_index="1,2"),
            _bigvul_row(target=1, func="int b(){}", flaw_index=""),  # positive, no lines
            _bigvul_row(target=0, func="int c(){}", project="other"),
        ],
    )

    stats = scan_split(path).as_dict()

    assert stats["rows"] == 3
    assert stats["vulnerable"] == 2
    assert stats["vulnerable_with_lines"] == 1
    assert stats["vulnerable_without_lines"] == 1
    assert stats["nonvulnerable_with_lines"] == 0
    assert stats["flagged_lines"] == 2
    assert stats["distinct_projects"] == 2


def test_scan_split_chunking_does_not_change_totals(tmp_path):
    rows = [_bigvul_row(target=i % 2, func=f"int f{i}(){{}}", flaw_index="1") for i in range(50)]
    path = _write_bigvul(tmp_path / "chunky.csv", rows)

    whole = scan_split(path, chunksize=1000).as_dict()
    chunked = scan_split(path, chunksize=7).as_dict()

    assert whole == chunked


def test_scan_split_rejects_a_corpus_without_line_columns(tmp_path):
    # The bstee615/bigvul mirror and the original CSV lack flaw_line_index. Using one
    # would silently produce a corpus incomparable with every published baseline.
    path = tmp_path / "wrong.csv"
    pd.DataFrame([{"target": 1, "func_before": "int a(){}"}]).to_csv(path, index=False)

    with pytest.raises(DataError, match="LineVul release"):
        scan_split(path)


def test_scan_split_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        scan_split(tmp_path / "absent.csv")


def _write_primevul(directory, records_by_split):
    directory.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        with (directory / f"primevul_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return directory


def test_load_primevul_labels_takes_the_vulnerable_claim(tmp_path):
    same = function_hash("int a(){}")
    directory = _write_primevul(
        tmp_path / "primevul",
        {
            "train": [{"func_hash": same, "target": 0}],
            "valid": [{"func_hash": same, "target": 1}],
            "test": [{"func_hash": function_hash("int b(){}"), "target": 0}],
        },
    )

    labels = load_primevul_labels(directory)

    assert labels[same] == 1
    assert len(labels) == 2


def test_load_primevul_labels_missing_split(tmp_path):
    directory = _write_primevul(tmp_path / "primevul", {"train": []})

    with pytest.raises(DataError, match="not found"):
        load_primevul_labels(directory)


def test_overlap_partitions_into_clean_excluded_and_absent(tmp_path):
    agree, contradict, absent = "int a(){}", "int b(){}", "int c(){}"
    path = _write_bigvul(
        tmp_path / "bigvul.csv",
        [
            _bigvul_row(target=1, func=agree, flaw_index="1,2,3"),
            _bigvul_row(target=1, func=contradict, flaw_index="4"),
            _bigvul_row(target=1, func=absent, flaw_index="5"),
            _bigvul_row(target=0, func="int d(){}"),  # negative: not a candidate
            _bigvul_row(target=1, func="int e(){}", flaw_index=""),  # no lines: not a candidate
        ],
    )
    labels = {function_hash(agree): 1, function_hash(contradict): 0}

    report = overlap_with_primevul([path], labels)

    assert report.candidates == 3
    assert report.agrees_vulnerable == 1
    assert report.says_benign == 1
    assert report.absent == 1
    assert report.retained_flagged_lines == 3
    assert report.clean_eval_hashes == {function_hash(agree)}
    assert report.excluded_hashes == {function_hash(contradict)}


def test_noise_rate_excludes_absent_functions():
    # ADR-003: absence conflates "rejected" with "deduplicated", so it must never be
    # counted as disagreement. A thousand absent functions must not move the rate.
    report = OverlapReport(candidates=1002, agrees_vulnerable=8, says_benign=2, absent=992)

    assert report.decided == 10
    assert report.noise_rate == pytest.approx(0.2)


def test_noise_rate_is_zero_when_nothing_is_decided():
    assert OverlapReport(candidates=5, absent=5).noise_rate == 0.0


def test_overlap_is_chunk_size_invariant(tmp_path):
    rows = [_bigvul_row(target=1, func=f"int f{i}(){{}}", flaw_index="1") for i in range(40)]
    path = _write_bigvul(tmp_path / "bigvul.csv", rows)
    labels = {function_hash(f"int f{i}(){{}}"): i % 2 for i in range(40)}

    small = overlap_with_primevul([path], labels, chunksize=3).as_dict()
    large = overlap_with_primevul([path], labels, chunksize=1000).as_dict()

    assert small == large
