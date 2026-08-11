"""Guards the Big-Vul loader (ADR-001 §D1.1).

The important behaviours are defensive: refusing the wrong Big-Vul distribution, and
keeping unit ids stable across the re-split that ADR-001 §D1.3 requires.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vulnloc.data import DATASET_REGISTRY
from vulnloc.data.audit import function_hash
from vulnloc.data.labeling import LabelIssue
from vulnloc.data.loaders.bigvul import BigVulLoader
from vulnloc.data.splits import assign_splits
from vulnloc.utils.exceptions import DataError

ROWS = [
    {
        "index": 10,
        "target": 1,
        "flaw_line": "strcpy(buf, n);",
        "flaw_line_index": "2",
        "processed_func": "int f(int n) {\n  char buf[8];\n  strcpy(buf, n);\n}",
        "project": "Chrome",
        "commit_id": "abc123",
        "CVE ID": "CVE-2019-0001",
        "CWE ID": "CWE-119",
        "Publish Date": "2019-03-14",
        "lang": "C",
    },
    {
        "index": 11,
        "target": 0,
        "flaw_line": None,
        "flaw_line_index": None,
        "processed_func": "int g(void) {\n  return 0;\n}",
        "project": "linux",
        "commit_id": "def456",
        "CVE ID": None,
        "CWE ID": None,
        "Publish Date": "2020-01-02",
        "lang": "C",
    },
    {
        "index": 12,
        "target": 1,
        "flaw_line": None,
        "flaw_line_index": None,  # positive with no line labels — 24% of Big-Vul
        "processed_func": "int h(void) {\n  return 1;\n}",
        "project": "linux",
        "commit_id": "ghi789",
        "CVE ID": "CVE-2020-0002",
        "CWE ID": "CWE-125",
        "Publish Date": "2020-06-30",
        "lang": "C",
    },
]


@pytest.fixture
def corpus(tmp_path):
    pd.DataFrame(ROWS).to_csv(tmp_path / "bigvul_val.csv", index=False)
    return tmp_path


@pytest.fixture
def loader(corpus):
    return BigVulLoader(corpus, splits=("bigvul_val",))


def test_loader_is_registered():
    assert DATASET_REGISTRY.get("bigvul") is BigVulLoader


def test_iter_functions_reads_every_row(loader):
    assert len(list(loader.iter_functions())) == 3


def test_line_labels_are_one_indexed(loader):
    unit = next(u for u in loader.iter_functions() if u.unit_id == "bigvul-10")

    assert unit.vulnerable_lines == {3}
    assert unit.line_label_vector() == [0, 0, 1, 0]
    assert unit.issues == ()


def test_positive_without_line_labels_is_marked(loader):
    unit = next(u for u in loader.iter_functions() if u.unit_id == "bigvul-12")

    assert unit.target == 1
    assert not unit.has_line_labels
    assert LabelIssue.NO_LINE_LABELS in unit.issues


def test_metadata_is_populated(loader):
    unit = next(u for u in loader.iter_functions() if u.unit_id == "bigvul-10")

    assert unit.metadata["project"] == "Chrome"
    assert unit.metadata["CVE ID"] == "CVE-2019-0001"
    assert unit.metadata["Publish Date"] == "2019-03-14"


def test_missing_metadata_becomes_none_not_the_string_nan(loader):
    # pandas turns absent CSV cells into float('nan'); str() of that is "nan", which
    # would silently become a CVE id of "nan" in the data card.
    unit = next(u for u in loader.iter_functions() if u.unit_id == "bigvul-11")

    assert unit.metadata["CVE ID"] is None
    assert unit.metadata["CWE ID"] is None


def test_iter_units_matches_iter_functions(loader):
    units = {u.unit_id: u for u in loader.iter_units()}
    functions = {f.unit_id: f for f in loader.iter_functions()}

    assert set(units) == set(functions)
    for unit_id, unit in units.items():
        assert unit.target == functions[unit_id].target
        assert unit.has_line_labels == functions[unit_id].has_line_labels


def test_unit_hash_matches_the_primevul_join_key(loader):
    unit = next(u for u in loader.iter_units() if u.unit_id == "bigvul-10")
    function = next(f for f in loader.iter_functions() if f.unit_id == "bigvul-10")

    assert unit.func_hash == function_hash(function.source)


def test_units_carry_dates_for_the_temporal_split(loader):
    units = list(loader.iter_units())

    assert all(u.date for u in units)
    assert assign_splits(units, strategy="temporal").counts()


def test_unit_ids_come_from_the_corpus_index_not_the_row_position(corpus):
    # ADR-001 §D1.3 re-splits the corpus, so ids must survive being repartitioned.
    ids = {u.unit_id for u in BigVulLoader(corpus, splits=("bigvul_val",)).iter_units()}

    assert ids == {"bigvul-10", "bigvul-11", "bigvul-12"}


def test_chunking_does_not_change_the_result(corpus):
    small = list(BigVulLoader(corpus, splits=("bigvul_val",), chunksize=1).iter_functions())
    large = list(BigVulLoader(corpus, splits=("bigvul_val",), chunksize=999).iter_functions())

    assert [u.unit_id for u in small] == [u.unit_id for u in large]
    assert [u.vulnerable_lines for u in small] == [u.vulnerable_lines for u in large]


def test_rows_without_source_are_skipped(tmp_path):
    rows = [*ROWS, {**ROWS[0], "index": 99, "processed_func": None}]
    pd.DataFrame(rows).to_csv(tmp_path / "bigvul_val.csv", index=False)

    ids = {u.unit_id for u in BigVulLoader(tmp_path, splits=("bigvul_val",)).iter_functions()}

    assert "bigvul-99" not in ids


def test_wrong_bigvul_distribution_is_rejected(tmp_path):
    # The bstee615/bigvul mirror and the original ZeoVan CSV have no line columns.
    pd.DataFrame([{"target": 1, "func_before": "int a(){}", "func_after": "int a(){}"}]).to_csv(
        tmp_path / "bigvul_val.csv", index=False
    )

    with pytest.raises(DataError, match="LineVul release"):
        list(BigVulLoader(tmp_path, splits=("bigvul_val",)).iter_functions())


def test_missing_split_file_is_reported(tmp_path):
    with pytest.raises(DataError, match="not found"):
        list(BigVulLoader(tmp_path, splits=("bigvul_train",)).iter_units())
