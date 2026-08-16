"""Guards the preprocessed shards (Phases.md Phase 1 deliverable).

Shards are where the corpus stops being a file format and starts being a dataset. The
risk they carry is silent loss: a record dropped here is invisible to every later phase,
which has no way to know it should have been there. So the tests are mostly about
round-tripping without loss and refusing to write a partial cache.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from vulnloc.data.cache import MANIFEST_FILENAME, iter_shard, shard_path, write_shards
from vulnloc.data.labeling import LabelIssue, derive_line_labels
from vulnloc.utils.exceptions import DataError

SOURCE = "int f(int n) {\n  char buf[8];\n  strcpy(buf, n);\n  return 0;\n}"


def unit(unit_id="u1", **overrides):
    kwargs = {
        "unit_id": unit_id,
        "source": SOURCE,
        "target": 1,
        "flaw_line_index": "2",
        "flaw_line": "strcpy(buf, n);",
        "metadata": {"project": "linux", "CVE ID": "CVE-2019-1", "commit_id": None},
    }
    kwargs.update(overrides)
    return derive_line_labels(**kwargs)


def corpus():
    return [
        unit("u1"),
        unit("u2", target=0, flaw_line_index=None, flaw_line=None),
        unit("u3", flaw_line_index="99", flaw_line=None),  # out of range
    ]


ASSIGNMENT = {"u1": "train", "u2": "val", "u3": "test"}


def test_every_record_round_trips(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    restored = next(iter_shard(tmp_path, "train"))
    original = corpus()[0]

    assert restored == original


def test_line_labels_survive_the_round_trip(tmp_path):
    # The line index is the atomic unit of this project (Rules.md §2). A shard that
    # loses or shifts it corrupts every downstream label with nothing raised.
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    restored = next(iter_shard(tmp_path, "train"))

    assert restored.vulnerable_lines == {3}
    assert restored.line_label_vector() == [0, 0, 1, 0, 0]


def test_issues_survive_as_enum_members_not_strings(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    restored = next(iter_shard(tmp_path, "test"))

    assert LabelIssue.OUT_OF_RANGE in restored.issues
    assert not restored.is_trustworthy


def test_records_go_to_the_partition_the_assignment_names(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    for unit_id, split in ASSIGNMENT.items():
        assert [u.unit_id for u in iter_shard(tmp_path, split)] == [unit_id]


def test_a_unit_missing_from_the_assignment_is_refused(tmp_path):
    # Skipping it would produce a cache that silently omits data, and nothing
    # downstream could detect the loss.
    with pytest.raises(DataError, match="absent from the split assignment"):
        write_shards(corpus(), {"u1": "train"}, tmp_path)


def test_selected_partitions_write_only_those_shards(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path, splits=["train"])

    assert shard_path(tmp_path, "train").exists()
    assert not shard_path(tmp_path, "val").exists()


def test_writing_twice_produces_identical_shards(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)
    first = shard_path(tmp_path, "train").read_bytes()
    stats = write_shards(corpus(), ASSIGNMENT, tmp_path)

    assert shard_path(tmp_path, "train").read_bytes() == first
    assert stats["train"].digest == hashlib.sha256(first).hexdigest()


def test_manifest_records_counts_and_digests(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest["total_functions"] == 3
    assert manifest["shards"]["train"]["line_labelled"] == 1
    assert manifest["shards"]["train"]["flagged_lines"] == 1
    assert len(manifest["shards"]["train"]["digest"]) == 64


def test_shards_use_lf_endings(tmp_path):
    write_shards(corpus(), ASSIGNMENT, tmp_path)

    assert b"\r\n" not in shard_path(tmp_path, "train").read_bytes()


def test_reading_a_missing_shard_says_how_to_make_it(tmp_path):
    with pytest.raises(DataError, match="build_cache"):
        list(iter_shard(tmp_path, "train"))


def test_unicode_sources_survive(tmp_path):
    # Big-Vul contains non-ASCII identifiers and comments; a mangled source would break
    # the byte-offset to line mapping in Phase 2 rather than failing here.
    source = "int f() {\n  // élève café 中文\n  return 0;\n}"
    units = [unit("u1", source=source, flaw_line_index="1", flaw_line=None)]

    write_shards(units, {"u1": "train"}, tmp_path)

    assert next(iter_shard(tmp_path, "train")).source == source
