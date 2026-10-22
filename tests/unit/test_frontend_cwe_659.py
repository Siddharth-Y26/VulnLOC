"""MITRE's CWE-659 view in the Findings response — reference data, never detection, never drawn.

Asked for on 2026-10-22: the list in the backend, returned with ``/api/findings`` and not
shown on the page. What these tests guard is that it cannot be read as more than it is: an
entry is ``checked`` only when a pattern the service tested carries its exact identifier,
and the page's files never mention it.
"""

from __future__ import annotations

import re
from pathlib import Path

from frontend.cwe_659 import MEMBERS, NOTE, VIEW, reference

from vulnloc.explanation.templates import all_templates

STATIC = Path(__file__).resolve().parents[2] / "frontend" / "static"


def tested() -> list[dict[str, str]]:
    """What the service reports as ``patterns.checked`` (``serving.report.patterns_checked``)."""
    return [{"id": t.id, "label": t.label, "cwe": t.cwe} for t in all_templates()]


def test_the_view_holds_mitres_97_weaknesses_once_each_in_identifier_order():
    ids = [cwe for cwe, _ in MEMBERS]

    assert len(ids) == 97
    assert len(set(ids)) == 97
    assert all(re.fullmatch(r"CWE-[1-9]\d*", cwe) for cwe in ids)
    assert ids == sorted(ids, key=lambda cwe: int(cwe.removeprefix("CWE-")))
    assert all(name and name == name.strip() for _, name in MEMBERS)


def test_an_entry_is_checked_only_when_a_tested_pattern_carries_its_exact_identifier():
    record = reference(tested(), cards=[])
    by_cwe = {member["cwe"]: member for member in record["members"]}

    checked = {cwe for cwe, member in by_cwe.items() if member["checked"]}
    assert checked == {t.cwe for t in all_templates()} & set(by_cwe)
    assert all(member["patterns"] == [] for member in by_cwe.values() if not member["checked"])
    # Exact identifiers only: the allocation-size pattern is filed under CWE-190, which is
    # not in the view, and it is not counted as the view's CWE-680 however close the two are.
    assert not by_cwe["CWE-680"]["checked"]
    assert by_cwe["CWE-134"]["patterns"] == [
        "format-string-printf",
        "format-string-stream",
        "format-string-snprintf",
    ]


def test_with_no_pattern_tested_nothing_is_checked():
    assert not any(member["checked"] for member in reference([], cards=[])["members"])


def test_named_here_lists_the_views_identifiers_from_the_cards_once_in_card_order():
    cards = [{"cwe": "CWE-134"}, {"cwe": "CWE-78"}, {"cwe": "CWE-120"}, {"cwe": "CWE-134"}]

    # CWE-78 (command injection) is a pattern the site has, and not a member of the view.
    assert reference(tested(), cards)["named_here"] == ["CWE-134", "CWE-120"]


def test_the_note_travels_with_the_list():
    record = reference(tested(), cards=[])

    assert record["view"] == VIEW
    assert record["note"] == NOTE
    assert NOTE.startswith("Reference list, not detection.")


def test_the_page_draws_none_of_it():
    files = [path for path in STATIC.iterdir() if path.is_file()]

    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "cwe_659" not in text, path.name
        assert "named_here" not in text, path.name
