"""The Findings tab's assembly — what reaches a developer's screen, with the model stubbed.

The model is not under test here; the joins are. A card must show the model's own rank, a
template must land on the line it matched, a lone brace must never become a finding, and a
pattern must never be listed twice.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import frontend.scoring  # noqa: E402
from frontend.pipeline import findings  # noqa: E402

SOURCE = "void f(char *s)\n{\nchar buf[8];\nstrcpy(buf, s);\nprintf(s);\nreturn;\n}\n"


def fake_scores(order: list[int], top_k: int = 5):
    """A ``score_lines`` stand-in ranking lines in ``order``, best first."""

    def score_lines(unit: Any) -> dict[str, Any]:
        rank = {line: i for i, line in enumerate(order, start=1)}
        scores = [
            {"line": n, "score": 1.0 - 0.1 * rank[n] if n in rank else None, "rank": rank.get(n)}
            for n in range(1, unit.n_lines + 1)
        ]
        top = sorted((s for s in scores if s["rank"]), key=lambda s: s["rank"])[:5]
        return {
            "scores": scores,
            "top": top,
            "unscored_lines": [s["line"] for s in scores if s["rank"] is None],
            "beyond_line_cap": 0,
            "top_k": top_k,
            "model": {"name": "stub", "run_id": "stub", "checkpoint": "", "device": "cpu"},
            "measured": {},
        }

    return score_lines


def run(monkeypatch, order: list[int], top_k: int = 5) -> dict[str, Any]:
    monkeypatch.setattr(frontend.scoring, "score_lines", fake_scores(order, top_k))
    return findings(SOURCE)


def test_a_line_holding_only_punctuation_never_becomes_a_card(monkeypatch):
    report = run(monkeypatch, [7, 4, 2, 3, 5, 6, 1])

    assert [card["line"] for card in report["top"]] == [4, 3, 5, 6, 1]
    assert report["skipped_punctuation_lines"] == [7, 2]


def test_a_card_keeps_the_models_rank_rather_than_renumbering(monkeypatch):
    report = run(monkeypatch, [7, 4, 2, 3, 5, 6, 1])

    assert [card["rank"] for card in report["top"]] == [2, 4, 5, 6, 7]


def test_templates_land_on_the_line_they_matched(monkeypatch):
    report = run(monkeypatch, [4, 5, 3, 1, 6])
    by_line = {card["line"]: {p["cwe"] for p in card["patterns"]} for card in report["top"]}

    assert by_line[4] == {"CWE-120"}
    assert by_line[5] == {"CWE-134"}
    assert by_line[3] == set()


def test_a_pattern_outside_the_cards_is_listed_once_and_never_duplicated(monkeypatch):
    # Three cards (lines 1, 3, 6), so the printf on line 5 falls outside them.
    report = run(monkeypatch, [1, 3, 6, 4, 2, 7, 5], top_k=3)

    card_lines = {card["line"] for card in report["top"]}
    elsewhere = [item["line"] for item in report["patterns_elsewhere"]]
    assert elsewhere == [4, 5]
    assert not card_lines & set(elsewhere)
    assert [item["rank"] for item in report["patterns_elsewhere"]] == [4, 7]


def test_nothing_to_analyse_is_an_input_error(monkeypatch):
    monkeypatch.setattr(frontend.scoring, "score_lines", fake_scores([]))

    with pytest.raises(ValueError, match="Paste a function"):
        findings("   \n")
