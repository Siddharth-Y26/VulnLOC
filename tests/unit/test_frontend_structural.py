"""The Detailed report's structural ranking — assembly with the model stubbed, and one real check.

The stubbed tests pin the joins: the model's own ranks are kept, punctuation-only lines never
reach the table, and nodeless lines are reported. The slow test pins the thing that fails
silently: a pasted function must reach the checkpoint as exactly the input training built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import frontend.structural  # noqa: E402
from frontend.pipeline import measured_results, structural_findings  # noqa: E402

SOURCE = "void f(char *s)\n{\nchar buf[8];\nstrcpy(buf, s);\nprintf(s);\nreturn;\n}\n"


def fake_scores(order: list[int], without_node: list[int] | None = None):
    """A ``score_lines`` stand-in ranking lines in ``order``, best first."""

    def score_lines(unit: Any) -> dict[str, Any]:
        rank = {line: i for i, line in enumerate(order, start=1)}
        return {
            "scores": [
                {"line": n, "score": 0.9 if n in rank else None, "rank": rank.get(n)}
                for n in range(1, unit.n_lines + 1)
            ],
            "top": [],
            "lines_without_node": without_node or [],
            "top_k": 3,
            "model": {"name": "stub", "run_id": "stub", "seed": 42, "device": "cpu"},
        }

    return score_lines


def test_punctuation_lines_are_skipped_and_ranks_are_kept(monkeypatch):
    monkeypatch.setattr(frontend.structural, "score_lines", fake_scores([2, 4, 7, 5, 3]))

    report = structural_findings(SOURCE)

    assert [row["line"] for row in report["top"]] == [4, 5, 3]
    assert [row["rank"] for row in report["top"]] == [2, 4, 5]
    assert report["top"][0]["text"] == "strcpy(buf, s);"


def test_code_lines_without_a_node_are_reported_and_blank_ones_are_not(monkeypatch):
    # Line 8 is the empty line after the final newline.
    monkeypatch.setattr(frontend.structural, "score_lines", fake_scores([4], without_node=[6, 8]))

    assert structural_findings(SOURCE)["lines_without_node"] == [6]


def test_nothing_to_analyse_is_an_input_error(monkeypatch):
    monkeypatch.setattr(frontend.structural, "score_lines", fake_scores([]))

    with pytest.raises(ValueError, match="Paste a function"):
        structural_findings("  \n")


def test_measured_results_are_read_from_disk_or_reported_missing():
    results = measured_results()

    assert {"semantic", "structural", "cross_attention", "concat"} <= set(results)
    for entry in results.values():
        if "missing" in entry:
            continue
        assert 0.0 <= entry["top_1"] <= entry["top_5"] <= 1.0
        assert entry["run_ids"]


_CACHE = Path("artifacts/data/cache")


@pytest.mark.slow
@pytest.mark.skipif(
    not (frontend.structural.CHECKPOINT.exists() and _CACHE.exists()),
    reason="needs run 77370d56's checkpoint and the preprocessed shards",
)
def test_a_paste_scores_exactly_as_the_training_data_path_does():
    """Scores through the paste path must equal scores through ``GraphLineDataset``.

    If the grammar policy, builder, vocabulary or subtoken table differed, the model would
    still produce plausible scores — this is the only check that would notice.
    """
    import torch

    from vulnloc.data.cache import iter_shard
    from vulnloc.representation.units import CodeUnit
    from vulnloc.training.graph_datasets import GraphLineDataset, collate_graphs

    state = frontend.structural._load()
    n = 10
    dataset = GraphLineDataset(
        _CACHE,
        "val",
        state["node_vocab"],
        subtoken_vocab=state["subtoken_vocab"],
        builder="ast",
        language="best_of_two",
        n_lines=512,
        limit=n,
        line_labelled_only=True,
    )
    sources: dict[str, str] = {}
    for labelled in iter_shard(_CACHE, "val"):
        if labelled.vulnerable_lines:
            unit = CodeUnit.from_labelled_function(labelled, normalize_whitespace=True)
            sources[unit.unit_id] = unit.source
        if len(sources) >= n:
            break

    for graph_unit in dataset.units:
        paste = CodeUnit.from_source(
            "paste", sources[graph_unit.unit_id], language="C", normalize_whitespace=True
        )
        ours = {
            e["line"]: e["score"]
            for e in frontend.structural.score_lines(paste)["scores"]
            if e["score"] is not None
        }
        with torch.no_grad():
            out = state["model"](collate_graphs([graph_unit]))
        reference = {
            line: float(torch.sigmoid(out["line_logits"][0][line - 1]))
            for line in range(1, min(graph_unit.n_lines, 512) + 1)
            if bool(out["line_mask"][0][line - 1])
        }
        assert set(ours) == set(reference), graph_unit.unit_id
        assert max(abs(ours[k] - reference[k]) for k in reference) < 1e-5, graph_unit.unit_id
