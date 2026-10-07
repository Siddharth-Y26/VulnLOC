"""Guards Phase 7a: evidence, the template explainer, the verifier and the §5.4 contract.

The exit criterion this file exists for is *"grounding test suite passes (no explanation
names an unflagged line)"* — FR-15. It is tested three ways, because each can fail without
the others noticing: the explainer refuses an unflagged line; the verifier rejects an
explanation of one even when handed it directly; and a property test draws arbitrary
scores and thresholds and checks that everything the pipeline emits survives the verifier
and mentions no line it has no evidence for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vulnloc.explanation import (
    EXPLAINER_REGISTRY,
    AttendedLine,
    Evidence,
    EvidenceSettings,
    Explanation,
    Prediction,
    TemplateExplainer,
    extract_evidence,
    output_contract,
    verify,
)
from vulnloc.explanation.contract import (
    CONTRACT_KEYS,
    EVIDENCE_KEYS,
    LINE_KEYS,
    NOT_FLAGGED_TEXT,
)
from vulnloc.explanation.evidence import (
    attended_lines,
    dataflow_path,
    identifiers,
    line_nodes,
    top_tokens,
)
from vulnloc.explanation.generator import (
    FORMATTING_ARTIFACT,
    FORMATTING_ARTIFACT_LABEL,
    UNCLASSIFIED,
    join_numbers,
)
from vulnloc.explanation.templates import NO_PATTERN
from vulnloc.explanation.verifier import mentioned_lines, quoted_fragments
from vulnloc.representation import CodeUnit, TreeSitterParser, build_ast_graph, build_dfg_graph
from vulnloc.utils.exceptions import ExplanationError

SOURCE = (
    "void copy(char *dst, const char *src, int n)\n"  # 1
    "{\n"  # 2
    "char buf[16];\n"  # 3
    "int len = strlen(src);\n"  # 4
    "if (n > len)\n"  # 5
    "n = len;\n"  # 6
    "memcpy(buf, src, n);\n"  # 7
    "strcpy(dst, buf);\n"  # 8
    "}"  # 9
)
N_LINES = 9

SETTINGS = EvidenceSettings(
    top_k_attended=2,
    min_attention_weight=0.0,
    max_dataflow_hops=2,
    max_graph_nodes=3,
    max_tokens=6,
)


@pytest.fixture(scope="module")
def unit() -> CodeUnit:
    return CodeUnit.from_source("u1", SOURCE, language="C", target=1)


@pytest.fixture(scope="module")
def graphs(unit):
    parsed = TreeSitterParser().parse(unit.source, language="C", unit_id=unit.unit_id)
    return build_ast_graph(parsed), build_dfg_graph(parsed)


def prediction(scores: dict[int, float], *, threshold: float = 0.5) -> Prediction:
    """A prediction over SOURCE with the given line scores and 0.01 everywhere else."""
    values = [scores.get(line, 0.01) for line in range(1, N_LINES + 1)]
    return Prediction("u1", tuple(values), (True,) * N_LINES, threshold)


def attention(weights: dict[tuple[int, int], float], *, heads: int = 2) -> np.ndarray:
    """``[H, L, L]`` with the given (query line, key line) weights on every head."""
    grid = np.zeros((heads, N_LINES, N_LINES), dtype=np.float32)
    for (query, key), weight in weights.items():
        grid[:, query - 1, key - 1] = weight
    return grid


def explain(unit, graphs, pred, line, **maps):
    ast, dfg = graphs
    evidence = extract_evidence(unit, pred, line, settings=SETTINGS, ast=ast, dfg=dfg, **maps)
    return TemplateExplainer(max_context_lines=2).explain(unit, pred, evidence), evidence


# ── Prediction ───────────────────────────────────────────────────────────────────────


def test_flagged_lines_are_ranked_and_a_tie_goes_to_the_earlier_line():
    pred = prediction({7: 0.9, 8: 0.9, 4: 0.6})

    assert pred.flagged_lines == (7, 8, 4)
    assert pred.rank(7) == 1
    assert pred.rank(8) == 2


def test_a_line_exactly_at_the_threshold_is_flagged():
    pred = prediction({7: 0.5})

    assert pred.is_flagged(7)
    assert not pred.is_flagged(8)


def test_an_unscoreable_line_is_never_flagged_whatever_number_sits_in_its_slot():
    # The head writes a masked sentinel there; after a sigmoid that can look like a score.
    pred = Prediction("u1", (0.99, 0.2), (False, True), 0.5)

    assert pred.flagged_lines == ()
    assert pred.ranking == (2,)
    with pytest.raises(ExplanationError, match="no score"):
        pred.score(1)


def test_a_line_outside_the_prediction_is_not_flagged_rather_than_an_index_error():
    pred = prediction({7: 0.9})

    assert not pred.is_flagged(0)
    assert not pred.is_flagged(N_LINES + 1)


@pytest.mark.parametrize(
    ("probabilities", "scoreable", "threshold", "message"),
    [
        ((0.1, 0.2), (True,), 0.5, "one length"),
        ((0.1,), (True,), 1.5, r"\[0, 1\]"),
        ((float("nan"),), (True,), 0.5, "number in"),
        ((1.2,), (True,), 0.5, "number in"),
    ],
)
def test_a_malformed_prediction_is_refused(probabilities, scoreable, threshold, message):
    with pytest.raises(ExplanationError, match=message):
        Prediction("u1", probabilities, scoreable, threshold)


def test_a_non_number_in_an_unscoreable_slot_is_tolerated():
    pred = Prediction("u1", (float("nan"), 0.7), (False, True), 0.5)

    assert pred.flagged_lines == (2,)


# ── Evidence: attention ──────────────────────────────────────────────────────────────


def test_attended_lines_are_the_heaviest_other_lines_averaged_over_heads():
    grid = np.zeros((2, N_LINES, N_LINES), dtype=np.float32)
    grid[0, 6, 3], grid[1, 6, 3] = 0.2, 0.6  # line 7 -> line 4, mean 0.4
    grid[:, 6, 5] = 0.3  # line 7 -> line 6
    grid[:, 6, 6] = 0.9  # line 7 -> itself: not context

    found = attended_lines(grid, 7, n_lines=N_LINES, direction="sem_from_str", top_k=2)

    assert [(entry.line, round(entry.weight, 6)) for entry in found] == [(4, 0.4), (6, 0.3)]
    assert {entry.direction for entry in found} == {"sem_from_str"}


def test_padding_past_the_units_last_line_is_never_attended_to():
    grid = np.zeros((1, 12, 12), dtype=np.float32)
    grid[0, 6, 10] = 0.9  # a key position beyond the unit's 9 lines
    grid[0, 6, 2] = 0.1

    found = attended_lines(grid, 7, n_lines=N_LINES, direction="str_from_sem", top_k=3)

    assert [entry.line for entry in found] == [3]


def test_a_query_line_beyond_the_trimmed_axis_has_no_attention_evidence():
    grid = np.full((1, 4, 4), 0.25, dtype=np.float32)

    assert attended_lines(grid, 7, n_lines=N_LINES, direction="sem_from_str", top_k=2) == ()


def test_a_nan_in_the_attention_row_raises_instead_of_ranking_garbage():
    grid = np.zeros((1, N_LINES, N_LINES), dtype=np.float32)
    grid[0, 6, 2] = np.nan

    with pytest.raises(ExplanationError, match="finite"):
        attended_lines(grid, 7, n_lines=N_LINES, direction="sem_from_str", top_k=2)


def test_an_attention_map_of_the_wrong_shape_is_refused():
    with pytest.raises(ExplanationError, match=r"\[H, L, L\]"):
        attended_lines(np.zeros((2, 3, 4)), 1, n_lines=3, direction="sem_from_str", top_k=1)


def test_an_unknown_attention_direction_is_refused():
    with pytest.raises(ExplanationError, match="direction"):
        AttendedLine(line=1, weight=0.5, direction="both")


# ── Evidence: data flow, nodes, tokens ───────────────────────────────────────────────


def test_dataflow_path_steps_to_the_nearest_earlier_definition(graphs):
    _, dfg = graphs

    path, symbols = dataflow_path(dfg, 7, max_hops=2)

    assert path == (4, 6, 7)
    assert symbols == ("len", "n")


def test_dataflow_path_is_bounded_by_max_hops(graphs):
    _, dfg = graphs

    path, symbols = dataflow_path(dfg, 7, max_hops=1)

    assert path == (6, 7)
    assert symbols == ("n",)


def test_a_line_nothing_flows_into_has_no_path_rather_than_a_path_of_itself(graphs):
    _, dfg = graphs

    assert dataflow_path(dfg, 2, max_hops=3) == ((), ())


def test_line_nodes_are_named_nodes_on_that_line_calls_first(graphs):
    ast, _ = graphs

    nodes = line_nodes(ast, 7, limit=2)

    assert len(nodes) == 2
    first = ast.nodes[int(nodes[0].removeprefix("n_"))]
    assert first.node_type == "call_expression"
    assert first.line_no == 7


def test_a_line_without_nodes_reports_none(graphs):
    ast, _ = graphs

    assert line_nodes(ast, N_LINES + 5, limit=3) == ()


def test_identifiers_drop_keywords_and_repeats():
    assert identifiers("if (n > len) return buf[n];") == ("n", "len", "buf")


def test_top_tokens_start_with_the_flagged_line_then_follow_attention_weight(unit):
    attended = (
        AttendedLine(line=3, weight=0.2, direction="sem_from_str"),
        AttendedLine(line=4, weight=0.5, direction="str_from_sem"),
    )

    tokens = top_tokens(unit, 8, attended, limit=6)

    assert tokens == ("strcpy", "dst", "buf", "len", "strlen", "src")


def test_evidence_is_refused_for_a_line_that_is_not_flagged(unit, graphs):
    ast, dfg = graphs
    pred = prediction({7: 0.9})

    with pytest.raises(ExplanationError, match="FR-15"):
        extract_evidence(unit, pred, 8, settings=SETTINGS, ast=ast, dfg=dfg)


def test_evidence_without_graphs_or_attention_is_empty_not_invented(unit):
    pred = prediction({7: 0.9})

    evidence = extract_evidence(unit, pred, 7, settings=SETTINGS)

    assert evidence.attended == ()
    assert evidence.dataflow_path == ()
    assert evidence.graph_nodes == ()
    assert evidence.top_tokens == ("memcpy", "buf", "src", "n")


def test_a_dataflow_path_that_does_not_end_at_its_line_is_refused():
    with pytest.raises(ExplanationError, match="last line"):
        Evidence("u1", 7, dataflow_path=(4, 6), dataflow_symbols=("len",))


def test_negative_evidence_limits_are_refused():
    with pytest.raises(ExplanationError, match="negative"):
        dataclasses.replace(SETTINGS, max_tokens=-1)


# ── The explainer ────────────────────────────────────────────────────────────────────


def test_the_registry_resolves_the_template_explainer():
    assert EXPLAINER_REGISTRY.get("template") is TemplateExplainer


def test_a_matched_line_is_explained_with_its_pattern_score_attention_and_data_flow(unit, graphs):
    pred = prediction({7: 0.91, 8: 0.6})
    maps = {"attn_sem_from_str": attention({(7, 4): 0.5, (7, 3): 0.2})}

    explanation, _ = explain(unit, graphs, pred, 7, **maps)

    assert explanation.label == "Out-of-bounds write"
    assert explanation.cwe == "CWE-787"
    assert explanation.template_id == "memcpy-length"
    assert explanation.text == (
        "Matches the pattern 'Out-of-bounds write' (CWE-787): The length passed to `memcpy()` "
        "is not written as a sizeof on this line, so whether it fits the destination is decided "
        "somewhere else. The model ranks line 7 at position 1 of 9 scored lines (calibrated "
        "score 0.91, threshold 0.50). Its cross-attention puts most weight on lines 4 and 3. "
        "Name-based data flow reaches it through lines 4 and 6 (`len`, `n`)."
    )
    assert explanation.cited_lines == (4, 3, 6)
    assert explanation.cited_symbols == ("memcpy", "len", "n")


def test_a_line_no_pattern_matches_says_so_and_names_no_cause(unit, graphs):
    pred = prediction({5: 0.8})

    explanation, _ = explain(unit, graphs, pred, 5)

    assert explanation.label == UNCLASSIFIED
    assert explanation.cwe == ""
    assert explanation.template_id == NO_PATTERN
    assert explanation.text.startswith("No known weakness pattern matches this line's text")


def test_the_explainer_refuses_a_line_that_is_not_flagged(unit):
    pred = prediction({7: 0.9})
    evidence = Evidence("u1", 8)

    with pytest.raises(ExplanationError, match="FR-15"):
        TemplateExplainer(max_context_lines=2).explain(unit, pred, evidence)


def test_the_explainer_refuses_evidence_from_another_unit(unit):
    pred = prediction({7: 0.9})

    with pytest.raises(ExplanationError, match="one unit"):
        TemplateExplainer(max_context_lines=2).explain(unit, pred, Evidence("other", 7))


def test_the_same_inputs_give_the_same_text(unit, graphs):
    pred = prediction({8: 0.7})

    first, _ = explain(unit, graphs, pred, 8)
    second, _ = explain(unit, graphs, pred, 8)

    assert first == second


def test_join_numbers_reads_as_prose():
    assert [join_numbers(n) for n in ([3], [3, 7], [3, 7, 9])] == ["3", "3 and 7", "3, 7 and 9"]


# ── The verifier (FR-15) ─────────────────────────────────────────────────────────────


def test_a_generated_explanation_passes_the_verifier(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(
        unit, graphs, pred, 7, attn_str_from_sem=attention({(7, 8): 0.4})
    )

    report = verify(explanation, unit, pred, evidence)

    assert report.mentioned_lines == (4, 6, 7, 8)
    assert report.quoted == ("memcpy()", "len", "n")


def test_an_explanation_of_an_unflagged_line_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    stricter = dataclasses.replace(pred, threshold=0.95)

    with pytest.raises(ExplanationError, match="not scored above the threshold"):
        verify(explanation, unit, stricter, evidence)


def test_text_that_mentions_a_line_it_does_not_cite_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    tampered = dataclasses.replace(explanation, text=explanation.text + " See also line 2.")

    with pytest.raises(ExplanationError, match="mentions a line it does not cite"):
        verify(tampered, unit, pred, evidence)


def test_a_cited_line_with_no_evidence_behind_it_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    tampered = dataclasses.replace(
        explanation,
        text=explanation.text + " Compare line 2.",
        cited_lines=(*explanation.cited_lines, 2),
    )

    with pytest.raises(ExplanationError, match="no evidence and no quoted symbol"):
        verify(tampered, unit, pred, evidence)


def test_a_context_line_may_be_cited_but_never_called_flagged(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    tampered = dataclasses.replace(
        explanation, text=explanation.text + " Line 4 is vulnerable as well."
    )

    with pytest.raises(ExplanationError, match="verdict mentions another line"):
        verify(tampered, unit, pred, evidence)


def test_a_quoted_fragment_that_is_on_no_cited_line_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    tampered = dataclasses.replace(explanation, text=explanation.text + " See `strcpy()`.")

    with pytest.raises(ExplanationError, match="quotes a fragment"):
        verify(tampered, unit, pred, evidence)


def test_attack_vocabulary_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    tampered = dataclasses.replace(
        explanation, text=explanation.text + " An attacker can craft input to reach it."
    )

    with pytest.raises(ExplanationError, match="attack-instruction"):
        verify(tampered, unit, pred, evidence)


def test_text_that_never_names_its_own_line_is_rejected(unit, graphs):
    pred = prediction({7: 0.91})
    _, evidence = explain(unit, graphs, pred, 7)
    silent = Explanation("u1", 7, UNCLASSIFIED, "", "Something is wrong here.", NO_PATTERN)

    with pytest.raises(ExplanationError, match="never names the line"):
        verify(silent, unit, pred, evidence)


def test_evidence_about_another_line_is_rejected(unit, graphs):
    pred = prediction({7: 0.91, 8: 0.8})
    explanation, _ = explain(unit, graphs, pred, 7)
    _, other = explain(unit, graphs, pred, 8)

    with pytest.raises(ExplanationError, match="another line"):
        verify(explanation, unit, pred, other)


def test_line_numbers_inside_quoted_code_are_not_citations():
    text = "`lines 3` is read. The model ranks line 7 at position 1 of 9 scored lines."

    assert mentioned_lines(text) == (7,)
    assert quoted_fragments(text) == ("lines 3",)


def test_a_list_of_lines_is_read_whole():
    assert mentioned_lines("through lines 4, 6 and 12 (`a`).") == (4, 6, 12)


@settings(max_examples=150, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=N_LINES,
        max_size=N_LINES,
    ),
    scoreable=st.lists(st.booleans(), min_size=N_LINES, max_size=N_LINES),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_nothing_the_pipeline_emits_names_an_unflagged_line(
    unit, graphs, scores, scoreable, threshold, seed
):
    # The property FR-15 states, over arbitrary scores: every explanation is of a flagged
    # line, passes the verifier, and mentions only that line and lines it has evidence for.
    ast, dfg = graphs
    pred = Prediction("u1", tuple(scores), tuple(scoreable), threshold)
    rng = np.random.default_rng(seed)
    maps = rng.dirichlet(np.ones(N_LINES), size=(2, 2, N_LINES)).astype(np.float32)
    explainer = TemplateExplainer(max_context_lines=2)

    for line in pred.flagged_lines:
        evidence = extract_evidence(
            unit,
            pred,
            line,
            settings=SETTINGS,
            ast=ast,
            dfg=dfg,
            attn_sem_from_str=maps[0],
            attn_str_from_sem=maps[1],
        )
        explanation = explainer.explain(unit, pred, evidence)
        report = verify(explanation, unit, pred, evidence)

        assert scoreable[line - 1] and scores[line - 1] >= threshold
        assert set(report.mentioned_lines) <= {line} | evidence.context_lines

    for line in range(1, N_LINES + 1):
        if line not in pred.flagged_lines:
            with pytest.raises(ExplanationError):
                extract_evidence(unit, pred, line, settings=SETTINGS, ast=ast, dfg=dfg)


# ── The §5.4 contract ────────────────────────────────────────────────────────────────


def test_the_contract_has_exactly_the_keys_the_prd_lists(unit, graphs):
    pred = prediction({7: 0.91, 8: 0.6})
    explanations, evidence = {}, {}
    for line in pred.flagged_lines:
        explanations[line], evidence[line] = explain(unit, graphs, pred, line)

    contract = output_contract(unit, pred, explanations, evidence)

    assert tuple(contract) == CONTRACT_KEYS
    assert tuple(contract["evidence"]) == EVIDENCE_KEYS
    assert all(tuple(entry) == LINE_KEYS for entry in contract["vulnerable_lines"])
    assert contract["is_vulnerable"] is True
    assert contract["risk_score"] == pytest.approx(0.91)
    assert contract["vulnerability_label"] == "Out-of-bounds write"
    assert [entry["line"] for entry in contract["vulnerable_lines"]] == [7, 8]
    assert contract["vulnerable_lines"][1]["code"] == "strcpy(dst, buf);"
    assert contract["explanation"] == explanations[7].text
    assert contract["evidence"]["dataflow_path"] == [4, 6, 7]


def test_a_unit_with_no_flagged_line_is_not_called_safe(unit):
    pred = prediction({7: 0.3})

    contract = output_contract(unit, pred, {}, {})

    assert tuple(contract) == CONTRACT_KEYS
    assert contract["is_vulnerable"] is False
    assert contract["risk_score"] == pytest.approx(0.3)
    assert contract["vulnerable_lines"] == []
    assert contract["explanation"] == NOT_FLAGGED_TEXT
    assert "not evidence that the function is safe" in contract["explanation"]


def test_the_contract_refuses_an_explanation_for_an_unflagged_line(unit, graphs):
    pred = prediction({7: 0.91})
    explanation, evidence = explain(unit, graphs, pred, 7)
    stale = dataclasses.replace(explanation, line=8)

    with pytest.raises(ExplanationError, match="FR-15"):
        output_contract(unit, pred, {7: explanation, 8: stale}, {7: evidence})


def test_the_contract_refuses_a_flagged_unit_whose_top_line_has_no_explanation(unit):
    pred = prediction({7: 0.91})

    with pytest.raises(ExplanationError, match="no verified explanation"):
        output_contract(unit, pred, {}, {})


def test_a_unit_with_no_scoreable_line_reports_zero_risk(unit):
    pred = Prediction("u1", (0.9,) * N_LINES, (False,) * N_LINES, 0.5)

    contract = output_contract(unit, pred, {}, {})

    assert contract["risk_score"] == 0.0
    assert contract["is_vulnerable"] is False


# ── The comment-terminator tell (Memory.md §1.25) ────────────────────────────────────

ARTIFACT_SOURCE = (
    "void f(struct ctx *c)\n"  # 1
    "{\n"  # 2
    "qio_channel_wait(c->ioc);  /* needed by nbd_reply_ready.  *\n"  # 3  tell, no pattern
    "memcpy(c->buf, c->src, c->n);  /* copy it *\n"  # 4  tell and a pattern
    "}"  # 5
)


def _artifact_case(line: int):
    unit = CodeUnit.from_source("u2", ARTIFACT_SOURCE, language="C", target=1)
    pred = Prediction("u2", (0.01, 0.01, 0.9, 0.8, 0.01), (True,) * 5, 0.5)
    evidence = extract_evidence(unit, pred, line, settings=SETTINGS)
    explanation = TemplateExplainer(max_context_lines=2).explain(unit, pred, evidence)
    return unit, pred, evidence, explanation


def test_a_flagged_line_carrying_the_tell_is_told_its_score_may_rest_on_formatting():
    unit, pred, evidence, explanation = _artifact_case(3)

    assert explanation.template_id == FORMATTING_ARTIFACT
    assert explanation.label == FORMATTING_ARTIFACT_LABEL
    assert "formatting artifact of the dataset" in explanation.text
    verify(explanation, unit, pred, evidence)


def test_a_pattern_on_a_tell_line_keeps_its_label_and_still_carries_the_warning():
    unit, pred, evidence, explanation = _artifact_case(4)

    assert explanation.template_id == "memcpy-length"
    assert explanation.label == "Out-of-bounds write"
    assert "formatting artifact of the dataset" in explanation.text
    verify(explanation, unit, pred, evidence)


def test_stripped_text_has_no_tell_left_to_warn_about():
    unit = CodeUnit.from_source("u2", ARTIFACT_SOURCE, language="C", target=1, strip_comments=True)
    pred = Prediction("u2", (0.01, 0.01, 0.9, 0.8, 0.01), (True,) * 5, 0.5)
    evidence = extract_evidence(unit, pred, 3, settings=SETTINGS)

    explanation = TemplateExplainer(max_context_lines=2).explain(unit, pred, evidence)

    assert explanation.template_id == NO_PATTERN
    assert "formatting artifact" not in explanation.text
