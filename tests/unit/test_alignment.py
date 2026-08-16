"""Guards the shared line axis (``Architecture.md`` §4.3).

This is the module the docs single out for the most thorough tests in the repo, because
it is the one place where both branches can be silently misaligned against each other and
still produce plausible tensors of the right shape. Nothing downstream can detect an
off-by-one here — the loss would simply be computed against the wrong lines and the model
would learn to be confidently wrong.

The tests are therefore built around values, not shapes: features are chosen so that a
correct mean is a number you can read, and every assertion names the line it is checking.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vulnloc.models.alignment import (
    LineAligner,
    align_nodes_to_lines,
    align_tokens_to_lines,
    scatter_mean_lines,
)
from vulnloc.representation.tokenization import NO_LINE
from vulnloc.utils.exceptions import AlignmentError


def rows(*values: float) -> torch.Tensor:
    """Build ``[M, 1]`` features whose means are readable by eye."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


# --- the 1-indexing contract ----------------------------------------------------------


def test_line_one_lands_at_output_index_zero():
    """The single most consequential convention in the project.

    Lines are 1-indexed; tensors are 0-indexed. If this test is ever "fixed" by shifting
    the expectation, every line-level metric in the paper moves by one row.
    """
    features = rows(5.0)
    line_index = torch.tensor([1])
    segment_index = torch.tensor([0])

    pooled, mask = scatter_mean_lines(features, line_index, segment_index, n_segments=1, n_lines=3)

    assert pooled[0, 0, 0] == 5.0
    assert mask[0, 0]
    assert not mask[0, 1] and not mask[0, 2]


def test_the_last_line_is_reachable():
    # An accumulator sized L instead of L+1 drops the final line and nothing complains.
    features = rows(9.0)
    pooled, mask = scatter_mean_lines(
        features, torch.tensor([4]), torch.tensor([0]), n_segments=1, n_lines=4
    )

    assert pooled[0, 3, 0] == 9.0
    assert mask[0, 3]


def test_lines_map_to_their_own_rows_and_nothing_else():
    features = rows(10.0, 20.0, 30.0)
    line_index = torch.tensor([1, 2, 3])

    pooled, mask = scatter_mean_lines(
        features, line_index, torch.zeros(3, dtype=torch.long), n_segments=1, n_lines=3
    )

    assert pooled[0, :, 0].tolist() == [10.0, 20.0, 30.0]
    assert mask[0].all()


# --- pooling correctness --------------------------------------------------------------


def test_several_rows_on_one_line_are_averaged_not_summed():
    features = rows(2.0, 4.0, 6.0)
    line_index = torch.tensor([2, 2, 2])

    pooled, _ = scatter_mean_lines(
        features, line_index, torch.zeros(3, dtype=torch.long), n_segments=1, n_lines=3
    )

    assert pooled[0, 1, 0] == 4.0


def test_no_line_rows_do_not_dilute_a_real_mean():
    """The contamination failure: specials and padding pulling a line toward zero."""
    features = rows(4.0, 100.0, 6.0, 100.0)
    line_index = torch.tensor([1, NO_LINE, 1, NO_LINE])

    pooled, mask = scatter_mean_lines(
        features, line_index, torch.zeros(4, dtype=torch.long), n_segments=1, n_lines=2
    )

    assert pooled[0, 0, 0] == 5.0, "the two NO_LINE rows must not enter the mean"
    assert not mask[0, 1]


def test_an_empty_line_is_exactly_zero_and_masked_false():
    features = rows(7.0)

    pooled, mask = scatter_mean_lines(
        features, torch.tensor([1]), torch.tensor([0]), n_segments=1, n_lines=3
    )

    assert pooled[0, 1].abs().sum() == 0.0
    assert not mask[0, 1]
    # Zero is a plausible feature value, so the mask is the only honest signal.
    assert mask.tolist() == [[True, False, False]]


@pytest.mark.parametrize("bad_line", [0, -1, NO_LINE, -99])
def test_non_positive_lines_are_dropped(bad_line):
    # Line 0 is the dangerous one: it is what a naive 0-indexing bug would produce.
    features = rows(3.0, 50.0)
    line_index = torch.tensor([1, bad_line])

    pooled, mask = scatter_mean_lines(
        features, line_index, torch.zeros(2, dtype=torch.long), n_segments=1, n_lines=2
    )

    assert pooled[0, 0, 0] == 3.0
    assert not mask[0, 1]


def test_lines_beyond_the_budget_are_dropped_not_wrapped():
    """A line past ``n_lines`` must vanish, never fold back onto a real line."""
    features = rows(1.0, 500.0)
    line_index = torch.tensor([1, 9])

    pooled, mask = scatter_mean_lines(
        features, line_index, torch.zeros(2, dtype=torch.long), n_segments=1, n_lines=2
    )

    assert pooled[0, 0, 0] == 1.0
    assert not mask[0, 1]
    assert pooled.sum() == 1.0


def test_segments_do_not_leak_into_each_other():
    features = rows(1.0, 2.0, 100.0, 200.0)
    line_index = torch.tensor([1, 2, 1, 2])
    segment_index = torch.tensor([0, 0, 1, 1])

    pooled, mask = scatter_mean_lines(features, line_index, segment_index, n_segments=2, n_lines=2)

    assert pooled[0, :, 0].tolist() == [1.0, 2.0]
    assert pooled[1, :, 0].tolist() == [100.0, 200.0]
    assert mask.all()


def test_gradients_flow_back_to_the_contributing_rows_only():
    features = torch.tensor([[1.0], [1.0], [1.0]], requires_grad=True)
    line_index = torch.tensor([1, 1, NO_LINE])

    pooled, _ = scatter_mean_lines(
        features, line_index, torch.zeros(3, dtype=torch.long), n_segments=1, n_lines=1
    )
    pooled.sum().backward()

    assert features.grad is not None
    assert features.grad[0, 0] == pytest.approx(0.5)
    assert features.grad[1, 0] == pytest.approx(0.5)
    assert features.grad[2, 0] == 0.0, "a NO_LINE row must receive no gradient"


# --- the semantic side ----------------------------------------------------------------


def test_align_tokens_pools_one_window_per_unit():
    token_states = torch.tensor([[[1.0], [3.0], [50.0]]])  # [1, 3, 1]
    token_to_line = torch.tensor([[1, 1, NO_LINE]])

    pooled, mask = align_tokens_to_lines(token_states, token_to_line, n_lines=2)

    assert pooled.shape == (1, 2, 1)
    assert pooled[0, 0, 0] == 2.0
    assert mask.tolist() == [[True, False]]


def test_overlapping_windows_of_one_unit_are_averaged_together():
    """Striding's payoff: a line at a window seam is pooled over both windows.

    Without ``window_to_unit`` these two windows would be two separate units and the
    seam line would hold two different values instead of one.
    """
    token_states = torch.tensor([[[2.0]], [[6.0]]])  # two windows, one token each
    token_to_line = torch.tensor([[3], [3]])
    window_to_unit = torch.tensor([0, 0])

    pooled, mask = align_tokens_to_lines(
        token_states, token_to_line, n_lines=3, window_to_unit=window_to_unit, n_units=1
    )

    assert pooled.shape == (1, 3, 1)
    assert pooled[0, 2, 0] == 4.0
    assert mask[0, 2]


def test_windows_of_different_units_stay_separate():
    token_states = torch.tensor([[[2.0]], [[6.0]]])
    token_to_line = torch.tensor([[1], [1]])
    window_to_unit = torch.tensor([0, 1])

    pooled, _ = align_tokens_to_lines(
        token_states, token_to_line, n_lines=1, window_to_unit=window_to_unit, n_units=2
    )

    assert pooled[0, 0, 0] == 2.0
    assert pooled[1, 0, 0] == 6.0


def test_align_tokens_rejects_a_mismatched_line_map():
    with pytest.raises(AlignmentError):
        align_tokens_to_lines(torch.zeros(2, 3, 4), torch.zeros(2, 5, dtype=torch.long), n_lines=3)


def test_align_tokens_rejects_wrong_rank():
    with pytest.raises(AlignmentError):
        align_tokens_to_lines(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long), n_lines=3)


# --- the structural side --------------------------------------------------------------


def test_align_nodes_pools_a_pyg_style_batch():
    node_states = rows(1.0, 3.0, 10.0)
    node_to_line = torch.tensor([1, 1, 2])
    batch_index = torch.tensor([0, 0, 1])

    pooled, mask = align_nodes_to_lines(node_states, node_to_line, batch_index, n_lines=2)

    assert pooled.shape == (2, 2, 1)
    assert pooled[0, 0, 0] == 2.0
    assert pooled[1, 1, 0] == 10.0
    assert mask.tolist() == [[True, False], [False, True]]


def test_align_nodes_rejects_wrong_rank():
    with pytest.raises(AlignmentError):
        align_nodes_to_lines(
            torch.zeros(3, 2, 2),
            torch.zeros(3, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            n_lines=2,
        )


def test_both_branches_agree_on_the_same_line_assignment():
    """The property that makes fusion meaningful.

    A token on line 2 and a node on line 2 must land in the same output row. If the two
    helpers ever disagree, cross-attention would be attending across a shift and the
    resulting attention maps — the project's headline evidence — would be nonsense.
    """
    token_states = torch.tensor([[[7.0], [0.0]]])
    token_to_line = torch.tensor([[2, NO_LINE]])
    node_states = rows(7.0)
    node_to_line = torch.tensor([2])

    from_tokens, token_mask = align_tokens_to_lines(token_states, token_to_line, n_lines=4)
    from_nodes, node_mask = align_nodes_to_lines(
        node_states, node_to_line, torch.tensor([0]), n_lines=4
    )

    assert torch.equal(from_tokens, from_nodes)
    assert torch.equal(token_mask, node_mask)


# --- input validation -----------------------------------------------------------------


def test_out_of_range_segment_index_raises():
    with pytest.raises(AlignmentError):
        scatter_mean_lines(rows(1.0), torch.tensor([1]), torch.tensor([5]), n_segments=2, n_lines=2)


@pytest.mark.parametrize(("n_segments", "n_lines"), [(0, 2), (2, 0), (-1, 2)])
def test_non_positive_dimensions_raise(n_segments, n_lines):
    with pytest.raises(AlignmentError):
        scatter_mean_lines(
            rows(1.0), torch.tensor([1]), torch.tensor([0]), n_segments=n_segments, n_lines=n_lines
        )


def test_mismatched_index_lengths_raise():
    with pytest.raises(AlignmentError):
        scatter_mean_lines(
            rows(1.0, 2.0), torch.tensor([1]), torch.tensor([0, 0]), n_segments=1, n_lines=2
        )


# --- the module wrapper ---------------------------------------------------------------


def test_line_aligner_holds_no_parameters():
    aligner = LineAligner(n_lines=8)

    assert list(aligner.parameters()) == []
    assert "n_lines=8" in repr(aligner)


def test_line_aligner_delegates_to_the_same_functions():
    aligner = LineAligner(n_lines=3)
    token_states = torch.tensor([[[4.0], [6.0]]])
    token_to_line = torch.tensor([[1, 1]])

    pooled, mask = aligner.tokens(token_states, token_to_line)

    assert pooled[0, 0, 0] == 5.0
    assert mask[0, 0]


def test_line_aligner_rejects_a_zero_line_budget():
    with pytest.raises(AlignmentError):
        LineAligner(n_lines=0)


# --- properties -----------------------------------------------------------------------


@settings(max_examples=80, deadline=None)
@given(
    assignments=st.lists(st.integers(min_value=-2, max_value=6), min_size=1, max_size=40),
    n_lines=st.integers(min_value=1, max_value=6),
)
def test_property_pooled_line_equals_the_mean_of_its_own_rows(assignments, n_lines):
    """Checked against a plain Python recomputation, sharing no code with the tensor path."""
    features = torch.arange(len(assignments), dtype=torch.float32).unsqueeze(-1)
    line_index = torch.tensor(assignments)
    segment_index = torch.zeros(len(assignments), dtype=torch.long)

    pooled, mask = scatter_mean_lines(
        features, line_index, segment_index, n_segments=1, n_lines=n_lines
    )

    for line in range(1, n_lines + 1):
        contributors = [
            float(index) for index, assigned in enumerate(assignments) if assigned == line
        ]
        if contributors:
            assert mask[0, line - 1]
            assert pooled[0, line - 1, 0] == pytest.approx(sum(contributors) / len(contributors))
        else:
            assert not mask[0, line - 1]
            assert pooled[0, line - 1, 0] == 0.0


@settings(max_examples=40, deadline=None)
@given(n_lines=st.integers(min_value=1, max_value=8))
def test_property_masked_positions_are_always_exactly_zero(n_lines):
    features = torch.randn(12, 5)
    line_index = torch.randint(-1, n_lines + 3, (12,))

    pooled, mask = scatter_mean_lines(
        features, line_index, torch.zeros(12, dtype=torch.long), n_segments=1, n_lines=n_lines
    )

    assert pooled[~mask].abs().sum() == 0.0
