"""Guards the per-line head.

Two properties matter more than the arithmetic: a line with no evidence must not be
scoreable, and the function-level decision must not be able to contradict the line scores
it was derived from. Both are the kind of thing that produces plausible numbers when
broken.
"""

from __future__ import annotations

import pytest
import torch

from vulnloc.models.heads.line_head import MASKED_LOGIT, LineHead, masked_logsumexp_pool
from vulnloc.utils.exceptions import ModelError


@pytest.fixture
def head():
    torch.manual_seed(0)
    return LineHead(8, hidden_dim=16)


def test_masked_lines_are_pushed_to_the_floor(head):
    features = torch.randn(2, 5, 8)
    line_mask = torch.tensor([[True, True, False, False, False], [True] * 5])

    logits, _ = head(features, line_mask)

    assert (logits[0, 2:] == MASKED_LOGIT).all()
    assert (logits[1] != MASKED_LOGIT).all()


def test_a_masked_line_can_never_win_a_ranking(head):
    """The failure this prevents: a padded line topping the evidence list."""
    features = torch.randn(1, 6, 8)
    line_mask = torch.tensor([[True, False, False, False, False, False]])

    logits, _ = head(features, line_mask)

    assert int(torch.argmax(logits[0])) == 0


def test_masked_logit_is_finite_so_gradients_stay_clean():
    # -inf survives the forward pass and produces NaN in the backward pass the moment it
    # meets a multiplication by zero, poisoning every parameter in one step.
    assert torch.isfinite(torch.tensor(MASKED_LOGIT))
    assert torch.sigmoid(torch.tensor(MASKED_LOGIT)) == 0.0


def test_masked_logit_survives_float16():
    # -1e5 overflows to -inf under autocast and reintroduces the NaN problem.
    assert torch.isfinite(torch.tensor(MASKED_LOGIT, dtype=torch.float16))


def test_function_logit_follows_the_strongest_line(head):
    features = torch.randn(1, 4, 8)
    line_mask = torch.ones(1, 4, dtype=torch.bool)

    logits, function_logit = head(features, line_mask)

    # logsumexp is bounded below by the max and above by max + log(n).
    assert function_logit[0] >= logits[0].max()
    assert function_logit[0] <= logits[0].max() + torch.log(torch.tensor(4.0))


def test_pooling_ignores_masked_lines():
    logits = torch.tensor([[5.0, -5.0]])
    all_lines = torch.tensor([[True, True]])
    first_only = torch.tensor([[True, False]])

    assert masked_logsumexp_pool(logits, all_lines)[0] == pytest.approx(
        float(masked_logsumexp_pool(logits, first_only)[0]), abs=1e-3
    )


def test_a_unit_with_no_usable_lines_pools_to_the_floor():
    logits = torch.zeros(1, 3)
    empty = torch.zeros(1, 3, dtype=torch.bool)

    assert masked_logsumexp_pool(logits, empty)[0] == MASKED_LOGIT


def test_gradient_reaches_every_evidenced_line():
    """A hard max would train one line per function; on 0.795% positives that is too thin."""
    logits = torch.tensor([[1.0, 0.9, 0.8]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)

    masked_logsumexp_pool(logits, mask).sum().backward()

    assert logits.grad is not None
    assert (logits.grad > 0).all()


def test_zero_hidden_dim_is_a_plain_linear_head():
    head = LineHead(8, hidden_dim=0)

    assert not any(isinstance(m, torch.nn.GELU) for m in head.net)


@pytest.mark.parametrize("bad_dim", [0, -4])
def test_non_positive_input_dim_raises(bad_dim):
    with pytest.raises(ModelError):
        LineHead(bad_dim)


def test_wrong_feature_width_raises(head):
    with pytest.raises(ModelError):
        head(torch.randn(1, 3, 9), torch.ones(1, 3, dtype=torch.bool))


def test_mismatched_mask_raises(head):
    with pytest.raises(ModelError):
        head(torch.randn(1, 3, 8), torch.ones(1, 4, dtype=torch.bool))


def test_wrong_rank_raises(head):
    with pytest.raises(ModelError):
        head(torch.randn(3, 8), torch.ones(3, dtype=torch.bool))
