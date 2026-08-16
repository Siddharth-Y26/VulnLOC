"""Guards the objective and the metrics.

Both are places where an extremely imbalanced task can be made to look solved. The loss
must not average padding into its denominator, and the metrics must not fold the coverage
ceiling into the model's recall — those two mistakes flatter a model in opposite
directions and neither raises anything.
"""

from __future__ import annotations

import pytest
import torch

from vulnloc.training.losses import (
    LineLoss,
    focal_loss_with_logits,
    weighted_bce_with_logits,
)
from vulnloc.training.metrics import (
    MetricAccumulator,
    initial_false_alarms,
    top_k_accuracy,
)
from vulnloc.utils.exceptions import ModelError

# --- losses ---------------------------------------------------------------------------


def test_masked_positions_do_not_change_the_loss():
    """Padding must not dilute the loss, or the effective LR depends on batch shape."""
    logits = torch.tensor([[2.0, -2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    two_valid = torch.tensor([[True, True, False]])

    with_padding = weighted_bce_with_logits(logits, targets, two_valid)
    without = weighted_bce_with_logits(logits[:, :2], targets[:, :2], two_valid[:, :2])

    assert with_padding == pytest.approx(float(without))


def test_a_wildly_wrong_masked_position_is_ignored():
    logits = torch.tensor([[0.0, 500.0]])
    targets = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])

    assert float(weighted_bce_with_logits(logits, targets, mask)) < 1.0


def test_an_all_padding_batch_contributes_no_gradient_and_no_nan():
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    targets = torch.zeros(1, 2)
    mask = torch.zeros(1, 2, dtype=torch.bool)

    loss = weighted_bce_with_logits(logits, targets, mask)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) == 0.0


def test_pos_weight_raises_the_cost_of_missing_a_positive():
    logits = torch.tensor([[-3.0]])
    targets = torch.tensor([[1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)

    plain = weighted_bce_with_logits(logits, targets, mask, pos_weight=1.0)
    weighted = weighted_bce_with_logits(logits, targets, mask, pos_weight=10.0)

    assert float(weighted) > float(plain) * 5


def test_focal_downweights_an_easy_negative_more_than_bce():
    easy = torch.tensor([[-8.0]])
    targets = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)

    ratio_focal = float(focal_loss_with_logits(easy, targets, mask))
    ratio_bce = float(weighted_bce_with_logits(easy, targets, mask))

    assert ratio_focal < ratio_bce


def test_focal_with_gamma_zero_is_weighted_bce():
    logits = torch.tensor([[0.7, -1.3]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    focal = focal_loss_with_logits(logits, targets, mask, alpha=0.5, gamma=0.0)
    bce = weighted_bce_with_logits(logits, targets, mask) * 0.5

    assert float(focal) == pytest.approx(float(bce), abs=1e-6)


def test_line_loss_reports_its_parts():
    criterion = LineLoss(kind="bce", function_weight=0.5)

    out = criterion(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2),
        torch.zeros(2),
    )

    assert set(out) == {"loss", "line_loss", "function_loss"}
    assert float(out["loss"]) == pytest.approx(
        float(out["line_loss"]) + 0.5 * float(out["function_loss"])
    )


def test_function_weight_zero_isolates_the_line_term():
    criterion = LineLoss(kind="bce", function_weight=0.0)

    out = criterion(
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([-9.0]),
        torch.tensor([1.0]),
    )

    assert float(out["loss"]) == pytest.approx(float(out["line_loss"]))


def test_unknown_loss_raises():
    with pytest.raises(ModelError):
        LineLoss(kind="hinge")


# --- metrics --------------------------------------------------------------------------


def test_top_k_is_undefined_without_a_positive_line():
    """None, not 0.0 — averaging a zero in would punish the model for an empty question."""
    assert top_k_accuracy(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), 1) is None
    assert initial_false_alarms(torch.tensor([1.0]), torch.tensor([0.0])) is None


def test_top_1_hits_when_the_best_line_is_true():
    scores = torch.tensor([0.1, 0.9, 0.3])
    targets = torch.tensor([0.0, 1.0, 0.0])

    assert top_k_accuracy(scores, targets, 1) == 1.0
    assert initial_false_alarms(scores, targets) == 0.0


def test_ifa_counts_the_clean_lines_read_first():
    scores = torch.tensor([0.9, 0.8, 0.7])
    targets = torch.tensor([0.0, 0.0, 1.0])

    assert initial_false_alarms(scores, targets) == 2.0
    assert top_k_accuracy(scores, targets, 1) == 0.0
    assert top_k_accuracy(scores, targets, 5) == 1.0


def test_unscoreable_flagged_lines_are_not_counted_as_false_negatives():
    """The coverage ceiling must stay separable from the model's recall.

    A flagged line with no features cannot be predicted. Charging it to the model as a
    false negative merges two different quantities that Memory.md §7 #21 insists on
    keeping apart.
    """
    accumulator = MetricAccumulator()
    line_logits = torch.tensor([[-9.0, -9.0]])
    line_targets = torch.tensor([[1.0, 1.0]])
    line_mask = torch.tensor([[True, False]])  # second flagged line has no features
    line_valid = torch.tensor([[True, True]])

    accumulator.update(
        line_logits, line_targets, line_mask, line_valid, torch.tensor([0.0]), torch.tensor([1.0])
    )
    result = accumulator.result()

    assert result["line/fn"] == 1.0, "only the scoreable flagged line counts against recall"
    assert result["coverage/flagged_lines"] == 2.0
    assert result["coverage/flagged_lines_without_features"] == 1.0
    assert result["coverage/flagged_line_coverage_pct"] == 50.0


def test_accuracy_is_not_reported():
    # With 0.795% positive lines, predicting "clean" everywhere scores 99.2%.
    accumulator = MetricAccumulator()
    accumulator.update(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1),
        torch.zeros(1),
    )

    assert not any("accuracy" in key and "top" not in key for key in accumulator.result())


def test_function_confusion_matrix_adds_up():
    accumulator = MetricAccumulator()
    accumulator.update(
        torch.zeros(4, 2),
        torch.zeros(4, 2),
        torch.ones(4, 2, dtype=torch.bool),
        torch.ones(4, 2, dtype=torch.bool),
        torch.tensor([9.0, 9.0, -9.0, -9.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    result = accumulator.result()

    assert (result["function/tp"], result["function/fp"]) == (1.0, 1.0)
    assert (result["function/fn"], result["function/tn"]) == (1.0, 1.0)


def test_units_with_no_positive_line_are_excluded_from_ranking_metrics():
    accumulator = MetricAccumulator()
    accumulator.update(
        torch.randn(2, 4),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        torch.ones(2, 4, dtype=torch.bool),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
    )

    assert accumulator.result()["line/scored_units"] == 1.0
