"""Calibration and the threshold, pinned against cases small enough to do by hand.

``Rules.md`` §6 asks for metrics tested against hand-computed examples, degenerate cases
included. These functions decide the cut-off every reported line F1 will use and the
probabilities the explanation layer will quote, and each of them has a way of being
quietly wrong: a calibrator that reorders the lines, a reliability curve that hides its
empty bins, a threshold that splits a tie.
"""

from __future__ import annotations

import math

import pytest
import torch

from vulnloc.models.heads.calibration import (
    LogitCalibrator,
    best_f1_threshold,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
)
from vulnloc.utils.exceptions import ModelError

# --- the calibrator -------------------------------------------------------------------


def _synthetic(scale: float, shift: float, n: int = 20_000, seed: int = 0):
    """Logits whose true label probability is ``sigmoid(scale * z + shift)``."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, generator=generator, dtype=torch.float64) * 2.0
    probability = torch.sigmoid(scale * logits + shift)
    labels = (torch.rand(n, generator=generator, dtype=torch.float64) < probability).float()
    return logits, labels


def test_platt_scaling_recovers_the_map_the_labels_were_drawn_from():
    logits, labels = _synthetic(scale=0.5, shift=-2.0)

    fitted = fit_calibrator(logits, labels, kind="platt")

    assert fitted.kind == "platt"
    assert fitted.scale == pytest.approx(0.5, abs=0.03)
    assert fitted.shift == pytest.approx(-2.0, abs=0.06)


def test_temperature_scaling_fits_the_slope_and_leaves_the_offset_alone():
    logits, labels = _synthetic(scale=2.0, shift=0.0)

    fitted = fit_calibrator(logits, labels, kind="temperature")

    assert fitted.shift == 0.0
    assert fitted.scale == pytest.approx(2.0, abs=0.1)


def test_fitting_is_deterministic():
    logits, labels = _synthetic(scale=0.7, shift=-1.0, n=2_000)

    assert fit_calibrator(logits, labels) == fit_calibrator(logits, labels)


def test_a_fitted_calibrator_never_reorders_the_lines():
    """Even on labels that argue for it: here higher logits are *less* likely flagged."""
    logits = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
    labels = (logits < 0).float()

    fitted = fit_calibrator(logits, labels, kind="platt")
    probabilities = fitted.probabilities(logits)

    assert fitted.scale > 0.0
    assert bool((probabilities[1:] >= probabilities[:-1]).all())


def test_a_calibrator_that_would_reorder_the_lines_cannot_be_built():
    with pytest.raises(ModelError, match="scale must be positive"):
        LogitCalibrator(scale=-1.0)
    with pytest.raises(ModelError, match="scale must be positive"):
        LogitCalibrator(scale=0.0)


def test_the_identity_calibrator_is_the_plain_sigmoid():
    logits = torch.tensor([-2.0, 0.0, 3.0])

    assert torch.allclose(LogitCalibrator().probabilities(logits), torch.sigmoid(logits))
    assert LogitCalibrator().to_dict() == {"kind": "identity", "scale": 1.0, "shift": 0.0}


@pytest.mark.parametrize("labels", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], ids=["no positive", "all"])
def test_a_single_class_cannot_be_calibrated_against(labels):
    """The likelihood has no maximum: the fit would run the shift to infinity."""
    with pytest.raises(ModelError, match="single class"):
        fit_calibrator(torch.tensor([0.1, 0.2, 0.3]), torch.tensor(labels))


def test_malformed_input_is_refused():
    with pytest.raises(ModelError, match="equal length"):
        fit_calibrator(torch.zeros(3), torch.zeros(4))
    with pytest.raises(ModelError, match="no scored lines"):
        fit_calibrator(torch.zeros(0), torch.zeros(0))
    with pytest.raises(ModelError, match="unknown calibrator kind"):
        fit_calibrator(torch.zeros(2), torch.tensor([0.0, 1.0]), kind="isotonic")  # type: ignore[arg-type]


# --- the reliability curve ------------------------------------------------------------


def test_the_reliability_curve_by_hand():
    """Two bins. Low bin: p = .1, .2, .4 with one positive. High bin: p = .6, 1.0, both."""
    probabilities = torch.tensor([0.1, 0.2, 0.4, 0.6, 1.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])

    low, high = reliability_bins(probabilities, labels, n_bins=2)

    assert (low.lower, low.upper, low.count) == (0.0, 0.5, 3)
    assert low.mean_probability == pytest.approx(0.7 / 3)
    assert low.positive_rate == pytest.approx(1 / 3)
    assert (high.lower, high.upper, high.count) == (0.5, 1.0, 2)
    assert high.mean_probability == pytest.approx(0.8)
    assert high.positive_rate == pytest.approx(1.0)
    # ECE = (3 * |0.2333 - 0.3333| + 2 * |0.8 - 1.0|) / 5
    assert expected_calibration_error([low, high]) == pytest.approx((3 * 0.1 + 2 * 0.2) / 5)


def test_an_empty_bin_is_reported_empty_not_dropped():
    bins = reliability_bins(torch.tensor([0.05, 0.95]), torch.tensor([0.0, 1.0]), n_bins=4)

    assert [calibration_bin.count for calibration_bin in bins] == [1, 0, 0, 1]
    assert bins[1].mean_probability == 0.0 and bins[1].positive_rate == 0.0


def test_a_probability_of_exactly_one_lands_in_the_last_bin():
    bins = reliability_bins(torch.tensor([1.0]), torch.tensor([1.0]), n_bins=10)

    assert bins[-1].count == 1


def test_perfect_calibration_has_zero_error():
    """Ten lines at p = 0.3 of which exactly three are flagged."""
    probabilities = torch.full((10,), 0.3, dtype=torch.float64)
    labels = torch.tensor([1.0] * 3 + [0.0] * 7)

    error = expected_calibration_error(reliability_bins(probabilities, labels))

    assert error == pytest.approx(0.0, abs=1e-12)


def test_an_empty_curve_has_no_error_to_report():
    assert expected_calibration_error([]) == 0.0
    with pytest.raises(ModelError, match="n_bins must be positive"):
        reliability_bins(torch.tensor([0.5]), torch.tensor([1.0]), n_bins=0)


# --- the threshold --------------------------------------------------------------------


def test_the_f1_threshold_by_hand():
    """Ranked: .9(+) .8(-) .7(+) .4(+) .2(-).

    Cut after 1: P 1/1, R 1/3, F1 0.5. After 2: 1/2, 1/3 -> 0.4. After 3: 2/3, 2/3 -> 0.667.
    After 4: 3/4, 3/3 -> 0.857. After 5: 3/5, 1 -> 0.75. The best is 0.4.
    """
    probabilities = torch.tensor([0.2, 0.9, 0.4, 0.8, 0.7])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])

    choice = best_f1_threshold(probabilities, labels)

    assert choice.threshold == pytest.approx(0.4)
    assert choice.precision == pytest.approx(0.75)
    assert choice.recall == pytest.approx(1.0)
    assert choice.f1 == pytest.approx(6 / 7)
    assert choice.predicted_positive == 4


def test_a_threshold_cannot_split_a_tie():
    """Three lines share p = 0.5 and one of them is flagged: it is all three or none."""
    probabilities = torch.tensor([0.5, 0.5, 0.5, 0.1])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])

    choice = best_f1_threshold(probabilities, labels)

    assert choice.threshold == pytest.approx(0.5)
    assert choice.predicted_positive == 3
    assert choice.precision == pytest.approx(1 / 3)
    assert choice.f1 == pytest.approx(0.5)


def test_equal_f1_goes_to_the_higher_threshold():
    """Fewer lines for a reviewer to read at the same score.

    Ranked .8(+) .6(-) .5(-) .4(+). Cut after 1: P 1, R 1/2, F1 2/3. After 2: 1/2, 1/2 ->
    1/2. After 3: 1/3, 1/2 -> 0.4. After 4: 1/2, 1 -> 2/3. The first and the last tie.
    """
    probabilities = torch.tensor([0.8, 0.6, 0.5, 0.4])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])

    choice = best_f1_threshold(probabilities, labels)

    assert choice.f1 == pytest.approx(2 / 3)
    assert choice.threshold == pytest.approx(0.8)
    assert choice.predicted_positive == 1


def test_a_threshold_needs_a_positive_line():
    with pytest.raises(ModelError, match="without a positive line"):
        best_f1_threshold(torch.tensor([0.2, 0.7]), torch.tensor([0.0, 0.0]))


def test_the_chosen_threshold_reproduces_its_own_numbers():
    """Apply the threshold the ordinary way and count: the report must be what you get."""
    logits, labels = _synthetic(scale=1.0, shift=-2.5, n=3_000, seed=3)
    probabilities = torch.sigmoid(logits)

    choice = best_f1_threshold(probabilities, labels)
    predicted = probabilities.to(torch.float64) >= choice.threshold
    true = labels > 0
    tp = int((predicted & true).sum())
    precision = tp / int(predicted.sum())
    recall = tp / int(true.sum())

    assert int(predicted.sum()) == choice.predicted_positive
    assert precision == pytest.approx(choice.precision)
    assert recall == pytest.approx(choice.recall)
    assert math.isclose(2 * precision * recall / (precision + recall), choice.f1, rel_tol=1e-9)
