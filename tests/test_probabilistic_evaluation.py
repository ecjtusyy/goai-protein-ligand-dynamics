import math

import pytest
import torch

from src.probabilistic_evaluation import probabilistic_metrics


def test_zero_standardized_error_has_full_radial_coverage() -> None:
    target = torch.zeros((2, 3, 3))
    scale = torch.ones((2, 3))

    metrics = probabilistic_metrics(target, scale, target)

    assert metrics["nll"] == pytest.approx(1.5 * math.log(2.0 * math.pi))
    assert metrics["mean_sigma"] == 1.0
    assert metrics["coverage_50"] == 1.0
    assert metrics["coverage_95"] == 1.0


def test_radial_coverage_uses_three_dimensional_chi_square_thresholds() -> None:
    mean = torch.zeros((1, 2, 3))
    target = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    scale = torch.ones((1, 2))

    metrics = probabilistic_metrics(mean, scale, target)

    assert metrics["coverage_50"] == 0.5
    assert metrics["coverage_95"] == 0.5


def test_probabilistic_metrics_reject_nonpositive_scale() -> None:
    positions = torch.zeros((1, 1, 3))

    with pytest.raises(ValueError, match="strictly positive"):
        probabilistic_metrics(positions, torch.zeros((1, 1)), positions)
