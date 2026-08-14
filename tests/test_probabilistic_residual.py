import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src/probabilistic_residual.py"
SPEC = importlib.util.spec_from_file_location("probabilistic_residual", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

corrected_positions = MODULE.corrected_positions
isotropic_gaussian_nll = MODULE.isotropic_gaussian_nll
residual_target = MODULE.residual_target


def test_residual_mean_exactly_recovers_target() -> None:
    prediction = np.array([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]])
    target = np.array([[[1.5, 1.0, 4.0]], [[3.0, 3.5, 2.0]]])

    residual = residual_target(prediction, target)
    corrected = corrected_positions(prediction, residual)

    np.testing.assert_allclose(corrected, target)


def test_isotropic_nll_matches_closed_form() -> None:
    residual = np.zeros((2, 1, 3), dtype=np.float64)
    mean = np.zeros_like(residual)
    scale = np.ones((2, 1), dtype=np.float64)

    actual = isotropic_gaussian_nll(residual, mean, scale)
    expected = 1.5 * math.log(2.0 * math.pi)

    assert actual == pytest.approx(expected)


def test_isotropic_nll_is_rotation_invariant() -> None:
    residual = np.array([[[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]]])
    mean = np.array([[[0.2, -1.5, 0.0], [0.5, 0.5, 2.0]]])
    scale = np.array([[0.7, 1.3]])
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = isotropic_gaussian_nll(residual, mean, scale)
    rotated = isotropic_gaussian_nll(residual @ rotation.T, mean @ rotation.T, scale)

    assert rotated == pytest.approx(original)


def test_contract_rejects_shape_and_scale_errors() -> None:
    positions = np.zeros((2, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        residual_target(positions, positions[:, :2])
    with pytest.raises(ValueError, match="strictly positive"):
        isotropic_gaussian_nll(positions, positions, np.zeros((2, 3), dtype=np.float32))
