"""质量加权 COM 残差分解及 Oracle 上界。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class COMDecomposition:
    """R = com + internal，其中 internal 的质量加权均值严格为零。"""

    com: np.ndarray
    internal: np.ndarray


def _trajectory(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [frames, atoms, 3], got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _masses(value, atoms: int) -> np.ndarray:
    masses = np.asarray(value)
    if masses.shape != (atoms,):
        raise ValueError(f"masses must have shape {(atoms,)}, got {masses.shape}")
    if not np.issubdtype(masses.dtype, np.floating):
        raise TypeError(f"masses must be floating point, got {masses.dtype}")
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("masses must be finite and strictly positive")
    return masses


def mass_weighted_com_decomposition(residual, masses) -> COMDecomposition:
    """按原子质量把每帧残差唯一分成整体平移与零 COM 内部形变。"""

    residual = _trajectory("residual", residual)
    masses = _masses(masses, residual.shape[1])
    weights = masses / masses.sum()
    com = np.einsum("a,fad->fd", weights, residual)
    internal = residual - com[:, None, :]
    return COMDecomposition(com=com, internal=internal)


def weighted_pythagorean_terms(residual, masses) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回总误差、COM 误差和内部误差；三者逐帧满足 total = com + internal。"""

    residual = _trajectory("residual", residual)
    masses = _masses(masses, residual.shape[1])
    decomposition = mass_weighted_com_decomposition(residual, masses)
    total = np.einsum("a,fad,fad->f", masses, residual, residual)
    com = masses.sum() * np.einsum("fd,fd->f", decomposition.com, decomposition.com)
    internal = np.einsum(
        "a,fad,fad->f", masses, decomposition.internal, decomposition.internal
    )
    return total, com, internal


def radius_of_gyration(positions) -> np.ndarray:
    """与官方评估一致的非质量加权配体回转半径。"""

    positions = _trajectory("positions", positions)
    center = positions.mean(axis=1, keepdims=True)
    return np.sqrt(np.square(positions - center).sum(axis=-1).mean(axis=1))


def analyze_com_oracle(
    neuralmd_positions,
    true_positions,
    residual,
    masses,
) -> dict[str, np.ndarray]:
    """计算 NeuralMD 基线与“已知真实 COM 漂移”Oracle 的逐帧指标。"""

    prediction = _trajectory("neuralmd_positions", neuralmd_positions)
    target = _trajectory("true_positions", true_positions)
    residual = _trajectory("residual", residual)
    if prediction.shape != target.shape or prediction.shape != residual.shape:
        raise ValueError("prediction, target and residual shapes must match")
    np.testing.assert_allclose(residual, target - prediction, rtol=1e-5, atol=1e-6)

    decomposition = mass_weighted_com_decomposition(residual, masses)
    corrected = prediction + decomposition.com[:, None, :]
    baseline_error = target - prediction
    oracle_error = target - corrected

    total, com, internal = weighted_pythagorean_terms(residual, masses)
    identity_error = np.abs(total - com - internal)
    tolerance = 1e-5 * np.maximum(total, 1.0)
    if np.any(identity_error > tolerance):
        raise RuntimeError("mass-weighted COM decomposition failed its Pythagorean identity")

    baseline_point = np.linalg.norm(baseline_error, axis=-1).mean(axis=1)
    oracle_point = np.linalg.norm(oracle_error, axis=-1).mean(axis=1)
    improvement = np.divide(
        baseline_point - oracle_point,
        baseline_point,
        out=np.zeros_like(baseline_point),
        where=baseline_point > 0,
    ) * 100.0

    prediction_center = prediction.mean(axis=1)
    corrected_center = corrected.mean(axis=1)
    target_center = target.mean(axis=1)
    prediction_rg = radius_of_gyration(prediction)
    corrected_rg = radius_of_gyration(corrected)
    target_rg = radius_of_gyration(target)

    return {
        "baseline_point_rmse": baseline_point,
        "oracle_point_rmse": oracle_point,
        "point_improvement_pct": improvement,
        "mass_com_shift": np.linalg.norm(decomposition.com, axis=-1),
        "baseline_official_com_error": np.linalg.norm(
            prediction_center - target_center, axis=-1
        ),
        "oracle_official_com_error": np.linalg.norm(corrected_center - target_center, axis=-1),
        "baseline_rg_error": np.abs(prediction_rg - target_rg),
        "oracle_rg_error": np.abs(corrected_rg - target_rg),
        "weighted_com_error_fraction": np.divide(
            com,
            total,
            out=np.zeros_like(com),
            where=total > 0,
        ),
        "pythagorean_abs_error": identity_error,
    }
