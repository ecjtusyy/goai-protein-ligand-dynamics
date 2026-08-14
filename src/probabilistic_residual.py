"""NeuralMD 后处理概率残差的最小数学合同。"""

from __future__ import annotations

import math

import numpy as np


def _positions(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [frames, atoms, 3], got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def residual_target(neuralmd_positions, true_positions) -> np.ndarray:
    """构造训练标签 R = X_true - X_NeuralMD。"""

    prediction = _positions("neuralmd_positions", neuralmd_positions)
    target = _positions("true_positions", true_positions)
    if prediction.shape != target.shape:
        raise ValueError(f"trajectory shape mismatch: {prediction.shape} != {target.shape}")
    return target - prediction


def corrected_positions(neuralmd_positions, residual_mean) -> np.ndarray:
    """返回最佳点预测 X_corrected = X_NeuralMD + mu。"""

    prediction = _positions("neuralmd_positions", neuralmd_positions)
    mean = _positions("residual_mean", residual_mean)
    if prediction.shape != mean.shape:
        raise ValueError(f"trajectory shape mismatch: {prediction.shape} != {mean.shape}")
    return prediction + mean


def isotropic_gaussian_nll(residual, mean, scale) -> float:
    """计算每个原子三维各向同性高斯分布的平均 NLL。"""

    residual = _positions("residual", residual)
    mean = _positions("mean", mean)
    if residual.shape != mean.shape:
        raise ValueError(f"trajectory shape mismatch: {residual.shape} != {mean.shape}")

    scale = np.asarray(scale)
    expected = residual.shape[:2]
    if scale.shape != expected:
        raise ValueError(f"scale must have shape {expected}, got {scale.shape}")
    if not np.issubdtype(scale.dtype, np.floating):
        raise TypeError(f"scale must be floating point, got {scale.dtype}")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scale must be finite and strictly positive")

    squared_error = np.square(residual - mean).sum(axis=-1)
    variance = np.square(scale)
    per_atom = 0.5 * (squared_error / variance + 3.0 * np.log(2.0 * math.pi * variance))
    return float(per_atom.mean())
