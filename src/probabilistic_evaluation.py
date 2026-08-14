"""三维各向同性残差分布的 NLL 与径向校准指标。"""

from __future__ import annotations

import math

import torch


# chi-square(df=3) 的固定分位点，避免为四个常数引入 SciPy 运行时依赖。
CHI_SQUARE_3_QUANTILES = {
    0.50: 2.365973884,
    0.80: 4.641627676,
    0.90: 6.251388631,
    0.95: 7.814727903,
}


def probabilistic_metrics(
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
    true_residual: torch.Tensor,
) -> dict[str, float]:
    """评估 ||R-mu||²/sigma² 是否符合三自由度卡方分布。"""

    if residual_mean.shape != true_residual.shape:
        raise ValueError("residual mean and target shapes do not match")
    if residual_scale.shape != true_residual.shape[:2]:
        raise ValueError("residual scale must have shape [frames, atoms]")
    if not torch.isfinite(residual_scale).all() or (residual_scale <= 0).any():
        raise ValueError("residual scale must be finite and strictly positive")

    error = true_residual - residual_mean
    variance = residual_scale.square()
    squared_mahalanobis = error.square().sum(dim=-1) / variance
    nll = 0.5 * (
        squared_mahalanobis + 3.0 * torch.log(2.0 * math.pi * variance)
    ).mean()

    metrics = {
        "nll": float(nll.cpu()),
        "mean_sigma": float(residual_scale.mean().cpu()),
    }
    calibration_errors = []
    for nominal, threshold in CHI_SQUARE_3_QUANTILES.items():
        observed = float((squared_mahalanobis <= threshold).float().mean().cpu())
        metrics[f"coverage_{int(100 * nominal)}"] = observed
        calibration_errors.append(abs(observed - nominal))
    metrics["coverage_mae"] = sum(calibration_errors) / len(calibration_errors)
    return metrics
