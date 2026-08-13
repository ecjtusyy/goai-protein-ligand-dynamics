import numpy as np


def static_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    _check_input(history, horizon)
    return np.repeat(history[-1][None], horizon, axis=0)


def linear_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    _check_input(history, horizon, min_frames=2)
    velocity = history[-1] - history[-2]
    steps = np.arange(1, horizon + 1, dtype=history.dtype)[:, None, None]
    return history[-1][None] + steps * velocity[None]


def _check_input(
    history: np.ndarray,
    horizon: int,
    min_frames: int = 1,
) -> None:
    if history.ndim != 3 or history.shape[-1] != 3:
        raise ValueError(f"历史轨迹形状异常: {history.shape}")
    if len(history) < min_frames:
        raise ValueError(f"至少需要 {min_frames} 帧历史轨迹")
    if horizon < 1:
        raise ValueError("预测步数必须大于 0")
