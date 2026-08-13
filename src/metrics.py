import numpy as np


def rmsd_curve(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(
            f"预测与目标形状不同: {prediction.shape} != {target.shape}"
        )
    squared_distance = np.sum((prediction - target) ** 2, axis=-1)
    return np.sqrt(np.mean(squared_distance, axis=-1))
