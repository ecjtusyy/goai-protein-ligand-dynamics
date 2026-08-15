"""概率残差缓存加载与训练损失。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .residual_cache import CACHE_ARRAYS, HISTORY_CACHE_ARRAYS, TRAINING_SPLITS
from .temporal_residual_model import ResidualPrediction


CACHE_KEYS = tuple(key for key in CACHE_ARRAYS if key not in {"protein_n_positions", "protein_c_positions"})


class ResidualCacheDataset(Dataset):
    """按复合物懒加载压缩缓存，支持不同配体原子数。"""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        expected_split: str,
        require_history: bool = False,
    ) -> None:
        if expected_split not in TRAINING_SPLITS:
            raise ValueError(f"expected_split must be one of {TRAINING_SPLITS}")
        self.split_dir = Path(split_dir).resolve()
        self.require_history = require_history
        manifest_path = self.split_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("split") != expected_split:
            raise ValueError(
                f"cache split is {self.manifest.get('split')!r}, expected {expected_split!r}"
            )

        files = self.manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("cache manifest has no files")
        if self.manifest.get("complexes") != len(files):
            raise ValueError("cache manifest complex count does not match file list")
        self.files = []
        for relative in files:
            path = (self.split_dir / relative).resolve()
            if not path.is_relative_to(self.split_dir):
                raise ValueError(f"cache path escapes split directory: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"missing cache file: {path}")
            self.files.append(path)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as archive:
            required = CACHE_KEYS + (HISTORY_CACHE_ARRAYS if self.require_history else ())
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f"{path} is missing cache arrays: {missing}")
            history_present = [key for key in HISTORY_CACHE_ARRAYS if key in archive]
            if history_present and len(history_present) != len(HISTORY_CACHE_ARRAYS):
                raise ValueError(f"{path} contains a partial observed-history contract")
            keys = CACHE_KEYS + (HISTORY_CACHE_ARRAYS if history_present else ())
            item = {key: torch.from_numpy(archive[key].copy()) for key in keys}
        item["pdb_id"] = path.stem
        return item


@dataclass
class LossTerms:
    total: torch.Tensor
    mean_mse: torch.Tensor
    point_rmse: torch.Tensor
    nll: torch.Tensor | None


def residual_loss(
    prediction: ResidualPrediction,
    residual: torch.Tensor,
    *,
    uncertainty_weight: float = 0.1,
) -> LossTerms:
    """用 MSE 守住点预测；NLL 只校准已分离梯度的尺度分支。"""

    if prediction.mean.shape != residual.shape:
        raise ValueError(f"mean {prediction.mean.shape} != residual {residual.shape}")
    if uncertainty_weight < 0:
        raise ValueError("uncertainty_weight must be non-negative")

    error = prediction.mean - residual
    mean_mse = error.square().mean()
    point_rmse = torch.linalg.vector_norm(error, dim=-1).mean()
    if prediction.scale is None:
        return LossTerms(mean_mse, mean_mse, point_rmse, None)
    if prediction.scale.shape != residual.shape[:2]:
        raise ValueError("scale must have shape [frames, atoms]")
    if not torch.isfinite(prediction.scale).all() or (prediction.scale <= 0).any():
        raise ValueError("scale must be finite and strictly positive")

    scale_error = residual - prediction.mean.detach()
    variance = prediction.scale.square()
    squared_distance = scale_error.square().sum(dim=-1)
    nll = 0.5 * (
        squared_distance / variance + 3.0 * torch.log(2.0 * math.pi * variance)
    ).mean()
    total = mean_mse + uncertainty_weight * nll
    return LossTerms(total, mean_mse, point_rmse, nll)
