import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.residual_training import ResidualCacheDataset, residual_loss
from src.temporal_residual_model import ResidualPrediction


def make_cache(split_dir: Path, split: str = "train") -> None:
    complexes = split_dir / "complexes"
    complexes.mkdir(parents=True)
    arrays = {
        "neuralmd_positions": np.zeros((2, 1, 3), dtype=np.float32),
        "true_positions": np.ones((2, 1, 3), dtype=np.float32),
        "residual": np.ones((2, 1, 3), dtype=np.float32),
        "target_frames": np.array([20, 21]),
        "ligand_atom_types": np.array([5]),
        "ligand_masses": np.array([12.0], dtype=np.float32),
        "protein_ca_positions": np.zeros((1, 3), dtype=np.float32),
        "protein_residue_types": np.array([1]),
    }
    np.savez_compressed(complexes / "1ABC.npz", **arrays)
    manifest = {"split": split, "complexes": 1, "files": ["complexes/1ABC.npz"]}
    (split_dir / "manifest.json").write_text(json.dumps(manifest))


def test_dataset_loads_one_variable_size_complex(tmp_path: Path) -> None:
    make_cache(tmp_path)

    dataset = ResidualCacheDataset(tmp_path, expected_split="train")
    item = dataset[0]

    assert len(dataset) == 1
    assert item["pdb_id"] == "1ABC"
    torch.testing.assert_close(item["residual"], torch.ones(2, 1, 3))


def test_dataset_refuses_split_mismatch(tmp_path: Path) -> None:
    make_cache(tmp_path, split="val")

    with pytest.raises(ValueError, match="expected 'train'"):
        ResidualCacheDataset(tmp_path, expected_split="train")


def test_probabilistic_loss_matches_isotropic_closed_form() -> None:
    mean = torch.zeros((2, 1, 3), requires_grad=True)
    scale = torch.ones((2, 1), requires_grad=True)
    residual = torch.zeros_like(mean)

    terms = residual_loss(ResidualPrediction(mean, scale), residual, uncertainty_weight=1.0)

    assert terms.nll is not None
    assert terms.nll.item() == pytest.approx(1.5 * math.log(2.0 * math.pi))
    assert terms.point_rmse.item() == 0.0


def test_nll_does_not_reweight_mean_gradient() -> None:
    residual = torch.ones((1, 1, 3))
    mean = torch.zeros_like(residual, requires_grad=True)
    scale = torch.full((1, 1), 0.2, requires_grad=True)

    terms = residual_loss(ResidualPrediction(mean, scale), residual, uncertainty_weight=10.0)
    terms.total.backward()
    actual_mean_gradient = mean.grad.clone()

    reference_mean = torch.zeros_like(residual, requires_grad=True)
    reference_loss = residual_loss(
        ResidualPrediction(reference_mean, None), residual, uncertainty_weight=0.0
    )
    reference_loss.total.backward()

    torch.testing.assert_close(actual_mean_gradient, reference_mean.grad)
    assert scale.grad is not None
