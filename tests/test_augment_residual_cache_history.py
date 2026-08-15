import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.augment_residual_cache_history import augment_split
from src.residual_cache import augment_complex_cache_history
from src.residual_training import ResidualCacheDataset


def write_base_cache(split_dir: Path) -> np.ndarray:
    (split_dir / "complexes").mkdir(parents=True)
    trajectory = np.arange(30, dtype=np.float32).reshape(5, 2, 3)
    arrays = {
        "neuralmd_positions": np.zeros((3, 2, 3), dtype=np.float32),
        "true_positions": trajectory[2:],
        "residual": trajectory[2:],
        "target_frames": np.array([2, 3, 4]),
        "ligand_atom_types": np.array([5, 7]),
        "ligand_masses": np.array([12.0, 16.0], dtype=np.float32),
        "protein_n_positions": np.zeros((1, 3), dtype=np.float32),
        "protein_ca_positions": np.zeros((1, 3), dtype=np.float32),
        "protein_c_positions": np.zeros((1, 3), dtype=np.float32),
        "protein_residue_types": np.array([1]),
    }
    np.savez_compressed(split_dir / "complexes/1ABC.npz", **arrays)
    manifest = {
        "split": "train",
        "task": "T3",
        "complexes": 1,
        "files": ["complexes/1ABC.npz"],
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest))
    return trajectory


def test_atomic_history_augmentation_is_resumable_and_leak_free(tmp_path: Path) -> None:
    trajectory = write_base_cache(tmp_path)
    path = tmp_path / "complexes/1ABC.npz"

    changed = augment_complex_cache_history(
        path,
        trajectory[:2],
        np.array([0, 1]),
        expected_target_positions=trajectory[2:],
    )

    assert changed
    assert not augment_complex_cache_history(
        path,
        trajectory[:2],
        np.array([0, 1]),
        expected_target_positions=trajectory[2:],
        resume=True,
    )
    wrong_target = trajectory[2:].copy()
    wrong_target[0, 0, 0] += 1.0
    with pytest.raises(AssertionError, match="official trajectory order"):
        augment_complex_cache_history(
            path,
            trajectory[:2],
            np.array([0, 1]),
            expected_target_positions=wrong_target,
            resume=True,
        )
    dataset = ResidualCacheDataset(tmp_path, expected_split="train", require_history=True)
    torch.testing.assert_close(dataset[0]["observed_positions"], torch.from_numpy(trajectory[:2]))
    with pytest.raises(ValueError, match="overlaps"):
        augment_complex_cache_history(
            path,
            trajectory[:3],
            np.array([0, 1, 2]),
            overwrite=True,
        )


def test_split_augmentation_checks_official_target_and_updates_manifest(tmp_path: Path) -> None:
    trajectory = write_base_cache(tmp_path)
    official = SimpleNamespace(
        ligand_trajectory_pos=torch.from_numpy(trajectory.transpose(1, 0, 2))
    )

    written, skipped = augment_split(
        [official],
        ["1ABC"],
        tmp_path,
        history_frames=2,
        target_start=2,
        target_stop=5,
        resume=True,
        overwrite=False,
    )

    assert (written, skipped) == (1, 0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["observed_history"]["frames"] == [0, 1]
    assert manifest["observed_history"]["target_overlap"] is False


def test_split_augmentation_rejects_pdb_order_mismatch(tmp_path: Path) -> None:
    trajectory = write_base_cache(tmp_path)
    official = SimpleNamespace(
        ligand_trajectory_pos=torch.from_numpy(trajectory.transpose(1, 0, 2))
    )

    with pytest.raises(ValueError, match="do not align"):
        augment_split(
            [official],
            ["9XYZ"],
            tmp_path,
            history_frames=2,
            target_start=2,
            target_stop=5,
            resume=True,
            overwrite=False,
        )
