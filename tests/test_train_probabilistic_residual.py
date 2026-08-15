import json
from pathlib import Path

import numpy as np
import torch

from scripts import train_probabilistic_residual as TRAINER


def make_split(root: Path, split: str, count: int) -> None:
    split_dir = root / split
    complexes = split_dir / "complexes"
    complexes.mkdir(parents=True)
    files = []
    for index in range(count):
        pdb_id = f"X{index:03d}"
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        residual = np.full_like(positions, 0.05 * (index + 1))
        arrays = {
            "neuralmd_positions": positions,
            "true_positions": positions + residual,
            "residual": residual,
            "target_frames": np.array([20, 21]),
            "ligand_atom_types": np.array([5, 7]),
            "ligand_masses": np.array([12.0, 16.0], dtype=np.float32),
            "protein_ca_positions": np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
            "protein_residue_types": np.array([1]),
        }
        relative = f"complexes/{pdb_id}.npz"
        np.savez_compressed(split_dir / relative, **arrays)
        files.append(relative)
    manifest = {"split": split, "task": "T3", "complexes": count, "files": files}
    (split_dir / "manifest.json").write_text(json.dumps(manifest))


def test_restore_generator_state_keeps_cpu_byte_tensor() -> None:
    source = torch.Generator().manual_seed(123)
    restored = torch.Generator().manual_seed(999)

    TRAINER._restore_generator_state(restored, source.get_state())

    assert restored.get_state().device.type == "cpu"
    assert restored.get_state().dtype == torch.uint8
    assert torch.equal(restored.get_state(), source.get_state())


def test_end_to_end_cpu_training_smoke(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "output"
    make_split(cache_root, "train", 2)
    make_split(cache_root, "val", 1)

    TRAINER.main(
        [
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(output_dir),
            "--variant",
            "ode_temporal_mu_sigma",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--patience",
            "5",
            "--hidden-dim",
            "8",
            "--rbf-channels",
            "4",
        ]
    )

    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=True)
    history = (output_dir / "history.csv").read_text().splitlines()
    config = json.loads((output_dir / "config.json").read_text())

    assert checkpoint["variant"] == "ode_temporal_mu_sigma"
    assert checkpoint["val_point_rmse"] >= 0
    assert checkpoint["model_config"]["frame_chunk_size"] == 16
    assert checkpoint["model_config"]["min_scale"] == 1e-3
    assert len(history) == 3
    assert config["selection_metric"] == "val_point_rmse"

    TRAINER.main(
        [
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(output_dir),
            "--variant",
            "ode_temporal_mu_sigma",
            "--device",
            "cpu",
            "--epochs",
            "3",
            "--patience",
            "5",
            "--hidden-dim",
            "8",
            "--rbf-channels",
            "4",
            "--resume",
        ]
    )

    resumed_history = (output_dir / "history.csv").read_text().splitlines()
    assert len(resumed_history) == 4
