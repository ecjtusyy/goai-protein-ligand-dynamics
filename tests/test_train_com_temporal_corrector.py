import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts import train_com_temporal_corrector as TRAINER


def make_split(root: Path, split: str, count: int) -> None:
    split_dir = root / split
    complexes = split_dir / "complexes"
    complexes.mkdir(parents=True)
    files = []
    for index in range(count):
        pdb_id = f"C{index:03d}"
        steps = np.arange(5, dtype=np.float32)
        positions = np.stack(
            (
                np.stack((0.1 * steps, np.zeros(5), np.zeros(5)), axis=-1),
                np.stack((1.0 + 0.1 * steps, np.full(5, 0.2), np.zeros(5)), axis=-1),
            ),
            axis=1,
        ).astype(np.float32)
        drift = np.stack((0.05 * steps, 0.02 * steps, np.zeros(5)), axis=-1)
        residual = np.broadcast_to(drift[:, None, :], positions.shape).copy()
        arrays = {
            "neuralmd_positions": positions,
            "true_positions": positions + residual,
            "residual": residual,
            "target_frames": np.arange(20, 25),
            "ligand_atom_types": np.array([5, 7]),
            "ligand_masses": np.array([12.0, 16.0], dtype=np.float32),
            "protein_ca_positions": np.array(
                [[0.0, 1.0, 2.0], [2.0, -1.0, 1.0]], dtype=np.float32
            ),
            "protein_residue_types": np.array([1, 2]),
        }
        relative = f"complexes/{pdb_id}.npz"
        np.savez_compressed(split_dir / relative, **arrays)
        files.append(relative)
    manifest = {"split": split, "task": "T3", "complexes": count, "files": files}
    (split_dir / "manifest.json").write_text(json.dumps(manifest))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_selection_score_uses_worse_of_mean_and_final() -> None:
    metrics = {
        "baseline_mean_point_rmse": 4.0,
        "corrected_mean_point_rmse": 3.0,
        "baseline_final_point_rmse": 8.0,
        "corrected_final_point_rmse": 7.2,
    }

    assert TRAINER._selection_score(metrics) == 0.9


def test_end_to_end_cpu_training_resume_and_artifacts(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "output"
    make_split(cache_root, "train", 2)
    make_split(cache_root, "val", 1)
    common = [
        "--cache-root",
        str(cache_root),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--patience",
        "5",
        "--hidden-dim",
        "8",
        "--rbf-channels",
        "4",
    ]

    TRAINER.main(common + ["--epochs", "2"])

    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=True)
    config = json.loads((output_dir / "config.json").read_text())
    assert 0 <= checkpoint["epoch"] <= 2
    assert checkpoint["model_config"]["rbf_channels"] == 4
    assert config["safe_fallback"] == "epoch_0_zero_correction"
    assert len(read_rows(output_dir / "history.csv")) == 2
    assert len(read_rows(output_dir / "com_temporal_val_frames.csv")) == 5
    assert len(read_rows(output_dir / "com_temporal_val_complexes.csv")) == 1
    assert len(read_rows(output_dir / "com_temporal_summary.csv")) == 1
    assert len(read_rows(output_dir / "com_temporal_decision.csv")) == 1
    assert (output_dir / "com_temporal_val_curves.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    TRAINER.main(common + ["--epochs", "3", "--resume"])

    assert len(read_rows(output_dir / "history.csv")) == 3
