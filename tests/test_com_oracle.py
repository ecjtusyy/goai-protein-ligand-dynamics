import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_com_oracle as RUNNER
from src.com_oracle import (
    analyze_com_oracle,
    mass_weighted_com_decomposition,
    weighted_pythagorean_terms,
)


def synthetic_trajectory():
    prediction = np.zeros((3, 2, 3), dtype=np.float32)
    masses = np.array([1.0, 3.0], dtype=np.float32)
    drift = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    local = np.array(
        [
            [[0.3, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            [[0.6, 0.0, 0.0], [-0.2, 0.0, 0.0]],
            [[0.9, 0.0, 0.0], [-0.3, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    residual = drift[:, None, :] + local
    target = prediction + residual
    return prediction, target, residual, masses, drift, local


def test_mass_com_decomposition_is_unique_and_pythagorean() -> None:
    _, _, residual, masses, drift, local = synthetic_trajectory()

    decomposition = mass_weighted_com_decomposition(residual, masses)
    total, com, internal = weighted_pythagorean_terms(residual, masses)

    np.testing.assert_allclose(decomposition.com, drift, atol=1e-6)
    np.testing.assert_allclose(decomposition.internal, local, atol=1e-6)
    np.testing.assert_allclose(
        np.einsum("a,fad->fd", masses, decomposition.internal),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(total, com + internal, rtol=1e-6, atol=1e-6)


def test_oracle_removes_only_translation_and_preserves_rg() -> None:
    prediction, target, residual, masses, _, local = synthetic_trajectory()

    metrics = analyze_com_oracle(prediction, target, residual, masses)

    expected_oracle = np.linalg.norm(local, axis=-1).mean(axis=1)
    np.testing.assert_allclose(metrics["oracle_point_rmse"], expected_oracle, atol=1e-6)
    np.testing.assert_allclose(
        metrics["oracle_rg_error"], metrics["baseline_rg_error"], atol=1e-6
    )
    assert np.all(metrics["point_improvement_pct"] > 0)
    assert np.max(metrics["pythagorean_abs_error"]) < 1e-5


def write_split(root: Path, split: str) -> None:
    split_root = root / split
    complexes = split_root / "complexes"
    complexes.mkdir(parents=True)
    prediction, target, residual, masses, _, _ = synthetic_trajectory()
    relative = "complexes/X001.npz"
    np.savez_compressed(
        split_root / relative,
        neuralmd_positions=prediction,
        true_positions=target,
        residual=residual,
        target_frames=np.array([20, 21, 22]),
        ligand_atom_types=np.array([5, 7]),
        ligand_masses=masses,
        protein_ca_positions=np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        protein_residue_types=np.array([1]),
    )
    manifest = {"split": split, "task": "T3", "complexes": 1, "files": [relative]}
    (split_root / "manifest.json").write_text(json.dumps(manifest))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_runner_writes_train_val_csv_png_and_decision(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output = tmp_path / "output"
    write_split(cache_root, "train")
    write_split(cache_root, "val")

    RUNNER.main(
        [
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(output),
            "--success-threshold-pct",
            "5",
        ]
    )

    assert len(read_rows(output / "com_oracle_frames.csv")) == 6
    assert len(read_rows(output / "com_oracle_complexes.csv")) == 2
    assert {row["split"] for row in read_rows(output / "com_oracle_summary.csv")} == {
        "train",
        "val",
    }
    decision = read_rows(output / "com_oracle_decision.csv")[0]
    assert decision["decision_split"] == "val"
    assert decision["decision"] == "PROCEED"
    image = output / "com_oracle_curves.png"
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_forbids_test_split() -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            [
                "--cache-root",
                "/tmp/cache",
                "--output-dir",
                "/tmp/output",
                "--splits",
                "test",
            ]
        )
