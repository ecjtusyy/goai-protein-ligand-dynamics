from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.residual_cache import (
    build_cache_payload,
    read_training_split_ids,
    validate_complex_cache,
    write_complex_cache,
)


def fake_batch() -> SimpleNamespace:
    protein_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    return SimpleNamespace(
        ligand_x=np.array([6, 8]),
        ligand_mass=np.array([12.0, 16.0]),
        protein_pos=protein_pos,
        mask_n=np.array([True, False, False, True, False, False]),
        mask_ca=np.array([False, True, False, False, True, False]),
        mask_c=np.array([False, False, True, False, False, True]),
        protein_backbone_residue=np.array([1, 12]),
    )


def test_training_split_reader_excludes_test(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train_MD.txt").write_text("1abc\n2def\n")

    assert read_training_split_ids(tmp_path, "train") == ["1ABC", "2DEF"]
    with pytest.raises(ValueError, match="train.*val"):
        read_training_split_ids(tmp_path, "test")


def test_payload_contains_exact_residual_and_context() -> None:
    prediction = np.zeros((3, 2, 3), dtype=np.float32)
    target = np.arange(18, dtype=np.float32).reshape(3, 2, 3)

    payload = build_cache_payload(fake_batch(), prediction, target, [20, 21, 22])

    np.testing.assert_array_equal(payload["residual"], target - prediction)
    np.testing.assert_array_equal(payload["target_frames"], [20, 21, 22])
    assert payload["protein_ca_positions"].shape == (2, 3)
    assert payload["ligand_atom_types"].dtype == np.int64


def test_cache_write_is_compressed_and_refuses_accidental_overwrite(tmp_path: Path) -> None:
    prediction = np.zeros((1, 2, 3), dtype=np.float32)
    payload = build_cache_payload(fake_batch(), prediction, prediction, [20])

    destination = write_complex_cache(tmp_path, "1abc", payload)

    assert destination == tmp_path / "complexes/1ABC.npz"
    with np.load(destination) as stored:
        np.testing.assert_array_equal(stored["residual"], prediction)
    with pytest.raises(FileExistsError, match="already exists"):
        write_complex_cache(tmp_path, "1ABC", payload)


def test_resume_validation_rejects_wrong_target_frames(tmp_path: Path) -> None:
    prediction = np.zeros((1, 2, 3), dtype=np.float32)
    payload = build_cache_payload(fake_batch(), prediction, prediction, [20])
    destination = write_complex_cache(tmp_path, "1ABC", payload)

    validate_complex_cache(destination, [20])
    with pytest.raises(ValueError, match="target frames"):
        validate_complex_cache(destination, [21])
