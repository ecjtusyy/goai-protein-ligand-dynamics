import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src/neuralmd_official.py"
SPEC = importlib.util.spec_from_file_location("neuralmd_official", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ROLLOUT_WINDOWS = MODULE.ROLLOUT_WINDOWS
rollout_contract = MODULE.rollout_contract
verify_misato1000 = MODULE.verify_misato1000


def _make_dataset(root: Path) -> Path:
    dataset = root / "MISATO_1000"
    raw = dataset / "raw"
    raw.mkdir(parents=True)
    (raw / "MD.hdf5").write_bytes(b"test-placeholder")

    offset = 0
    for split, count in (("train", 800), ("val", 100), ("test", 100)):
        ids = [f"X{index:04d}" for index in range(offset, offset + count)]
        (raw / f"{split}_MD.txt").write_text("\n".join(ids) + "\n")
        offset += count
    return dataset


def test_rollout_windows_end_at_snapshot_100() -> None:
    assert ROLLOUT_WINDOWS["paper"].position_frame == 0
    assert ROLLOUT_WINDOWS["paper"].velocity_to_frame == 1
    assert ROLLOUT_WINDOWS["T1"].position_frame == 9
    assert ROLLOUT_WINDOWS["T1"].velocity_to_frame == 9
    assert ROLLOUT_WINDOWS["T1"].target_stop == 20
    assert ROLLOUT_WINDOWS["T2"].target_stop == 100
    assert ROLLOUT_WINDOWS["T3"].target_stop == 100
    assert rollout_contract()["paper"]["target_stop"] == 100


def test_verify_misato1000_checks_official_split_contract(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    metadata = verify_misato1000(dataset, strict_size=False)

    assert metadata["total_complexes"] == 1000
    assert metadata["split_counts"] == {"train": 800, "val": 100, "test": 100}


def test_verify_misato1000_rejects_overlap(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    val_path = dataset / "raw/val_MD.txt"
    val_ids = val_path.read_text().splitlines()
    val_ids[0] = "X0000"
    val_path.write_text("\n".join(val_ids) + "\n")

    with pytest.raises(ValueError, match="overlap"):
        verify_misato1000(dataset, strict_size=False)
