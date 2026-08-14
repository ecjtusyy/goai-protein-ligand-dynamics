"""Contracts for reproducing the official NeuralMD MISATO_1000 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


MISATO_1000_HDF5_BYTES = 7_455_614_516
MISATO_1000_SPLIT_COUNTS = {"train": 800, "val": 100, "test": 100}
NEURALMD_ODE_CHECKPOINT_BYTES = 8_955_570


@dataclass(frozen=True)
class RolloutWindow:
    """Frames needed to initialize and score one NeuralMD rollout."""

    name: str
    history_frames: int
    position_frame: int
    velocity_from_frame: int
    velocity_to_frame: int
    target_start: int
    horizon: int

    @property
    def target_stop(self) -> int:
        return self.target_start + self.horizon


ROLLOUT_WINDOWS = {
    # The paper's public evaluator initializes x_0 and v_0 = x_1 - x_0,
    # then scores outputs corresponding to snapshots 1..99.
    "paper": RolloutWindow("paper", 2, 0, 0, 1, 1, 99),
    "T1": RolloutWindow("T1", 10, 9, 8, 9, 10, 10),
    "T2": RolloutWindow("T2", 80, 79, 78, 79, 80, 20),
    "T3": RolloutWindow("T3", 20, 19, 18, 19, 20, 80),
}


def _read_ids(path: Path) -> list[str]:
    return [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]


def verify_misato1000(dataset_dir: str | Path, *, strict_size: bool = True) -> dict:
    """Fail fast when a mounted/downloaded MISATO_1000 is incomplete."""

    dataset_dir = Path(dataset_dir)
    raw_dir = dataset_dir / "raw"
    hdf5_path = raw_dir / "MD.hdf5"
    required = [hdf5_path] + [raw_dir / f"{split}_MD.txt" for split in MISATO_1000_SPLIT_COUNTS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("MISATO_1000 is incomplete; missing: " + ", ".join(missing))

    actual_bytes = hdf5_path.stat().st_size
    if strict_size and actual_bytes != MISATO_1000_HDF5_BYTES:
        raise ValueError(
            f"MD.hdf5 has {actual_bytes:,} bytes; expected {MISATO_1000_HDF5_BYTES:,}. "
            "Delete the partial download and retry."
        )

    splits = {split: _read_ids(raw_dir / f"{split}_MD.txt") for split in MISATO_1000_SPLIT_COUNTS}
    for split, expected in MISATO_1000_SPLIT_COUNTS.items():
        if len(splits[split]) != expected:
            raise ValueError(f"{split} split has {len(splits[split])} IDs; expected {expected}")

    all_ids = [pdb_id for ids in splits.values() for pdb_id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("MISATO_1000 split files overlap or contain duplicate IDs")

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "hdf5_bytes": actual_bytes,
        "split_counts": {split: len(ids) for split, ids in splits.items()},
        "total_complexes": len(all_ids),
    }


def rollout_contract() -> dict[str, dict]:
    """JSON-serializable task definition stored next to every result."""

    return {name: asdict(window) | {"target_stop": window.target_stop} for name, window in ROLLOUT_WINDOWS.items()}
