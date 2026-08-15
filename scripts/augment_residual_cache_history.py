"""不重跑 NeuralMD，把真实观测窗口原子式补入现有 train/val 缓存。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.neuralmd_official import ROLLOUT_WINDOWS, verify_misato1000
from src.residual_cache import (
    TRAINING_SPLITS,
    augment_complex_cache_history,
    read_training_split_ids,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=TRAINING_SPLITS, default=list(TRAINING_SPLITS))
    parser.add_argument("--task", choices=ROLLOUT_WINDOWS, default="T3")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_dataset(official_repo: Path, dataset_dir: Path, split: str):
    required = official_repo / "NeuralMD/datasets/MISATO/dataset_MISATO_semi_flexible.py"
    if not required.is_file():
        raise FileNotFoundError(f"official NeuralMD checkout is incomplete: {required}")
    sys.path.insert(0, str(official_repo.resolve()))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory

    return DatasetMISATOSemiFlexibleMultiTrajectory(str(dataset_dir), mode=split)


def augment_split(
    dataset,
    split_ids: list[str],
    split_dir: Path,
    *,
    history_frames: int,
    target_start: int,
    target_stop: int,
    resume: bool,
    overwrite: bool,
) -> tuple[int, int]:
    """升级一个 split，并以 target 真值反查 PDB 顺序没有错位。"""

    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    files = {Path(relative).stem.upper(): relative for relative in manifest["files"]}
    if len(dataset) != len(split_ids) or set(files) != set(split_ids):
        raise ValueError("official dataset, split IDs and residual cache do not align")

    observed_frames = np.arange(history_frames, dtype=np.int64)
    written = 0
    skipped = 0
    for index, (pdb_id, item) in enumerate(zip(split_ids, dataset), start=1):
        trajectory = item.ligand_trajectory_pos.detach().cpu().numpy()
        if trajectory.ndim != 3 or trajectory.shape[1] < target_stop:
            raise ValueError(f"{pdb_id} has invalid official trajectory shape {trajectory.shape}")
        observed = trajectory[:, :history_frames, :].transpose(1, 0, 2)
        target = trajectory[:, target_start:target_stop, :].transpose(1, 0, 2)
        path = (split_dir / files[pdb_id]).resolve()
        if not path.is_relative_to(split_dir.resolve()):
            raise ValueError(f"cache path escapes split directory: {files[pdb_id]}")
        changed = augment_complex_cache_history(
            path,
            observed,
            observed_frames,
            expected_target_positions=target,
            resume=resume,
            overwrite=overwrite,
        )
        written += int(changed)
        skipped += int(not changed)
        print(
            f"[{index:03d}/{len(split_ids):03d}] {pdb_id} "
            f"{'written' if changed else 'verified'}",
            flush=True,
        )

    manifest["schema_version"] = 2
    manifest["observed_history"] = {
        "frames": observed_frames.tolist(),
        "shape": "[history, ligand_atoms, 3]",
        "source": "official_ground_truth_context_only",
        "target_overlap": False,
    }
    _write_json_atomic(manifest_path, manifest)
    return written, skipped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verify_misato1000(args.dataset_dir, strict_size=True)
    window = ROLLOUT_WINDOWS[args.task]
    if window.history_frames != window.target_start:
        raise ValueError("history augmentation requires context frames to end at target_start")

    summary = []
    for split in args.splits:
        split_ids = read_training_split_ids(args.dataset_dir, split)
        dataset = _load_dataset(args.official_repo, args.dataset_dir, split)
        written, skipped = augment_split(
            dataset,
            split_ids,
            args.cache_root / split,
            history_frames=window.history_frames,
            target_start=window.target_start,
            target_stop=window.target_stop,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        manifest_path = args.cache_root / split / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["history_augmentation_commit"] = _git_commit(REPO_ROOT)
        manifest["history_official_repo_commit"] = _git_commit(args.official_repo)
        _write_json_atomic(manifest_path, manifest)
        summary.append({"split": split, "written": written, "verified": skipped})
    print(json.dumps({"task": args.task, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
