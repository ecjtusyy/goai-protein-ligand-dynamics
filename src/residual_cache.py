"""冻结 NeuralMD 后供概率残差模型使用的缓存合同。"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .probabilistic_residual import residual_target


TRAINING_SPLITS = ("train", "val")
_SAFE_PDB_ID = re.compile(r"^[A-Z0-9]+$")
CACHE_ARRAYS = (
    "neuralmd_positions",
    "true_positions",
    "residual",
    "target_frames",
    "ligand_atom_types",
    "ligand_masses",
    "protein_n_positions",
    "protein_ca_positions",
    "protein_c_positions",
    "protein_residue_types",
)
HISTORY_CACHE_ARRAYS = (
    "observed_positions",
    "observed_frames",
)


def read_training_split_ids(dataset_dir: str | Path, split: str) -> list[str]:
    """只读取 train/val；test 永远不进入残差模型训练缓存。"""

    if split not in TRAINING_SPLITS:
        raise ValueError(f"split must be one of {TRAINING_SPLITS}, got {split!r}")
    path = Path(dataset_dir) / "raw" / f"{split}_MD.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")

    identifiers = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    if not identifiers:
        raise ValueError(f"split file is empty: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"split file contains duplicate IDs: {path}")
    return identifiers


def _numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def build_cache_payload(
    batch,
    neuralmd_positions,
    true_positions,
    target_frames,
) -> dict[str, np.ndarray]:
    """构造一个复合物的可移植 NumPy 缓存，不保存任何模型对象。"""

    prediction = _numpy(neuralmd_positions, dtype=np.float32)
    target = _numpy(true_positions, dtype=np.float32)
    residual = residual_target(prediction, target).astype(np.float32, copy=False)
    frames = _numpy(target_frames, dtype=np.int64)
    if frames.shape != (prediction.shape[0],):
        raise ValueError(f"target_frames must have shape {(prediction.shape[0],)}, got {frames.shape}")

    payload = {
        "neuralmd_positions": prediction,
        "true_positions": target,
        "residual": residual,
        "target_frames": frames,
        "ligand_atom_types": _numpy(batch.ligand_x, dtype=np.int64),
        "ligand_masses": _numpy(batch.ligand_mass, dtype=np.float32),
        "protein_n_positions": _numpy(batch.protein_pos[batch.mask_n], dtype=np.float32),
        "protein_ca_positions": _numpy(batch.protein_pos[batch.mask_ca], dtype=np.float32),
        "protein_c_positions": _numpy(batch.protein_pos[batch.mask_c], dtype=np.float32),
        "protein_residue_types": _numpy(batch.protein_backbone_residue, dtype=np.int64),
    }
    ligand_atoms = prediction.shape[1]
    if payload["ligand_atom_types"].shape != (ligand_atoms,):
        raise ValueError("ligand atom features do not match trajectory atom count")
    if payload["ligand_masses"].shape != (ligand_atoms,):
        raise ValueError("ligand masses do not match trajectory atom count")

    residues = payload["protein_residue_types"].shape[0]
    for key in ("protein_n_positions", "protein_ca_positions", "protein_c_positions"):
        if payload[key].shape != (residues, 3):
            raise ValueError(f"{key} does not match protein residue count")
    return payload


def write_complex_cache(
    output_dir: str | Path,
    pdb_id: str,
    payload: dict[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    """以原子替换方式写入单复合物压缩缓存；默认拒绝覆盖。"""

    pdb_id = pdb_id.upper()
    if not _SAFE_PDB_ID.fullmatch(pdb_id):
        raise ValueError(f"unsafe PDB ID: {pdb_id!r}")
    directory = Path(output_dir) / "complexes"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{pdb_id}.npz"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"cache already exists: {destination}")

    temporary = directory / f".{pdb_id}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)
    return destination


def validate_complex_cache(path: str | Path, expected_frames) -> None:
    """校验可恢复缓存，防止把半写入或旧合同文件当作已完成。"""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in CACHE_ARRAYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing cache arrays: {missing}")
        prediction = archive["neuralmd_positions"]
        target = archive["true_positions"]
        residual = archive["residual"]
        frames = archive["target_frames"]
        if prediction.shape != target.shape or residual.shape != target.shape:
            raise ValueError(f"{path} contains inconsistent trajectory shapes")
        if not np.array_equal(frames, np.asarray(expected_frames, dtype=np.int64)):
            raise ValueError(f"{path} target frames do not match the requested rollout")
        np.testing.assert_allclose(residual, target - prediction, rtol=1e-5, atol=1e-6)


def validate_observed_history(
    observed_positions,
    observed_frames,
    *,
    ligand_atoms: int,
    first_target_frame: int,
) -> None:
    """历史必须完整位于监督窗口之前，防止把未来真值泄漏给模型。"""

    positions = np.asarray(observed_positions)
    frames = np.asarray(observed_frames)
    if positions.ndim != 3 or positions.shape[1:] != (ligand_atoms, 3):
        raise ValueError(
            "observed_positions must have shape "
            f"[history, {ligand_atoms}, 3], got {positions.shape}"
        )
    if not np.issubdtype(positions.dtype, np.floating):
        raise TypeError("observed_positions must be floating point")
    if not np.isfinite(positions).all():
        raise ValueError("observed_positions contains NaN or Inf")
    if frames.shape != (positions.shape[0],) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("observed_frames must be one integer per history frame")
    if not np.array_equal(frames, np.arange(positions.shape[0], dtype=frames.dtype)):
        raise ValueError("observed_frames must be contiguous and start at zero")
    if positions.shape[0] < 2:
        raise ValueError("observed history needs at least two frames")
    if int(frames[-1]) >= first_target_frame:
        raise ValueError("observed history overlaps the target window")


def augment_complex_cache_history(
    path: str | Path,
    observed_positions,
    observed_frames,
    *,
    expected_target_positions=None,
    resume: bool = False,
    overwrite: bool = False,
) -> bool:
    """原子替换一个 NPZ；返回 True 表示本次确实写入了历史。"""

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in CACHE_ARRAYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing cache arrays: {missing}")
        payload = {key: archive[key].copy() for key in archive.files}

    target = payload["true_positions"]
    target_frames = payload["target_frames"]
    validate_observed_history(
        observed_positions,
        observed_frames,
        ligand_atoms=target.shape[1],
        first_target_frame=int(target_frames[0]),
    )
    observed_positions = np.asarray(observed_positions, dtype=np.float32)
    observed_frames = np.asarray(observed_frames, dtype=np.int64)

    if expected_target_positions is not None:
        np.testing.assert_allclose(
            target,
            np.asarray(expected_target_positions, dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
            err_msg=f"{path} does not match the official trajectory order",
        )

    existing = [key for key in HISTORY_CACHE_ARRAYS if key in payload]
    if existing and len(existing) != len(HISTORY_CACHE_ARRAYS):
        raise ValueError(f"{path} contains a partial observed-history contract")
    if len(existing) == len(HISTORY_CACHE_ARRAYS) and not overwrite:
        validate_observed_history(
            payload["observed_positions"],
            payload["observed_frames"],
            ligand_atoms=target.shape[1],
            first_target_frame=int(target_frames[0]),
        )
        if not resume:
            raise FileExistsError(f"observed history already exists: {path}")
        np.testing.assert_allclose(
            payload["observed_positions"], observed_positions, rtol=0.0, atol=1e-6
        )
        np.testing.assert_array_equal(payload["observed_frames"], observed_frames)
        return False

    payload["observed_positions"] = observed_positions
    payload["observed_frames"] = observed_frames
    temporary = path.with_name(f".{path.stem}.history.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    return True
