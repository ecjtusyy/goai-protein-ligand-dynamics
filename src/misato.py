from pathlib import Path

import h5py
import numpy as np


def load_ligand_trajectory(
    path: str | Path,
    pdb_id: str | None = None,
    heavy_only: bool = True,
) -> tuple[str, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        ids = sorted(h5.keys())
        if not ids:
            raise ValueError("MD.hdf5 中没有复合物")

        pdb_id = (pdb_id or ids[0]).upper()
        if pdb_id not in h5:
            raise KeyError(f"找不到复合物 {pdb_id}")

        group = h5[pdb_id]
        ligand_begin = int(group["molecules_begin_atom_index"][-1])
        atom_numbers = group["atoms_number"][ligand_begin:]
        coordinates = group["trajectory_coordinates"][:, ligand_begin:, :]

    if heavy_only:
        mask = atom_numbers != 1
        atom_numbers = atom_numbers[mask]
        coordinates = coordinates[:, mask]

    coordinates = np.asarray(coordinates, dtype=np.float32)
    atom_numbers = np.asarray(atom_numbers, dtype=np.int64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"轨迹形状异常: {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("轨迹包含 NaN 或 Inf")

    return pdb_id, coordinates, atom_numbers
