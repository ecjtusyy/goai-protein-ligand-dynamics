import h5py
import numpy as np
import pytest

from src.misato import load_ligand_trajectory


@pytest.fixture
def tiny_misato(tmp_path):
    path = tmp_path / "MD.hdf5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("1ABC")
        group.create_dataset("molecules_begin_atom_index", data=[0, 2])
        group.create_dataset("atoms_number", data=[6, 7, 6, 1, 8])
        group.create_dataset(
            "trajectory_coordinates",
            data=np.arange(60, dtype=float).reshape(4, 5, 3),
        )
    return path


def test_loads_ligand_heavy_atom_trajectory(tiny_misato):
    pdb_id, trajectory, atom_numbers = load_ligand_trajectory(tiny_misato)

    assert pdb_id == "1ABC"
    assert trajectory.shape == (4, 2, 3)
    assert atom_numbers.tolist() == [6, 8]
    assert trajectory.dtype == np.float32


def test_rejects_unknown_complex(tiny_misato):
    with pytest.raises(KeyError, match="找不到复合物"):
        load_ligand_trajectory(tiny_misato, "9XYZ")
