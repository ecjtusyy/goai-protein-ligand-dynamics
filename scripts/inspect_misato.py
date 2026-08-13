import argparse
from pathlib import Path

from src.misato import load_ligand_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/MISATO_100/raw/MD.hdf5"),
    )
    parser.add_argument("--pdb-id")
    args = parser.parse_args()

    pdb_id, trajectory, atom_numbers = load_ligand_trajectory(
        args.data,
        args.pdb_id,
    )
    print(f"PDB ID: {pdb_id}")
    print(f"轨迹形状: {trajectory.shape}")
    print(f"帧数: {trajectory.shape[0]}")
    print(f"配体重原子数: {trajectory.shape[1]}")
    print(f"原子序数: {atom_numbers.tolist()}")


if __name__ == "__main__":
    main()
