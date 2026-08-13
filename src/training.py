from pathlib import Path

import numpy as np
import torch

from .misato import load_ligand_trajectory


def read_split(path: str | Path) -> list[str]:
    return [line.strip().upper() for line in Path(path).read_text().splitlines() if line.strip()]


def load_split(
    h5_path: str | Path,
    split_path: str | Path,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    trajectories = []
    for pdb_id in read_split(split_path):
        _, coordinates, atom_numbers = load_ligand_trajectory(h5_path, pdb_id)
        trajectories.append(
            (
                pdb_id,
                torch.from_numpy(coordinates),
                torch.from_numpy(atom_numbers),
            )
        )
    return trajectories


def velocity_examples(
    trajectories: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    previous_list = []
    current_list = []
    target_list = []
    atom_list = []

    for _, positions, atom_numbers in trajectories:
        velocity = positions[1:] - positions[:-1]
        time_steps = len(velocity) - 2
        atoms = len(atom_numbers)
        previous_list.append(velocity[:-2].reshape(-1, 3))
        current_list.append(velocity[1:-1].reshape(-1, 3))
        target_list.append(velocity[2:].reshape(-1, 3))
        atom_list.append(atom_numbers.repeat(time_steps))

        if previous_list[-1].shape[0] != time_steps * atoms:
            raise ValueError("速度样本数量异常")

    return tuple(
        torch.cat(values)
        for values in (previous_list, current_list, target_list, atom_list)
    )


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
