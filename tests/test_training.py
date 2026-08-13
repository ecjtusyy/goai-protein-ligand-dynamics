import torch

from src.training import velocity_examples


def test_velocity_examples_keep_atom_and_time_alignment():
    time = torch.arange(6, dtype=torch.float32)[:, None, None]
    velocity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    positions = time * velocity
    trajectories = [("1ABC", positions, torch.tensor([6, 8]))]

    previous, current, target, atoms = velocity_examples(trajectories)

    assert previous.shape == current.shape == target.shape == (6, 3)
    assert atoms.tolist() == [6, 8, 6, 8, 6, 8]
    torch.testing.assert_close(previous, current)
    torch.testing.assert_close(current, target)
