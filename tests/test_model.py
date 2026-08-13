import numpy as np
import torch

from src.geometry import bond_length_error, infer_bonds
from src.model import VelocityMLP


def test_zero_residual_model_reduces_to_linear_rollout():
    model = VelocityMLP(velocity_scale=1.0, hidden_size=8)
    for parameter in model.network.parameters():
        torch.nn.init.zeros_(parameter)

    history = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    prediction = model.rollout(history, torch.tensor([6]), horizon=2)

    torch.testing.assert_close(
        prediction,
        torch.tensor([[[3.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]),
    )


def test_infers_bond_and_measures_length_error():
    coordinates = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
    edges = infer_bonds(coordinates, np.array([6, 6, 8]))

    np.testing.assert_array_equal(edges, np.array([[0], [1]]))
    prediction = torch.tensor([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0], [5.0, 0.0, 0.0]])
    error = bond_length_error(
        prediction,
        torch.tensor(coordinates, dtype=torch.float32),
        torch.tensor(edges),
    )
    torch.testing.assert_close(error, torch.tensor(0.2))
