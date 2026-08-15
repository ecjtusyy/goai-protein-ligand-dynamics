import pytest
import torch

from src.com_temporal_model import (
    COMTemporalCorrector,
    com_temporal_loss,
    mass_weighted_com,
)


def inputs(dtype=torch.float32):
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.3, 0.0]],
            [[0.3, 0.1, 0.0], [1.3, 0.4, 0.1]],
            [[0.6, 0.2, 0.0], [1.6, 0.5, 0.2]],
        ],
        dtype=dtype,
    )
    masses = torch.tensor([12.0, 16.0], dtype=dtype)
    protein = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, -1.0, 1.0], [-1.0, 0.5, 0.0]], dtype=dtype
    )
    return positions, masses, protein


def randomize_head(model: COMTemporalCorrector) -> None:
    torch.manual_seed(7)
    torch.nn.init.normal_(model.coefficient_head.weight, std=0.1)
    torch.nn.init.normal_(model.coefficient_head.bias, std=0.1)


def test_zero_initialized_model_is_exact_neuralmd_fallback() -> None:
    model = COMTemporalCorrector(hidden_dim=8, rbf_channels=4)
    positions, masses, protein = inputs()

    prediction = model(positions, masses, protein)

    torch.testing.assert_close(prediction.mean, torch.zeros_like(prediction.mean))


def test_rotation_equivariance_and_translation_invariance() -> None:
    model = COMTemporalCorrector(hidden_dim=8, rbf_channels=4).eval()
    randomize_head(model)
    positions, masses, protein = inputs(dtype=torch.float64)
    model = model.double()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    shift = torch.tensor([10.0, -4.0, 2.0], dtype=torch.float64)

    original = model(positions, masses, protein).mean
    transformed = model(
        positions @ rotation.T + shift,
        masses,
        protein @ rotation.T + shift,
    ).mean

    torch.testing.assert_close(transformed, original @ rotation.T, rtol=1e-6, atol=1e-7)


def test_temporal_model_is_causal() -> None:
    model = COMTemporalCorrector(hidden_dim=8, rbf_channels=4).eval()
    randomize_head(model)
    positions, masses, protein = inputs()
    changed_future = positions.clone()
    changed_future[3] += torch.tensor([5.0, -2.0, 1.0])

    original = model(positions, masses, protein).mean
    changed = model(changed_future, masses, protein).mean

    torch.testing.assert_close(changed[:3], original[:3])
    assert not torch.allclose(changed[3], original[3])


def test_mass_com_and_loss_contract() -> None:
    positions, masses, _ = inputs()
    drift = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    local = torch.tensor([[[0.4, 0.0, 0.0], [-0.3, 0.0, 0.0]]]).repeat(4, 1, 1)
    residual = drift[:, None, :] + local

    target = mass_weighted_com(residual, masses)
    total, mean_mse, final_mse = com_temporal_loss(target, residual, masses)

    torch.testing.assert_close(target, drift)
    torch.testing.assert_close(total, torch.tensor(0.0))
    torch.testing.assert_close(mean_mse, torch.tensor(0.0))
    torch.testing.assert_close(final_mse, torch.tensor(0.0))


def test_small_trajectory_can_be_overfit() -> None:
    """证明零初始化保险不会阻断时序编码器后续学习。"""

    torch.manual_seed(0)
    frames = 12
    steps = torch.arange(frames, dtype=torch.float32)
    positions = torch.stack(
        (
            torch.stack((0.1 * steps, torch.zeros(frames), torch.zeros(frames)), dim=-1),
            torch.stack(
                (1.0 + 0.1 * steps, torch.full((frames,), 0.2), torch.zeros(frames)),
                dim=-1,
            ),
        ),
        dim=1,
    )
    masses = torch.tensor([12.0, 16.0])
    protein = torch.tensor([[0.0, 1.0, 2.0], [2.0, -1.0, 1.0]])
    drift = torch.stack((0.05 * steps, 0.02 * steps, torch.zeros(frames)), dim=-1)
    residual = drift[:, None, :].expand_as(positions)
    model = COMTemporalCorrector(hidden_dim=16, rbf_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []

    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(positions, masses, protein)
        loss, _, _ = com_temporal_loss(prediction.mean, residual, masses)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.1 * losses[0]


def observed_history(dtype=torch.float32):
    positions, _, _ = inputs(dtype=dtype)
    first = positions[0]
    return torch.stack(
        (
            first - torch.tensor([0.3, 0.1, 0.0], dtype=dtype),
            first - torch.tensor([0.2, 0.05, 0.0], dtype=dtype),
            first - torch.tensor([0.1, 0.02, 0.0], dtype=dtype),
        )
    )


def test_history_conditioning_is_equivariant_and_uses_observed_path() -> None:
    model = COMTemporalCorrector(
        hidden_dim=8,
        rbf_channels=4,
        history_conditioning=True,
    ).double().eval()
    randomize_head(model)
    positions, masses, protein = inputs(dtype=torch.float64)
    observed = observed_history(dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    shift = torch.tensor([3.0, -4.0, 2.0], dtype=torch.float64)

    original = model(
        positions, masses, protein, observed_positions=observed
    ).mean
    transformed = model(
        positions @ rotation.T + shift,
        masses,
        protein @ rotation.T + shift,
        observed_positions=observed @ rotation.T + shift,
    ).mean
    altered = observed.clone()
    altered[0] += torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64)
    altered_prediction = model(
        positions, masses, protein, observed_positions=altered
    ).mean

    torch.testing.assert_close(transformed, original @ rotation.T, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(altered_prediction, original)


def test_history_model_requires_three_observed_frames() -> None:
    model = COMTemporalCorrector(
        hidden_dim=8,
        rbf_channels=4,
        history_conditioning=True,
    )
    positions, masses, protein = inputs()

    with pytest.raises(ValueError, match="required"):
        model(positions, masses, protein)

    with pytest.raises(ValueError, match="at least three"):
        model(
            positions,
            masses,
            protein,
            observed_positions=observed_history()[:2],
        )
