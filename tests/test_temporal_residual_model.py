import torch

from src.temporal_residual_model import TemporalProbabilisticResidual


def example_inputs():
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 1.1, 0.1]],
            [[0.2, 0.1, 0.0], [1.2, 0.1, 0.1], [0.1, 1.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    return (
        positions,
        torch.tensor([5, 7, 6]),
        torch.tensor([12.0, 16.0, 14.0]),
        torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([1, 12]),
    )


def test_output_contract_and_positive_isotropic_scale() -> None:
    model = TemporalProbabilisticResidual(hidden_dim=16, rbf_channels=8)
    prediction = model(*example_inputs())

    assert prediction.mean.shape == (3, 3, 3)
    assert prediction.scale is not None
    assert prediction.scale.shape == (3, 3)
    assert torch.all(prediction.scale > 0)


def test_mean_is_rotation_equivariant_and_scale_is_invariant() -> None:
    torch.manual_seed(7)
    model = TemporalProbabilisticResidual(hidden_dim=16, rbf_channels=8).eval()
    inputs = example_inputs()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotated = (inputs[0] @ rotation.T, inputs[1], inputs[2], inputs[3] @ rotation.T, inputs[4])

    original_prediction = model(*inputs)
    rotated_prediction = model(*rotated)

    torch.testing.assert_close(rotated_prediction.mean, original_prediction.mean @ rotation.T)
    torch.testing.assert_close(rotated_prediction.scale, original_prediction.scale)


def test_gru_is_causal_across_trajectory_frames() -> None:
    torch.manual_seed(11)
    model = TemporalProbabilisticResidual(hidden_dim=16, rbf_channels=8).eval()
    inputs = list(example_inputs())
    original = model(*inputs)
    inputs[0] = inputs[0].clone()
    inputs[0][-1, 0] += torch.tensor([3.0, -2.0, 1.0])

    changed = model(*inputs)

    torch.testing.assert_close(changed.mean[:-1], original.mean[:-1])
    torch.testing.assert_close(changed.scale[:-1], original.scale[:-1])


def test_distribution_is_invariant_to_global_translation() -> None:
    torch.manual_seed(13)
    model = TemporalProbabilisticResidual(hidden_dim=16, rbf_channels=8).eval()
    inputs = example_inputs()
    translation = torch.tensor([10.0, -4.0, 2.5])
    translated = (inputs[0] + translation, inputs[1], inputs[2], inputs[3] + translation, inputs[4])

    original_prediction = model(*inputs)
    translated_prediction = model(*translated)

    torch.testing.assert_close(translated_prediction.mean, original_prediction.mean)
    torch.testing.assert_close(translated_prediction.scale, original_prediction.scale)


def test_deterministic_ablation_omits_scale_head() -> None:
    model = TemporalProbabilisticResidual(
        hidden_dim=16,
        rbf_channels=8,
        temporal=False,
        probabilistic=False,
    )

    prediction = model(*example_inputs())

    assert prediction.mean.shape == (3, 3, 3)
    assert prediction.scale is None


def test_uncertainty_gradient_is_isolated_from_shared_mean_features() -> None:
    model = TemporalProbabilisticResidual(hidden_dim=16, rbf_channels=8)
    assert model.scale_head is not None
    torch.nn.init.constant_(model.scale_head.weight, 0.2)

    prediction = model(*example_inputs())
    assert prediction.scale is not None
    prediction.scale.sum().backward()

    assert model.scale_head.weight.grad is not None
    assert model.atom_embedding.weight.grad is None
    assert model.vector_head.weight.grad is None


def test_frame_chunking_preserves_model_outputs() -> None:
    torch.manual_seed(17)
    framewise = TemporalProbabilisticResidual(
        hidden_dim=16,
        rbf_channels=8,
        frame_chunk_size=1,
    ).eval()
    chunked = TemporalProbabilisticResidual(
        hidden_dim=16,
        rbf_channels=8,
        frame_chunk_size=2,
    ).eval()
    chunked.load_state_dict(framewise.state_dict())

    framewise_prediction = framewise(*example_inputs())
    chunked_prediction = chunked(*example_inputs())

    torch.testing.assert_close(chunked_prediction.mean, framewise_prediction.mean)
    torch.testing.assert_close(chunked_prediction.scale, framewise_prediction.scale)


def test_message_mlps_only_receive_edges_inside_cutoff() -> None:
    model = TemporalProbabilisticResidual(
        hidden_dim=16,
        rbf_channels=8,
        temporal=False,
        frame_chunk_size=8,
    ).eval()
    positions, atom_types, masses, _, _ = example_inputs()
    protein_ca = torch.tensor(
        [[0.0, 0.0, 2.0], [100.0, 100.0, 100.0]],
        dtype=torch.float32,
    )
    residue_types = torch.tensor([1, 12])
    rows = {}

    def record_rows(name):
        def hook(_module, inputs):
            rows[name] = inputs[0].shape[0]

        return hook

    ligand_hook = model.ligand_message.register_forward_pre_hook(record_rows("ligand"))
    protein_hook = model.protein_message.register_forward_pre_hook(record_rows("protein"))
    try:
        model(positions, atom_types, masses, protein_ca, residue_types)
    finally:
        ligand_hook.remove()
        protein_hook.remove()

    ligand_relative = positions[:, None, :, :] - positions[:, :, None, :]
    ligand_distance = torch.linalg.vector_norm(ligand_relative, dim=-1)
    expected_ligand = ((ligand_distance > 0) & (ligand_distance < 6.0)).sum().item()
    protein_relative = protein_ca[None, None, :, :] - positions[:, :, None, :]
    protein_distance = torch.linalg.vector_norm(protein_relative, dim=-1)
    expected_protein = (protein_distance < 8.0).sum().item()

    assert rows == {"ligand": expected_ligand, "protein": expected_protein}
    assert rows["protein"] < positions.shape[0] * positions.shape[1] * protein_ca.shape[0]
