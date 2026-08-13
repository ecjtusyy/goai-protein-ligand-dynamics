import numpy as np
import torch


COVALENT_RADII = {
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    34: 1.20,
    35: 1.20,
    53: 1.39,
}


def infer_bonds(
    coordinates: np.ndarray,
    atom_numbers: np.ndarray,
    tolerance: float = 1.25,
) -> np.ndarray:
    radii = np.array([COVALENT_RADII.get(int(z), 0.77) for z in atom_numbers])
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    cutoff = tolerance * (radii[:, None] + radii[None])
    bonded = (distance > 0.4) & (distance < cutoff)
    return np.stack(np.where(np.triu(bonded, k=1))).astype(np.int64)


def bond_length_error(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    edges: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    if edges.numel() == 0:
        return prediction.new_zeros(())

    source, target = edges
    predicted_length = torch.linalg.vector_norm(
        prediction[..., source, :] - prediction[..., target, :],
        dim=-1,
    )
    reference_length = torch.linalg.vector_norm(
        reference[source] - reference[target],
        dim=-1,
    )
    error = predicted_length - reference_length
    return error.square().mean() if squared else error.abs().mean()
