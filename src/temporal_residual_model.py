"""无需 PyG 扩展的几何 GNN + GRU 概率残差模型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ResidualPrediction:
    """每帧、每个配体原子的残差分布参数。"""

    mean: torch.Tensor
    scale: torch.Tensor | None


class GaussianRBF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, 1.0, channels)
        self.register_buffer("centers", centers)
        self.gamma = float(channels**2)

    def forward(self, normalized_distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (normalized_distance[..., None] - self.centers) ** 2)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TemporalProbabilisticResidual(nn.Module):
    """在完整 NeuralMD 轨迹上预测等变残差均值和各向同性尺度。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        rbf_channels: int = 16,
        ligand_cutoff: float = 6.0,
        protein_cutoff: float = 8.0,
        atom_vocab: int = 118,
        residue_vocab: int = 32,
        temporal: bool = True,
        probabilistic: bool = True,
        detach_uncertainty_features: bool = True,
        frame_chunk_size: int = 16,
        min_scale: float = 1e-3,
        initial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or rbf_channels < 1 or frame_chunk_size < 1:
            raise ValueError("hidden_dim, rbf_channels and frame_chunk_size must be positive")
        if ligand_cutoff <= 0 or protein_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        if not 0 < min_scale < initial_scale:
            raise ValueError("require 0 < min_scale < initial_scale")

        self.hidden_dim = hidden_dim
        self.ligand_cutoff = ligand_cutoff
        self.protein_cutoff = protein_cutoff
        self.temporal = temporal
        self.probabilistic = probabilistic
        self.detach_uncertainty_features = detach_uncertainty_features
        self.frame_chunk_size = frame_chunk_size
        self.min_scale = min_scale

        self.atom_embedding = nn.Embedding(atom_vocab, hidden_dim)
        self.mass_embedding = _mlp(1, hidden_dim, hidden_dim)
        self.residue_embedding = nn.Embedding(residue_vocab, hidden_dim)
        self.rbf = GaussianRBF(rbf_channels)

        self.ligand_message = _mlp(2 * hidden_dim + rbf_channels, hidden_dim, hidden_dim)
        self.protein_message = _mlp(2 * hidden_dim + rbf_channels, hidden_dim, hidden_dim)
        self.ligand_vector_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.protein_vector_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.spatial_update = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True) if temporal else None
        self.vector_head = nn.Linear(hidden_dim, 3)

        self.scale_head = nn.Linear(hidden_dim, 1) if probabilistic else None
        if self.scale_head is not None:
            raw_initial = torch.log(torch.expm1(torch.tensor(initial_scale - min_scale)))
            nn.init.zeros_(self.scale_head.weight)
            nn.init.constant_(self.scale_head.bias, float(raw_initial))

    @staticmethod
    def _validate_inputs(
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        masses: torch.Tensor,
        protein_ca: torch.Tensor,
        residue_types: torch.Tensor,
    ) -> None:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [frames, atoms, 3]")
        frames, atoms, _ = positions.shape
        if frames < 1 or atoms < 1:
            raise ValueError("trajectory must contain at least one frame and atom")
        if atom_types.shape != (atoms,) or masses.shape != (atoms,):
            raise ValueError("atom features do not match trajectory atom count")
        if protein_ca.ndim != 2 or protein_ca.shape[-1] != 3:
            raise ValueError("protein_ca must have shape [residues, 3]")
        if residue_types.shape != (protein_ca.shape[0],):
            raise ValueError("protein residue features do not match CA positions")
        if not torch.is_floating_point(positions) or not torch.is_floating_point(masses):
            raise TypeError("positions and masses must be floating point")
        if not torch.isfinite(positions).all() or not torch.isfinite(masses).all():
            raise ValueError("positions and masses must be finite")
        if (masses <= 0).any():
            raise ValueError("masses must be strictly positive")

    def _spatial_chunk(
        self,
        positions: torch.Tensor,
        atom_features: torch.Tensor,
        protein_ca: torch.Tensor,
        residue_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames, atoms, _ = positions.shape
        residues = protein_ca.shape[0]

        ligand_relative = positions[:, None, :, :] - positions[:, :, None, :]
        ligand_distance = torch.linalg.vector_norm(ligand_relative, dim=-1)
        ligand_mask = (ligand_distance > 0) & (ligand_distance < self.ligand_cutoff)
        ligand_rbf = self.rbf(ligand_distance / self.ligand_cutoff)
        source = atom_features[None, :, None, :].expand(frames, atoms, atoms, -1)
        neighbor = atom_features[None, None, :, :].expand(frames, atoms, atoms, -1)
        ligand_message = self.ligand_message(torch.cat((source, neighbor, ligand_rbf), dim=-1))
        ligand_message = ligand_message * ligand_mask[..., None]
        ligand_count = ligand_mask.sum(dim=2, keepdim=True).clamp_min(1)
        ligand_aggregate = ligand_message.sum(dim=2) / ligand_count
        ligand_weight = self.ligand_vector_weight(ligand_message).squeeze(-1)
        ligand_vector = (ligand_weight[..., None] * ligand_relative).sum(dim=2) / ligand_count

        protein_relative = protein_ca[None, None, :, :] - positions[:, :, None, :]
        protein_distance = torch.linalg.vector_norm(protein_relative, dim=-1)
        protein_mask = protein_distance < self.protein_cutoff
        protein_rbf = self.rbf(protein_distance / self.protein_cutoff)
        ligand = atom_features[None, :, None, :].expand(frames, atoms, residues, -1)
        residue = residue_features[None, None, :, :].expand(frames, atoms, residues, -1)
        protein_message = self.protein_message(torch.cat((ligand, residue, protein_rbf), dim=-1))
        protein_message = protein_message * protein_mask[..., None]
        protein_count = protein_mask.sum(dim=2, keepdim=True).clamp_min(1)
        protein_aggregate = protein_message.sum(dim=2) / protein_count
        protein_weight = self.protein_vector_weight(protein_message).squeeze(-1)
        protein_vector = (protein_weight[..., None] * protein_relative).sum(dim=2) / protein_count

        repeated_atom_features = atom_features[None, :, :].expand(frames, atoms, -1)
        hidden = self.spatial_update(
            torch.cat((repeated_atom_features, ligand_aggregate, protein_aggregate), dim=-1)
        )
        return hidden, ligand_vector, protein_vector

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        masses: torch.Tensor,
        protein_ca: torch.Tensor,
        residue_types: torch.Tensor,
    ) -> ResidualPrediction:
        self._validate_inputs(positions, atom_types, masses, protein_ca, residue_types)
        atom_features = self.atom_embedding(atom_types) + self.mass_embedding(torch.log1p(masses[:, None]))
        residue_features = self.residue_embedding(residue_types)

        hidden_chunks = []
        ligand_vector_chunks = []
        protein_vector_chunks = []
        for start in range(0, positions.shape[0], self.frame_chunk_size):
            hidden, ligand_vector, protein_vector = self._spatial_chunk(
                positions[start : start + self.frame_chunk_size],
                atom_features,
                protein_ca,
                residue_features,
            )
            hidden_chunks.append(hidden)
            ligand_vector_chunks.append(ligand_vector)
            protein_vector_chunks.append(protein_vector)

        hidden = torch.cat(hidden_chunks)
        if self.gru is not None:
            hidden, _ = self.gru(hidden.transpose(0, 1))
            hidden = hidden.transpose(0, 1)

        velocity = torch.zeros_like(positions)
        velocity[1:] = positions[1:] - positions[:-1]
        basis = torch.stack(
            (torch.cat(ligand_vector_chunks), torch.cat(protein_vector_chunks), velocity), dim=-2
        )
        coefficients = torch.tanh(self.vector_head(hidden))
        mean = (coefficients[..., None] * basis).sum(dim=-2)

        scale = None
        if self.scale_head is not None:
            scale_features = hidden.detach() if self.detach_uncertainty_features else hidden
            scale = F.softplus(self.scale_head(scale_features).squeeze(-1)) + self.min_scale
        return ResidualPrediction(mean=mean, scale=scale)
