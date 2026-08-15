"""轻量、因果、旋转等变的质量中心时序修正器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class COMPrediction:
    """每帧的质量中心残差均值。"""

    mean: torch.Tensor


def mass_weighted_com(values: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """把 ``[frame, atom, xyz]`` 张量压缩成质量加权 COM。"""

    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("values must have shape [frames, atoms, 3]")
    if masses.shape != (values.shape[1],):
        raise ValueError("masses do not match the atom dimension")
    if not torch.is_floating_point(values) or not torch.is_floating_point(masses):
        raise TypeError("values and masses must be floating point")
    if not torch.isfinite(values).all() or not torch.isfinite(masses).all():
        raise ValueError("values and masses must be finite")
    if (masses <= 0).any():
        raise ValueError("masses must be strictly positive")
    weights = masses / masses.sum()
    return torch.einsum("a,fad->fd", weights, values)


class COMTemporalCorrector(nn.Module):
    """从冻结的 NeuralMD 轨迹预测一个不改变内部构象的刚性平移。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        rbf_channels: int = 8,
        protein_cutoff: float = 12.0,
        history_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or rbf_channels < 1:
            raise ValueError("hidden_dim and rbf_channels must be positive")
        if protein_cutoff <= 0:
            raise ValueError("protein_cutoff must be positive")

        self.hidden_dim = hidden_dim
        self.rbf_channels = rbf_channels
        self.protein_cutoff = protein_cutoff
        self.history_conditioning = history_conditioning
        self.register_buffer("rbf_centers", torch.linspace(0.0, 1.0, rbf_channels))
        self.rbf_gamma = float(rbf_channels**2)

        # RBF 密度 + 速度/加速度范数 + Rg + 最近蛋白距离 + 时间。
        scalar_dim = rbf_channels + 5
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        vector_channels = rbf_channels + 2 + (3 if history_conditioning else 0)
        self.coefficient_head = nn.Linear(hidden_dim, vector_channels)

        # 安全默认值：训练开始前严格退化为原始 NeuralMD（零修正）。
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    @staticmethod
    def _validate_inputs(
        positions: torch.Tensor,
        masses: torch.Tensor,
        protein_ca: torch.Tensor,
    ) -> None:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [frames, atoms, 3]")
        if positions.shape[0] < 1 or positions.shape[1] < 1:
            raise ValueError("trajectory must contain at least one frame and atom")
        if masses.shape != (positions.shape[1],):
            raise ValueError("masses do not match the trajectory")
        if protein_ca.ndim != 2 or protein_ca.shape[-1] != 3 or protein_ca.shape[0] < 1:
            raise ValueError("protein_ca must have shape [residues, 3]")
        for name, value in (
            ("positions", positions),
            ("masses", masses),
            ("protein_ca", protein_ca),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if (masses <= 0).any():
            raise ValueError("masses must be strictly positive")

    def _geometric_features(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        protein_ca: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回旋转不变量标量和同步旋转的向量基。"""

        center = mass_weighted_com(positions, masses)
        relative = protein_ca[None, :, :] - center[:, None, :]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        normalized_distance = distance / self.protein_cutoff
        envelope = (1.0 - normalized_distance).clamp_min(0.0)
        rbf = torch.exp(
            -self.rbf_gamma
            * (normalized_distance[..., None] - self.rbf_centers) ** 2
        ) * envelope[..., None]

        # 每个径向壳层产生一个蛋白指向向量；多壳层比单一蛋白质心更有表达力。
        unit = relative / distance.clamp_min(1e-8)[..., None]
        weight_sum = rbf.sum(dim=1).clamp_min(1e-8)
        pocket_basis = torch.einsum("frk,frd->fkd", rbf, unit) / weight_sum[..., None]

        velocity = torch.zeros_like(center)
        velocity[1:] = center[1:] - center[:-1]
        acceleration = torch.zeros_like(center)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        vector_basis = torch.cat(
            (pocket_basis, velocity[:, None, :], acceleration[:, None, :]), dim=1
        )

        mass_weights = masses / masses.sum()
        ligand_relative = positions - center[:, None, :]
        radius = torch.sqrt(
            torch.einsum(
                "a,fa->f",
                mass_weights,
                ligand_relative.square().sum(dim=-1),
            ).clamp_min(0.0)
        )
        time = torch.linspace(
            0.0,
            1.0,
            positions.shape[0],
            device=positions.device,
            dtype=positions.dtype,
        )
        scalars = torch.cat(
            (
                rbf.mean(dim=1),
                torch.linalg.vector_norm(velocity, dim=-1, keepdim=True),
                torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True),
                (radius / self.protein_cutoff)[:, None],
                (distance.min(dim=1).values / self.protein_cutoff)[:, None],
                time[:, None],
            ),
            dim=-1,
        )
        return scalars, vector_basis

    def forward(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        protein_ca: torch.Tensor,
        *,
        observed_positions: torch.Tensor | None = None,
    ) -> COMPrediction:
        self._validate_inputs(positions, masses, protein_ca)
        scalars, vector_basis = self._geometric_features(positions, masses, protein_ca)
        if self.history_conditioning:
            if observed_positions is None:
                raise ValueError("observed_positions is required for history conditioning")
            self._validate_inputs(observed_positions, masses, protein_ca)
            if observed_positions.shape[1:] != positions.shape[1:]:
                raise ValueError("observed and predicted trajectories must use the same atoms")
            if observed_positions.shape[0] < 3:
                raise ValueError("history conditioning requires at least three observed frames")
            if observed_positions.device != positions.device or observed_positions.dtype != positions.dtype:
                raise ValueError("observed and predicted trajectories must share device and dtype")

            observed_scalars, _ = self._geometric_features(
                observed_positions, masses, protein_ca
            )
            total_frames = observed_positions.shape[0] + positions.shape[0]
            timeline = torch.linspace(
                0.0,
                1.0,
                total_frames,
                device=positions.device,
                dtype=positions.dtype,
            )
            observed_scalars = observed_scalars.clone()
            scalars = scalars.clone()
            observed_scalars[:, -1] = timeline[: observed_positions.shape[0]]
            scalars[:, -1] = timeline[observed_positions.shape[0] :]
            sequence = torch.cat((observed_scalars, scalars), dim=0)
            hidden = self.scalar_encoder(sequence)
            hidden, _ = self.temporal(hidden[None, :, :])
            hidden = hidden[0, observed_positions.shape[0] :]

            observed_center = mass_weighted_com(observed_positions, masses)
            predicted_center = mass_weighted_com(positions, masses)
            history_velocity = observed_center[-1] - observed_center[-2]
            previous_velocity = observed_center[-2] - observed_center[-3]
            history_acceleration = history_velocity - previous_velocity
            first_prediction_jump = predicted_center[0] - observed_center[-1]
            history_basis = torch.stack(
                (history_velocity, history_acceleration, first_prediction_jump), dim=0
            )[None, :, :].expand(positions.shape[0], -1, -1)
            vector_basis = torch.cat((vector_basis, history_basis), dim=1)
        else:
            if observed_positions is not None:
                raise ValueError("observed_positions was provided to a non-history model")
            hidden = self.scalar_encoder(scalars)
            hidden, _ = self.temporal(hidden[None, :, :])
            hidden = hidden[0]

        coefficients = self.coefficient_head(hidden)
        mean = torch.einsum("fk,fkd->fd", coefficients, vector_basis)
        return COMPrediction(mean=mean)


def com_temporal_loss(
    predicted_com: torch.Tensor,
    residual: torch.Tensor,
    masses: torch.Tensor,
    *,
    final_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """训练全程 COM，并单独照顾初赛要求的最终帧。"""

    if final_weight < 0:
        raise ValueError("final_weight must be non-negative")
    target_com = mass_weighted_com(residual, masses)
    if predicted_com.shape != target_com.shape:
        raise ValueError("predicted COM shape does not match the target")
    mean_mse = (predicted_com - target_com).square().mean()
    final_mse = (predicted_com[-1] - target_com[-1]).square().mean()
    return mean_mse + final_weight * final_mse, mean_mse, final_mse
