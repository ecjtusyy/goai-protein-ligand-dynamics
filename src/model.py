import torch
from torch import nn


class VelocityMLP(nn.Module):
    def __init__(self, velocity_scale: float, hidden_size: int = 64) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(119, 8)
        self.network = nn.Sequential(
            nn.Linear(14, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3),
        )
        self.register_buffer("velocity_scale", torch.tensor(float(velocity_scale)))

    def predict_velocity(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        atom_numbers: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.velocity_scale.clamp_min(1e-6)
        features = torch.cat(
            [
                current / scale,
                (current - previous) / scale,
                self.atom_embedding(atom_numbers),
            ],
            dim=-1,
        )
        return current + scale * self.network(features)

    def rollout(
        self,
        history: torch.Tensor,
        atom_numbers: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        if len(history) < 3:
            raise ValueError("MLP rollout 至少需要 3 帧历史轨迹")

        previous_velocity = history[-2] - history[-3]
        current_velocity = history[-1] - history[-2]
        position = history[-1]
        predictions = []

        for _ in range(horizon):
            next_velocity = self.predict_velocity(
                previous_velocity,
                current_velocity,
                atom_numbers,
            )
            position = position + next_velocity
            predictions.append(position)
            previous_velocity, current_velocity = current_velocity, next_velocity

        return torch.stack(predictions)
