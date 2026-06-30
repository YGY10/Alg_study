from __future__ import annotations
import torch
from torch import nn


class TrajectoryDecoder(nn.Module):
    """
    VectorNet trajectory decoder
    输入：
        target feature:[B, D]
        目标agent对应的global graph feature
    输出：
        trajectory:[B, T, 2]
        目标agent的未来轨迹点
    """

    def __init__(self, hidden_dim: int = 64, future_steps: int = 30) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, target_feature: torch.Tensor) -> torch.Tensor:
        batch_size = target_feature.shape[0]
        trajectory = self.mlp(target_feature)
        trajectory = trajectory.reshape(batch_size, self.future_steps, 2)
        return trajectory
