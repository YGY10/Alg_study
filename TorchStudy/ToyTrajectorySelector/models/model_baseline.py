from __future__ import annotations
import torch
import torch.nn as nn
from anchors import NUM_ANCHORS, NUM_POINTS


class BaselineTrajectorySelector(nn.Module):
    def __init__(
        self,
        num_anchors: int = NUM_ANCHORS,
        num_points: int = NUM_POINTS,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_points = num_points
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 8 * 8, hidden_dim),
            nn.ReLU(),
        )

        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_points * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_grid: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one score for each candidate trajectory.

        Args:
            input_grid: [B, 2, 64, 64]
            anchors: [B, N, T, 2]

        Returns:
            scores: [B, N]
        """
        batch_size, num_anchors, num_points, _ = anchors.shape
        scene_feature: torch.Tensor = self.scene_encoder(input_grid)
        scene_feature = scene_feature.unsqueeze(1).expand(-1, num_anchors, -1)

        flat_anchors = anchors.reshape(batch_size, num_anchors, num_points * 2)
        anchor_feature = self.anchor_encoder(flat_anchors)

        fused_feature = torch.cat([scene_feature, anchor_feature], dim=-1)
        scores = self.scorer(fused_feature).squeeze(-1)
        return scores


if __name__ == "__main__":
    input_grid = torch.randn(4, 2, 64, 64)
    anchors = torch.randn(4, NUM_ANCHORS, NUM_POINTS, 2)

    model = BaselineTrajectorySelector()
    scores = model(input_grid, anchors)

    print("scores shape:", scores.shape)
