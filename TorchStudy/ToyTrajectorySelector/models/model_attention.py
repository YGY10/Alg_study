from __future__ import annotations

import torch
from torch import nn

from anchors import NUM_ANCHORS, NUM_POINTS


class AttentionTrajectorySelector(nn.Module):
    def __init__(
        self,
        num_anchors: int = NUM_ANCHORS,
        num_points: int = NUM_POINTS,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.num_anchors = num_anchors
        self.num_points = num_points
        self.hidden_dim = hidden_dim

        self.scene_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.position_embedding = nn.Parameter(torch.randn(1, 8 * 8, hidden_dim) * 0.02)

        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_points * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_grid: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_anchors, num_points, _ = anchors.shape

        scene_map = self.scene_encoder(input_grid)
        spatial_tokens = scene_map.flatten(2).transpose(1, 2)
        spatial_tokens = spatial_tokens + self.position_embedding

        flat_anchors = anchors.reshape(batch_size, num_anchors, num_points * 2)
        anchor_queries = self.anchor_encoder(flat_anchors)

        attended_feature, _ = self.cross_attention(
            query=anchor_queries,
            key=spatial_tokens,
            value=spatial_tokens,
        )

        anchor_feature = self.norm1(anchor_queries + attended_feature)
        anchor_feature = self.norm2(anchor_feature + self.ffn(anchor_feature))

        scores = self.scorer(anchor_feature).squeeze(-1)

        return scores


if __name__ == "__main__":
    input_grid = torch.randn(4, 2, 64, 64)
    anchors = torch.randn(4, NUM_ANCHORS, NUM_POINTS, 2)

    model = AttentionTrajectorySelector()
    scores = model(input_grid, anchors)

    print("scores shape:", scores.shape)
