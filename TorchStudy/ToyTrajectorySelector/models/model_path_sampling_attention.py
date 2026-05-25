from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from anchors import NUM_ANCHORS, NUM_POINTS

X_MAX = 32.0
Y_MAX = 16.0


class PathSamplingAttentionTrajectorySelector(nn.Module):
    def __init__(
        self,
        num_anchors: int = NUM_ANCHORS,
        num_points: int = NUM_POINTS,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.num_anchors = num_anchors
        self.num_points = num_points
        self.hidden_dim = hidden_dim

        # Keep spatial resolution for accurate sampling along each path.
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_points * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # The 31 sampled points have different meanings: near ego, middle, endpoint.
        self.sample_point_embedding = nn.Parameter(
            torch.randn(1, 1, num_points, hidden_dim) * 0.02
        )

        # Each path queries only the features sampled from its own route.
        self.path_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # After each path knows its local environment, paths compare each other.
        self.path_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def anchors_to_sample_grid(self, anchors: torch.Tensor) -> torch.Tensor:
        """Convert path points [x_forward, y_left] to grid_sample coordinates.

        Input:
            anchors: [B, N, T, 2]

        Output:
            sample_grid: [B, N, T, 2]
            last dimension is [horizontal, vertical], normalized to [-1, 1].
        """
        forward_x = anchors[..., 0]
        lateral_y = anchors[..., 1]

        # Image left corresponds to positive lateral y.
        normalized_col = -lateral_y / Y_MAX

        # Image top corresponds to large forward x.
        normalized_row = 1.0 - 2.0 * forward_x / X_MAX

        return torch.stack([normalized_col, normalized_row], dim=-1)

    def sample_path_features(
        self,
        scene_map: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Sample scene feature along every candidate trajectory.

        Returns:
            sampled_features: [B, N, T, C]
        """
        sample_grid = self.anchors_to_sample_grid(anchors)

        # grid_sample treats N paths as output rows and T path points as columns.
        sampled_features = F.grid_sample(
            scene_map,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # [B, C, N, T] -> [B, N, T, C]
        sampled_features = sampled_features.permute(0, 2, 3, 1)

        return sampled_features

    def forward(
        self,
        input_grid: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one selection score for each path anchor.

        Args:
            input_grid: [B, 2, 64, 64]
            anchors: [B, N, T, 2]

        Returns:
            scores: [B, N]
        """
        batch_size, num_anchors, num_points, _ = anchors.shape

        scene_map = self.scene_encoder(input_grid)
        sampled_features = self.sample_path_features(scene_map, anchors)
        sampled_features = sampled_features + self.sample_point_embedding

        flat_anchors = anchors.reshape(batch_size, num_anchors, num_points * 2)
        anchor_embed = self.anchor_encoder(flat_anchors)

        # Each path independently reads its own T sampled feature tokens.
        path_queries = anchor_embed.reshape(
            batch_size * num_anchors, 1, self.hidden_dim
        )
        path_tokens = sampled_features.reshape(
            batch_size * num_anchors,
            num_points,
            self.hidden_dim,
        )

        path_evidence, _ = self.path_cross_attention(
            query=path_queries,
            key=path_tokens,
            value=path_tokens,
        )

        path_evidence = path_evidence.reshape(batch_size, num_anchors, self.hidden_dim)
        conditioned_anchor = self.cross_norm(anchor_embed + path_evidence)

        compared_anchor, _ = self.path_self_attention(
            query=conditioned_anchor,
            key=conditioned_anchor,
            value=conditioned_anchor,
        )
        compared_anchor = self.self_norm(conditioned_anchor + compared_anchor)

        refined_anchor = self.ffn_norm(compared_anchor + self.ffn(compared_anchor))

        scores = self.scorer(refined_anchor).squeeze(-1)

        return scores


# if __name__ == "__main__":
#     input_grid = torch.randn(4, 2, 64, 64)
#     anchors = torch.randn(4, NUM_ANCHORS, NUM_POINTS, 2)

#     model = PathSamplingAttentionTrajectorySelector()
#     scores = model(input_grid, anchors)

#     print("scores shape:", scores.shape)

if __name__ == "__main__":
    from anchors import build_trajectory_anchors

    model = PathSamplingAttentionTrajectorySelector()

    anchors = torch.from_numpy(build_trajectory_anchors()).unsqueeze(0).float()

    input_grid = torch.zeros(1, 2, 64, 64)

    # Put obstacle channel on the center trajectory around x=15, y=0.
    input_grid[0, 0, 33, 32] = 1.0

    with torch.no_grad():
        scene_map = input_grid[:, :1].repeat(1, model.hidden_dim, 1, 1)
        sampled = model.sample_path_features(scene_map, anchors)

    center_index = anchors.shape[1] // 2
    center_max = sampled[0, center_index, :, 0].max().item()
    far_side_max = sampled[0, 0, :, 0].max().item()

    print("sampled shape:", sampled.shape)
    print("center path sampled max:", center_max)
    print("far side path sampled max:", far_side_max)
