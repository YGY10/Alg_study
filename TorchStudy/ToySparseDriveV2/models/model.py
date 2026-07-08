from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn


CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, TOY_ROOT / "vocab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vocab import SparseDriveVocab


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels),
                    num_channels=out_channels,
                ),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x) + self.shortcut(x))


class ToySparseDriveV2Model(nn.Module):
    def __init__(
        self,
        num_path: int = 1024,
        len_path: int = 50,
        num_velocity: int = 256,
        len_velocity: int = 8,
        input_channels: int = 4,
        ego_state_dim: int = 3,
        hidden_dim: int = 128,
        x_min: float = -50.0,
        x_max: float = 65.0,
        y_min: float = -60.0,
        y_max: float = 60.0,
        topk_path: int = 8,
    ) -> None:
        super().__init__()

        self.num_path = num_path
        self.len_path = len_path
        self.num_velocity = num_velocity
        self.len_velocity = len_velocity
        self.hidden_dim = hidden_dim
        self.ego_state_dim = ego_state_dim
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.topk_path = topk_path

        self.scene_encoder = nn.Sequential(
            ConvBlock(input_channels, 64),
            ConvBlock(64, 64),
            ConvBlock(64, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )

        self.scene_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scene_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.scene_ego_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.path_geometry_encoder = nn.Sequential(
            nn.Linear(len_path * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.path_explicit_feature_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.path_point_embedding = nn.Parameter(
            torch.randn(1, 1, len_path, hidden_dim) * 0.02
        )
        self.path_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.path_context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.velocity_encoder = nn.Sequential(
            nn.Linear(len_velocity, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.velocity_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.trajectory_point_embedding = nn.Parameter(
            torch.randn(1, 1, len_velocity, hidden_dim) * 0.02
        )
        self.trajectory_point_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.trajectory_temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.trajectory_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.no_collision_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def xy_to_sample_grid(self, xy: torch.Tensor) -> torch.Tensor:
        """Convert xy coordinates to grid_sample coordinates.

        Args:
            xy: [..., T, 2] or [..., T, 3]

        Returns:
            sample_grid: [..., T, 2], last dim is [col_norm, row_norm].
        """
        x = xy[..., 0]
        y = xy[..., 1]

        col_norm = 2.0 * (self.y_max - y) / (self.y_max - self.y_min) - 1.0
        row_norm = 2.0 * (self.x_max - x) / (self.x_max - self.x_min) - 1.0

        return torch.stack([col_norm, row_norm], dim=-1)

    def encode_ego_state(
        self,
        ego_state: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if ego_state is None:
            ego_state = torch.zeros(
                batch_size,
                self.ego_state_dim,
                device=device,
            )
        return self.ego_encoder(ego_state.float())

    def encode_scene(
        self,
        input_grid: torch.Tensor,
        ego_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_map = self.scene_encoder(input_grid)
        scene_feature = self.scene_pool(scene_map).flatten(1)
        scene_feature = self.scene_mlp(scene_feature)
        ego_feature = self.encode_ego_state(
            ego_state,
            batch_size=input_grid.shape[0],
            device=input_grid.device,
        )
        scene_feature = self.scene_ego_fusion(
            torch.cat([scene_feature, ego_feature], dim=-1)
        )
        return scene_map, scene_feature, ego_feature

    def sample_path_features(
        self,
        scene_map: torch.Tensor,
        path_vocab: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scene_map.shape[0]
        sample_grid = self.xy_to_sample_grid(path_vocab).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        sampled_features = F.grid_sample(
            scene_map,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return sampled_features.permute(0, 2, 3, 1)

    def compute_explicit_path_features(
        self,
        path_vocab: torch.Tensor,
        batch_size: int,
        goal_xy: torch.Tensor | None = None,
        route_path: torch.Tensor | None = None,
        obstacle_centers: torch.Tensor | None = None,
        obstacle_sizes: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = path_vocab.device
        dtype = path_vocab.dtype
        path_xy = path_vocab[..., :2]
        endpoint_xy = path_xy[:, -1]
        start_xy = path_xy[:, 0]
        path_length = torch.linalg.norm(
            path_xy[:, 1:] - path_xy[:, :-1],
            dim=-1,
        ).sum(dim=-1)

        endpoint_xy_b = endpoint_xy[None].expand(batch_size, -1, -1)
        start_xy_b = start_xy[None].expand(batch_size, -1, -1)
        path_length_b = path_length[None, :, None].expand(batch_size, -1, 1)

        if goal_xy is None:
            goal_xy = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        goal_xy = goal_xy.to(device=device, dtype=dtype)
        endpoint_to_goal = goal_xy[:, None, :] - endpoint_xy_b
        endpoint_goal_l2 = torch.linalg.norm(endpoint_to_goal, dim=-1, keepdim=True)

        heading_start = max(self.len_path - 4, 0)
        path_heading = path_xy[:, -1] - path_xy[:, heading_start]
        path_heading = path_heading / torch.linalg.norm(
            path_heading,
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0e-6)
        goal_heading = goal_xy[:, None, :] - endpoint_xy_b
        goal_heading = goal_heading / torch.linalg.norm(
            goal_heading,
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0e-6)
        path_heading_b = path_heading[None].expand(batch_size, -1, -1)
        heading_cos = (path_heading_b * goal_heading).sum(dim=-1, keepdim=True)
        heading_sin = (
            path_heading_b[..., 0:1] * goal_heading[..., 1:2]
            - path_heading_b[..., 1:2] * goal_heading[..., 0:1]
        )

        if route_path is None:
            mean_route_l2 = torch.zeros(
                batch_size, self.num_path, 1, device=device, dtype=dtype
            )
            endpoint_route_l2 = mean_route_l2
        else:
            route_xy = route_path[..., :2].to(device=device, dtype=dtype)
            route_mask = torch.isfinite(route_xy).all(dim=-1)
            route_xy = torch.where(
                route_mask[..., None],
                route_xy,
                torch.zeros_like(route_xy),
            )

            mean_route_chunks = []
            endpoint_route_chunks = []
            path_chunk_size = 128
            for start in range(0, self.num_path, path_chunk_size):
                end = min(start + path_chunk_size, self.num_path)
                path_xy_chunk = path_xy[start:end]
                route_diff = (
                    path_xy_chunk[None, :, :, None, :]
                    - route_xy[:, None, None, :, :]
                )
                route_l2 = torch.linalg.norm(route_diff, dim=-1)
                route_l2 = route_l2.masked_fill(
                    ~route_mask[:, None, None, :],
                    1.0e6,
                )
                mean_route_chunks.append(
                    route_l2.min(dim=-1).values.mean(dim=-1, keepdim=True)
                )

                endpoint_diff = endpoint_xy[start:end][None, :, None, :] - route_xy[:, None, :, :]
                endpoint_l2 = torch.linalg.norm(endpoint_diff, dim=-1)
                endpoint_l2 = endpoint_l2.masked_fill(~route_mask[:, None, :], 1.0e6)
                endpoint_route_chunks.append(endpoint_l2.min(dim=-1, keepdim=True).values)

            mean_route_l2 = torch.cat(mean_route_chunks, dim=1)
            endpoint_route_l2 = torch.cat(endpoint_route_chunks, dim=1)

        if (
            obstacle_centers is None
            or obstacle_sizes is None
            or obstacle_mask is None
        ):
            min_clearance = torch.full(
                (batch_size, self.num_path, 1),
                10.0,
                device=device,
                dtype=dtype,
            )
            collision_flag = torch.zeros(
                batch_size, self.num_path, 1, device=device, dtype=dtype
            )
        else:
            centers = obstacle_centers.to(device=device, dtype=dtype)
            sizes = obstacle_sizes.to(device=device, dtype=dtype)
            mask = obstacle_mask.to(device=device).bool()
            delta = (
                path_xy[None, :, :, None, :] - centers[:, None, None, :, :]
            ).abs()
            half_size = 0.5 * sizes[:, None, None, :, :]
            outside = (delta - half_size).clamp(min=0.0)
            outside_dist = torch.linalg.norm(outside, dim=-1)
            inside_margin = (half_size - delta).amin(dim=-1)
            signed_clearance = torch.where(
                inside_margin > 0.0,
                -inside_margin,
                outside_dist,
            )
            signed_clearance = signed_clearance.masked_fill(~mask[:, None, None, :], 1.0e6)
            min_clearance = signed_clearance.amin(dim=-1).amin(dim=-1, keepdim=True)
            collision_flag = (min_clearance < 0.0).to(dtype)
            min_clearance = min_clearance.clamp(min=-5.0, max=20.0)

        return torch.cat(
            [
                endpoint_xy_b / 60.0,
                start_xy_b / 60.0,
                endpoint_to_goal / 80.0,
                endpoint_goal_l2 / 80.0,
                heading_cos,
                heading_sin,
                mean_route_l2.clamp(max=80.0) / 80.0,
                endpoint_route_l2.clamp(max=80.0) / 80.0,
                min_clearance / 20.0,
                collision_flag,
                path_length_b / 80.0,
            ],
            dim=-1,
        )


    def score_paths(
        self,
        scene_map: torch.Tensor,
        path_vocab: torch.Tensor,
        ego_feature: torch.Tensor,
        goal_xy: torch.Tensor | None = None,
        route_path: torch.Tensor | None = None,
        obstacle_centers: torch.Tensor | None = None,
        obstacle_sizes: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = scene_map.shape[0]
        sampled_features = self.sample_path_features(scene_map, path_vocab)
        sampled_features = sampled_features + self.path_point_embedding
        sampled_features = sampled_features + ego_feature[:, None, None, :]

        path_evidence = self.path_token_mlp(sampled_features).mean(dim=2)

        flat_path = path_vocab.reshape(path_vocab.shape[0], -1)
        path_geometry = self.path_geometry_encoder(flat_path)
        path_geometry = path_geometry[None, :, :].expand(
            batch_size,
            -1,
            -1,
        )
        explicit_features = self.compute_explicit_path_features(
            path_vocab=path_vocab,
            batch_size=batch_size,
            goal_xy=goal_xy,
            route_path=route_path,
            obstacle_centers=obstacle_centers,
            obstacle_sizes=obstacle_sizes,
            obstacle_mask=obstacle_mask,
        )
        path_explicit = self.path_explicit_feature_encoder(explicit_features)

        path_feature = torch.cat([path_evidence, path_geometry, path_explicit], dim=-1)
        path_scores = self.path_score_head(path_feature).squeeze(-1)
        path_context = self.path_context_projection(path_feature)

        return path_scores, path_context

    def score_velocities(
        self,
        scene_feature: torch.Tensor,
        velocity_vocab: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = scene_feature.shape[0]
        velocity_vocab_feature = self.velocity_encoder(velocity_vocab)

        scene_feature = scene_feature[:, None, :].expand(
            batch_size,
            velocity_vocab_feature.shape[0],
            self.hidden_dim,
        )
        velocity_feature = velocity_vocab_feature[None, :, :].expand(
            batch_size,
            -1,
            -1,
        )

        fused_feature = torch.cat([scene_feature, velocity_feature], dim=-1)
        velocity_scores = self.velocity_score_head(fused_feature).squeeze(-1)
        return velocity_scores, velocity_vocab_feature

    def sample_trajectory_features(
        self,
        scene_map: torch.Tensor,
        candidate_trajectories: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scene_map.shape[0]
        num_candidates = candidate_trajectories.shape[1]
        sample_grid = self.xy_to_sample_grid(candidate_trajectories)

        sampled_features = F.grid_sample(
            scene_map,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return sampled_features.permute(0, 2, 3, 1).reshape(
            batch_size,
            num_candidates,
            self.len_velocity,
            self.hidden_dim,
        )

    def score_trajectories(
        self,
        scene_map: torch.Tensor,
        path_scores: torch.Tensor,
        path_context: torch.Tensor,
        velocity_feature: torch.Tensor,
        trajectory_vocab: torch.Tensor,
        ego_feature: torch.Tensor,
        forced_path_indices: torch.Tensor | None = None,
        extra_path_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = scene_map.shape[0]
        num_velocity = velocity_feature.shape[0]
        if forced_path_indices is None:
            topk_path = min(self.topk_path, path_scores.shape[1])
            topk_path_scores, topk_path_indices = torch.topk(
                path_scores,
                topk_path,
                dim=1,
            )
            model_topk_path_indices = topk_path_indices
            model_topk_path_scores = topk_path_scores
            if extra_path_indices is not None:
                extra_path_indices = extra_path_indices.to(path_scores.device).long()
                topk_path_indices = torch.cat(
                    [topk_path_indices, extra_path_indices],
                    dim=1,
                )
                topk_path = topk_path_indices.shape[1]
                topk_path_scores = torch.gather(path_scores, 1, topk_path_indices)
        else:
            topk_path_indices = forced_path_indices.to(path_scores.device).long()
            topk_path = topk_path_indices.shape[1]
            topk_path_scores = torch.gather(path_scores, 1, topk_path_indices)
            model_topk_path_indices = topk_path_indices
            model_topk_path_scores = topk_path_scores
        selected_path_context = torch.gather(
            path_context,
            1,
            topk_path_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )

        candidate_trajectories = trajectory_vocab[topk_path_indices]
        candidate_trajectories = candidate_trajectories.reshape(
            batch_size,
            topk_path * num_velocity,
            self.len_velocity,
            3,
        )

        trajectory_evidence = self.sample_trajectory_features(
            scene_map,
            candidate_trajectories,
        )
        trajectory_evidence = trajectory_evidence + self.trajectory_point_embedding
        trajectory_evidence = trajectory_evidence + ego_feature[:, None, None, :]
        candidate_point_geometry = torch.stack(
            [
                candidate_trajectories[..., 0] / 60.0,
                candidate_trajectories[..., 1] / 60.0,
                candidate_trajectories[..., 2] / torch.pi,
            ],
            dim=-1,
        )
        trajectory_evidence = self.trajectory_point_encoder(
            torch.cat([trajectory_evidence, candidate_point_geometry], dim=-1)
        )
        trajectory_evidence = trajectory_evidence.reshape(
            batch_size * topk_path * num_velocity,
            self.len_velocity,
            self.hidden_dim,
        )
        trajectory_evidence = self.trajectory_temporal_encoder(
            trajectory_evidence.transpose(1, 2)
        ).mean(dim=-1)
        trajectory_evidence = trajectory_evidence.reshape(
            batch_size,
            topk_path * num_velocity,
            self.hidden_dim,
        )

        selected_path_context = selected_path_context[:, :, None, :].expand(
            batch_size,
            topk_path,
            num_velocity,
            self.hidden_dim,
        )
        velocity_context = velocity_feature[None, None, :, :].expand(
            batch_size,
            topk_path,
            num_velocity,
            self.hidden_dim,
        )

        trajectory_context = torch.cat(
            [
                trajectory_evidence,
                selected_path_context.reshape(
                    batch_size,
                    topk_path * num_velocity,
                    self.hidden_dim,
                ),
                velocity_context.reshape(
                    batch_size,
                    topk_path * num_velocity,
                    self.hidden_dim,
                ),
            ],
            dim=-1,
        )

        trajectory_scores = self.trajectory_score_head(trajectory_context).squeeze(-1)
        no_collision_logits = self.no_collision_head(trajectory_context).squeeze(-1)

        velocity_indices = torch.arange(
            num_velocity,
            device=trajectory_scores.device,
        )
        velocity_indices = velocity_indices[None, None, :].expand(
            batch_size,
            topk_path,
            num_velocity,
        )
        candidate_path_indices = topk_path_indices[:, :, None].expand(
            batch_size,
            topk_path,
            num_velocity,
        )

        return {
            "trajectory_scores": trajectory_scores,
            "no_collision_logits": no_collision_logits,
            "candidate_trajectories": candidate_trajectories,
            "candidate_path_indices": candidate_path_indices.reshape(
                batch_size,
                topk_path * num_velocity,
            ),
            "candidate_velocity_indices": velocity_indices.reshape(
                batch_size,
                topk_path * num_velocity,
            ),
            "topk_path_scores": topk_path_scores,
            "topk_path_indices": topk_path_indices,
            "model_topk_path_scores": model_topk_path_scores,
            "model_topk_path_indices": model_topk_path_indices,
        }

    def forward(
        self,
        input_grid: torch.Tensor,
        path_vocab: torch.Tensor,
        velocity_vocab: torch.Tensor,
        trajectory_vocab: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
        goal_xy: torch.Tensor | None = None,
        route_path: torch.Tensor | None = None,
        obstacle_centers: torch.Tensor | None = None,
        obstacle_sizes: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
        forced_path_indices: torch.Tensor | None = None,
        extra_path_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene_map, scene_feature, ego_feature = self.encode_scene(input_grid, ego_state)
        path_scores, path_context = self.score_paths(
            scene_map=scene_map,
            path_vocab=path_vocab,
            ego_feature=ego_feature,
            goal_xy=goal_xy,
            route_path=route_path,
            obstacle_centers=obstacle_centers,
            obstacle_sizes=obstacle_sizes,
            obstacle_mask=obstacle_mask,
        )
        velocity_scores, velocity_feature = self.score_velocities(
            scene_feature,
            velocity_vocab,
        )

        output = {
            "path_scores": path_scores,
            "velocity_scores": velocity_scores,
        }

        if trajectory_vocab is not None:
            output.update(
                self.score_trajectories(
                    scene_map=scene_map,
                    path_scores=path_scores,
                    path_context=path_context,
                    velocity_feature=velocity_feature,
                    trajectory_vocab=trajectory_vocab,
                    ego_feature=ego_feature,
                    forced_path_indices=forced_path_indices,
                    extra_path_indices=extra_path_indices,
                )
            )

        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = SparseDriveVocab.load().to(device)
    model = ToySparseDriveV2Model().to(device)

    input_grid = torch.randn(2, 4, 128, 128, device=device)

    output = model(
        input_grid=input_grid,
        path_vocab=vocab.path,
        velocity_vocab=vocab.velocity,
        trajectory_vocab=vocab.trajectory,
    )

    for key, value in output.items():
        print(key, tuple(value.shape), value.dtype, value.device)
