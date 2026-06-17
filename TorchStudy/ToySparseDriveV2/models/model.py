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
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
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
        self.path_point_embedding = nn.Parameter(
            torch.randn(1, 1, len_path, hidden_dim) * 0.02
        )
        self.path_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.path_context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        self.trajectory_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trajectory_score_head = nn.Sequential(
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

    def score_paths(
        self,
        scene_map: torch.Tensor,
        path_vocab: torch.Tensor,
        ego_feature: torch.Tensor,
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

        path_feature = torch.cat([path_evidence, path_geometry], dim=-1)
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
        else:
            topk_path_indices = forced_path_indices.to(path_scores.device).long()
            topk_path = topk_path_indices.shape[1]
            topk_path_scores = torch.gather(path_scores, 1, topk_path_indices)
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
        trajectory_evidence = self.trajectory_token_mlp(trajectory_evidence).mean(dim=2)

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
        }

    def forward(
        self,
        input_grid: torch.Tensor,
        path_vocab: torch.Tensor,
        velocity_vocab: torch.Tensor,
        trajectory_vocab: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
        forced_path_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene_map, scene_feature, ego_feature = self.encode_scene(input_grid, ego_state)
        path_scores, path_context = self.score_paths(scene_map, path_vocab, ego_feature)
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
