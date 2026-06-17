# 加载真实SparseDriveV2 vocab, 封装成dataclass
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VOCAB_DIR = REPO_ROOT / "SparseDriveV2" / "ckpt" / "kmeans"


@dataclass(frozen=True)
class SparseDriveVocab:
    path: torch.Tensor  # [1024, 50, 3], x/y/yaw
    velocity: torch.Tensor  # [256, 8]
    trajectory: torch.Tensor  # [1024, 256, 8, 3]
    trajectory_mask: torch.Tensor  # [1024, 256, 8]

    @classmethod
    def load(
        cls,
        vocab_dir: str | Path = DEFAULT_VOCAB_DIR,
        dtype: torch.dtype = torch.float32,
    ) -> "SparseDriveVocab":
        vocab_dir = Path(vocab_dir)

        path = np.load(vocab_dir / "path_1024.npy")
        velocity = np.load(vocab_dir / "velocity_256.npy")
        trajectory_data = np.load(vocab_dir / "trajectory_1024_256.npz")

        return cls(
            path=torch.as_tensor(path, dtype=dtype),
            velocity=torch.as_tensor(velocity, dtype=dtype),
            trajectory=torch.as_tensor(trajectory_data["trajectory"], dtype=dtype),
            trajectory_mask=torch.as_tensor(
                trajectory_data["trajectory_mask"],
                dtype=dtype,
            ),
        )

    @property
    def num_path(self) -> int:
        return int(self.path.shape[0])

    @property
    def num_velocity(self) -> int:
        return int(self.velocity.shape[0])

    @property
    def len_path(self) -> int:
        return int(self.path.shape[1])

    @property
    def len_velocity(self) -> int:
        return int(self.velocity.shape[1])

    def to(self, device: torch.device | str) -> "SparseDriveVocab":
        return SparseDriveVocab(
            path=self.path.to(device),
            velocity=self.velocity.to(device),
            trajectory=self.trajectory.to(device),
            trajectory_mask=self.trajectory_mask.to(device),
        )

    def summary(self) -> dict[str, object]:
        path_xy = self.path[..., :2]
        traj_xy = self.trajectory[..., :2]
        velocity_mean = self.velocity.mean(dim=1)

        return {
            "path_shape": tuple(self.path.shape),
            "velocity_shape": tuple(self.velocity.shape),
            "trajectory_shape": tuple(self.trajectory.shape),
            "trajectory_mask_shape": tuple(self.trajectory_mask.shape),
            "path_x_range": (
                float(path_xy[..., 0].min()),
                float(path_xy[..., 0].max()),
            ),
            "path_y_range": (
                float(path_xy[..., 1].min()),
                float(path_xy[..., 1].max()),
            ),
            "trajectory_x_range": (
                float(traj_xy[..., 0].min()),
                float(traj_xy[..., 0].max()),
            ),
            "trajectory_y_range": (
                float(traj_xy[..., 1].min()),
                float(traj_xy[..., 1].max()),
            ),
            "velocity_range": (
                float(self.velocity.min()),
                float(self.velocity.max()),
            ),
            "velocity_mean_range": (
                float(velocity_mean.min()),
                float(velocity_mean.max()),
            ),
            "trajectory_mask_valid_ratio": float(self.trajectory_mask.mean()),
        }

    def check_grid_coverage(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> dict[str, float | bool]:
        path_xy = self.path[..., :2]
        traj_xy = self.trajectory[..., :2]

        path_inside = (
            (path_xy[..., 0] >= x_min)
            & (path_xy[..., 0] <= x_max)
            & (path_xy[..., 1] >= y_min)
            & (path_xy[..., 1] <= y_max)
        )

        traj_inside = (
            (traj_xy[..., 0] >= x_min)
            & (traj_xy[..., 0] <= x_max)
            & (traj_xy[..., 1] >= y_min)
            & (traj_xy[..., 1] <= y_max)
        )

        return {
            "all_path_points_inside": bool(path_inside.all()),
            "all_trajectory_points_inside": bool(traj_inside.all()),
            "path_point_coverage": float(path_inside.float().mean()),
            "trajectory_point_coverage": float(traj_inside.float().mean()),
        }


if __name__ == "__main__":
    vocab = SparseDriveVocab.load()

    for key, value in vocab.summary().items():
        print(f"{key}: {value}")

    coverage = vocab.check_grid_coverage(
        x_min=-40.0,
        x_max=55.0,
        y_min=-50.0,
        y_max=50.0,
    )

    for key, value in coverage.items():
        print(f"{key}: {value}")
