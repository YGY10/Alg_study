from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grid import GridConfig, draw_points, draw_polyline, draw_rectangle, make_empty_grid
from teacher import EgoState, TeacherConfig, default_route_path, score_teacher_candidates
from vocab import SparseDriveVocab

TIME_STEPS = np.arange(1, 9, dtype=np.float32) * 0.5


@dataclass(frozen=True)
class Obstacle:
    center_xy: np.ndarray
    size_xy: tuple[float, float]
    velocity_xy: np.ndarray


def obstacle_center_at_time(obstacle: Obstacle, time_s: float) -> np.ndarray:
    return obstacle.center_xy + obstacle.velocity_xy * time_s


def sample_obstacle(
    rng: np.random.Generator,
    grid_config: GridConfig,
) -> Obstacle:
    center_xy = np.array(
        [
            rng.uniform(grid_config.x_min + 8.0, grid_config.x_max - 8.0),
            rng.uniform(grid_config.y_min + 8.0, grid_config.y_max - 8.0),
        ],
        dtype=np.float32,
    )
    size_xy = (
        float(rng.uniform(3.0, 8.0)),
        float(rng.uniform(2.0, 5.0)),
    )
    velocity_xy = np.array(
        [
            rng.uniform(-4.0, 4.0),
            rng.uniform(-3.0, 3.0),
        ],
        dtype=np.float32,
    )
    return Obstacle(
        center_xy=center_xy,
        size_xy=size_xy,
        velocity_xy=velocity_xy,
    )


def rasterize_dynamic_obstacles(
    input_grid: np.ndarray,
    obstacles: list[Obstacle],
    grid_config: GridConfig,
) -> None:
    for obstacle in obstacles:
        draw_rectangle(
            input_grid[0],
            obstacle.center_xy,
            obstacle.size_xy,
            value=1.0,
            config=grid_config,
        )

        for time_s in TIME_STEPS:
            future_center = obstacle_center_at_time(obstacle, float(time_s))
            draw_rectangle(
                input_grid[1],
                future_center,
                obstacle.size_xy,
                value=1.0,
                config=grid_config,
            )


def obstacles_to_tensors(
    obstacles: list[Obstacle],
    max_obstacles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = np.zeros((max_obstacles, 2), dtype=np.float32)
    sizes = np.zeros((max_obstacles, 2), dtype=np.float32)
    velocities = np.zeros((max_obstacles, 2), dtype=np.float32)
    mask = np.zeros((max_obstacles,), dtype=np.float32)

    for i, obstacle in enumerate(obstacles[:max_obstacles]):
        centers[i] = obstacle.center_xy
        sizes[i] = np.array(obstacle.size_xy, dtype=np.float32)
        velocities[i] = obstacle.velocity_xy
        mask[i] = 1.0

    return (
        torch.from_numpy(centers),
        torch.from_numpy(sizes),
        torch.from_numpy(velocities),
        torch.from_numpy(mask),
    )


def obstacles_to_teacher_dicts(
    obstacles: list[Obstacle],
) -> list[dict[str, tuple[float, float]]]:
    return [
        {
            "center_xy": (float(obstacle.center_xy[0]), float(obstacle.center_xy[1])),
            "size_xy": obstacle.size_xy,
            "velocity_xy": (
                float(obstacle.velocity_xy[0]),
                float(obstacle.velocity_xy[1]),
            ),
        }
        for obstacle in obstacles
    ]


def build_path_probability(
    path_costs: np.ndarray,
    temperature: float,
) -> np.ndarray:
    score = -path_costs / max(float(temperature), 1.0e-6)
    score = score - score.max()
    prob = np.exp(score)
    return (prob / max(float(prob.sum()), 1.0e-12)).astype(np.float32)


class ToySparseDriveV2Dataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seed_offset: int = 0,
        grid_config: GridConfig = GridConfig(),
        min_obstacles: int = 4,
        max_obstacles: int = 12,
        teacher_config: TeacherConfig | None = None,
        ego_state: EgoState = EgoState(speed=0.0),
    ) -> None:
        self.num_samples = num_samples
        self.seed_offset = seed_offset
        self.grid_config = grid_config
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.vocab = SparseDriveVocab.load()
        self.teacher_config = teacher_config or TeacherConfig(
            num_path_candidates=8,
            num_top_trajectories=8,
            temperature=2.0,
        )
        self.ego_state = ego_state

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed_offset + index)

        route_path_index = int(rng.integers(0, self.vocab.num_path))
        route_path = self.vocab.path[route_path_index].cpu().numpy()
        goal_xy = route_path[-1, :2].astype(np.float32)

        num_obstacles = int(rng.integers(self.min_obstacles, self.max_obstacles + 1))
        obstacles = [
            sample_obstacle(rng, self.grid_config) for _ in range(num_obstacles)
        ]

        teacher_output = score_teacher_candidates(
            vocab=self.vocab,
            goal_xy=goal_xy,
            obstacles=obstacles_to_teacher_dicts(obstacles),
            ego_state=self.ego_state,
            route_path=route_path[:, :2],
            config=self.teacher_config,
        )

        path_index = teacher_output.best_path_index
        velocity_index = teacher_output.best_velocity_index
        trajectory_index = teacher_output.best_flat_index

        target_path = self.vocab.path[path_index].cpu().numpy()
        target_velocity = self.vocab.velocity[velocity_index].cpu().numpy()
        target_trajectory = (
            self.vocab.trajectory[path_index, velocity_index].cpu().numpy()
        )
        target_trajectory_mask = (
            self.vocab.trajectory_mask[path_index, velocity_index].cpu().numpy()
        )
        target_path_mask = np.ones(self.vocab.len_path, dtype=np.float32)

        input_grid = make_empty_grid(channels=4, config=self.grid_config)
        rasterize_dynamic_obstacles(input_grid, obstacles, self.grid_config)

        draw_polyline(
            input_grid[2],
            route_path[:, :2],
            value=1.0,
            radius=1,
            samples_per_segment=4,
            config=self.grid_config,
        )
        draw_points(
            input_grid[3],
            goal_xy[None],
            value=1.0,
            radius=3,
            config=self.grid_config,
        )

        obstacle_centers, obstacle_sizes, obstacle_velocities, obstacle_mask = (
            obstacles_to_tensors(obstacles, self.max_obstacles)
        )
        ego_state_vector = np.array(
            [
                float(self.ego_state.speed),
                float(self.ego_state.size_xy[0]),
                float(self.ego_state.size_xy[1]),
            ],
            dtype=np.float32,
        )

        teacher_path_probs = build_path_probability(
            teacher_output.path_costs,
            temperature=self.teacher_config.temperature,
        )

        return {
            "input_grid": torch.from_numpy(input_grid).float(),
            "target_path": torch.from_numpy(target_path).float(),
            "target_path_mask": torch.from_numpy(target_path_mask).float(),
            "target_velocity": torch.from_numpy(target_velocity).float(),
            "target_trajectory": torch.from_numpy(target_trajectory).float(),
            "target_trajectory_mask": torch.from_numpy(
                target_trajectory_mask,
            ).float(),
            "path_index": torch.tensor(path_index, dtype=torch.long),
            "velocity_index": torch.tensor(velocity_index, dtype=torch.long),
            "trajectory_index": torch.tensor(trajectory_index, dtype=torch.long),
            "route_path_index": torch.tensor(route_path_index, dtype=torch.long),
            "route_path": torch.from_numpy(route_path).float(),
            "goal_xy": torch.from_numpy(goal_xy).float(),
            "ego_state": torch.from_numpy(ego_state_vector).float(),
            "teacher_path_indices": torch.from_numpy(
                teacher_output.path_indices,
            ).long(),
            "teacher_path_probs": torch.from_numpy(teacher_path_probs).float(),
            "teacher_candidate_probs": torch.from_numpy(
                teacher_output.candidate_probs,
            ).float(),
            "teacher_best_candidate_index": torch.tensor(
                teacher_output.best_candidate_index,
                dtype=torch.long,
            ),
            "obstacle_centers": obstacle_centers.float(),
            "obstacle_sizes": obstacle_sizes.float(),
            "obstacle_velocities": obstacle_velocities.float(),
            "obstacle_mask": obstacle_mask.float(),
        }


if __name__ == "__main__":
    dataset = ToySparseDriveV2Dataset(num_samples=4, seed_offset=0)

    sample = dataset[0]
    for key, value in sample.items():
        print(key, tuple(value.shape), value.dtype)

    print("path_index:", int(sample["path_index"]))
    print("velocity_index:", int(sample["velocity_index"]))
    print("trajectory_index:", int(sample["trajectory_index"]))
    print("teacher_best_candidate_index:", int(sample["teacher_best_candidate_index"]))
    print(
        "teacher_candidate_probs sum:",
        sample["teacher_candidate_probs"].sum().item(),
    )
    print(
        "input_grid min/max:",
        sample["input_grid"].min().item(),
        sample["input_grid"].max().item(),
    )
