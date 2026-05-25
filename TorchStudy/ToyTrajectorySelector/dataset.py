from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from anchors import build_trajectory_anchors

GRID_SIZE = 64

X_RANGE = (0.0, 32.0)  # forward range
Y_RANGE = (-16.0, 16.0)  # lateral range


@dataclass
class RectangleObstacle:
    center_x: float
    center_y: float
    length: float
    width: float


@dataclass
class SceneSample:
    grid: np.ndarray
    goal: np.ndarray
    obstacles: list[RectangleObstacle]
    anchors: np.ndarray
    collision_flags: np.ndarray
    costs: np.ndarray
    best_index: int


def world_to_grid(points: np.ndarray) -> np.ndarray:
    """Convert world points [x, y] into image grid indices [row, col].

    Grid visualization convention:
        top of image: farther forward
        left of image: positive lateral y
    """
    x = points[..., 0]
    y = points[..., 1]

    row = (X_RANGE[1] - x) / (X_RANGE[1] - X_RANGE[0]) * (GRID_SIZE - 1)
    col = (Y_RANGE[1] - y) / (Y_RANGE[1] - Y_RANGE[0]) * (GRID_SIZE - 1)

    row = np.clip(np.round(row), 0, GRID_SIZE - 1).astype(np.int64)
    col = np.clip(np.round(col), 0, GRID_SIZE - 1).astype(np.int64)

    return np.stack([row, col], axis=-1)


def random_goal(rng: np.random.Generator) -> np.ndarray:
    """Generate a goal near the front boundary."""
    return np.array(
        [
            30.0,
            rng.uniform(-10.0, 10.0),
        ],
        dtype=np.float32,
    )


def random_obstacles(rng: np.random.Generator) -> list[RectangleObstacle]:
    """Generate random rectangular obstacles in front of ego."""
    num_obstacles = int(rng.integers(1, 4))
    obstacles = []

    for _ in range(num_obstacles):
        obstacles.append(
            RectangleObstacle(
                center_x=float(rng.uniform(7.0, 25.0)),
                center_y=float(rng.uniform(-10.0, 10.0)),
                length=float(rng.uniform(2.0, 5.0)),
                width=float(rng.uniform(2.0, 5.0)),
            )
        )

    return obstacles


def rasterize_obstacles(obstacles: list[RectangleObstacle]) -> np.ndarray:
    """Rasterize rectangular obstacles into a binary occupancy grid."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for obstacle in obstacles:
        corner_a = np.array(
            [
                obstacle.center_x - obstacle.length / 2.0,
                obstacle.center_y - obstacle.width / 2.0,
            ],
            dtype=np.float32,
        )
        corner_b = np.array(
            [
                obstacle.center_x + obstacle.length / 2.0,
                obstacle.center_y + obstacle.width / 2.0,
            ],
            dtype=np.float32,
        )

        grid_corners = world_to_grid(np.stack([corner_a, corner_b], axis=0))
        row_a, col_a = grid_corners[0]
        row_b, col_b = grid_corners[1]

        row_min, row_max = sorted([row_a, row_b])
        col_min, col_max = sorted([col_a, col_b])

        grid[row_min : row_max + 1, col_min : col_max + 1] = 1.0

    return grid


def rasterize_goal(goal: np.ndarray, radius: int = 2) -> np.ndarray:
    """Rasterize the goal position into a binary goal map."""
    goal_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    row, col = world_to_grid(goal)

    row_min = max(0, row - radius)
    row_max = min(GRID_SIZE, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(GRID_SIZE, col + radius + 1)

    goal_map[row_min:row_max, col_min:col_max] = 1.0

    return goal_map


def build_model_input(grid: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Combine obstacle map and goal map into model input channels."""
    goal_map = rasterize_goal(goal)

    input_grid = np.stack(
        [
            grid,
            goal_map,
        ],
        axis=0,
    )

    return input_grid.astype(np.float32)


def check_collisions(anchors: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Check whether each trajectory intersects an occupied grid cell."""
    trajectory_pixels = world_to_grid(anchors)
    collision_flags = []

    for pixels in trajectory_pixels:
        hit = False

        for row, col in pixels:
            if grid[row, col] > 0.5:
                hit = True
                break

        collision_flags.append(hit)

    return np.array(collision_flags, dtype=bool)


def compute_costs(
    anchors: np.ndarray,
    goal: np.ndarray,
    collision_flags: np.ndarray,
) -> np.ndarray:
    """Compute rule-based cost for every candidate trajectory."""
    terminal_positions = anchors[:, -1, :]
    goal_distances = np.linalg.norm(terminal_positions - goal[None, :], axis=-1)

    collision_penalty = collision_flags.astype(np.float32) * 1000.0

    return goal_distances + collision_penalty


def generate_sample(seed: int | None = None) -> SceneSample:
    """Generate one valid toy planning scene."""
    rng = np.random.default_rng(seed)
    anchors = build_trajectory_anchors()

    while True:
        goal = random_goal(rng)
        obstacles = random_obstacles(rng)
        grid = rasterize_obstacles(obstacles)

        collision_flags = check_collisions(anchors, grid)

        # The sample must contain at least one feasible trajectory.
        if (~collision_flags).any():
            break

    costs = compute_costs(anchors, goal, collision_flags)
    best_index = int(np.argmin(costs))

    return SceneSample(
        grid=grid,
        goal=goal,
        obstacles=obstacles,
        anchors=anchors,
        collision_flags=collision_flags,
        costs=costs,
        best_index=best_index,
    )


class ToyTrajectoryDataset(Dataset):
    """Randomly generated trajectory-selection dataset."""

    def __init__(self, num_samples: int, seed_offset: int = 0) -> None:
        self.num_samples = num_samples
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = generate_sample(seed=self.seed_offset + index)

        input_grid = build_model_input(sample.grid, sample.goal)

        return {
            "input_grid": torch.from_numpy(input_grid),
            "best_index": torch.tensor(sample.best_index, dtype=torch.long),
            "anchors": torch.from_numpy(sample.anchors).float(),
            "costs": torch.from_numpy(sample.costs).float(),
            "collision_flags": torch.from_numpy(sample.collision_flags),
            "goal": torch.from_numpy(sample.goal).float(),
        }


if __name__ == "__main__":
    train_dataset = ToyTrajectoryDataset(
        num_samples=10000,
        seed_offset=0,
    )
    val_dataset = ToyTrajectoryDataset(
        num_samples=2000,
        seed_offset=10000,
    )

    sample = train_dataset[0]

    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))
    print("input grid shape:", sample["input_grid"].shape)
    print("obstacle channel shape:", sample["input_grid"][0].shape)
    print("goal channel shape:", sample["input_grid"][1].shape)
    print("anchors shape:", sample["anchors"].shape)
    print("best index:", sample["best_index"])
    print("goal [x, y]:", sample["goal"])
    print("collision flags:", sample["collision_flags"])
