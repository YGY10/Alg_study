from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, TOY_ROOT / "dataset", TOY_ROOT / "vocab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import Obstacle, obstacles_to_tensors, rasterize_dynamic_obstacles
from grid import GridConfig, draw_points, draw_polyline, make_empty_grid
from vocab import SparseDriveVocab

FUTURE_TIMES = np.arange(1, 9, dtype=np.float32) * 0.5


@dataclass(frozen=True)
class HumanSampleMeta:
    episode_path: Path
    step_index: int
    current_time: float


def rotation_world_to_ego(yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw],
        ],
        dtype=np.float32,
    )


def transform_points_to_ego(
    points_xy: np.ndarray,
    ego_xy: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    ego_xy = np.asarray(ego_xy, dtype=np.float32)
    rotation = rotation_world_to_ego(ego_yaw)
    return (points_xy - ego_xy) @ rotation.T


def transform_vectors_to_ego(
    vectors_xy: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    vectors_xy = np.asarray(vectors_xy, dtype=np.float32)
    rotation = rotation_world_to_ego(ego_yaw)
    return vectors_xy @ rotation.T


def interp_columns(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.interp(query_times, times, values[:, column])
            for column in range(values.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)


def make_obstacles_in_current_ego_frame(
    raw_obstacles: list[dict[str, Any]],
    current_time: float,
    ego_xy: np.ndarray,
    ego_yaw: float,
) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    for raw_obstacle in raw_obstacles:
        center_xy = np.asarray(raw_obstacle["center_xy"], dtype=np.float32)
        velocity_xy = np.asarray(raw_obstacle["velocity_xy"], dtype=np.float32)
        center_at_time = center_xy + velocity_xy * float(current_time)
        obstacles.append(
            Obstacle(
                center_xy=transform_points_to_ego(
                    center_at_time[None],
                    ego_xy=ego_xy,
                    ego_yaw=ego_yaw,
                )[0],
                size_xy=(
                    float(raw_obstacle["size_xy"][0]),
                    float(raw_obstacle["size_xy"][1]),
                ),
                velocity_xy=transform_vectors_to_ego(
                    velocity_xy[None],
                    ego_yaw=ego_yaw,
                )[0],
            )
        )
    return obstacles


def trajectory_l2_distance(
    trajectory_vocab: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    diff = trajectory_vocab[..., :2] - target_trajectory[None, None, :, :2]
    dist = diff.pow(2).sum(dim=-1)
    dist = dist * target_mask[None, None, :]
    return dist.sum(dim=-1) / target_mask.sum().clamp(min=1.0)


def cumulative_distance(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    segment = np.linalg.norm(np.diff(points_xy, axis=0), axis=-1)
    return np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment).astype(np.float32)],
        axis=0,
    )


def interp_points_by_distance(
    points_xy: np.ndarray,
    query_distance: np.ndarray,
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    query_distance = np.asarray(query_distance, dtype=np.float32)
    distance = cumulative_distance(points_xy)
    if len(points_xy) == 0:
        return np.zeros((len(query_distance), 2), dtype=np.float32)
    if float(distance[-1]) <= 1.0e-6:
        return np.repeat(points_xy[:1], repeats=len(query_distance), axis=0)
    clipped_query = np.clip(query_distance, 0.0, float(distance[-1]))
    return np.stack(
        [
            np.interp(clipped_query, distance, points_xy[:, 0]),
            np.interp(clipped_query, distance, points_xy[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return (exp_values / np.sum(exp_values)).astype(np.float32)


class HumanDriveDataset(Dataset):
    def __init__(
        self,
        episode_paths: list[str | Path],
        vocab: SparseDriveVocab | None = None,
        grid_config: GridConfig = GridConfig(),
        cache_path: str | Path | None = None,
        rebuild_cache: bool = False,
        path_chunk_size: int = 32,
        label_batch_size: int = 64,
        long_path_horizon: float = 8.0,
        long_path_compare_points: int = 32,
        long_path_positive_topk: int = 8,
        long_path_temperature: float = 2.0,
        long_path_min_progress: float = 5.0,
    ) -> None:
        self.episode_paths = [Path(path) for path in episode_paths]
        self.vocab = vocab or SparseDriveVocab.load()
        self.grid_config = grid_config
        self.path_chunk_size = path_chunk_size
        self.label_batch_size = label_batch_size
        self.long_path_horizon = float(long_path_horizon)
        self.long_path_compare_points = int(long_path_compare_points)
        self.long_path_positive_topk = int(long_path_positive_topk)
        self.long_path_temperature = float(long_path_temperature)
        self.long_path_min_progress = float(long_path_min_progress)
        self.cache_path = Path(cache_path) if cache_path is not None else None

        self.samples: list[dict[str, Any]] = []
        self.metas: list[HumanSampleMeta] = []
        self._build_samples()

        self.path_indices: np.ndarray
        self.velocity_indices: np.ndarray
        self.long_path_indices: np.ndarray
        self.long_path_weights: np.ndarray
        self.long_path_costs: np.ndarray
        self.long_path_valid: np.ndarray
        if (
            self.cache_path is not None
            and self.cache_path.is_file()
            and not rebuild_cache
        ):
            self._load_label_cache()
        else:
            self._build_label_cache()

    def _build_samples(self) -> None:
        for episode_path in self.episode_paths:
            data = json.loads(episode_path.read_text())
            ego_history = np.asarray(data["ego_history"], dtype=np.float32)
            dt = float(data["dt"])
            times = np.arange(len(ego_history), dtype=np.float32) * dt
            last_time = float(times[-1])

            for step_index, current_state in enumerate(ego_history):
                current_time = float(times[step_index])
                if current_time + float(FUTURE_TIMES[-1]) > last_time + 1.0e-4:
                    continue

                self.samples.append(
                    {
                        "episode": data,
                        "ego_history": ego_history,
                        "times": times,
                        "step_index": step_index,
                    }
                )
                self.metas.append(
                    HumanSampleMeta(
                        episode_path=episode_path,
                        step_index=step_index,
                        current_time=current_time,
                    )
                )

    def _load_label_cache(self) -> None:
        assert self.cache_path is not None
        with np.load(self.cache_path, allow_pickle=False) as data:
            required_keys = {
                "path_indices",
                "velocity_indices",
                "long_path_indices",
                "long_path_weights",
                "long_path_costs",
                "long_path_valid",
                "long_path_horizon",
                "long_path_compare_points",
                "long_path_positive_topk",
                "long_path_temperature",
                "long_path_min_progress",
            }
            missing_keys = required_keys.difference(data.files)
            if missing_keys:
                raise ValueError(
                    f"Human label cache misses {sorted(missing_keys)}; "
                    f"rerun with rebuild_cache=True: {self.cache_path}"
                )
            cache_episode_paths = [Path(item) for item in data["episode_paths"]]
            if cache_episode_paths != self.episode_paths:
                raise ValueError(
                    f"Human label cache episode list mismatch: {self.cache_path}"
                )
            if int(data["num_samples"]) != len(self.samples):
                raise ValueError(
                    f"Human label cache sample count mismatch: {self.cache_path}"
                )
            if abs(float(data["long_path_horizon"]) - self.long_path_horizon) > 1.0e-6:
                raise ValueError(f"Human label cache long_path_horizon mismatch: {self.cache_path}")
            if int(data["long_path_compare_points"]) != self.long_path_compare_points:
                raise ValueError(f"Human label cache long_path_compare_points mismatch: {self.cache_path}")
            if int(data["long_path_positive_topk"]) != self.long_path_positive_topk:
                raise ValueError(f"Human label cache long_path_positive_topk mismatch: {self.cache_path}")
            if abs(float(data["long_path_temperature"]) - self.long_path_temperature) > 1.0e-6:
                raise ValueError(f"Human label cache long_path_temperature mismatch: {self.cache_path}")
            if abs(float(data["long_path_min_progress"]) - self.long_path_min_progress) > 1.0e-6:
                raise ValueError(f"Human label cache long_path_min_progress mismatch: {self.cache_path}")
            self.path_indices = data["path_indices"].astype(np.int64)
            self.velocity_indices = data["velocity_indices"].astype(np.int64)
            self.long_path_indices = data["long_path_indices"].astype(np.int64)
            self.long_path_weights = data["long_path_weights"].astype(np.float32)
            self.long_path_costs = data["long_path_costs"].astype(np.float32)
            self.long_path_valid = data["long_path_valid"].astype(bool)

    def _build_label_cache(self) -> None:
        num_samples = len(self.samples)
        path_indices = np.zeros((num_samples,), dtype=np.int64)
        velocity_indices = np.zeros((num_samples,), dtype=np.int64)
        long_path_indices = np.zeros(
            (num_samples, self.long_path_positive_topk),
            dtype=np.int64,
        )
        long_path_weights = np.full(
            (num_samples, self.long_path_positive_topk),
            1.0 / max(self.long_path_positive_topk, 1),
            dtype=np.float32,
        )
        long_path_valid = np.zeros((num_samples,), dtype=bool)

        target_trajectories = []
        target_masks = []
        for sample_index in range(num_samples):
            target_trajectory, target_mask = self.build_target_trajectory(sample_index)
            target_trajectories.append(target_trajectory)
            target_masks.append(target_mask)

        target_trajectories_t = torch.from_numpy(
            np.stack(target_trajectories, axis=0),
        ).float()
        target_masks_t = torch.from_numpy(np.stack(target_masks, axis=0)).float()
        trajectory_vocab = self.vocab.trajectory.cpu().float()

        with torch.no_grad():
            for batch_start in range(0, num_samples, self.label_batch_size):
                batch_end = min(batch_start + self.label_batch_size, num_samples)
                batch_target = target_trajectories_t[batch_start:batch_end]
                batch_mask = target_masks_t[batch_start:batch_end]
                batch_size = batch_target.shape[0]

                best_cost = torch.full((batch_size,), float("inf"))
                best_path = torch.zeros((batch_size,), dtype=torch.long)
                best_velocity = torch.zeros((batch_size,), dtype=torch.long)

                for path_start in range(0, self.vocab.num_path, self.path_chunk_size):
                    path_end = min(path_start + self.path_chunk_size, self.vocab.num_path)
                    chunk = trajectory_vocab[path_start:path_end]
                    diff = (
                        chunk[None, ..., :2]
                        - batch_target[:, None, None, :, :2]
                    )
                    dist = diff.pow(2).sum(dim=-1)
                    dist = dist * batch_mask[:, None, None, :]
                    dist = dist.sum(dim=-1) / batch_mask.sum(dim=-1)[:, None, None].clamp(min=1.0)

                    flat_dist = dist.reshape(batch_size, -1)
                    chunk_cost, flat_index = flat_dist.min(dim=1)
                    improved = chunk_cost < best_cost
                    if bool(improved.any()):
                        path_offset = flat_index // self.vocab.num_velocity
                        velocity_index = flat_index % self.vocab.num_velocity
                        best_cost[improved] = chunk_cost[improved]
                        best_path[improved] = path_start + path_offset[improved]
                        best_velocity[improved] = velocity_index[improved]

                path_indices[batch_start:batch_end] = best_path.numpy()
                velocity_indices[batch_start:batch_end] = best_velocity.numpy()

        self.path_indices = path_indices
        self.velocity_indices = velocity_indices
        (
            long_path_indices,
            long_path_weights,
            long_path_costs,
            long_path_valid,
        ) = self._build_long_path_labels()

        self.long_path_indices = long_path_indices
        self.long_path_weights = long_path_weights
        self.long_path_costs = long_path_costs
        self.long_path_valid = long_path_valid

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                num_samples=np.asarray(len(self.samples), dtype=np.int64),
                episode_paths=np.asarray(
                    [str(path) for path in self.episode_paths],
                    dtype="<U512",
                ),
                path_indices=self.path_indices,
                velocity_indices=self.velocity_indices,
                long_path_indices=self.long_path_indices,
                long_path_weights=self.long_path_weights,
                long_path_costs=self.long_path_costs,
                long_path_valid=self.long_path_valid,
                long_path_horizon=np.asarray(self.long_path_horizon, dtype=np.float32),
                long_path_compare_points=np.asarray(
                    self.long_path_compare_points,
                    dtype=np.int64,
                ),
                long_path_positive_topk=np.asarray(
                    self.long_path_positive_topk,
                    dtype=np.int64,
                ),
                long_path_temperature=np.asarray(
                    self.long_path_temperature,
                    dtype=np.float32,
                ),
                long_path_min_progress=np.asarray(
                    self.long_path_min_progress,
                    dtype=np.float32,
                ),
            )

    def _build_long_path_labels(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = len(self.samples)
        topk = min(self.long_path_positive_topk, self.vocab.num_path)
        long_path_indices = np.zeros((num_samples, topk), dtype=np.int64)
        long_path_weights = np.full((num_samples, topk), 1.0 / topk, dtype=np.float32)
        long_path_costs = np.full(
            (num_samples, self.vocab.num_path),
            np.inf,
            dtype=np.float32,
        )
        long_path_valid = np.zeros((num_samples,), dtype=bool)

        path_xy = self.vocab.path[..., :2].cpu().float()
        path_distance = torch.stack(
            [
                torch.from_numpy(cumulative_distance(path_xy[path_index].numpy()))
                for path_index in range(self.vocab.num_path)
            ],
            dim=0,
        ).float()
        path_count = path_xy.shape[0]

        for sample_index in range(num_samples):
            human_xy, valid = self.build_long_path_target(sample_index)
            if not valid:
                long_path_indices[sample_index, :topk] = self.path_indices[sample_index]
                continue

            human_distance = cumulative_distance(human_xy)
            human_progress = float(human_distance[-1])
            query_distance_np = np.linspace(
                0.0,
                human_progress,
                self.long_path_compare_points,
                dtype=np.float32,
            )
            human_points = torch.from_numpy(
                interp_points_by_distance(human_xy, query_distance_np)
            ).float()
            query_distance = torch.from_numpy(query_distance_np).float()

            query = query_distance[None, :].expand(path_count, -1)
            query = torch.minimum(
                query,
                path_distance[:, -1:].expand_as(query),
            )
            upper = torch.searchsorted(path_distance.contiguous(), query.contiguous())
            upper = upper.clamp(min=1, max=path_distance.shape[1] - 1)
            lower = upper - 1

            s0 = torch.gather(path_distance, 1, lower)
            s1 = torch.gather(path_distance, 1, upper)
            denom = (s1 - s0).clamp(min=1.0e-6)
            alpha = ((query - s0) / denom).unsqueeze(-1)

            lower_xy = torch.gather(
                path_xy,
                1,
                lower[..., None].expand(-1, -1, 2),
            )
            upper_xy = torch.gather(
                path_xy,
                1,
                upper[..., None].expand(-1, -1, 2),
            )
            path_points = lower_xy + alpha * (upper_xy - lower_xy)
            cost = torch.linalg.norm(path_points - human_points[None, :, :], dim=-1).mean(dim=-1)
            long_path_costs[sample_index] = cost.numpy().astype(np.float32)

            top_cost, top_indices = torch.topk(cost, k=topk, largest=False)
            top_indices_np = top_indices.numpy().astype(np.int64)
            long_path_indices[sample_index, :topk] = top_indices_np
            long_path_weights[sample_index, :topk] = softmax_numpy(
                -top_cost.numpy().astype(np.float32) / max(self.long_path_temperature, 1.0e-6)
            )
            long_path_valid[sample_index] = True

        return long_path_indices, long_path_weights, long_path_costs, long_path_valid

    def __len__(self) -> int:
        return len(self.samples)

    def build_target_trajectory(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[index]
        ego_history = sample["ego_history"]
        times = sample["times"]
        step_index = sample["step_index"]
        current_state = ego_history[step_index]
        current_time = float(times[step_index])
        query_times = current_time + FUTURE_TIMES

        future_states = interp_columns(times, ego_history, query_times)
        current_xy = current_state[:2]
        current_yaw = float(current_state[2])
        future_xy = transform_points_to_ego(
            future_states[:, :2],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )
        future_yaw = future_states[:, 2] - current_yaw
        target_trajectory = np.concatenate(
            [future_xy, future_yaw[:, None]],
            axis=-1,
        ).astype(np.float32)
        target_mask = np.ones((len(FUTURE_TIMES),), dtype=np.float32)
        return target_trajectory, target_mask

    def build_long_path_target(self, index: int) -> tuple[np.ndarray, bool]:
        sample = self.samples[index]
        ego_history = sample["ego_history"]
        times = sample["times"]
        step_index = sample["step_index"]
        current_state = ego_history[step_index]
        current_time = float(times[step_index])
        end_time = min(
            current_time + self.long_path_horizon,
            float(times[-1]),
        )

        future_mask = (times >= current_time - 1.0e-6) & (times <= end_time + 1.0e-6)
        future_states = ego_history[future_mask]
        if len(future_states) < 2:
            return np.zeros((0, 2), dtype=np.float32), False

        current_xy = current_state[:2]
        current_yaw = float(current_state[2])
        future_xy = transform_points_to_ego(
            future_states[:, :2],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )
        progress = float(cumulative_distance(future_xy)[-1])
        valid = progress >= self.long_path_min_progress
        return future_xy.astype(np.float32), valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        data = sample["episode"]
        ego_history = sample["ego_history"]
        times = sample["times"]
        step_index = sample["step_index"]
        current_state = ego_history[step_index]
        current_time = float(times[step_index])

        current_xy = current_state[:2]
        current_yaw = float(current_state[2])
        current_speed = float(current_state[3])

        route_path = np.asarray(data["route_path"], dtype=np.float32)
        route_xy = transform_points_to_ego(
            route_path[:, :2],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )
        route_yaw = route_path[:, 2] - current_yaw
        route_path_ego = np.concatenate([route_xy, route_yaw[:, None]], axis=-1)
        goal_xy = transform_points_to_ego(
            np.asarray(data["goal_xy"], dtype=np.float32)[None],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )[0]

        obstacles = make_obstacles_in_current_ego_frame(
            raw_obstacles=data["obstacles"],
            current_time=current_time,
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )

        input_grid = make_empty_grid(channels=4, config=self.grid_config)
        rasterize_dynamic_obstacles(input_grid, obstacles, self.grid_config)
        draw_polyline(
            input_grid[2],
            route_path_ego[:, :2],
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

        target_trajectory, target_trajectory_mask = self.build_target_trajectory(index)
        path_index = int(self.path_indices[index])
        velocity_index = int(self.velocity_indices[index])
        target_path = self.vocab.path[path_index].cpu().numpy()
        target_velocity = self.vocab.velocity[velocity_index].cpu().numpy()

        obstacle_centers, obstacle_sizes, obstacle_velocities, obstacle_mask = (
            obstacles_to_tensors(obstacles, max_obstacles=12)
        )
        ego_state = np.array(
            [
                current_speed,
                float(data["ego_size_xy"][0]),
                float(data["ego_size_xy"][1]),
            ],
            dtype=np.float32,
        )

        return {
            "input_grid": torch.from_numpy(input_grid).float(),
            "target_path": torch.from_numpy(target_path).float(),
            "target_path_mask": torch.ones(self.vocab.len_path).float(),
            "target_velocity": torch.from_numpy(target_velocity).float(),
            "target_trajectory": torch.from_numpy(target_trajectory).float(),
            "target_trajectory_mask": torch.from_numpy(target_trajectory_mask).float(),
            "path_index": torch.tensor(path_index, dtype=torch.long),
            "velocity_index": torch.tensor(velocity_index, dtype=torch.long),
            "trajectory_index": torch.tensor(
                path_index * self.vocab.num_velocity + velocity_index,
                dtype=torch.long,
            ),
            "long_path_indices": torch.from_numpy(self.long_path_indices[index]).long(),
            "long_path_weights": torch.from_numpy(self.long_path_weights[index]).float(),
            "long_path_costs": torch.from_numpy(self.long_path_costs[index]).float(),
            "long_path_valid": torch.tensor(
                bool(self.long_path_valid[index]),
                dtype=torch.bool,
            ),
            "route_path": torch.from_numpy(route_path_ego).float(),
            "goal_xy": torch.from_numpy(goal_xy).float(),
            "ego_state": torch.from_numpy(ego_state).float(),
            "obstacle_centers": obstacle_centers.float(),
            "obstacle_sizes": obstacle_sizes.float(),
            "obstacle_velocities": obstacle_velocities.float(),
            "obstacle_mask": obstacle_mask.float(),
        }


if __name__ == "__main__":
    episode_dir = CURRENT_DIR / "episodes"
    paths = sorted(episode_dir.glob("*.json"))[:4]
    dataset = HumanDriveDataset(
        episode_paths=paths,
        cache_path=CURRENT_DIR / "cache" / "human_dataset_debug.npz",
        rebuild_cache=True,
    )
    print("episodes:", len(paths))
    print("samples:", len(dataset))
    sample = dataset[0]
    for key, value in sample.items():
        print(key, tuple(value.shape), value.dtype)
