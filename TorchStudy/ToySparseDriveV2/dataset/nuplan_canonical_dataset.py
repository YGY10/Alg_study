from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, CURRENT_DIR, TOY_ROOT / "vocab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .dataset import Obstacle, obstacles_to_tensors
from .grid import (
    GridConfig,
    draw_points,
    draw_polyline,
    draw_rectangle,
    grid_to_world,
    make_empty_grid,
)
from vocab import SparseDriveVocab

FUTURE_TIMES = np.arange(1, 9, dtype=np.float32) * 0.5

RASTER_CHANNELS: dict[str, int] = {
    "drivable_area": 0,
    "lane_centerline": 1,
    "lane_boundary": 2,
    "route": 3,
    "goal": 4,
    "ego_history": 5,
    "vehicle_current": 6,
    "vehicle_history": 7,
    "vehicle_velocity_x": 8,
    "vehicle_velocity_y": 9,
    "pedestrian_bicycle_current": 10,
    "pedestrian_bicycle_history": 11,
    "static_obstacle_current": 12,
    "red_light_stopline": 13,
    "green_light_stopline": 14,
    "stop_sign_stopline": 15,
    "crosswalk": 16,
    "intersection": 17,
}
NUM_RASTER_CHANNELS = len(RASTER_CHANNELS)

VEHICLE_TYPES = {"vehicle", "car", "truck", "bus", "trailer"}
PEDESTRIAN_BICYCLE_TYPES = {"pedestrian", "bicycle", "cyclist", "motorcycle"}
STATIC_TYPES = {
    "traffic_cone",
    "barrier",
    "czone_sign",
    "construction_zone_sign",
    "generic_static",
    "static",
}


@dataclass(frozen=True)
class NuPlanSampleMeta:
    episode_path: Path
    step_index: int
    current_time: float
    scene_id: str


def angle_wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_world_to_ego(yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray(
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


def as_polyline_list(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return [value.astype(np.float32)]
        return [item.astype(np.float32) for item in value]
    if not value:
        return []
    first = value[0]
    if isinstance(first, (int, float)):
        return [np.asarray(value, dtype=np.float32)]
    if len(first) > 0 and isinstance(first[0], (int, float)):
        return [np.asarray(value, dtype=np.float32)]
    return [np.asarray(item, dtype=np.float32) for item in value if len(item) > 0]


def interp_columns(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.interp(query_times, times, values[:, column])
            for column in range(values.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)


def polygon_mask(poly_xy: np.ndarray, config: GridConfig) -> np.ndarray:
    if len(poly_xy) < 3:
        return np.zeros((config.height, config.width), dtype=bool)
    rows = np.arange(config.height, dtype=np.float32)
    cols = np.arange(config.width, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xy = grid_to_world(rr, cc, config)
    x = xy[..., 0]
    y = xy[..., 1]
    px = poly_xy[:, 0]
    py = poly_xy[:, 1]
    inside = np.zeros((config.height, config.width), dtype=bool)
    j = len(poly_xy) - 1
    for i in range(len(poly_xy)):
        crosses = ((py[i] > y) != (py[j] > y)) & (
            x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1.0e-8) + px[i]
        )
        inside ^= crosses
        j = i
    return inside


def draw_polylines(
    grid: np.ndarray,
    polylines: list[np.ndarray],
    *,
    value: float = 1.0,
    radius: int = 1,
    samples_per_segment: int = 4,
    config: GridConfig,
) -> None:
    for polyline in polylines:
        if len(polyline) == 0:
            continue
        draw_polyline(
            grid,
            polyline[:, :2],
            value=value,
            radius=radius,
            samples_per_segment=samples_per_segment,
            config=config,
        )


def draw_polygons(
    grid: np.ndarray,
    polygons: list[np.ndarray],
    *,
    value: float = 1.0,
    outline_radius: int = 1,
    config: GridConfig,
) -> None:
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        grid[polygon_mask(polygon[:, :2], config)] = value
        draw_polyline(
            grid,
            np.concatenate([polygon[:, :2], polygon[:1, :2]], axis=0),
            value=value,
            radius=outline_radius,
            samples_per_segment=4,
            config=config,
        )


class NuPlanCanonicalDataset(Dataset):
    """Dataset for canonical nuPlan-style planning episodes.

    The canonical JSON keeps global coordinates. This dataset builds one training
    sample per ego timestep, transforms map/agents/route into the current ego
    frame, and returns the same high-level fields consumed by ToySparseDriveV2.
    """

    def __init__(
        self,
        episode_paths: list[str | Path],
        vocab: SparseDriveVocab | None = None,
        grid_config: GridConfig = GridConfig(),
        future_times: np.ndarray = FUTURE_TIMES,
        history_sec: float = 2.0,
        long_path_horizon: float = 8.0,
        long_path_compare_points: int = 32,
        long_path_positive_topk: int = 8,
        long_path_temperature: float = 2.0,
        max_obstacles: int = 32,
    ) -> None:
        self.episode_paths = [Path(path) for path in episode_paths]
        self.vocab = vocab or SparseDriveVocab.load()
        self.grid_config = grid_config
        self.future_times = np.asarray(future_times, dtype=np.float32)
        self.history_sec = float(history_sec)
        self.long_path_horizon = float(long_path_horizon)
        self.long_path_compare_points = int(long_path_compare_points)
        self.long_path_positive_topk = int(long_path_positive_topk)
        self.long_path_temperature = float(long_path_temperature)
        self.max_obstacles = int(max_obstacles)

        self.samples: list[dict[str, Any]] = []
        self.metas: list[NuPlanSampleMeta] = []
        self._build_samples()
        self._build_labels()

    def _build_samples(self) -> None:
        for episode_path in self.episode_paths:
            data = json.loads(episode_path.read_text())
            ego_history = np.asarray(data["ego_history"], dtype=np.float32)
            dt = float(data.get("dt", 0.1))
            times = np.arange(len(ego_history), dtype=np.float32) * dt
            last_time = float(times[-1])
            scene_id = str(data.get("scene_id", episode_path.stem))
            for step_index in range(len(ego_history)):
                current_time = float(times[step_index])
                if current_time + float(self.future_times[-1]) > last_time + 1.0e-4:
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
                    NuPlanSampleMeta(
                        episode_path=episode_path,
                        step_index=step_index,
                        current_time=current_time,
                        scene_id=scene_id,
                    )
                )

    def _build_labels(self) -> None:
        num_samples = len(self.samples)
        self.path_indices = np.zeros((num_samples,), dtype=np.int64)
        self.velocity_indices = np.zeros((num_samples,), dtype=np.int64)
        self.long_path_indices = np.zeros(
            (num_samples, self.long_path_positive_topk),
            dtype=np.int64,
        )
        self.long_path_weights = np.full(
            (num_samples, self.long_path_positive_topk),
            1.0 / max(self.long_path_positive_topk, 1),
            dtype=np.float32,
        )
        self.long_path_costs = np.full(
            (num_samples, self.vocab.num_path),
            np.inf,
            dtype=np.float32,
        )
        trajectory_vocab = self.vocab.trajectory.cpu().float()
        path_xy = self.vocab.path[..., :2].cpu().float()
        path_distance = torch.stack(
            [
                torch.from_numpy(cumulative_distance(path_xy[path_index].numpy()))
                for path_index in range(self.vocab.num_path)
            ],
            dim=0,
        ).float()

        with torch.no_grad():
            for sample_index in range(num_samples):
                target_trajectory, target_mask = self.build_target_trajectory(sample_index)
                target_t = torch.from_numpy(target_trajectory).float()
                mask_t = torch.from_numpy(target_mask).float()
                diff = trajectory_vocab[..., :2] - target_t[None, None, :, :2]
                dist = diff.pow(2).sum(dim=-1)
                dist = dist * mask_t[None, None, :]
                dist = dist.sum(dim=-1) / mask_t.sum().clamp(min=1.0)
                flat_index = int(torch.argmin(dist.reshape(-1)).item())
                self.path_indices[sample_index] = flat_index // self.vocab.num_velocity
                self.velocity_indices[sample_index] = flat_index % self.vocab.num_velocity

                long_xy = self.build_long_path_target(sample_index)
                human_distance = cumulative_distance(long_xy)
                if len(long_xy) < 2 or float(human_distance[-1]) <= 1.0e-6:
                    self.long_path_indices[sample_index, :] = self.path_indices[sample_index]
                    self.long_path_costs[sample_index, self.path_indices[sample_index]] = 0.0
                    continue

                query_distance_np = np.linspace(
                    0.0,
                    float(human_distance[-1]),
                    self.long_path_compare_points,
                    dtype=np.float32,
                )
                human_points = torch.from_numpy(
                    interp_points_by_distance(long_xy, query_distance_np)
                ).float()
                query_distance = torch.from_numpy(query_distance_np).float()
                query = query_distance[None, :].expand(self.vocab.num_path, -1)
                query = torch.minimum(query, path_distance[:, -1:].expand_as(query))
                upper = torch.searchsorted(path_distance.contiguous(), query.contiguous())
                upper = upper.clamp(min=1, max=path_distance.shape[1] - 1)
                lower = upper - 1
                s0 = torch.gather(path_distance, 1, lower)
                s1 = torch.gather(path_distance, 1, upper)
                alpha = ((query - s0) / (s1 - s0).clamp(min=1.0e-6)).unsqueeze(-1)
                lower_xy = torch.gather(path_xy, 1, lower[..., None].expand(-1, -1, 2))
                upper_xy = torch.gather(path_xy, 1, upper[..., None].expand(-1, -1, 2))
                path_points = lower_xy + alpha * (upper_xy - lower_xy)
                cost = torch.linalg.norm(path_points - human_points[None, :, :], dim=-1).mean(dim=-1)
                self.long_path_costs[sample_index] = cost.numpy().astype(np.float32)
                top_cost, top_indices = torch.topk(
                    cost,
                    k=min(self.long_path_positive_topk, self.vocab.num_path),
                    largest=False,
                )
                top_indices_np = top_indices.numpy().astype(np.int64)
                self.long_path_indices[sample_index, : len(top_indices_np)] = top_indices_np
                self.long_path_weights[sample_index, : len(top_indices_np)] = softmax_numpy(
                    -top_cost.numpy().astype(np.float32) / max(self.long_path_temperature, 1.0e-6)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def current_state(self, index: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        sample = self.samples[index]
        ego_history = sample["ego_history"]
        times = sample["times"]
        step_index = int(sample["step_index"])
        state = ego_history[step_index]
        current_xy = state[:2]
        current_yaw = float(state[2])
        current_speed = float(state[3]) if state.shape[0] > 3 else 0.0
        current_time = float(times[step_index])
        return state, current_xy, current_yaw, current_speed, current_time

    def build_target_trajectory(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[index]
        ego_history = sample["ego_history"]
        times = sample["times"]
        _, current_xy, current_yaw, _, current_time = self.current_state(index)
        future_states = interp_columns(times, ego_history, current_time + self.future_times)
        future_xy = transform_points_to_ego(
            future_states[:, :2],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        )
        future_yaw = angle_wrap(future_states[:, 2] - current_yaw)
        target_trajectory = np.concatenate([future_xy, future_yaw[:, None]], axis=-1)
        target_mask = np.ones((len(self.future_times),), dtype=np.float32)
        return target_trajectory.astype(np.float32), target_mask

    def build_long_path_target(self, index: int) -> np.ndarray:
        sample = self.samples[index]
        ego_history = sample["ego_history"]
        times = sample["times"]
        _, current_xy, current_yaw, _, current_time = self.current_state(index)
        end_time = min(current_time + self.long_path_horizon, float(times[-1]))
        mask = (times >= current_time - 1.0e-6) & (times <= end_time + 1.0e-6)
        future_states = ego_history[mask]
        if len(future_states) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return transform_points_to_ego(
            future_states[:, :2],
            ego_xy=current_xy,
            ego_yaw=current_yaw,
        ).astype(np.float32)

    def _agent_state_at_time(
        self,
        states: np.ndarray,
        current_time: float,
        max_dt: float,
    ) -> np.ndarray | None:
        if len(states) == 0:
            return None
        valid = states[:, -1] > 0.5 if states.shape[1] >= 7 else np.ones((len(states),), dtype=bool)
        if not bool(valid.any()):
            return None
        valid_states = states[valid]
        nearest_index = int(np.argmin(np.abs(valid_states[:, 0] - current_time)))
        if abs(float(valid_states[nearest_index, 0]) - current_time) > max_dt:
            return None
        return valid_states[nearest_index]

    def _agents_to_obstacles_and_raster(
        self,
        input_grid: np.ndarray,
        agents: list[dict[str, Any]],
        current_time: float,
        current_xy: np.ndarray,
        current_yaw: float,
        dt: float,
    ) -> list[Obstacle]:
        obstacles: list[Obstacle] = []
        history_start = current_time - self.history_sec
        for agent in agents:
            states = np.asarray(agent.get("states", []), dtype=np.float32)
            state = self._agent_state_at_time(states, current_time, max_dt=max(0.15, dt * 1.5))
            if state is None:
                continue
            agent_type = str(agent.get("type", "vehicle")).lower()
            size = agent.get("size", agent.get("size_xy", [4.5, 2.0]))
            size_xy = (float(size[0]), float(size[1]))
            center_xy = transform_points_to_ego(
                state[1:3][None],
                ego_xy=current_xy,
                ego_yaw=current_yaw,
            )[0]
            velocity_xy = transform_vectors_to_ego(
                state[4:6][None],
                ego_yaw=current_yaw,
            )[0]
            obstacle = Obstacle(
                center_xy=center_xy.astype(np.float32),
                size_xy=size_xy,
                velocity_xy=velocity_xy.astype(np.float32),
            )
            obstacles.append(obstacle)
            speed = float(np.linalg.norm(velocity_xy))
            if agent_type in VEHICLE_TYPES:
                draw_rectangle(input_grid[RASTER_CHANNELS["vehicle_current"]], center_xy, size_xy, config=self.grid_config)
                draw_rectangle(
                    input_grid[RASTER_CHANNELS["vehicle_velocity_x"]],
                    center_xy,
                    size_xy,
                    value=float(np.clip(velocity_xy[0] / 15.0, -1.0, 1.0)),
                    config=self.grid_config,
                )
                draw_rectangle(
                    input_grid[RASTER_CHANNELS["vehicle_velocity_y"]],
                    center_xy,
                    size_xy,
                    value=float(np.clip(velocity_xy[1] / 15.0, -1.0, 1.0)),
                    config=self.grid_config,
                )
                history_channel = RASTER_CHANNELS["vehicle_history"]
            elif agent_type in PEDESTRIAN_BICYCLE_TYPES:
                draw_rectangle(input_grid[RASTER_CHANNELS["pedestrian_bicycle_current"]], center_xy, size_xy, config=self.grid_config)
                history_channel = RASTER_CHANNELS["pedestrian_bicycle_history"]
            elif agent_type in STATIC_TYPES or speed < 0.2:
                draw_rectangle(input_grid[RASTER_CHANNELS["static_obstacle_current"]], center_xy, size_xy, config=self.grid_config)
                history_channel = -1
            else:
                draw_rectangle(input_grid[RASTER_CHANNELS["vehicle_current"]], center_xy, size_xy, config=self.grid_config)
                history_channel = RASTER_CHANNELS["vehicle_history"]

            if history_channel >= 0 and len(states) > 0:
                history_mask = (
                    (states[:, 0] >= history_start - 1.0e-6)
                    & (states[:, 0] <= current_time + 1.0e-6)
                )
                if states.shape[1] >= 7:
                    history_mask &= states[:, -1] > 0.5
                history_xy = states[history_mask, 1:3]
                if len(history_xy) > 0:
                    history_local = transform_points_to_ego(
                        history_xy,
                        ego_xy=current_xy,
                        ego_yaw=current_yaw,
                    )
                    draw_polyline(
                        input_grid[history_channel],
                        history_local,
                        value=1.0,
                        radius=1,
                        samples_per_segment=2,
                        config=self.grid_config,
                    )
        return obstacles

    def _rasterize_map(
        self,
        input_grid: np.ndarray,
        data: dict[str, Any],
        current_xy: np.ndarray,
        current_yaw: float,
    ) -> None:
        map_data = data.get("map", {})

        def to_ego_polylines(key: str) -> list[np.ndarray]:
            return [
                transform_points_to_ego(polyline[:, :2], current_xy, current_yaw)
                for polyline in as_polyline_list(map_data.get(key))
            ]

        draw_polygons(
            input_grid[RASTER_CHANNELS["drivable_area"]],
            to_ego_polylines("drivable_area"),
            config=self.grid_config,
        )
        route_polygons = to_ego_polylines("route_polygons")
        draw_polygons(
            input_grid[RASTER_CHANNELS["route"]],
            route_polygons,
            config=self.grid_config,
        )
        draw_polylines(
            input_grid[RASTER_CHANNELS["lane_centerline"]],
            to_ego_polylines("lane_centerlines"),
            radius=1,
            config=self.grid_config,
        )
        draw_polylines(
            input_grid[RASTER_CHANNELS["lane_boundary"]],
            to_ego_polylines("lane_boundaries"),
            radius=1,
            config=self.grid_config,
        )
        draw_polygons(
            input_grid[RASTER_CHANNELS["crosswalk"]],
            to_ego_polylines("crosswalks"),
            config=self.grid_config,
        )
        draw_polygons(
            input_grid[RASTER_CHANNELS["intersection"]],
            to_ego_polylines("intersections"),
            config=self.grid_config,
        )
        for stop_line in map_data.get("stop_lines", []):
            points = np.asarray(stop_line.get("points", stop_line), dtype=np.float32)
            if len(points) == 0:
                continue
            stop_type = str(stop_line.get("type", "stop_sign")).lower() if isinstance(stop_line, dict) else "stop_sign"
            channel_name = "stop_sign_stopline"
            if "red" in stop_type:
                channel_name = "red_light_stopline"
            elif "green" in stop_type:
                channel_name = "green_light_stopline"
            points_ego = transform_points_to_ego(points[:, :2], current_xy, current_yaw)
            draw_polyline(
                input_grid[RASTER_CHANNELS[channel_name]],
                points_ego,
                value=1.0,
                radius=1,
                samples_per_segment=4,
                config=self.grid_config,
            )

    def _scalar_features(
        self,
        state: np.ndarray,
        ego_history: np.ndarray,
        times: np.ndarray,
        step_index: int,
        route_path_ego: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: list[Obstacle],
    ) -> np.ndarray:
        speed = float(state[3]) if state.shape[0] > 3 else 0.0
        accel = float(state[4]) if state.shape[0] > 4 else 0.0
        yaw_rate = float(state[5]) if state.shape[0] > 5 else 0.0
        if state.shape[0] <= 5 and step_index > 0:
            dt = max(float(times[step_index] - times[step_index - 1]), 1.0e-3)
            prev = ego_history[step_index - 1]
            prev_speed = float(prev[3]) if prev.shape[0] > 3 else speed
            accel = (speed - prev_speed) / dt
            yaw_rate = float(angle_wrap(float(state[2]) - float(prev[2])) / dt)

        route_heading_error = 0.0
        if len(route_path_ego) >= 2:
            delta = route_path_ego[min(3, len(route_path_ego) - 1), :2] - route_path_ego[0, :2]
            route_heading_error = math.atan2(float(delta[1]), float(delta[0]))

        front_distance = 80.0
        front_speed = 0.0
        for obstacle in obstacles:
            x = float(obstacle.center_xy[0])
            y = float(obstacle.center_xy[1])
            if x <= 0.0 or abs(y) > 4.0:
                continue
            if x < front_distance:
                front_distance = x
                front_speed = float(obstacle.velocity_xy[0])

        return np.asarray(
            [
                speed / 20.0,
                np.clip(accel / 8.0, -1.0, 1.0),
                np.clip(yaw_rate / 1.5, -1.0, 1.0),
                np.clip(np.linalg.norm(goal_xy) / 120.0, 0.0, 1.0),
                np.clip(route_heading_error / math.pi, -1.0, 1.0),
                np.clip(front_distance / 80.0, 0.0, 1.0),
                np.clip(front_speed / 20.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        data = sample["episode"]
        ego_history = sample["ego_history"]
        times = sample["times"]
        step_index = int(sample["step_index"])
        dt = float(data.get("dt", times[1] - times[0] if len(times) > 1 else 0.1))
        state, current_xy, current_yaw, _, current_time = self.current_state(index)

        route_path = np.asarray(data.get("route_path", ego_history[:, :3]), dtype=np.float32)
        route_xy = transform_points_to_ego(route_path[:, :2], current_xy, current_yaw)
        route_yaw = angle_wrap(route_path[:, 2] - current_yaw) if route_path.shape[1] >= 3 else np.zeros((len(route_path),), dtype=np.float32)
        route_path_ego = np.concatenate([route_xy, route_yaw[:, None]], axis=-1).astype(np.float32)
        goal_global = np.asarray(data.get("goal_xy", route_path[-1, :2]), dtype=np.float32)
        goal_xy = transform_points_to_ego(goal_global[None], current_xy, current_yaw)[0]

        input_grid = make_empty_grid(channels=NUM_RASTER_CHANNELS, config=self.grid_config)
        self._rasterize_map(input_grid, data, current_xy, current_yaw)
        if not data.get("map", {}).get("route_polygons"):
            draw_polyline(
                input_grid[RASTER_CHANNELS["route"]],
                route_path_ego[:, :2],
                value=1.0,
                radius=1,
                samples_per_segment=4,
                config=self.grid_config,
            )
        draw_points(input_grid[RASTER_CHANNELS["goal"]], goal_xy[None], radius=3, config=self.grid_config)

        history_mask = (
            (times >= current_time - self.history_sec - 1.0e-6)
            & (times <= current_time + 1.0e-6)
        )
        ego_history_local = transform_points_to_ego(
            ego_history[history_mask, :2],
            current_xy,
            current_yaw,
        )
        draw_polyline(
            input_grid[RASTER_CHANNELS["ego_history"]],
            ego_history_local,
            value=1.0,
            radius=1,
            samples_per_segment=2,
            config=self.grid_config,
        )

        obstacles = self._agents_to_obstacles_and_raster(
            input_grid=input_grid,
            agents=list(data.get("agents", [])),
            current_time=current_time,
            current_xy=current_xy,
            current_yaw=current_yaw,
            dt=dt,
        )
        obstacle_centers, obstacle_sizes, obstacle_velocities, obstacle_mask = obstacles_to_tensors(
            obstacles,
            max_obstacles=self.max_obstacles,
        )

        target_trajectory, target_trajectory_mask = self.build_target_trajectory(index)
        path_index = int(self.path_indices[index])
        velocity_index = int(self.velocity_indices[index])
        target_path = self.vocab.path[path_index].cpu().numpy()
        target_velocity = self.vocab.velocity[velocity_index].cpu().numpy()
        ego_state = self._scalar_features(
            state=state,
            ego_history=ego_history,
            times=times,
            step_index=step_index,
            route_path_ego=route_path_ego,
            goal_xy=goal_xy,
            obstacles=obstacles,
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
            "long_path_valid": torch.tensor(True, dtype=torch.bool),
            "route_path": torch.from_numpy(route_path_ego).float(),
            "goal_xy": torch.from_numpy(goal_xy.astype(np.float32)).float(),
            "ego_state": torch.from_numpy(ego_state).float(),
            "obstacle_centers": obstacle_centers.float(),
            "obstacle_sizes": obstacle_sizes.float(),
            "obstacle_velocities": obstacle_velocities.float(),
            "obstacle_mask": obstacle_mask.float(),
        }
