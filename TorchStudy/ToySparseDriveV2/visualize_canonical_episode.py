from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from dataset.nuplan_canonical_dataset import (
    NuPlanCanonicalDataset,
    RASTER_CHANNELS,
    transform_points_to_ego,
)

DEFAULT_EPISODE_DIR = Path("outputs/nuplan_canonical/episodes")
DEFAULT_OUTPUT_DIR = Path("outputs/nuplan_canonical/preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one canonical nuPlan-style episode and its raster sample."
    )
    parser.add_argument("--episode-path", type=Path, default=None)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-agents", type=int, default=80)
    return parser.parse_args()


def find_episode_path(args: argparse.Namespace) -> Path:
    if args.episode_path is not None:
        if not args.episode_path.is_file():
            raise FileNotFoundError(args.episode_path)
        return args.episode_path

    paths = sorted(args.episode_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No canonical episode json found under {args.episode_dir}")
    index = min(max(int(args.episode_index), 0), len(paths) - 1)
    return paths[index]


def agent_state_at_time(
    agent: dict[str, Any],
    current_time: float,
    max_dt: float,
) -> list[float] | None:
    states = agent.get("states", [])
    if not states:
        return None
    nearest = min(states, key=lambda state: abs(float(state[0]) - current_time))
    if abs(float(nearest[0]) - current_time) > max_dt:
        return None
    if len(nearest) >= 7 and float(nearest[6]) < 0.5:
        return None
    return nearest


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    *,
    color: str,
    alpha: float,
    label: str | None = None,
) -> None:
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label=label,
        zorder=4,
    )
    transform = matplotlib.transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)


def raster_overlay(input_grid: np.ndarray) -> np.ndarray:
    route = input_grid[RASTER_CHANNELS["route"]]
    goal = input_grid[RASTER_CHANNELS["goal"]]
    ego = input_grid[RASTER_CHANNELS["ego_history"]]
    vehicle = input_grid[RASTER_CHANNELS["vehicle_current"]]
    vehicle_history = input_grid[RASTER_CHANNELS["vehicle_history"]]
    static = input_grid[RASTER_CHANNELS["static_obstacle_current"]]
    drivable = input_grid[RASTER_CHANNELS["drivable_area"]]
    lane = input_grid[RASTER_CHANNELS["lane_centerline"]]
    boundary = input_grid[RASTER_CHANNELS["lane_boundary"]]

    rgb = np.zeros((*route.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.maximum.reduce([vehicle, static, goal])
    rgb[..., 1] = np.maximum.reduce([route, ego, lane])
    rgb[..., 2] = np.maximum.reduce([drivable * 0.35, boundary, vehicle_history])
    return np.clip(rgb, 0.0, 1.0)


def raster_agents(input_grid: np.ndarray) -> np.ndarray:
    vehicle = input_grid[RASTER_CHANNELS["vehicle_current"]]
    vehicle_history = input_grid[RASTER_CHANNELS["vehicle_history"]]
    pedestrian = input_grid[RASTER_CHANNELS["pedestrian_bicycle_current"]]
    static = input_grid[RASTER_CHANNELS["static_obstacle_current"]]
    rgb = np.zeros((*vehicle.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.maximum.reduce([vehicle, pedestrian, static])
    rgb[..., 1] = pedestrian
    rgb[..., 2] = vehicle_history
    return np.clip(rgb, 0.0, 1.0)


def visualize_episode(episode_path: Path, args: argparse.Namespace) -> Path:
    data = json.loads(episode_path.read_text())
    ego_history = np.asarray(data["ego_history"], dtype=np.float32)
    route_path = np.asarray(data.get("route_path", ego_history[:, :3]), dtype=np.float32)
    goal_xy = np.asarray(data.get("goal_xy", route_path[-1, :2]), dtype=np.float32)
    dt = float(data.get("dt", 0.1))
    times = np.arange(len(ego_history), dtype=np.float32) * dt

    dataset = NuPlanCanonicalDataset([episode_path])
    sample_index = min(max(int(args.sample_index), 0), len(dataset) - 1)
    sample = dataset[sample_index]
    current_time = float(times[sample_index])
    current_state = ego_history[sample_index]
    current_xy = current_state[:2]
    current_yaw = float(current_state[2])

    current_agents = []
    for agent in data.get("agents", []):
        state = agent_state_at_time(agent, current_time=current_time, max_dt=max(0.15, dt * 1.5))
        if state is None:
            continue
        current_agents.append((agent, state))
    current_agents = current_agents[: max(0, int(args.max_agents))]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    ax_global, ax_ego, ax_overlay = axes[0]
    ax_route_raster, ax_lane_raster, ax_agent_raster = axes[1]

    past_slice = slice(0, sample_index + 1)
    future_slice = slice(sample_index, None)
    ax_global.plot(
        ego_history[:, 0],
        ego_history[:, 1],
        color="#bbbbbb",
        lw=1.0,
        alpha=0.45,
        label="full ego",
    )
    ax_global.plot(
        ego_history[past_slice, 0],
        ego_history[past_slice, 1],
        color="#2ca02c",
        lw=2.0,
        label="ego past",
    )
    ax_global.plot(
        ego_history[future_slice, 0],
        ego_history[future_slice, 1],
        color="#ff7f0e",
        lw=2.0,
        label="ego future",
    )
    ax_global.plot(
        route_path[:, 0],
        route_path[:, 1],
        color="#1f77b4",
        lw=2.6,
        linestyle="--",
        alpha=0.9,
        label="route",
        zorder=7,
    )
    ax_global.scatter([current_xy[0]], [current_xy[1]], color="#2ca02c", s=60, zorder=8, label="current")
    ax_global.scatter([goal_xy[0]], [goal_xy[1]], color="#d62728", marker="*", s=140, zorder=9, label="goal")
    for agent, state in current_agents:
        length, width = agent.get("size", [4.5, 2.0])
        color = "#d62728" if agent.get("type") != "vehicle" else "#777777"
        draw_box(
            ax_global,
            float(state[1]),
            float(state[2]),
            float(state[3]),
            float(length),
            float(width),
            color=color,
            alpha=0.18,
        )
    ax_global.set_title("global trajectory")
    ax_global.set_aspect("equal", adjustable="box")
    ax_global.grid(True, alpha=0.25)
    ax_global.legend(loc="best")

    route_ego = transform_points_to_ego(route_path[:, :2], current_xy, current_yaw)
    ego_ego = transform_points_to_ego(ego_history[:, :2], current_xy, current_yaw)
    goal_ego = transform_points_to_ego(goal_xy[None], current_xy, current_yaw)[0]
    ax_ego.plot(
        ego_ego[:, 1],
        ego_ego[:, 0],
        color="#bbbbbb",
        lw=1.0,
        alpha=0.4,
        label="full ego",
    )
    ax_ego.plot(
        ego_ego[past_slice, 1],
        ego_ego[past_slice, 0],
        color="#2ca02c",
        lw=2.0,
        label="ego past",
    )
    ax_ego.plot(
        ego_ego[future_slice, 1],
        ego_ego[future_slice, 0],
        color="#ff7f0e",
        lw=2.0,
        label="ego future",
    )
    ax_ego.plot(
        route_ego[:, 1],
        route_ego[:, 0],
        color="#1f77b4",
        lw=2.8,
        linestyle="--",
        alpha=0.95,
        label="route",
        zorder=8,
    )
    ax_ego.scatter([0.0], [0.0], color="#2ca02c", s=60, zorder=8, label="current")
    ax_ego.scatter([goal_ego[1]], [goal_ego[0]], color="#d62728", marker="*", s=140, zorder=9, label="goal")
    for agent, state in current_agents:
        local_xy = transform_points_to_ego(
            np.asarray([[float(state[1]), float(state[2])]], dtype=np.float32),
            current_xy,
            current_yaw,
        )[0]
        length, width = agent.get("size", [4.5, 2.0])
        color = "#d62728" if agent.get("type") != "vehicle" else "#777777"
        draw_box(
            ax_ego,
            float(local_xy[1]),
            float(local_xy[0]),
            float(state[3]) - current_yaw,
            float(length),
            float(width),
            color=color,
            alpha=0.28,
        )
    ax_ego.set_title("ego frame geometry")
    ax_ego.set_xlabel("y left [m]")
    ax_ego.set_ylabel("x forward [m]")
    ax_ego.set_xlim(-60, 60)
    ax_ego.set_ylim(-20, 90)
    ax_ego.set_aspect("equal", adjustable="box")
    ax_ego.grid(True, alpha=0.25)
    ax_ego.legend(loc="best")

    input_grid = sample["input_grid"].numpy()
    overlay = raster_overlay(input_grid)
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(
        "raster overlay\nred=agents/static/goal, green=route/ego/lane, blue=history/map"
    )
    ax_overlay.set_axis_off()

    route_goal = np.maximum(
        input_grid[RASTER_CHANNELS["route"]],
        input_grid[RASTER_CHANNELS["goal"]],
    )
    ax_route_raster.imshow(route_goal, cmap="Greens", vmin=0.0, vmax=1.0)
    ax_route_raster.set_title("raster channel: route + goal")
    ax_route_raster.set_axis_off()

    ax_lane_raster.imshow(
        input_grid[RASTER_CHANNELS["lane_centerline"]],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    ax_lane_raster.set_title("raster channel: route-road lane centerlines")
    ax_lane_raster.set_axis_off()

    ax_agent_raster.imshow(raster_agents(input_grid))
    ax_agent_raster.set_title("raster channels: agents\nred=current/static, blue=history")
    ax_agent_raster.set_axis_off()

    fig.suptitle(
        f"{data.get('scene_id', episode_path.stem)} | sample={sample_index} "
        f"t={current_time:.2f}s | agents={len(current_agents)} | tags={','.join(data.get('scenario_tags', [])[:4])}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{episode_path.stem}_sample_{sample_index:04d}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    episode_path = find_episode_path(args)
    output_path = visualize_episode(episode_path, args)
    print(f"episode: {episode_path}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
