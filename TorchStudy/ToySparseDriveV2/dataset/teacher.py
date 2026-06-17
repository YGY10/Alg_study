from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, TOY_ROOT / "vocab", CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grid import GridConfig
from vocab import SparseDriveVocab

TIME_STEPS = np.arange(1, 9, dtype=np.float32) * 0.5
OUTPUT_DIR = TOY_ROOT / "outputs" / "teacher"


@dataclass(frozen=True)
class DynamicObstacle:
    center_xy: np.ndarray
    size_xy: tuple[float, float]
    velocity_xy: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicObstacle":
        return cls(
            center_xy=np.asarray(data["center_xy"], dtype=np.float32),
            size_xy=(float(data["size_xy"][0]), float(data["size_xy"][1])),
            velocity_xy=np.asarray(data["velocity_xy"], dtype=np.float32),
        )

    def center_at(self, time_s: np.ndarray | float) -> np.ndarray:
        return self.center_xy + self.velocity_xy * np.asarray(time_s)[..., None]


@dataclass(frozen=True)
class EgoState:
    xy: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    speed: float = 0.0
    size_xy: tuple[float, float] = (4.8, 2.0)


@dataclass(frozen=True)
class TeacherConfig:
    num_path_candidates: int = 32
    num_top_trajectories: int = 16
    temperature: float = 2.0
    segment_samples: int = 6
    safety_margin: float = 0.5
    desired_clearance: float = 2.0
    collision_weight: float = 1.0e6
    path_goal_weight: float = 1.0
    path_route_weight: float = 0.5
    path_collision_weight: float = 1000.0
    goal_weight: float = 10
    route_weight: float = 0.8
    clearance_weight: float = 8.0
    progress_weight: float = 0.25
    comfort_weight: float = 0.15
    speed_weight: float = 0.02
    initial_accel_weight: float = 0.2
    accel_weight: float = 0.08
    jerk_weight: float = 0.02


@dataclass(frozen=True)
class TeacherOutput:
    path_indices: np.ndarray
    path_costs: np.ndarray
    candidate_flat_indices: np.ndarray
    candidate_path_indices: np.ndarray
    candidate_velocity_indices: np.ndarray
    candidate_costs: np.ndarray
    candidate_scores: np.ndarray
    candidate_probs: np.ndarray
    best_candidate_index: int
    best_flat_index: int
    best_path_index: int
    best_velocity_index: int
    topk_flat_indices: np.ndarray
    debug: dict[str, Any]


def normalize_obstacles(
    obstacles: list[DynamicObstacle | dict[str, Any]],
) -> list[DynamicObstacle]:
    normalized = []
    for obstacle in obstacles:
        if isinstance(obstacle, DynamicObstacle):
            normalized.append(obstacle)
        else:
            normalized.append(DynamicObstacle.from_dict(obstacle))
    return normalized


def default_route_path(
    ego_state: EgoState,
    goal_xy: np.ndarray,
    num_points: int = 64,
) -> np.ndarray:
    start_xy = np.asarray(ego_state.xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    return start_xy[None] * (1.0 - alpha[:, None]) + goal_xy[None] * alpha[:, None]


def pairwise_nearest_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    diff = points[..., None, :] - reference[None, None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    return distance.min(axis=-1)


def softmax_from_cost(cost: np.ndarray, temperature: float) -> np.ndarray:
    score = -cost / max(float(temperature), 1.0e-6)
    score = score - score.max()
    prob = np.exp(score)
    return prob / max(float(prob.sum()), 1.0e-12)


def interpolate_trajectory_samples(
    trajectories: np.ndarray,
    masks: np.ndarray,
    ego_state: EgoState,
    segment_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_xy = np.asarray(ego_state.xy, dtype=np.float32)
    samples = []
    valid_samples = []
    sample_times = []

    prev_xy = np.broadcast_to(ego_xy[None, :], (trajectories.shape[0], 2))
    prev_time = 0.0
    for step, end_time in enumerate(TIME_STEPS):
        end_xy = trajectories[:, step, :2]
        alpha = np.linspace(
            1.0 / segment_samples,
            1.0,
            segment_samples,
            dtype=np.float32,
        )
        segment_xy = prev_xy[:, None, :] * (1.0 - alpha[None, :, None])
        segment_xy += end_xy[:, None, :] * alpha[None, :, None]
        samples.append(segment_xy)

        valid = masks[:, step] > 0.5
        valid_samples.append(np.broadcast_to(valid[:, None], segment_xy.shape[:2]))

        times = prev_time * (1.0 - alpha) + float(end_time) * alpha
        sample_times.append(times)

        prev_xy = end_xy
        prev_time = float(end_time)

    return (
        np.concatenate(samples, axis=1),
        np.concatenate(valid_samples, axis=1),
        np.concatenate(sample_times, axis=0),
    )


def expanded_obstacle_distance(
    points_xy: np.ndarray,
    times: np.ndarray,
    obstacle: DynamicObstacle,
    ego_state: EgoState,
    safety_margin: float,
) -> np.ndarray:
    centers = obstacle.center_at(times)
    obstacle_size = np.asarray(obstacle.size_xy, dtype=np.float32)
    ego_size = np.asarray(ego_state.size_xy, dtype=np.float32)
    half_size = obstacle_size * 0.5 + ego_size * 0.5 + float(safety_margin)

    diff = np.abs(points_xy - centers[None, :, :]) - half_size[None, None, :]
    outside = np.maximum(diff, 0.0)
    outside_distance = np.linalg.norm(outside, axis=-1)
    inside = (diff[..., 0] <= 0.0) & (diff[..., 1] <= 0.0)
    return np.where(inside, 0.0, outside_distance)


def compute_collision_and_clearance(
    sample_xy: np.ndarray,
    sample_valid: np.ndarray,
    sample_times: np.ndarray,
    obstacles: list[DynamicObstacle],
    ego_state: EgoState,
    safety_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(obstacles) == 0:
        inf_clearance = np.full((sample_xy.shape[0],), 1.0e6, dtype=np.float32)
        no_collision = np.zeros((sample_xy.shape[0],), dtype=bool)
        return no_collision, inf_clearance

    collision = np.zeros((sample_xy.shape[0],), dtype=bool)
    min_clearance = np.full((sample_xy.shape[0],), 1.0e6, dtype=np.float32)

    for obstacle in obstacles:
        distance = expanded_obstacle_distance(
            sample_xy,
            sample_times,
            obstacle,
            ego_state,
            safety_margin=safety_margin,
        )
        valid_distance = np.where(sample_valid, distance, 1.0e6)
        obstacle_min = valid_distance.min(axis=1)
        min_clearance = np.minimum(min_clearance, obstacle_min)
        collision |= obstacle_min <= 1.0e-6

    return collision, min_clearance


def score_paths(
    vocab: SparseDriveVocab,
    goal_xy: np.ndarray,
    obstacles: list[DynamicObstacle],
    ego_state: EgoState,
    route_path: np.ndarray,
    config: TeacherConfig,
) -> tuple[np.ndarray, np.ndarray]:
    path = vocab.path.detach().cpu().numpy()
    path_xy = path[..., :2]

    endpoint_xy = path_xy[:, -1]
    goal_cost = np.linalg.norm(endpoint_xy - goal_xy[None, :], axis=-1)

    route_dist = pairwise_nearest_distance(path_xy, route_path)
    route_cost = route_dist.mean(axis=-1)

    path_times = np.linspace(0.0, TIME_STEPS[-1], path_xy.shape[1], dtype=np.float32)
    path_valid = np.ones(path_xy.shape[:2], dtype=bool)
    collision, clearance = compute_collision_and_clearance(
        path_xy,
        path_valid,
        path_times,
        obstacles,
        ego_state,
        safety_margin=config.safety_margin,
    )
    clearance_cost = np.maximum(config.desired_clearance - clearance, 0.0)

    cost = (
        config.path_goal_weight * goal_cost
        + config.path_route_weight * route_cost
        + config.clearance_weight * clearance_cost
        + config.path_collision_weight * collision.astype(np.float32)
    )
    path_indices = np.argsort(cost)[: config.num_path_candidates]
    return path_indices.astype(np.int64), cost.astype(np.float32)


def score_trajectories(
    vocab: SparseDriveVocab,
    path_indices: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[DynamicObstacle],
    ego_state: EgoState,
    route_path: np.ndarray,
    config: TeacherConfig,
) -> dict[str, np.ndarray]:
    trajectories = vocab.trajectory[path_indices].detach().cpu().numpy()
    masks = vocab.trajectory_mask[path_indices].detach().cpu().numpy()

    num_path = len(path_indices)
    num_velocity = vocab.num_velocity
    trajectories = trajectories.reshape(num_path * num_velocity, 8, 3)
    masks = masks.reshape(num_path * num_velocity, 8)

    candidate_path_indices = np.repeat(path_indices, num_velocity)
    candidate_velocity_indices = np.tile(np.arange(num_velocity), num_path)
    candidate_flat_indices = candidate_path_indices * num_velocity
    candidate_flat_indices = candidate_flat_indices + candidate_velocity_indices

    sample_xy, sample_valid, sample_times = interpolate_trajectory_samples(
        trajectories,
        masks,
        ego_state,
        segment_samples=config.segment_samples,
    )
    collision, clearance = compute_collision_and_clearance(
        sample_xy,
        sample_valid,
        sample_times,
        obstacles,
        ego_state,
        safety_margin=config.safety_margin,
    )

    valid_count = masks.sum(axis=-1).astype(np.int64)
    valid_count = np.clip(valid_count, 1, masks.shape[1])
    endpoint_index = valid_count - 1
    endpoint_xy = trajectories[np.arange(trajectories.shape[0]), endpoint_index, :2]

    goal_cost = np.linalg.norm(endpoint_xy - goal_xy[None, :], axis=-1)
    route_dist = pairwise_nearest_distance(trajectories[..., :2], route_path)
    route_cost = (route_dist * masks).sum(axis=-1) / np.maximum(masks.sum(axis=-1), 1.0)
    clearance_cost = np.maximum(config.desired_clearance - clearance, 0.0) ** 2

    goal_direction = goal_xy - np.asarray(ego_state.xy, dtype=np.float32)
    goal_norm = max(float(np.linalg.norm(goal_direction)), 1.0e-6)
    goal_direction = goal_direction / goal_norm
    progress = (
        endpoint_xy - np.asarray(ego_state.xy, dtype=np.float32)
    ) @ goal_direction

    delta_xy = np.diff(trajectories[..., :2], axis=1)
    comfort_cost = np.linalg.norm(np.diff(delta_xy, axis=1), axis=-1).mean(axis=-1)

    velocity_vocab = vocab.velocity.detach().cpu().numpy()
    candidate_velocity = velocity_vocab[candidate_velocity_indices]
    speed_mean = candidate_velocity.mean(axis=-1)
    dt = float(TIME_STEPS[0])
    initial_accel = (candidate_velocity[:, 0] - float(ego_state.speed)) / dt
    accel = np.diff(candidate_velocity, axis=1) / dt
    jerk = np.diff(accel, axis=1) / dt
    initial_accel_cost = initial_accel**2
    accel_cost = (accel**2).mean(axis=-1)
    jerk_cost = (jerk**2).mean(axis=-1)

    cost = (
        config.collision_weight * collision.astype(np.float32)
        + config.goal_weight * goal_cost
        + config.route_weight * route_cost
        + config.clearance_weight * clearance_cost
        - config.progress_weight * progress
        + config.comfort_weight * comfort_cost
        + config.speed_weight * speed_mean
        + config.initial_accel_weight * initial_accel_cost
        + config.accel_weight * accel_cost
        + config.jerk_weight * jerk_cost
    )

    return {
        "trajectories": trajectories,
        "masks": masks,
        "candidate_path_indices": candidate_path_indices.astype(np.int64),
        "candidate_velocity_indices": candidate_velocity_indices.astype(np.int64),
        "candidate_flat_indices": candidate_flat_indices.astype(np.int64),
        "cost": cost.astype(np.float32),
        "collision": collision,
        "clearance": clearance.astype(np.float32),
        "goal_cost": goal_cost.astype(np.float32),
        "route_cost": route_cost.astype(np.float32),
        "progress": progress.astype(np.float32),
        "comfort_cost": comfort_cost.astype(np.float32),
        "speed_mean": speed_mean.astype(np.float32),
        "initial_accel": initial_accel.astype(np.float32),
        "initial_accel_cost": initial_accel_cost.astype(np.float32),
        "accel_cost": accel_cost.astype(np.float32),
        "jerk_cost": jerk_cost.astype(np.float32),
    }


def score_teacher_candidates(
    vocab: SparseDriveVocab,
    goal_xy: tuple[float, float] | np.ndarray,
    obstacles: list[DynamicObstacle | dict[str, Any]],
    ego_state: EgoState = EgoState(),
    route_path: np.ndarray | None = None,
    config: TeacherConfig = TeacherConfig(),
) -> TeacherOutput:
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    obstacles = normalize_obstacles(obstacles)
    if route_path is None:
        route_path = default_route_path(ego_state, goal_xy)
    else:
        route_path = np.asarray(route_path, dtype=np.float32)

    path_indices, path_costs = score_paths(
        vocab=vocab,
        goal_xy=goal_xy,
        obstacles=obstacles,
        ego_state=ego_state,
        route_path=route_path,
        config=config,
    )
    scored = score_trajectories(
        vocab=vocab,
        path_indices=path_indices,
        goal_xy=goal_xy,
        obstacles=obstacles,
        ego_state=ego_state,
        route_path=route_path,
        config=config,
    )

    candidate_costs = scored["cost"]
    candidate_scores = -candidate_costs
    candidate_probs = softmax_from_cost(candidate_costs, config.temperature)
    order = np.argsort(candidate_costs)
    best_candidate_index = int(order[0])
    topk = order[: config.num_top_trajectories]

    best_flat_index = int(scored["candidate_flat_indices"][best_candidate_index])
    best_path_index = int(scored["candidate_path_indices"][best_candidate_index])
    best_velocity_index = int(
        scored["candidate_velocity_indices"][best_candidate_index]
    )

    debug = {
        "collision": scored["collision"],
        "clearance": scored["clearance"],
        "goal_cost": scored["goal_cost"],
        "route_cost": scored["route_cost"],
        "progress": scored["progress"],
        "comfort_cost": scored["comfort_cost"],
        "speed_mean": scored["speed_mean"],
        "initial_accel": scored["initial_accel"],
        "initial_accel_cost": scored["initial_accel_cost"],
        "accel_cost": scored["accel_cost"],
        "jerk_cost": scored["jerk_cost"],
        "trajectories": scored["trajectories"],
        "masks": scored["masks"],
        "route_path": route_path,
        "goal_xy": goal_xy,
    }

    return TeacherOutput(
        path_indices=path_indices,
        path_costs=path_costs,
        candidate_flat_indices=scored["candidate_flat_indices"],
        candidate_path_indices=scored["candidate_path_indices"],
        candidate_velocity_indices=scored["candidate_velocity_indices"],
        candidate_costs=candidate_costs,
        candidate_scores=candidate_scores,
        candidate_probs=candidate_probs,
        best_candidate_index=best_candidate_index,
        best_flat_index=best_flat_index,
        best_path_index=best_path_index,
        best_velocity_index=best_velocity_index,
        topk_flat_indices=scored["candidate_flat_indices"][topk],
        debug=debug,
    )


def draw_rectangle_world(
    ax: plt.Axes,
    center_xy: np.ndarray,
    size_xy: tuple[float, float],
    color: str,
    alpha: float,
) -> None:
    x = float(center_xy[0])
    y = float(center_xy[1])
    length_x = float(size_xy[0])
    width_y = float(size_xy[1])
    rect = plt.Rectangle(
        (y - width_y / 2.0, x - length_x / 2.0),
        width_y,
        length_x,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.8,
    )
    ax.add_patch(rect)


def visualize_teacher_output(
    output: TeacherOutput,
    vocab: SparseDriveVocab,
    obstacles: list[DynamicObstacle],
    output_path: Path,
    grid_config: GridConfig = GridConfig(),
) -> None:
    best = output.best_candidate_index
    best_trajectory = output.debug["trajectories"][best]
    best_mask = output.debug["masks"][best]
    best_valid = best_trajectory[best_mask > 0.5]
    route_path = output.debug["route_path"]
    goal_xy = output.debug["goal_xy"]
    best_path = vocab.path[output.best_path_index].detach().cpu().numpy()
    best_velocity = vocab.velocity[output.best_velocity_index].detach().cpu().numpy()

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.0, 1.0),
    )
    ax = fig.add_subplot(gs[:, 0])
    ax_path = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])

    for obstacle in obstacles:
        draw_rectangle_world(
            ax,
            obstacle.center_xy,
            obstacle.size_xy,
            color="#666666",
            alpha=0.45,
        )
        for time_s in TIME_STEPS:
            future_center = obstacle.center_at(float(time_s))
            draw_rectangle_world(
                ax,
                future_center,
                obstacle.size_xy,
                color="#ff9900",
                alpha=0.07,
            )

    ax.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linewidth=1.2,
        label="route",
    )
    ax.plot(
        best_valid[:, 1],
        best_valid[:, 0],
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="teacher best",
    )
    ax.scatter(
        [goal_xy[1]],
        [goal_xy[0]],
        color="#d62728",
        marker="*",
        s=120,
        label="goal",
    )

    ax.set_title(
        f"teacher best p{output.best_path_index} v{output.best_velocity_index}"
    )
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_xlim(grid_config.y_max, grid_config.y_min)
    ax.set_ylim(grid_config.x_min, grid_config.x_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax_path.plot(
        best_path[:, 1],
        best_path[:, 0],
        color="#1f77b4",
        linewidth=1.8,
        marker=".",
        markersize=4,
    )
    ax_path.scatter(
        [best_path[0, 1]],
        [best_path[0, 0]],
        color="#2ca02c",
        s=45,
        label="start",
    )
    ax_path.scatter(
        [best_path[-1, 1]],
        [best_path[-1, 0]],
        color="#d62728",
        s=45,
        label="end",
    )
    ax_path.set_title(f"selected path p{output.best_path_index} | 50 points")
    ax_path.set_xlabel("y left [m]")
    ax_path.set_ylabel("x forward [m]")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.invert_xaxis()
    ax_path.grid(True, alpha=0.25)
    ax_path.legend(loc="best")

    ax_velocity.plot(
        np.arange(1, len(best_velocity) + 1),
        best_velocity,
        color="#9467bd",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    ax_velocity.set_title(f"selected velocity v{output.best_velocity_index} | 8 steps")
    ax_velocity.set_xlabel("step")
    ax_velocity.set_ylabel("speed [m/s]")
    ax_velocity.set_xticks(np.arange(1, len(best_velocity) + 1))
    ax_velocity.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    vocab = SparseDriveVocab.load()
    ego_state = EgoState(
        xy=(0.0, 0.0),
        yaw=0.0,
        speed=5.0,
        size_xy=(4.8, 2.0),
    )
    goal_xy = np.array([38.0, 0.0], dtype=np.float32)

    obstacles = normalize_obstacles(
        [
            {
                "center_xy": (15.0, 1.0),
                "size_xy": (5.0, 2.5),
                "velocity_xy": (1.0, 0.0),
            },
            {
                "center_xy": (10.0, -4.0),
                "size_xy": (5.0, 2.5),
                "velocity_xy": (1.0, 0.0),
            },
            # {
            #     "center_xy": (10.0, 4.0),
            #     "size_xy": (5.0, 2.5),
            #     "velocity_xy": (1.0, 0.0),
            # },
            # {
            #     "center_xy": (15.0, -6.0),
            #     "size_xy": (4.5, 2.2),
            #     "velocity_xy": (1.0, 0.0),
            # },
            # {
            #     "center_xy": (15.0, -3.0),
            #     "size_xy": (4.5, 2.2),
            #     "velocity_xy": (1.0, 0.0),
            # },
            # {
            #     "center_xy": (15.0, 0.0),
            #     "size_xy": (5.0, 2.5),
            #     "velocity_xy": (1.0, 0.0),
            # },
            # {
            #     "center_xy": (15.0, 4.0),
            #     "size_xy": (5.0, 2.5),
            #     "velocity_xy": (1.0, 0.0),
            # },
            # {
            #     "center_xy": (15.0, 6.0),
            #     "size_xy": (5.0, 2.5),
            #     "velocity_xy": (1.0, 0.0),
            # },
        ]
    )

    config = TeacherConfig(
        num_path_candidates=32,
        num_top_trajectories=10,
        temperature=2.0,
    )
    output = score_teacher_candidates(
        vocab=vocab,
        goal_xy=goal_xy,
        obstacles=obstacles,
        ego_state=ego_state,
        route_path=None,
        config=config,
    )

    print("best_flat_index:", output.best_flat_index)
    print("best_path_index:", output.best_path_index)
    print("best_velocity_index:", output.best_velocity_index)
    print("best_cost:", float(output.candidate_costs[output.best_candidate_index]))
    print("best_prob:", float(output.candidate_probs[output.best_candidate_index]))
    print(
        "best_collision:", bool(output.debug["collision"][output.best_candidate_index])
    )
    print(
        "best_clearance:", float(output.debug["clearance"][output.best_candidate_index])
    )
    print()
    print("top trajectories:")

    order = np.argsort(output.candidate_costs)[: config.num_top_trajectories]
    for rank, candidate_index in enumerate(order, start=1):
        flat_index = int(output.candidate_flat_indices[candidate_index])
        path_index = int(output.candidate_path_indices[candidate_index])
        velocity_index = int(output.candidate_velocity_indices[candidate_index])
        print(
            f"{rank:02d} flat={flat_index} p={path_index} v={velocity_index} "
            f"cost={output.candidate_costs[candidate_index]:.3f} "
            f"prob={output.candidate_probs[candidate_index]:.4f} "
            f"collision={bool(output.debug['collision'][candidate_index])} "
            f"clearance={output.debug['clearance'][candidate_index]:.3f} "
            f"goal={output.debug['goal_cost'][candidate_index]:.3f} "
            f"route={output.debug['route_cost'][candidate_index]:.3f} "
            f"progress={output.debug['progress'][candidate_index]:.3f} "
            f"speed={output.debug['speed_mean'][candidate_index]:.3f} "
            f"init_acc={output.debug['initial_accel'][candidate_index]:.3f} "
            f"accel_cost={output.debug['accel_cost'][candidate_index]:.3f} "
            f"jerk_cost={output.debug['jerk_cost'][candidate_index]:.3f}"
        )

    output_path = OUTPUT_DIR / "teacher_debug.png"
    visualize_teacher_output(output, vocab, obstacles, output_path)
    print()
    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
