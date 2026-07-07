from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
DATASET_DIR = TOY_ROOT / "dataset"
for path in (TOY_ROOT, DATASET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auto_policy import (
    AutoDriveObservation,
    AutoDriveScene,
    AutoPolicyConfig,
    plan_auto_action,
)
from simple_teacher import plan_simple_teacher_action
from dataset import (
    Obstacle,
    ToySparseDriveV2Dataset,
    make_scene_sampling_config,
    obstacle_is_valid_for_scene,
    obstacle_overlaps_ego_initially,
    sample_mixed_obstacle,
)
from drive_sim import (
    DEFAULT_EPISODE_DIR,
    DEFAULT_OUTPUT_DIR,
    DriveState,
    has_collision,
    load_saved_scene_keys,
    obstacle_center_at,
    obstacle_to_dict,
    reached_goal,
    save_episode,
    step_ego,
)
from grid import GridConfig
from teacher import TeacherConfig, draw_rectangle_world

AUTO_DRIVER_NAME = "auto_normal_v1"
AUTO_SOURCE = "auto_drive_v1"


def point_to_segment_distance(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Distance from a point to the finite segment [a, b]."""
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0e-8:
        return float(np.linalg.norm(point - a))

    ratio = float(np.dot(point - a, ab) / denom)
    ratio = max(0.0, min(1.0, ratio))
    closest = a + ratio * ab
    return float(np.linalg.norm(point - closest))


def reached_goal(
    state: DriveState,
    goal_xy: np.ndarray,
    threshold: float,
    prev_state: DriveState | None = None,
    passed_threshold: float | None = None,
    debug: bool = False,
) -> bool:
    """Position-only goal check that supports pass-through goals.

    The old check only tested the current ego point against one radius. That can
    miss the goal when the ego moves more than the radius between two simulation
    ticks. This version also checks whether the previous-current motion segment
    crossed near the goal and allows a slightly relaxed radius for pass-through
    scenes where stopping at the goal is not required.
    """
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    curr_xy = np.array([state.x, state.y], dtype=np.float32)

    threshold = float(threshold)
    relaxed_threshold = float(
        max(threshold, passed_threshold if passed_threshold is not None else threshold)
    )

    curr_dist = float(np.linalg.norm(curr_xy - goal_xy))
    if curr_dist <= threshold:
        if debug:
            print(
                "[reached_goal] current hit: "
                f"curr_dist={curr_dist:.2f}, threshold={threshold:.2f}"
            )
        return True

    if prev_state is None:
        return False

    prev_xy = np.array([prev_state.x, prev_state.y], dtype=np.float32)
    move_vec = curr_xy - prev_xy
    move_len2 = float(np.dot(move_vec, move_vec))
    if move_len2 < 1.0e-8:
        return False

    goal_vec = goal_xy - prev_xy
    t_raw = float(np.dot(goal_vec, move_vec) / move_len2)
    seg_dist = point_to_segment_distance(goal_xy, prev_xy, curr_xy)

    # The goal projection lies on the actual motion segment, and that segment
    # passes through the relaxed goal disk.
    if 0.0 <= t_raw <= 1.0 and seg_dist <= relaxed_threshold:
        if debug:
            print(
                "[reached_goal] segment hit: "
                f"seg_dist={seg_dist:.2f}, threshold={threshold:.2f}, "
                f"relaxed={relaxed_threshold:.2f}, t={t_raw:.2f}, "
                f"curr_dist={curr_dist:.2f}"
            )
        return True

    # If the ego has just passed the goal, allow a relaxed current-distance
    # check, but keep it bounded to avoid accepting a far lateral miss.
    if t_raw < 0.0 and curr_dist <= relaxed_threshold:
        if debug:
            print(
                "[reached_goal] passed relaxed hit: "
                f"curr_dist={curr_dist:.2f}, relaxed={relaxed_threshold:.2f}, "
                f"t={t_raw:.2f}, seg_dist={seg_dist:.2f}"
            )
        return True

    if debug:
        print(
            "[reached_goal] miss: "
            f"curr_dist={curr_dist:.2f}, seg_dist={seg_dist:.2f}, "
            f"threshold={threshold:.2f}, relaxed={relaxed_threshold:.2f}, "
            f"t={t_raw:.2f}"
        )
    return False


def status_episode_dir(base_dir: Path, status: str) -> Path:
    safe_status = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(status or "unknown")
    )
    return base_dir / safe_status


def load_auto_collection_progress(
    episode_dir: Path,
) -> tuple[Counter[str], Counter[str]]:
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    if not episode_dir.is_dir():
        return status_counts, mode_counts

    for path in sorted(episode_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            status_counts["<parse_failed>"] += 1
            continue
        status = str(record.get("status") or path.parent.name or "unknown")
        scene_mode = str(record.get("scene_mode") or "unknown")
        status_counts[status] += 1
        mode_counts[scene_mode] += 1
    return status_counts, mode_counts


def build_start_to_goal_route(
    goal_xy: np.ndarray,
    num_points: int = 70,
    extend_after_goal_m: float = 20.0,
) -> np.ndarray:
    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    direction = goal - start
    distance = float(np.linalg.norm(direction))
    if distance > 1.0e-6:
        unit = direction / distance
    else:
        unit = np.array([1.0, 0.0], dtype=np.float32)
    route_end = goal + unit * float(extend_after_goal_m)
    xy = np.linspace(start, route_end, int(num_points), dtype=np.float32)
    yaw = math.atan2(float(unit[1]), float(unit[0]))
    yaw_col = np.full((xy.shape[0], 1), yaw, dtype=np.float32)
    return np.concatenate([xy, yaw_col], axis=1)


def make_episode_record(
    args: argparse.Namespace,
    policy_config: AutoPolicyConfig,
    scene_index: int,
    route_path_index: int,
    route_path: np.ndarray,
    goal_xy: np.ndarray,
    ego_size_xy: tuple[float, float],
    obstacles: list[dict[str, Any]],
    status: str,
    steps: list[dict[str, Any]],
    ego_history: list[list[float]],
) -> dict[str, Any]:
    policy_name = str(getattr(args, "policy", "auto_policy"))
    return {
        "source": f"{AUTO_SOURCE}:{policy_name}",
        "driver": policy_name,
        "scene_mode": str(args.scene_mode),
        "seed_offset": int(args.seed_offset),
        "scene_index": int(scene_index),
        "scene_attempt": int(args.scene_attempt),
        "route_path_index": int(route_path_index),
        "route_path": route_path.astype(float).tolist(),
        "goal_xy": goal_xy.astype(float).tolist(),
        "ego_size_xy": [float(ego_size_xy[0]), float(ego_size_xy[1])],
        "dt": float(args.dt),
        "wheelbase": float(args.wheelbase),
        "obstacles": obstacles,
        "status": status,
        "steps": steps,
        "ego_history": ego_history,
        "controller": {
            **policy_config.to_metadata(),
            "policy": policy_name,
            "planning_route": "origin_to_goal_extended_line",
        },
    }


def plot_episode(
    output_path: Path,
    route_path: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[dict[str, Any]],
    ego_history: list[list[float]],
    title: str,
    planning_route_path: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for obstacle in obstacles:
        center = np.asarray(obstacle["center_xy"], dtype=np.float32)
        size_xy = tuple(float(v) for v in obstacle["size_xy"])
        draw_rectangle_world(ax, center, size_xy, color="#666666", alpha=0.45)
        ax.annotate(
            str(obstacle["id"]),
            xy=(float(center[1]), float(center[0])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#333333", "alpha": 0.9},
        )
    history = np.asarray(ego_history, dtype=np.float32)
    ax.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linewidth=1.5,
        label="original route",
    )
    if planning_route_path is not None:
        ax.plot(
            planning_route_path[:, 1],
            planning_route_path[:, 0],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.5,
            label="planner route",
        )
    ax.plot(
        history[:, 1],
        history[:, 0],
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=2.5,
        label="auto",
    )
    ax.scatter(
        [goal_xy[1]], [goal_xy[0]], color="#d62728", marker="*", s=130, label="goal"
    )
    ax.scatter([history[0, 1]], [history[0, 0]], color="#111111", s=45, label="start")
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


class AutoCollector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.policy_config = AutoPolicyConfig.from_args(args)
        self.grid_config = GridConfig()
        self.teacher_config = TeacherConfig()
        self.dataset = ToySparseDriveV2Dataset(
            num_samples=args.num_samples,
            seed_offset=args.seed_offset,
            grid_config=self.grid_config,
            teacher_config=self.teacher_config,
            scene_config=make_scene_sampling_config(args.scene_mode),
        )
        self.saved_scene_keys, self.legacy_scene_keys = load_saved_scene_keys(
            args.episode_dir
        )
        self.used_scene_indices: set[int] = set()
        self.stats = {
            "saved": 0,
            "skipped_existing": 0,
            "discarded_collision": 0,
            "discarded_timeout": 0,
            "discarded_stuck": 0,
            "discarded_other": 0,
        }

    def make_route_obstacle(
        self,
        route_xy: np.ndarray,
        distance_m: float,
        lateral_m: float,
        size_xy: tuple[float, float],
        velocity_forward: float,
        rng: np.random.Generator,
    ) -> Obstacle:
        start = route_xy[0]
        end = route_xy[-1]
        direction = end - start
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            unit = np.array([1.0, 0.0], dtype=np.float32)
        else:
            unit = (direction / norm).astype(np.float32)
        normal = np.array([-unit[1], unit[0]], dtype=np.float32)
        center_xy = (
            start
            + unit * float(distance_m)
            + normal * float(lateral_m)
            + unit * float(rng.uniform(-1.0, 1.0))
            + normal * float(rng.uniform(-0.35, 0.35))
        ).astype(np.float32)
        velocity_xy = (
            unit * float(velocity_forward) + normal * float(rng.uniform(-0.04, 0.04))
        ).astype(np.float32)
        return Obstacle(
            center_xy=center_xy,
            size_xy=(float(size_xy[0]), float(size_xy[1])),
            velocity_xy=velocity_xy,
        )

    def sample_structured_planner_obstacles(
        self,
        rng: np.random.Generator,
        planning_route_path: np.ndarray,
        mode: str,
    ) -> list[Obstacle] | None:
        route_xy = planning_route_path[:, :2].astype(np.float32)
        if mode == "low_speed_pass_gap":
            base_distance = float(rng.uniform(15.0, 20.0))
            gap_half_width = float(rng.uniform(2.0, 2.6))
            return [
                self.make_route_obstacle(
                    route_xy=route_xy,
                    distance_m=base_distance,
                    lateral_m=gap_half_width + float(rng.uniform(1.2, 1.8)),
                    size_xy=(
                        float(rng.uniform(4.5, 6.5)),
                        float(rng.uniform(2.5, 3.5)),
                    ),
                    velocity_forward=float(rng.uniform(0.0, 0.6)),
                    rng=rng,
                ),
                self.make_route_obstacle(
                    route_xy=route_xy,
                    distance_m=base_distance + float(rng.uniform(1.5, 4.0)),
                    lateral_m=-(gap_half_width + float(rng.uniform(1.2, 1.8))),
                    size_xy=(
                        float(rng.uniform(4.5, 6.5)),
                        float(rng.uniform(2.5, 3.5)),
                    ),
                    velocity_forward=float(rng.uniform(0.0, 0.6)),
                    rng=rng,
                ),
            ]
        if mode == "low_speed_yield_blocked":
            base_distance = float(rng.uniform(13.0, 17.0))
            side_sign = -1.0 if rng.random() < 0.5 else 1.0
            return [
                self.make_route_obstacle(
                    route_xy=route_xy,
                    distance_m=base_distance,
                    lateral_m=float(rng.uniform(-0.35, 0.35)),
                    size_xy=(
                        float(rng.uniform(4.5, 6.0)),
                        float(rng.uniform(2.4, 3.0)),
                    ),
                    velocity_forward=float(rng.uniform(0.8, 1.6)),
                    rng=rng,
                ),
                self.make_route_obstacle(
                    route_xy=route_xy,
                    distance_m=base_distance + float(rng.uniform(3.0, 6.0)),
                    lateral_m=side_sign * float(rng.uniform(3.8, 5.2)),
                    size_xy=(
                        float(rng.uniform(4.5, 6.0)),
                        float(rng.uniform(2.4, 3.0)),
                    ),
                    velocity_forward=float(rng.uniform(0.2, 0.8)),
                    rng=rng,
                ),
            ]
        return None

    def sample_obstacles_for_planner_route(
        self,
        scene_index: int,
        planning_route_path: np.ndarray,
        ego_state: Any,
    ) -> list[Any]:
        seed_sequence = np.random.SeedSequence(
            [
                int(self.args.seed_offset),
                int(scene_index),
                int(self.args.scene_attempt),
                7919,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        scene_config = self.dataset.scene_config
        structured_obstacles = self.sample_structured_planner_obstacles(
            rng=rng,
            planning_route_path=planning_route_path,
            mode=str(scene_config.mode),
        )
        if structured_obstacles is not None:
            if all(
                obstacle_is_valid_for_scene(obstacle)
                and not obstacle_overlaps_ego_initially(
                    obstacle=obstacle,
                    ego_state=ego_state,
                    safety_margin=self.teacher_config.safety_margin,
                )
                for obstacle in structured_obstacles
            ):
                return structured_obstacles

        min_obstacles = (
            scene_config.min_obstacles
            if scene_config.min_obstacles is not None
            else self.dataset.min_obstacles
        )
        max_obstacles = (
            scene_config.max_obstacles
            if scene_config.max_obstacles is not None
            else self.dataset.max_obstacles
        )
        num_obstacles = int(rng.integers(min_obstacles, max_obstacles + 1))
        obstacles = []
        for _ in range(num_obstacles):
            for _sample_attempt in range(100):
                obstacle = sample_mixed_obstacle(
                    rng=rng,
                    grid_config=self.grid_config,
                    route_path=planning_route_path,
                    config=scene_config,
                )
                if obstacle_is_valid_for_scene(
                    obstacle
                ) and not obstacle_overlaps_ego_initially(
                    obstacle=obstacle,
                    ego_state=ego_state,
                    safety_margin=self.teacher_config.safety_margin,
                ):
                    obstacles.append(obstacle)
                    break
            else:
                raise RuntimeError(
                    f"Unable to sample planner-route obstacle for "
                    f"sample={scene_index}, scene_attempt={self.args.scene_attempt}"
                )
        return obstacles

    def scene_key(self, scene_index: int) -> tuple[str, int, int, int]:
        return (
            str(self.args.scene_mode),
            int(self.args.seed_offset),
            int(scene_index),
            int(self.args.scene_attempt),
        )

    def scene_exists(self, scene_index: int) -> bool:
        key = self.scene_key(scene_index)
        legacy_key = (key[2], key[3])
        return key in self.saved_scene_keys or legacy_key in self.legacy_scene_keys

    def print_collection_progress(
        self, prefix: str = "[auto collection progress]"
    ) -> None:
        status_counts, mode_counts = load_auto_collection_progress(
            self.args.episode_dir
        )
        total = sum(status_counts.values())
        print(f"{prefix} dir={self.args.episode_dir} total_saved={total}")
        if status_counts:
            status_text = ", ".join(
                f"{key}={status_counts[key]}" for key in sorted(status_counts)
            )
            print(f"  by_status: {status_text}")
        if mode_counts:
            mode_text = ", ".join(
                f"{key}={mode_counts[key]}" for key in sorted(mode_counts)
            )
            print(f"  by_scene_mode: {mode_text}")
        print(
            "  current_session: "
            f"saved={self.stats['saved']} "
            f"skipped_existing={self.stats['skipped_existing']} "
            f"discarded_collision={self.stats['discarded_collision']} "
            f"discarded_stuck={self.stats['discarded_stuck']} "
            f"discarded_timeout={self.stats['discarded_timeout']} "
            f"discarded_other={self.stats['discarded_other']}"
        )

    def sample_scene_index(self) -> int | None:
        for _ in range(1000):
            scene_index = random.randrange(self.args.num_samples)
            if scene_index in self.used_scene_indices:
                continue
            self.used_scene_indices.add(scene_index)
            if self.scene_exists(scene_index):
                self.stats["skipped_existing"] += 1
                continue
            return scene_index
        for scene_index in range(self.args.num_samples):
            if scene_index in self.used_scene_indices:
                continue
            self.used_scene_indices.add(scene_index)
            if self.scene_exists(scene_index):
                self.stats["skipped_existing"] += 1
                continue
            return scene_index
        return None

    def generate_scene_state(
        self,
        scene_index: int,
    ) -> tuple[int, AutoDriveScene, DriveState, tuple[float, float]]:
        route_path_index, route_path, goal_xy, ego_state, raw_obstacles = (
            self.dataset.generate_scene(
                scene_index, scene_attempt=self.args.scene_attempt
            )
        )
        planning_route_path = build_start_to_goal_route(goal_xy)
        raw_obstacles = self.sample_obstacles_for_planner_route(
            scene_index=scene_index,
            planning_route_path=planning_route_path,
            ego_state=ego_state,
        )
        obstacles = [
            obstacle_to_dict(obstacle, obstacle_index)
            for obstacle_index, obstacle in enumerate(raw_obstacles)
        ]
        state = DriveState(
            x=float(ego_state.xy[0]),
            y=float(ego_state.xy[1]),
            yaw=float(ego_state.yaw),
            speed=float(ego_state.speed),
        )
        scene = AutoDriveScene(
            route_path=route_path,
            goal_xy=goal_xy,
            obstacles=obstacles,
            planning_route_path=planning_route_path,
        )
        return route_path_index, scene, state, ego_state.size_xy

    def step_policy(
        self,
        state: DriveState,
        scene: AutoDriveScene,
        time_s: float,
        ego_history: list[list[float]],
    ):
        observation = AutoDriveObservation(
            state=state,
            time_s=float(time_s),
            history=ego_history,
        )
        if str(getattr(self.args, "policy", "auto_policy")) == "simple_teacher":
            return plan_simple_teacher_action(observation, scene, self.policy_config)
        return plan_auto_action(observation, scene, self.policy_config)

    def run_scene(self, scene_index: int) -> tuple[str, dict[str, Any] | None]:
        route_path_index, scene, state, ego_size_xy = self.generate_scene_state(
            scene_index
        )
        steps: list[dict[str, Any]] = []
        ego_history = [[state.x, state.y, state.yaw, state.speed]]
        time_s = 0.0
        low_speed_steps = 0
        status = "timeout"

        max_steps = int(math.ceil(self.args.max_time / self.args.dt))
        for step_index in range(max_steps):
            prev_state = state
            before = asdict(state)
            action = self.step_policy(state, scene, time_s, ego_history)
            state = step_ego(
                state=state,
                action=action,
                dt=self.args.dt,
                wheelbase=self.args.wheelbase,
                max_speed=self.args.max_speed,
            )
            time_s += float(self.args.dt)
            ego_history.append([state.x, state.y, state.yaw, state.speed])
            collision = has_collision(
                state=state,
                ego_size_xy=ego_size_xy,
                obstacles=scene.obstacles,
                time_s=time_s,
                safety_margin=self.args.safety_margin,
            )
            goal_reached = reached_goal(
                state=state,
                goal_xy=scene.goal_xy,
                threshold=self.args.goal_threshold,
                prev_state=prev_state,
                passed_threshold=max(
                    float(self.args.goal_threshold),
                    float(self.args.goal_passed_threshold),
                ),
                debug=bool(getattr(self.args, "debug_goal_check", False)),
            )
            steps.append(
                {
                    "step": step_index,
                    "time_s": float(time_s),
                    "action": asdict(action),
                    "ego_before": before,
                    "ego_after": asdict(state),
                    "collision": bool(collision),
                    "goal_reached": bool(goal_reached),
                }
            )
            if collision:
                status = "collision"
                break
            if goal_reached:
                status = "goal_reached"
                break
            if state.speed < 0.25:
                low_speed_steps += 1
            else:
                low_speed_steps = 0
            if low_speed_steps * self.args.dt >= self.args.stuck_time:
                status = "stuck"
                break

        record = make_episode_record(
            args=self.args,
            policy_config=self.policy_config,
            scene_index=scene_index,
            route_path_index=route_path_index,
            route_path=scene.route_path,
            goal_xy=scene.goal_xy,
            ego_size_xy=ego_size_xy,
            obstacles=scene.obstacles,
            status=status,
            steps=steps,
            ego_history=ego_history,
        )
        record["_debug_route_path"] = scene.route_path
        record["_debug_planning_route_path"] = scene.planning_route_path
        record["_debug_goal_xy"] = scene.goal_xy
        return status, record

    def maybe_save_debug_plot(
        self, record: dict[str, Any], scene_index: int, status: str
    ) -> None:
        if self.stats["saved"] > self.args.save_debug_plots:
            return
        route_path = record.pop("_debug_route_path")
        planning_route_path = record.pop("_debug_planning_route_path", None)
        goal_xy = record.pop("_debug_goal_xy")
        output_path = (
            self.args.debug_plot_dir
            / f"auto_{self.args.scene_mode}_scene_{scene_index:06d}_{status}.png"
        )
        plot_episode(
            output_path=output_path,
            route_path=route_path,
            goal_xy=goal_xy,
            obstacles=record["obstacles"],
            ego_history=record["ego_history"],
            title=f"auto collect | {self.args.scene_mode} | scene {scene_index} | {status}",
            planning_route_path=planning_route_path,
        )
        print(f"debug plot: {output_path}")

    def collect(self) -> None:
        attempts = 0
        try:
            while (
                self.stats["saved"] < self.args.num_episodes
                and attempts < self.args.max_attempts
            ):
                attempts += 1
                scene_index = self.sample_scene_index()
                if scene_index is None:
                    print("no unused scene remains")
                    break
                status, record = self.run_scene(scene_index)
                if record is None:
                    self.stats["discarded_other"] += 1
                    continue
                if status == "goal_reached" or (
                    status == "timeout" and self.args.save_timeout
                ):
                    route_path = record.pop("_debug_route_path")
                    planning_route_path = record.pop("_debug_planning_route_path", None)
                    goal_xy = record.pop("_debug_goal_xy")
                    output_path = save_episode(
                        status_episode_dir(self.args.episode_dir, status),
                        record,
                    )
                    self.saved_scene_keys.add(self.scene_key(scene_index))
                    self.stats["saved"] += 1
                    record["_debug_route_path"] = route_path
                    record["_debug_planning_route_path"] = planning_route_path
                    record["_debug_goal_xy"] = goal_xy
                    if self.stats["saved"] <= self.args.save_debug_plots:
                        self.maybe_save_debug_plot(record, scene_index, status)
                    print(
                        f"[saved] {self.stats['saved']}/{self.args.num_episodes} "
                        f"scene={scene_index} status={status} steps={len(record['steps'])} "
                        f"path={output_path}"
                    )
                    continue
                if status == "collision":
                    self.stats["discarded_collision"] += 1
                elif status == "timeout":
                    self.stats["discarded_timeout"] += 1
                elif status == "stuck":
                    self.stats["discarded_stuck"] += 1
                else:
                    self.stats["discarded_other"] += 1
        except KeyboardInterrupt:
            print("\n[interrupted]")
        finally:
            print("summary:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")
            print(f"  attempts: {attempts}")
            self.print_collection_progress(prefix="[exit progress]")


class AutoReviewApp(AutoCollector):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.fig: plt.Figure
        self.ax_scene: plt.Axes
        self.ax_speed: plt.Axes
        self.scene_index = 0
        self.route_path_index = 0
        self.scene: AutoDriveScene | None = None
        self.ego_size_xy = (4.8, 2.0)
        self.state = DriveState()
        self.time_s = 0.0
        self.steps: list[dict[str, Any]] = []
        self.ego_history: list[list[float]] = []
        self.status = "driving"
        self.low_speed_steps = 0
        self.review_saved = 0
        self.review_discarded = 0
        self.review_summary_printed = False

    def print_review_summary(self) -> None:
        if self.review_summary_printed:
            return
        self.review_summary_printed = True
        print(
            f"[review summary] saved={self.review_saved} "
            f"discarded={self.review_discarded}"
        )
        self.print_collection_progress(prefix="[exit progress]")

    def reset_scene(self) -> None:
        scene_index = self.sample_scene_index()
        if scene_index is None:
            print("no unused scene remains")
            plt.close(self.fig)
            return
        self.scene_index = int(scene_index)
        self.route_path_index, self.scene, self.state, self.ego_size_xy = (
            self.generate_scene_state(self.scene_index)
        )
        self.time_s = 0.0
        self.steps = []
        self.ego_history = [
            [self.state.x, self.state.y, self.state.yaw, self.state.speed]
        ]
        self.status = "driving"
        self.low_speed_steps = 0
        print(f"[new auto scene] scene={self.scene_index} not saved yet")

    def build_current_record(self) -> dict[str, Any]:
        assert self.scene is not None
        return make_episode_record(
            args=self.args,
            policy_config=self.policy_config,
            scene_index=self.scene_index,
            route_path_index=self.route_path_index,
            route_path=self.scene.route_path,
            goal_xy=self.scene.goal_xy,
            ego_size_xy=self.ego_size_xy,
            obstacles=self.scene.obstacles,
            status=self.status,
            steps=self.steps,
            ego_history=self.ego_history,
        )

    def auto_step_once(self) -> None:
        if self.status != "driving" or self.scene is None:
            return
        prev_state = self.state
        before = asdict(self.state)
        action = self.step_policy(self.state, self.scene, self.time_s, self.ego_history)
        self.state = step_ego(
            state=self.state,
            action=action,
            dt=self.args.dt,
            wheelbase=self.args.wheelbase,
            max_speed=self.args.max_speed,
        )
        self.time_s += float(self.args.dt)
        self.ego_history.append(
            [self.state.x, self.state.y, self.state.yaw, self.state.speed]
        )
        collision = has_collision(
            state=self.state,
            ego_size_xy=self.ego_size_xy,
            obstacles=self.scene.obstacles,
            time_s=self.time_s,
            safety_margin=self.args.safety_margin,
        )
        goal_reached = reached_goal(
            state=self.state,
            goal_xy=self.scene.goal_xy,
            threshold=self.args.goal_threshold,
            prev_state=prev_state,
            passed_threshold=max(
                float(self.args.goal_threshold),
                float(self.args.goal_passed_threshold),
            ),
            debug=bool(getattr(self.args, "debug_goal_check", False)),
        )
        self.steps.append(
            {
                "step": len(self.steps),
                "time_s": float(self.time_s),
                "action": asdict(action),
                "ego_before": before,
                "ego_after": asdict(self.state),
                "collision": bool(collision),
                "goal_reached": bool(goal_reached),
            }
        )
        if collision:
            self.status = "collision"
        elif goal_reached:
            self.status = "goal_reached"
        elif self.time_s >= self.args.max_time:
            self.status = "timeout"
        elif self.state.speed < 0.05:
            self.low_speed_steps += 1
            if self.low_speed_steps * self.args.dt >= self.args.stuck_time:
                self.status = "stuck"
        else:
            self.low_speed_steps = 0

        if self.status != "driving":
            print(
                f"[finished] scene={self.scene_index} status={self.status} "
                f"steps={len(self.steps)}; press enter/f to save or r to discard"
            )

    def save_current(self) -> None:
        output_path = save_episode(
            status_episode_dir(self.args.episode_dir, self.status),
            self.build_current_record(),
        )
        self.saved_scene_keys.add(self.scene_key(self.scene_index))
        self.review_saved += 1
        print(
            f"[saved] scene={self.scene_index} status={self.status} "
            f"steps={len(self.steps)} path={output_path}"
        )

    def discard_current(self) -> None:
        self.review_discarded += 1
        print(
            f"[discarded] scene={self.scene_index} status={self.status} no json saved"
        )

    def on_key_press(self, event: Any) -> None:
        if event.key is None:
            return
        key = str(event.key).lower()
        if key in {"q", "escape"}:
            self.print_review_summary()
            plt.close(self.fig)
            return
        if key in {"enter", "\n", "\r", "f"}:
            if self.status == "driving":
                print("[not saved] auto scene is still running")
                return
            self.save_current()
            self.reset_scene()
            self.draw()
            return
        if key == "r":
            self.discard_current()
            self.reset_scene()
            self.draw()
            return

    def draw(self) -> None:
        assert self.scene is not None
        self.ax_scene.clear()
        self.ax_speed.clear()
        for obstacle in self.scene.obstacles:
            center = obstacle_center_at(obstacle, self.time_s)
            size_xy = tuple(float(v) for v in obstacle["size_xy"])
            draw_rectangle_world(
                self.ax_scene, center, size_xy, color="#666666", alpha=0.48
            )
            self.ax_scene.annotate(
                str(obstacle["id"]),
                xy=(float(center[1]), float(center[0])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#333333",
                    "alpha": 0.9,
                },
            )
        history = np.asarray(self.ego_history, dtype=np.float32)
        self.ax_scene.plot(
            self.scene.route_path[:, 1],
            self.scene.route_path[:, 0],
            color="#1f77b4",
            linewidth=1.4,
            label="original route",
        )
        if self.scene.planning_route_path is not None:
            self.ax_scene.plot(
                self.scene.planning_route_path[:, 1],
                self.scene.planning_route_path[:, 0],
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.5,
                label="planner route",
            )
        self.ax_scene.plot(
            history[:, 1],
            history[:, 0],
            color="#2ca02c",
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="auto ego",
        )
        self.ax_scene.scatter(
            [self.scene.goal_xy[1]],
            [self.scene.goal_xy[0]],
            color="#d62728",
            marker="*",
            s=130,
            label="goal",
        )
        self.ax_scene.set_title(
            f"auto review | scene={self.scene_index} | mode={self.args.scene_mode} | "
            f"t={self.time_s:.1f}s | speed={self.state.speed:.2f}m/s | {self.status}\n"
            "enter/f save, r discard/next, q/esc quit"
        )
        self.ax_scene.set_xlabel("y left [m]")
        self.ax_scene.set_ylabel("x forward [m]")
        self.ax_scene.set_aspect("equal", adjustable="box")
        self.ax_scene.grid(True, alpha=0.25)
        self.ax_scene.legend(loc="best", fontsize=8)

        times = [0.0] + [float(step["time_s"]) for step in self.steps]
        speeds = [self.ego_history[0][3]] + [
            float(step["ego_after"]["speed"]) for step in self.steps
        ]
        self.ax_speed.plot(
            times,
            speeds,
            color="#9467bd",
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        self.ax_speed.set_title("auto speed")
        self.ax_speed.set_xlabel("time [s]")
        self.ax_speed.set_ylabel("speed [m/s]")
        self.ax_speed.set_xlim(0.0, max(self.args.max_time, 1.0))
        self.ax_speed.set_ylim(0.0, max(self.args.max_speed, max(speeds) + 1.0))
        self.ax_speed.grid(True, alpha=0.25)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_timer(self) -> bool:
        if self.status == "driving":
            for _ in range(self.args.gui_steps_per_tick):
                self.auto_step_once()
                if self.status != "driving":
                    break
            self.draw()
        return True

    def run(self) -> None:
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.3, 0.7))
        self.ax_scene = self.fig.add_subplot(gs[0, 0])
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.reset_scene()
        self.draw()
        timer = self.fig.canvas.new_timer(interval=self.args.gui_interval_ms)
        timer.add_callback(self.on_timer)
        timer.start()
        plt.show()
        self.print_review_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto driving episode collector.")
    parser.add_argument("--review-gui", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("auto_policy", "simple_teacher"),
        default="auto_policy",
        help="Teacher policy used to generate driving actions.",
    )
    parser.add_argument(
        "--scene-mode",
        choices=(
            "random",
            "straight",
            "low_speed_avoid",
            "low_speed_pass_gap",
            "low_speed_yield_blocked",
            "follow_stop",
            "dense_front",
        ),
        default="straight",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--seed-offset", type=int, default=70_000)
    parser.add_argument("--scene-attempt", type=int, default=0)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "auto_collect"
    )
    parser.add_argument("--debug-plot-dir", type=Path, default=None)
    parser.add_argument("--save-debug-plots", type=int, default=3)
    parser.add_argument("--save-timeout", action="store_true")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-time", type=float, default=16.0)
    parser.add_argument("--accel", type=float, default=1.6)
    parser.add_argument("--brake", type=float, default=3.0)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--max-speed", type=float, default=12.0)
    parser.add_argument("--max-steer-deg", type=float, default=18.0)
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--goal-threshold", type=float, default=1.0)
    parser.add_argument(
        "--goal-passed-threshold",
        type=float,
        default=2.5,
        help="Relaxed goal radius used when ego segment passes near the goal.",
    )
    parser.add_argument(
        "--debug-goal-check",
        action="store_true",
        help="Print detailed reached_goal debug information.",
    )
    parser.add_argument("--cruise-speed", type=float, default=6.5)
    parser.add_argument("--speed-kp", type=float, default=0.9)
    parser.add_argument("--lookahead-base", type=float, default=7.0)
    parser.add_argument("--goal-bias", type=float, default=0.82)
    parser.add_argument("--avoid-lateral-offset", type=float, default=5.0)
    parser.add_argument("--corridor-half-width", type=float, default=2.5)
    parser.add_argument("--obstacle-lookahead", type=float, default=28.0)
    parser.add_argument("--stuck-time", type=float, default=4.0)
    parser.add_argument("--gui-interval-ms", type=int, default=80)
    parser.add_argument("--gui-steps-per-tick", type=int, default=1)
    args = parser.parse_args()
    if args.debug_plot_dir is None:
        args.debug_plot_dir = args.output_dir / "debug_plots"
    return args


def main() -> None:
    args = parse_args()
    if args.review_gui:
        app = AutoReviewApp(args)
        app.run()
        return
    collector = AutoCollector(args)
    collector.collect()


if __name__ == "__main__":
    main()
