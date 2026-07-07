from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import sys
import termios
import threading
import time
import tty
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

for keymap_name in (
    "keymap.back",
    "keymap.forward",
    "keymap.home",
    "keymap.pan",
    "keymap.quit",
    "keymap.quit_all",
    "keymap.save",
    "keymap.xscale",
    "keymap.yscale",
    "keymap.zoom",
):
    matplotlib.rcParams[keymap_name] = []

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
DATASET_DIR = TOY_ROOT / "dataset"
for path in (TOY_ROOT, DATASET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import Obstacle, ToySparseDriveV2Dataset, make_scene_sampling_config
from grid import GridConfig
from teacher import EgoState, TeacherConfig, draw_rectangle_world

DEFAULT_OUTPUT_DIR = TOY_ROOT / "outputs" / "human_drive"
DEFAULT_EPISODE_DIR = TOY_ROOT / "human_drive" / "episodes"
TIME_STEPS = np.arange(1, 9, dtype=np.float32) * 0.5
SceneKey = Tuple[str, int, int, int]
LegacySceneKey = Tuple[int, int]
COLLECTION_TARGETS = {
    "straight": (500, 800),
    "low_speed_avoid": (300, 500),
    "low_speed_pass_gap": (100, 200),
    "low_speed_yield_blocked": (100, 200),
    "follow_stop": (300, 500),
    "dense_front": (300, 500),
}
MISSING_SCENE_MODE = "<missing_scene_mode>"


@dataclass
class DriveState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0


@dataclass
class DriveAction:
    key: str
    accel: float
    steer: float


def obstacle_to_dict(obstacle: Obstacle, index: int) -> dict[str, Any]:
    return {
        "id": f"obs{index}",
        "center_xy": [
            float(obstacle.center_xy[0]),
            float(obstacle.center_xy[1]),
        ],
        "size_xy": [float(obstacle.size_xy[0]), float(obstacle.size_xy[1])],
        "velocity_xy": [
            float(obstacle.velocity_xy[0]),
            float(obstacle.velocity_xy[1]),
        ],
    }


def obstacle_center_at(obstacle: dict[str, Any], time_s: float) -> np.ndarray:
    center_xy = np.asarray(obstacle["center_xy"], dtype=np.float32)
    velocity_xy = np.asarray(obstacle["velocity_xy"], dtype=np.float32)
    return center_xy + velocity_xy * float(time_s)


def step_ego(
    state: DriveState,
    action: DriveAction,
    dt: float,
    wheelbase: float,
    max_speed: float,
) -> DriveState:
    next_speed = float(np.clip(state.speed + action.accel * dt, 0.0, max_speed))
    yaw_rate = 0.0
    if abs(action.steer) > 1.0e-6:
        yaw_rate = state.speed / max(wheelbase, 1.0e-6) * math.tan(action.steer)
    next_yaw = state.yaw + yaw_rate * dt
    average_speed = 0.5 * (state.speed + next_speed)
    next_x = state.x + average_speed * math.cos(next_yaw) * dt
    next_y = state.y + average_speed * math.sin(next_yaw) * dt
    return DriveState(
        x=float(next_x),
        y=float(next_y),
        yaw=float(next_yaw),
        speed=float(next_speed),
    )


def ego_polygon_yx(
    state: DriveState,
    size_xy: tuple[float, float],
) -> np.ndarray:
    length_x, width_y = size_xy
    corners_xy = np.array(
        [
            [length_x / 2.0, width_y / 2.0],
            [length_x / 2.0, -width_y / 2.0],
            [-length_x / 2.0, -width_y / 2.0],
            [-length_x / 2.0, width_y / 2.0],
            [length_x / 2.0, width_y / 2.0],
        ],
        dtype=np.float32,
    )
    c = math.cos(state.yaw)
    s = math.sin(state.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    world_xy = corners_xy @ rotation.T + np.array([state.x, state.y], dtype=np.float32)
    return world_xy[:, [1, 0]]


def has_collision(
    state: DriveState,
    ego_size_xy: tuple[float, float],
    obstacles: list[dict[str, Any]],
    time_s: float,
    safety_margin: float,
) -> bool:
    ego_center = np.array([state.x, state.y], dtype=np.float32)
    ego_size = np.asarray(ego_size_xy, dtype=np.float32)
    for obstacle in obstacles:
        obstacle_center = obstacle_center_at(obstacle, time_s)
        obstacle_size = np.asarray(obstacle["size_xy"], dtype=np.float32)
        half_extent = 0.5 * (ego_size + obstacle_size) + float(safety_margin)
        if np.all(np.abs(ego_center - obstacle_center) <= half_extent):
            return True
    return False


# def reached_goal(
#     state: DriveState,
#     goal_xy: np.ndarray,
#     threshold: float,
# ) -> bool:
#     ego_xy = np.array([state.x, state.y], dtype=np.float32)
#     return bool(np.linalg.norm(ego_xy - goal_xy) <= threshold)


def point_to_segment_distance(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0e-8:
        return float(np.linalg.norm(point - a))

    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def reached_goal(
    state: DriveState,
    goal_xy: np.ndarray,
    threshold: float,
    prev_state: DriveState | None = None,
    passed_threshold: float | None = None,
    debug: bool = False,
) -> bool:
    """Position-only goal check.

    Rules:
      1. current ego point inside threshold -> reached;
      2. previous-current motion segment crosses goal circle -> reached;
      3. if the ego has just passed the goal, allow a slightly relaxed radius.

    This is designed for pass-through goals where speed is not required to be 0.
    """
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    curr_xy = np.array([state.x, state.y], dtype=np.float32)

    threshold = float(threshold)
    relaxed_threshold = float(
        max(threshold, passed_threshold if passed_threshold is not None else threshold)
    )

    curr_dist = float(np.linalg.norm(curr_xy - goal_xy))

    # 1. 当前帧已经进圈
    if curr_dist <= threshold:
        if debug:
            print(
                f"[reached_goal] current hit: "
                f"curr_dist={curr_dist:.2f}, threshold={threshold:.2f}"
            )
        return True

    # 没有上一帧时，只能做当前点判断。
    if prev_state is None:
        return False

    prev_xy = np.array([prev_state.x, prev_state.y], dtype=np.float32)
    move_vec = curr_xy - prev_xy
    move_len2 = float(np.dot(move_vec, move_vec))

    if move_len2 < 1.0e-8:
        return False

    goal_vec = goal_xy - prev_xy
    t_raw = float(np.dot(goal_vec, move_vec) / move_len2)

    # goal 到上一帧-当前帧运动线段的距离
    seg_dist = point_to_segment_distance(goal_xy, prev_xy, curr_xy)

    # 2. 当前这一步穿过了 goal 附近。
    # 0 <= t_raw <= 1 表示 goal 的投影点落在 prev->curr 这一段运动里。
    if 0.0 <= t_raw <= 1.0 and seg_dist <= relaxed_threshold:
        if debug:
            print(
                f"[reached_goal] segment hit: "
                f"seg_dist={seg_dist:.2f}, "
                f"threshold={threshold:.2f}, "
                f"relaxed={relaxed_threshold:.2f}, "
                f"t={t_raw:.2f}, "
                f"curr_dist={curr_dist:.2f}"
            )
        return True

    # 3. 已经越过 goal 后，允许稍微放松。
    # t_raw < 0 表示 goal 在上一帧位置的后方，也就是当前已经越过过头。
    # 这里仍然要求当前距离不能太远，避免很远处误判。
    if t_raw < 0.0 and curr_dist <= relaxed_threshold:
        if debug:
            print(
                f"[reached_goal] passed relaxed hit: "
                f"curr_dist={curr_dist:.2f}, "
                f"relaxed={relaxed_threshold:.2f}, "
                f"t={t_raw:.2f}, "
                f"seg_dist={seg_dist:.2f}"
            )
        return True

    if debug:
        print(
            f"[reached_goal] miss: "
            f"curr_dist={curr_dist:.2f}, "
            f"seg_dist={seg_dist:.2f}, "
            f"threshold={threshold:.2f}, "
            f"relaxed={relaxed_threshold:.2f}, "
            f"t={t_raw:.2f}"
        )

    return False


def draw_obstacles(
    ax: plt.Axes,
    obstacles: list[dict[str, Any]],
    time_s: float,
) -> None:
    for obstacle in obstacles:
        center_xy = obstacle_center_at(obstacle, time_s)
        size_xy = (float(obstacle["size_xy"][0]), float(obstacle["size_xy"][1]))
        draw_rectangle_world(
            ax,
            center_xy,
            size_xy,
            color="#666666",
            alpha=0.48,
        )
        ax.annotate(
            str(obstacle["id"]),
            xy=(float(center_xy[1]), float(center_xy[0])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#333333",
                "edgecolor": "white",
                "linewidth": 0.5,
                "alpha": 0.9,
            },
            zorder=20,
        )
        for future_dt in TIME_STEPS:
            future_center = obstacle_center_at(obstacle, time_s + float(future_dt))
            draw_rectangle_world(
                ax,
                future_center,
                size_xy,
                color="#ff9900",
                alpha=0.055,
            )


def make_speed_command_map(accel_step: float, brake_step: float) -> dict[str, float]:
    return {
        "+": accel_step,
        "=": 0.0,
        "-": -brake_step,
    }


def make_steer_command_map(steer_step: float) -> dict[str, float]:
    return {
        "a": steer_step,
        "w": 0.0,
        "d": -steer_step,
    }


def save_episode(
    episode_dir: Path,
    record: dict[str, Any],
) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scene_index = int(record["scene_index"])
    output_path = episode_dir / f"episode_{timestamp}_scene_{scene_index:06d}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return output_path


def scene_index_from_filename(path: Path) -> int | None:
    stem = path.stem
    marker = "_scene_"
    if marker not in stem:
        return None
    try:
        return int(stem.rsplit(marker, maxsplit=1)[1])
    except ValueError:
        return None


def load_saved_scene_keys(
    episode_dir: Path,
) -> tuple[set[SceneKey], set[LegacySceneKey]]:
    scene_keys: set[SceneKey] = set()
    legacy_scene_keys: set[LegacySceneKey] = set()
    if not episode_dir.is_dir():
        return scene_keys, legacy_scene_keys

    for path in sorted(episode_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        scene_index = record.get("scene_index", scene_index_from_filename(path))
        scene_attempt = record.get("scene_attempt", 0)
        try:
            scene_index = int(scene_index)
            scene_attempt = int(scene_attempt)
        except (TypeError, ValueError):
            continue

        scene_mode = record.get("scene_mode")
        seed_offset = record.get("seed_offset")
        if scene_mode is None or seed_offset is None:
            legacy_scene_keys.add((scene_index, scene_attempt))
            continue

        try:
            scene_keys.add(
                (
                    str(scene_mode),
                    int(seed_offset),
                    scene_index,
                    scene_attempt,
                )
            )
        except (TypeError, ValueError):
            legacy_scene_keys.add((scene_index, scene_attempt))

    return scene_keys, legacy_scene_keys


def load_episode_progress(episode_dir: Path) -> Counter[str]:
    progress: Counter[str] = Counter()
    if not episode_dir.is_dir():
        return progress

    for path in sorted(episode_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            progress["<parse_failed>"] += 1
            continue
        progress[str(record.get("scene_mode") or MISSING_SCENE_MODE)] += 1
    return progress


def format_scene_key(
    scene_mode: str,
    seed_offset: int,
    scene_index: int,
    scene_attempt: int,
) -> str:
    return (
        f"mode={scene_mode} seed_offset={seed_offset} "
        f"scene={scene_index} attempt={scene_attempt}"
    )


class HumanDriveSimulator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.grid_config = GridConfig()
        self.teacher_config = TeacherConfig()
        self.dataset = ToySparseDriveV2Dataset(
            num_samples=args.num_samples,
            seed_offset=args.seed_offset,
            grid_config=self.grid_config,
            teacher_config=self.teacher_config,
            scene_config=make_scene_sampling_config(args.scene_mode),
        )
        self.speed_command_map = make_speed_command_map(
            accel_step=args.accel,
            brake_step=args.brake,
        )
        self.steer_command_map = make_steer_command_map(
            steer_step=math.radians(args.steer_deg),
        )
        self.pending_accel: float | None = None
        self.pending_accel_key: str | None = None
        self.pending_steer: float | None = None
        self.pending_steer_key: str | None = None
        self.fig: plt.Figure
        self.ax_scene: plt.Axes
        self.ax_speed: plt.Axes
        self.scene_index = 0
        self.scene_attempt = 0
        self.route_path_index = 0
        self.route_path = np.zeros((0, 3), dtype=np.float32)
        self.goal_xy = np.zeros((2,), dtype=np.float32)
        self.ego_size_xy = (4.8, 2.0)
        self.obstacles: list[dict[str, Any]] = []
        self.state = DriveState()
        self.time_s = 0.0
        self.steps: list[dict[str, Any]] = []
        self.ego_history: list[list[float]] = []
        self.done = False
        self.status = "driving"
        self.terminal_queue: queue.Queue[str] = queue.Queue()
        self.terminal_original_attrs: list[Any] | None = None
        self.terminal_reader_started = False
        self.used_scene_indices: set[int] = set()
        self.exit_progress_printed = False
        self.saved_scene_keys, self.legacy_saved_scene_keys = load_saved_scene_keys(
            self.args.episode_dir
        )
        self.episode_progress = load_episode_progress(self.args.episode_dir)
        print(
            f"[existing episodes] exact={len(self.saved_scene_keys)} "
            f"legacy={len(self.legacy_saved_scene_keys)} "
            f"dir={self.args.episode_dir}"
        )
        self.print_collection_progress(prefix="[current progress]")

    def current_scene_key(self, scene_index: int | None = None) -> SceneKey:
        return (
            str(self.args.scene_mode),
            int(self.args.seed_offset),
            int(self.scene_index if scene_index is None else scene_index),
            int(self.args.scene_attempt),
        )

    def scene_already_saved(self, scene_index: int | None = None) -> bool:
        key = self.current_scene_key(scene_index)
        legacy_key = (key[2], key[3])
        return (
            key in self.saved_scene_keys or legacy_key in self.legacy_saved_scene_keys
        )

    def print_collection_progress(self, prefix: str = "[collection progress]") -> None:
        total = sum(self.episode_progress.values())
        print(f"{prefix} total={total}")
        for mode, (target_min, target_max) in COLLECTION_TARGETS.items():
            count = int(self.episode_progress.get(mode, 0))
            need_min = max(target_min - count, 0)
            need_max = max(target_max - count, 0)
            print(
                f"  {mode}: {count}/{target_min}-{target_max} "
                f"need_min={need_min} need_high={need_max}"
            )
        missing = int(self.episode_progress.get(MISSING_SCENE_MODE, 0))
        if missing > 0:
            print(f"  {MISSING_SCENE_MODE}: {missing}")

    def print_exit_progress(self) -> None:
        if self.exit_progress_printed:
            return
        self.exit_progress_printed = True
        self.print_collection_progress(prefix="[exit progress]")

    def sample_unused_scene_index(self) -> int:
        if len(self.used_scene_indices) >= self.args.num_samples:
            self.used_scene_indices.clear()
        for _ in range(1000):
            scene_index = random.randrange(self.args.num_samples)
            if (
                scene_index not in self.used_scene_indices
                and not self.scene_already_saved(scene_index)
            ):
                return scene_index
        for scene_index in range(self.args.num_samples):
            if (
                scene_index not in self.used_scene_indices
                and not self.scene_already_saved(scene_index)
            ):
                return scene_index
        raise RuntimeError(
            "No unused scene remains for "
            f"mode={self.args.scene_mode}, seed_offset={self.args.seed_offset}, "
            f"scene_attempt={self.args.scene_attempt}"
        )

    def reset_scene(
        self,
        scene_index: int | None = None,
        mark_used: bool = True,
    ) -> None:
        self.scene_index = (
            int(scene_index)
            if scene_index is not None
            else self.sample_unused_scene_index()
        )
        if scene_index is not None and self.scene_already_saved(self.scene_index):
            print(
                "[existing scene] this requested scene already has an episode; "
                "it can be inspected or retried, but save will be skipped"
            )
        if mark_used:
            self.used_scene_indices.add(self.scene_index)
        self.scene_attempt = int(self.args.scene_attempt)
        (
            self.route_path_index,
            self.route_path,
            self.goal_xy,
            ego_state,
            raw_obstacles,
        ) = self.dataset.generate_scene(
            self.scene_index, scene_attempt=self.scene_attempt
        )
        self.ego_size_xy = ego_state.size_xy
        self.obstacles = [
            obstacle_to_dict(obstacle, obstacle_index)
            for obstacle_index, obstacle in enumerate(raw_obstacles)
        ]
        self.state = DriveState(
            x=float(ego_state.xy[0]),
            y=float(ego_state.xy[1]),
            yaw=float(ego_state.yaw),
            speed=float(ego_state.speed),
        )
        self.time_s = 0.0
        self.steps = []
        self.ego_history = [
            [self.state.x, self.state.y, self.state.yaw, self.state.speed]
        ]
        self.done = False
        self.status = "driving"
        self.pending_accel = None
        self.pending_accel_key = None
        self.pending_steer = None
        self.pending_steer_key = None

    def build_episode_record(self) -> dict[str, Any]:
        return {
            "source": "human_drive_v1",
            "scene_mode": str(self.args.scene_mode),
            "seed_offset": int(self.args.seed_offset),
            "scene_index": int(self.scene_index),
            "scene_attempt": int(self.scene_attempt),
            "route_path_index": int(self.route_path_index),
            "route_path": self.route_path.astype(float).tolist(),
            "goal_xy": self.goal_xy.astype(float).tolist(),
            "ego_size_xy": [float(self.ego_size_xy[0]), float(self.ego_size_xy[1])],
            "dt": float(self.args.dt),
            "wheelbase": float(self.args.wheelbase),
            "obstacles": self.obstacles,
            "status": self.status,
            "steps": self.steps,
            "ego_history": self.ego_history,
        }

    def handle_key(self, key: str) -> None:
        raw_key = key
        key = key.lower()
        if key == "escape" or raw_key == "Q":
            self.status = "quit"
            self.done = True
            self.print_exit_progress()
            self.restore_terminal_mode()
            plt.close(self.fig)
            return
        if key in {"enter", "\n", "\r", "f"}:
            if self.status == "collision":
                print(
                    f"[not saved] scene={self.scene_index} has collision; "
                    "press r to discard or backspace to retry"
                )
                return
            if self.status == "driving":
                self.status = "manual_finish"
            self.done = True
            if self.scene_already_saved(self.scene_index):
                key = self.current_scene_key(self.scene_index)
                print(
                    "[not saved] duplicate scene already exists: "
                    + format_scene_key(*key)
                )
                self.reset_scene()
                print(f"[new scene] scene={self.scene_index} not saved yet")
                self.draw()
                return
            output_path = save_episode(
                self.args.episode_dir,
                self.build_episode_record(),
            )
            self.saved_scene_keys.add(self.current_scene_key(self.scene_index))
            self.episode_progress[str(self.args.scene_mode)] += 1
            print(
                f"[saved] scene={self.scene_index} steps={len(self.steps)} "
                f"status={self.status} path={output_path}"
            )
            self.reset_scene()
            print(f"[new scene] scene={self.scene_index} not saved yet")
            self.draw()
            return
        if key == "backspace":
            print(f"[reset] scene={self.scene_index} discarded, no json saved")
            self.reset_scene(self.scene_index, mark_used=False)
            self.draw()
            return
        if key == "r":
            print(f"[discarded] scene={self.scene_index} no json saved")
            self.reset_scene()
            print(f"[new scene] scene={self.scene_index} not saved yet")
            self.draw()
            return
        if self.done:
            return
        if key in self.speed_command_map:
            self.pending_accel = self.speed_command_map[key]
            self.pending_accel_key = key
            self.try_step_pending_action()
            return

        if key in self.steer_command_map:
            self.pending_steer = self.steer_command_map[key]
            self.pending_steer_key = key
            self.try_step_pending_action()
            return

    def try_step_pending_action(self) -> None:
        if self.pending_accel is None or self.pending_steer is None:
            self.draw()
            return

        action_key = f"{self.pending_accel_key}+{self.pending_steer_key}"
        action = DriveAction(
            key=action_key,
            accel=float(self.pending_accel),
            steer=float(self.pending_steer),
        )
        self.pending_accel = None
        self.pending_accel_key = None
        self.pending_steer = None
        self.pending_steer_key = None

        before = asdict(self.state)
        next_state = step_ego(
            state=self.state,
            action=action,
            dt=self.args.dt,
            wheelbase=self.args.wheelbase,
            max_speed=self.args.max_speed,
        )
        self.time_s += float(self.args.dt)
        self.state = next_state
        self.ego_history.append(
            [self.state.x, self.state.y, self.state.yaw, self.state.speed]
        )
        collision = has_collision(
            state=self.state,
            ego_size_xy=self.ego_size_xy,
            obstacles=self.obstacles,
            time_s=self.time_s,
            safety_margin=self.args.safety_margin,
        )
        goal_reached = reached_goal(
            state=self.state,
            goal_xy=self.goal_xy,
            threshold=self.args.goal_threshold,
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
            self.done = True
            print(
                f"[not saved] scene={self.scene_index} collision at "
                f"step={len(self.steps)}; press r to discard or backspace to retry"
            )
        elif goal_reached:
            self.status = "goal_reached"
            self.done = True
            print(
                f"[finished] scene={self.scene_index} reached goal; "
                "press f to save or r to discard"
            )
        elif self.time_s >= self.args.max_time:
            self.status = "timeout"
            self.done = True
            print(
                f"[finished] scene={self.scene_index} timeout; "
                "press f to save or r to discard"
            )

        self.draw()

    def on_key_press(self, event: Any) -> None:
        if event.key is None:
            return
        self.handle_key(str(event.key))

    def start_terminal_reader(self) -> None:
        if self.terminal_reader_started or not sys.stdin.isatty():
            return
        self.terminal_reader_started = True
        self.terminal_original_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        def read_keys() -> None:
            while True:
                try:
                    key = sys.stdin.read(1)
                except Exception:
                    return
                if key:
                    if key == "\x1b":
                        self.terminal_queue.put("escape")
                    elif key == "\x7f":
                        self.terminal_queue.put("backspace")
                    else:
                        self.terminal_queue.put(key)

        threading.Thread(target=read_keys, daemon=True).start()
        print(
            "terminal key input enabled: "
            "+ accel, - brake, = keep speed, "
            "A/W/D steer left/straight/right, "
            "f save+next, r new, backspace reset, Q or esc quit"
        )

    def restore_terminal_mode(self) -> None:
        if self.terminal_original_attrs is None:
            return
        try:
            termios.tcsetattr(
                sys.stdin, termios.TCSADRAIN, self.terminal_original_attrs
            )
        finally:
            self.terminal_original_attrs = None

    def poll_terminal_keys(self) -> None:
        while True:
            try:
                key = self.terminal_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_key(key)

    def draw(self) -> None:
        self.ax_scene.clear()
        self.ax_speed.clear()

        draw_obstacles(
            ax=self.ax_scene,
            obstacles=self.obstacles,
            time_s=self.time_s,
        )
        self.ax_scene.plot(
            self.route_path[:, 1],
            self.route_path[:, 0],
            color="#1f77b4",
            linewidth=1.2,
            label="route",
        )
        self.ax_scene.scatter(
            [self.goal_xy[1]],
            [self.goal_xy[0]],
            color="#d62728",
            marker="*",
            s=120,
            label="goal",
            zorder=15,
        )
        history = np.asarray(self.ego_history, dtype=np.float32)
        self.ax_scene.plot(
            history[:, 1],
            history[:, 0],
            color="#2ca02c",
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="human ego",
        )
        polygon = ego_polygon_yx(self.state, self.ego_size_xy)
        self.ax_scene.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="#2ca02c",
            linewidth=2.0,
        )
        heading = np.array(
            [
                self.state.x + 4.0 * math.cos(self.state.yaw),
                self.state.y + 4.0 * math.sin(self.state.yaw),
            ],
            dtype=np.float32,
        )
        self.ax_scene.plot(
            [self.state.y, heading[1]],
            [self.state.x, heading[0]],
            color="#006400",
            linewidth=2.0,
        )
        self.ax_scene.set_title(
            f"scene {self.scene_index} | mode={self.args.scene_mode} | "
            f"t={self.time_s:.1f}s | "
            f"speed={self.state.speed:.2f}m/s | status={self.status}\n"
            f"pending speed={self.pending_accel_key or '_'} "
            f"steer={self.pending_steer_key or '_'} | "
            "+ accel, - brake, = keep speed; A/W/D left/straight/right; "
            "f save, r discard, backspace reset, Q/esc quit"
        )
        self.ax_scene.set_xlabel("y left [m]")
        self.ax_scene.set_ylabel("x forward [m]")
        self.ax_scene.set_xlim(self.grid_config.y_max, self.grid_config.y_min)
        self.ax_scene.set_ylim(self.grid_config.x_min, self.grid_config.x_max)
        self.ax_scene.set_aspect("equal", adjustable="box")
        self.ax_scene.grid(True, alpha=0.25)
        self.ax_scene.legend(loc="upper right", fontsize=8)

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
        self.ax_speed.set_title("ego speed")
        self.ax_speed.set_xlabel("time [s]")
        self.ax_speed.set_ylabel("speed [m/s]")
        self.ax_speed.set_xlim(0.0, max(self.args.max_time, 1.0))
        self.ax_speed.set_ylim(0.0, max(self.args.max_speed, max(speeds) + 1.0))
        self.ax_speed.grid(True, alpha=0.25)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self) -> None:
        self.reset_scene(self.args.scene_index)
        print(f"[new scene] scene={self.scene_index} not saved yet")
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.3, 0.7))
        self.ax_scene = self.fig.add_subplot(gs[0, 0])
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.draw()
        if self.args.smoke_test:
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.args.output_dir / "drive_sim_smoke.png"
            self.fig.savefig(output_path, dpi=160)
            plt.close(self.fig)
            print(f"smoke image: {output_path}")
            return
        if self.args.terminal_keys:
            self.start_terminal_reader()
            timer = self.fig.canvas.new_timer(interval=50)
            timer.add_callback(self.poll_terminal_keys)
            timer.start()
        try:
            plt.show()
        finally:
            self.print_exit_progress()
            self.restore_terminal_mode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive step-by-step human driving data collector.",
    )
    parser.add_argument("--seed-offset", type=int, default=70_000)
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--scene-attempt", type=int, default=0)
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
        default="random",
        help="Scene distribution used for newly sampled human driving episodes.",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--accel", type=float, default=1.5)
    parser.add_argument("--brake", type=float, default=3.0)
    parser.add_argument("--steer-deg", type=float, default=12.0)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--max-speed", type=float, default=16.0)
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--goal-threshold", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--no-terminal-keys",
        dest="terminal_keys",
        action="store_false",
    )
    parser.set_defaults(terminal_keys=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = HumanDriveSimulator(args)
    simulator.run()


if __name__ == "__main__":
    main()
