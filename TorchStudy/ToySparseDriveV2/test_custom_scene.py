from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import (
    Obstacle,
    TIME_STEPS,
    obstacles_to_tensors,
    rasterize_dynamic_obstacles,
)
from grid import GridConfig, draw_points, draw_polyline, make_empty_grid
from model import ToySparseDriveV2Model
from vocab import SparseDriveVocab

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints_auto_mix_straight_lowavoid_248"
    / "best_human_model.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "test_custom_scene" / "custom_scene.png"
FUTURE_TIMES = TIME_STEPS

# Edit this block for quick custom-scene experiments. CLI arguments are still
# available, but this file-level config is the default source of truth.
CUSTOM_SCENE = {
    "checkpoint_path": DEFAULT_CHECKPOINT,
    "output_path": DEFAULT_OUTPUT,
    "goal_xy": [35.0, -6.0],
    "ego_speed": 4.5,
    "ego_size_xy": [4.8, 2.0],
    "route_points": 70,
    "route_extend": 20.0,
    # Each obstacle is [x, y, length, width, vx, vy].
    "obstacles": [
        [16.0, -1.0, 5.0, 3.0, 0.0, 0.0],
        [21.0, 3.0, 5.0, 3.0, 1.0, 0.0],
        [22.0, -3.0, 5.0, 3.0, 1.0, -0.5],
    ],
    "model_topk_path": 80,
    "collision_penalty": 20.0,
    "use_safety_score": True,
    "show_topk_paths": 8,
}


def parse_obstacle(text: str) -> Obstacle:
    parts = [float(item.strip()) for item in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("--obstacle must be x,y,length,width,vx,vy")
    x, y, length, width, vx, vy = parts
    return Obstacle(
        center_xy=np.array([x, y], dtype=np.float32),
        size_xy=(float(length), float(width)),
        velocity_xy=np.array([vx, vy], dtype=np.float32),
    )


def obstacle_from_values(values: list[float] | tuple[float, ...]) -> Obstacle:
    if len(values) != 6:
        raise ValueError("Obstacle config must be [x, y, length, width, vx, vy]")
    x, y, length, width, vx, vy = [float(item) for item in values]
    return Obstacle(
        center_xy=np.array([x, y], dtype=np.float32),
        size_xy=(float(length), float(width)),
        velocity_xy=np.array([vx, vy], dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ToySparseDriveV2 on a hand-built open-space scene."
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--ego-speed", type=float, default=None)
    parser.add_argument("--ego-length", type=float, default=None)
    parser.add_argument("--ego-width", type=float, default=None)
    parser.add_argument("--route-points", type=int, default=None)
    parser.add_argument("--route-extend", type=float, default=None)
    parser.add_argument(
        "--obstacle",
        action="append",
        type=parse_obstacle,
        default=None,
        help="Obstacle as x,y,length,width,vx,vy. Can be repeated.",
    )
    parser.add_argument("--model-topk-path", type=int, default=None)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--no-use-safety-score", action="store_true")
    parser.add_argument("--show-topk-paths", type=int, default=None)
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def choose_arg(cli_value, config_value, checkpoint_value=None):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return checkpoint_value


def build_start_to_goal_route(
    goal_xy: np.ndarray,
    num_points: int,
    extend_after_goal_m: float,
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


def make_default_obstacles(goal_xy: np.ndarray) -> list[Obstacle]:
    goal = np.asarray(goal_xy, dtype=np.float32)
    direction = goal / max(float(np.linalg.norm(goal)), 1.0e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    center = direction * 16.0 + normal * 1.2
    return [
        Obstacle(
            center_xy=center.astype(np.float32),
            size_xy=(5.0, 3.0),
            velocity_xy=np.array([0.0, 0.0], dtype=np.float32),
        )
    ]


def build_sample(
    goal_xy: np.ndarray,
    route_path: np.ndarray,
    obstacles: list[Obstacle],
    ego_speed: float,
    ego_size_xy: tuple[float, float],
    grid_config: GridConfig,
    max_obstacles: int = 12,
) -> dict[str, torch.Tensor]:
    input_grid = make_empty_grid(channels=4, config=grid_config)
    rasterize_dynamic_obstacles(input_grid, obstacles, grid_config)
    draw_polyline(
        input_grid[2],
        route_path[:, :2],
        value=1.0,
        radius=1,
        samples_per_segment=4,
        config=grid_config,
    )
    draw_points(
        input_grid[3],
        goal_xy[None],
        value=1.0,
        radius=3,
        config=grid_config,
    )
    obstacle_centers, obstacle_sizes, obstacle_velocities, obstacle_mask = (
        obstacles_to_tensors(obstacles, max_obstacles=max_obstacles)
    )
    ego_state = torch.tensor(
        [float(ego_speed), float(ego_size_xy[0]), float(ego_size_xy[1])],
        dtype=torch.float32,
    )
    return {
        "input_grid": torch.from_numpy(input_grid).float(),
        "route_path": torch.from_numpy(route_path).float(),
        "goal_xy": torch.from_numpy(goal_xy).float(),
        "ego_state": ego_state,
        "obstacle_centers": obstacle_centers.float(),
        "obstacle_sizes": obstacle_sizes.float(),
        "obstacle_velocities": obstacle_velocities.float(),
        "obstacle_mask": obstacle_mask.float(),
    }


def compute_planning_scores(
    trajectory_scores: torch.Tensor,
    no_collision_logits: torch.Tensor,
    collision_penalty: float,
) -> torch.Tensor:
    collision_prob = torch.sigmoid(-no_collision_logits)
    return trajectory_scores - float(collision_penalty) * collision_prob


def select_prediction(
    output: dict[str, torch.Tensor],
    use_safety_score: bool,
    collision_penalty: float,
) -> dict[str, torch.Tensor]:
    scores = output["trajectory_scores"]
    if use_safety_score:
        scores = compute_planning_scores(
            trajectory_scores=scores,
            no_collision_logits=output["no_collision_logits"],
            collision_penalty=collision_penalty,
        )
    best_candidate = scores.argmax(dim=-1)
    batch_indices = torch.arange(scores.shape[0], device=scores.device)
    return {
        "candidate_index": best_candidate,
        "trajectory": output["candidate_trajectories"][batch_indices, best_candidate],
        "path_index": output["candidate_path_indices"][batch_indices, best_candidate],
        "velocity_index": output["candidate_velocity_indices"][
            batch_indices,
            best_candidate,
        ],
        "score": scores[batch_indices, best_candidate],
    }


def trajectory_collision_flags(
    trajectory_xy: np.ndarray,
    obstacles: list[Obstacle],
    ego_size_xy: tuple[float, float],
    safety_margin: float = 0.3,
) -> np.ndarray:
    flags = np.zeros((trajectory_xy.shape[0],), dtype=bool)
    ego_size = np.asarray(ego_size_xy, dtype=np.float32)
    for i, time_s in enumerate(FUTURE_TIMES):
        point = trajectory_xy[i]
        for obstacle in obstacles:
            center = obstacle.center_xy + obstacle.velocity_xy * float(time_s)
            obs_size = np.asarray(obstacle.size_xy, dtype=np.float32)
            half_extent = 0.5 * (ego_size + obs_size) + float(safety_margin)
            if np.all(np.abs(point - center) <= half_extent):
                flags[i] = True
                break
    return flags


def draw_rectangle_world(
    ax: plt.Axes,
    center_xy: np.ndarray,
    size_xy: tuple[float, float] | np.ndarray,
    color: str,
    alpha: float,
) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    size_xy = np.asarray(size_xy, dtype=np.float32)
    rect = plt.Rectangle(
        (
            float(center_xy[1] - size_xy[1] / 2.0),
            float(center_xy[0] - size_xy[0] / 2.0),
        ),
        float(size_xy[1]),
        float(size_xy[0]),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.9,
    )
    ax.add_patch(rect)


def visualize_custom_scene(
    output_path: Path,
    sample: dict[str, torch.Tensor],
    obstacles: list[Obstacle],
    vocab: SparseDriveVocab,
    output: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    ego_size_xy: tuple[float, float],
    use_safety_score: bool,
    collision_penalty: float,
    show_topk_paths: int,
) -> None:
    route_path = sample["route_path"].detach().cpu().numpy()
    goal_xy = sample["goal_xy"].detach().cpu().numpy()
    pred_trajectory = pred["trajectory"][0].detach().cpu().numpy()
    pred_path_index = int(pred["path_index"][0].detach().cpu())
    pred_velocity_index = int(pred["velocity_index"][0].detach().cpu())
    pred_path = vocab.path[pred_path_index].detach().cpu().numpy()
    pred_velocity = vocab.velocity[pred_velocity_index].detach().cpu().numpy()
    current_speed = float(sample["ego_state"][0])
    collision_flags = trajectory_collision_flags(
        pred_trajectory[:, :2],
        obstacles=obstacles,
        ego_size_xy=ego_size_xy,
    )

    topk_indices = output["model_topk_path_indices"][0].detach().cpu().numpy()
    topk_scores = output["model_topk_path_scores"][0].detach().cpu().numpy()
    num_show = max(0, min(int(show_topk_paths), len(topk_indices)))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.0, 1.15),
        height_ratios=(1.0, 1.0),
    )
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_paths = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])

    for obstacle_index, obstacle in enumerate(obstacles):
        draw_rectangle_world(
            ax_scene,
            center_xy=obstacle.center_xy,
            size_xy=obstacle.size_xy,
            color="#666666",
            alpha=0.45,
        )
        ax_scene.annotate(
            f"obs{obstacle_index}",
            xy=(float(obstacle.center_xy[1]), float(obstacle.center_xy[0])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#333333"},
        )
        for time_s in FUTURE_TIMES:
            future_center = obstacle.center_xy + obstacle.velocity_xy * float(time_s)
            draw_rectangle_world(
                ax_scene,
                center_xy=future_center,
                size_xy=obstacle.size_xy,
                color="#ff9900",
                alpha=0.06,
            )

    ax_scene.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="planner route",
    )
    ax_scene.plot(
        pred_trajectory[:, 1],
        pred_trajectory[:, 0],
        color="#d62728",
        linewidth=2.0,
        marker="x",
        markersize=5,
        label=f"pred p{pred_path_index} v{pred_velocity_index}",
    )
    if collision_flags.any():
        bad = pred_trajectory[collision_flags]
        ax_scene.scatter(
            bad[:, 1],
            bad[:, 0],
            color="#000000",
            s=55,
            marker="X",
            label="pred collision point",
            zorder=8,
        )
    ax_scene.scatter([0.0], [0.0], color="#2ca02c", s=70, label="ego")
    ax_scene.scatter(
        [float(goal_xy[1])],
        [float(goal_xy[0])],
        color="#d62728",
        marker="*",
        s=140,
        label="goal",
    )
    ax_scene.set_title(
        "custom scene | "
        f"safety={'on' if use_safety_score else 'off'} "
        f"penalty={collision_penalty:g}"
    )
    ax_scene.set_xlabel("y left [m]")
    ax_scene.set_ylabel("x forward [m]")
    ax_scene.grid(True, alpha=0.25)
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.legend(loc="best")

    ax_paths.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        label="planner route",
    )
    colors = plt.get_cmap("tab10")
    for rank in range(num_show):
        path_index = int(topk_indices[rank])
        path = vocab.path[path_index].detach().cpu().numpy()
        ax_paths.plot(
            path[:, 1],
            path[:, 0],
            color=colors(rank % 10),
            alpha=0.35,
            linewidth=1.2,
            label=f"top{rank + 1} p{path_index} s={topk_scores[rank]:.2f}",
        )
    ax_paths.plot(
        pred_path[:, 1],
        pred_path[:, 0],
        color="#d62728",
        linewidth=2.2,
        marker="x",
        markersize=2.5,
        label=f"selected path p{pred_path_index}",
    )
    ax_paths.scatter([0.0], [0.0], color="#2ca02c", s=50, label="start")
    ax_paths.scatter(
        [float(goal_xy[1])],
        [float(goal_xy[0])],
        color="#d62728",
        marker="*",
        s=100,
        label="goal",
    )
    ax_paths.set_title("path scorer top-k")
    ax_paths.set_xlabel("y left [m]")
    ax_paths.set_ylabel("x forward [m]")
    ax_paths.grid(True, alpha=0.25)
    ax_paths.set_aspect("equal", adjustable="box")
    ax_paths.legend(loc="best", fontsize=8)

    velocity_times = np.concatenate(
        [np.array([0.0], dtype=np.float32), FUTURE_TIMES.astype(np.float32)]
    )
    pred_velocity_with_current = np.concatenate(
        [np.array([current_speed], dtype=np.float32), pred_velocity.astype(np.float32)]
    )
    ax_velocity.plot(
        velocity_times,
        pred_velocity_with_current,
        color="#d62728",
        linewidth=2.0,
        marker="x",
        markersize=5,
        label=f"selected velocity v{pred_velocity_index}",
    )
    ax_velocity.scatter(
        [0.0],
        [current_speed],
        color="#1f77b4",
        s=50,
        label="current ego speed",
    )
    ax_velocity.set_title("selected velocity vocab")
    ax_velocity.set_xlabel("time [s]")
    ax_velocity.set_ylabel("speed [m/s]")
    ax_velocity.set_xticks(velocity_times)
    ax_velocity.grid(True, alpha=0.25)
    ax_velocity.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = CUSTOM_SCENE

    checkpoint_path = Path(args.checkpoint_path or config["checkpoint_path"])
    output_path = Path(args.output_path or config["output_path"])
    goal_config = config["goal_xy"]
    goal_x = float(args.goal_x if args.goal_x is not None else goal_config[0])
    goal_y = float(args.goal_y if args.goal_y is not None else goal_config[1])
    ego_speed = float(
        args.ego_speed if args.ego_speed is not None else config["ego_speed"]
    )
    ego_size_config = config["ego_size_xy"]
    ego_length = float(
        args.ego_length if args.ego_length is not None else ego_size_config[0]
    )
    ego_width = float(
        args.ego_width if args.ego_width is not None else ego_size_config[1]
    )
    route_points = int(
        args.route_points if args.route_points is not None else config["route_points"]
    )
    route_extend = float(
        args.route_extend if args.route_extend is not None else config["route_extend"]
    )
    show_topk_paths = int(
        args.show_topk_paths
        if args.show_topk_paths is not None
        else config["show_topk_paths"]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(checkpoint)
    model_topk_path = int(
        choose_arg(
            args.model_topk_path,
            config["model_topk_path"],
            ckpt_args.get("model_topk_path"),
        )
    )
    collision_penalty = float(
        choose_arg(
            args.collision_penalty,
            config["collision_penalty"],
            ckpt_args.get("collision_penalty"),
        )
    )
    use_safety_score = bool(config["use_safety_score"]) and not bool(
        args.no_use_safety_score
    )

    goal_xy = np.array([goal_x, goal_y], dtype=np.float32)
    route_path = build_start_to_goal_route(
        goal_xy=goal_xy,
        num_points=route_points,
        extend_after_goal_m=route_extend,
    )
    obstacles = (
        args.obstacle
        if args.obstacle is not None
        else [obstacle_from_values(item) for item in config["obstacles"]]
    )
    if len(obstacles) == 0:
        obstacles = make_default_obstacles(goal_xy)
    ego_size_xy = (ego_length, ego_width)

    grid_config = GridConfig()
    sample = build_sample(
        goal_xy=goal_xy,
        route_path=route_path,
        obstacles=obstacles,
        ego_speed=ego_speed,
        ego_size_xy=ego_size_xy,
        grid_config=grid_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_cpu = SparseDriveVocab.load()
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model(topk_path=model_topk_path).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    with torch.no_grad():
        output = model(
            input_grid=batch["input_grid"],
            path_vocab=vocab.path,
            velocity_vocab=vocab.velocity,
            trajectory_vocab=vocab.trajectory,
            ego_state=batch["ego_state"],
            goal_xy=batch["goal_xy"],
            route_path=batch["route_path"],
            obstacle_centers=batch["obstacle_centers"],
            obstacle_sizes=batch["obstacle_sizes"],
            obstacle_mask=batch["obstacle_mask"],
        )
        pred = select_prediction(
            output=output,
            use_safety_score=use_safety_score,
            collision_penalty=collision_penalty,
        )

    pred_candidate = int(pred["candidate_index"][0].detach().cpu())
    pred_path = int(pred["path_index"][0].detach().cpu())
    pred_velocity = int(pred["velocity_index"][0].detach().cpu())
    pred_score = float(pred["score"][0].detach().cpu())
    trajectory_score = float(
        output["trajectory_scores"][0, pred_candidate].detach().cpu()
    )
    no_collision_logit = float(
        output["no_collision_logits"][0, pred_candidate].detach().cpu()
    )
    collision_prob = float(torch.sigmoid(torch.tensor(-no_collision_logit)).item())
    pred_traj = pred["trajectory"][0].detach().cpu().numpy()
    collision_flags = trajectory_collision_flags(
        pred_traj[:, :2],
        obstacles=obstacles,
        ego_size_xy=ego_size_xy,
    )

    visualize_custom_scene(
        output_path=output_path,
        sample={key: value.detach().cpu() for key, value in sample.items()},
        obstacles=obstacles,
        vocab=vocab_cpu,
        output={key: value.detach().cpu() for key, value in output.items()},
        pred={key: value.detach().cpu() for key, value in pred.items()},
        ego_size_xy=ego_size_xy,
        use_safety_score=use_safety_score,
        collision_penalty=collision_penalty,
        show_topk_paths=show_topk_paths,
    )

    print(f"checkpoint epoch: {checkpoint.get('epoch', 'n/a')}")
    print(f"checkpoint val_metrics: {checkpoint.get('val_metrics', 'n/a')}")
    print(f"device: {device}")
    print(f"model_topk_path: {model_topk_path}")
    print(f"goal_xy: {goal_xy.tolist()}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"ego_speed: {ego_speed:.3f} m/s")
    print(f"obstacles: {len(obstacles)}")
    for index, obstacle in enumerate(obstacles):
        print(
            f"  obs{index}: center={obstacle.center_xy.tolist()} "
            f"size={list(obstacle.size_xy)} vel={obstacle.velocity_xy.tolist()}"
        )
    print(f"pred path: {pred_path}")
    print(f"pred velocity: {pred_velocity}")
    print(f"pred candidate index: {pred_candidate}")
    print(f"planning_score: {pred_score:.4f}")
    print(f"trajectory_score: {trajectory_score:.4f}")
    print(f"no_collision_logit: {no_collision_logit:.4f}")
    print(f"collision_prob_from_head: {collision_prob:.4f}")
    print(f"axis_aligned_collision_check: {bool(collision_flags.any())}")
    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
