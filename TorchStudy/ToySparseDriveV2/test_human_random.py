from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "human_drive",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from human_dataset import (
    FUTURE_TIMES,
    HumanDriveDataset,
    cumulative_distance,
    interp_points_by_distance,
)
from model import ToySparseDriveV2Model
from vocab import SparseDriveVocab

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints_human" / "best_human_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_human_random"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=PROJECT_ROOT / "human_drive" / "episodes",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "human_drive" / "cache",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--path-chunk-size", type=int, default=32)
    parser.add_argument("--label-batch-size", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--use-safety-score",
        dest="use_safety_score",
        action="store_true",
        help="Use no_collision_logits as a collision-risk penalty when selecting candidates.",
    )
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument(
        "--no-use-safety-score",
        dest="use_safety_score",
        action="store_false",
        help="Select using raw trajectory scores only.",
    )
    parser.set_defaults(use_safety_score=True)
    return parser.parse_args()


def split_episode_paths(
    episode_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    paths = list(episode_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    num_val = max(1, int(round(len(paths) * val_ratio)))
    return paths[num_val:], paths[:num_val]


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def checkpoint_path_list(checkpoint: dict[str, Any], key: str) -> list[Path] | None:
    raw_paths = checkpoint.get(key)
    if raw_paths is None:
        return None
    return [Path(path) for path in raw_paths]


def choose_arg(
    cli_value: Any,
    checkpoint_value: Any,
    default_value: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    if checkpoint_value is not None:
        return checkpoint_value
    return default_value


def draw_rectangle_world(
    ax: plt.Axes,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    size_xy = np.asarray(size_xy, dtype=np.float32)
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


def setup_scene_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_xlim(60.0, -60.0)
    ax.set_ylim(-50.0, 70.0)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def setup_local_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


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


def compute_errors(
    pred_trajectory: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[float, float]:
    mask = target_mask.float()
    diff = pred_trajectory[..., :2] - target_trajectory[..., :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    traj_l2 = (point_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    endpoint_index = mask.sum(dim=-1).long().clamp(min=1) - 1
    batch_indices = torch.arange(pred_trajectory.shape[0], device=pred_trajectory.device)
    pred_endpoint = pred_trajectory[batch_indices, endpoint_index, :2]
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    endpoint_l2 = (pred_endpoint - target_endpoint).pow(2).sum(dim=-1).sqrt()
    return float(traj_l2.mean().cpu()), float(endpoint_l2.mean().cpu())


def rank_of_score(scores: torch.Tensor, index: int) -> int:
    score = scores[index]
    return int((scores > score).sum().item()) + 1


def compute_long_path_label_audit(
    dataset: HumanDriveDataset,
    sample_index: int,
    vocab: SparseDriveVocab,
) -> tuple[bool, float, list[dict[str, float | int]], dict[int, int]]:
    human_xy, valid = dataset.build_long_path_target(sample_index)
    if not valid:
        return False, 0.0, [], {}

    human_distance = cumulative_distance(human_xy)
    human_progress = float(human_distance[-1])
    query_distance = np.linspace(
        0.0,
        human_progress,
        dataset.long_path_compare_points,
        dtype=np.float32,
    )
    human_points = interp_points_by_distance(human_xy, query_distance)
    path_xy = vocab.path[..., :2].detach().cpu().numpy().astype(np.float32)

    rows: list[dict[str, float | int]] = []
    for path_index in range(vocab.num_path):
        path_points = interp_points_by_distance(path_xy[path_index], query_distance)
        point_dist = np.linalg.norm(path_points - human_points, axis=-1)
        rows.append(
            {
                "path_index": path_index,
                "mean_l2": float(point_dist.mean()),
                "max_l2": float(point_dist.max()),
                "endpoint_l2": float(point_dist[-1]),
                "path_length": float(cumulative_distance(path_xy[path_index])[-1]),
            }
        )

    rows.sort(key=lambda item: float(item["mean_l2"]))
    ranks = {int(row["path_index"]): rank for rank, row in enumerate(rows, start=1)}
    return True, human_progress, rows, ranks


def find_target_candidate_index(
    output: dict[str, torch.Tensor],
    target_path_index: int,
    target_velocity_index: int,
) -> int | None:
    candidate_path = output["candidate_path_indices"][0].detach().cpu()
    candidate_velocity = output["candidate_velocity_indices"][0].detach().cpu()
    match = (candidate_path == target_path_index) & (
        candidate_velocity == target_velocity_index
    )
    if not bool(match.any()):
        return None
    return int(match.nonzero(as_tuple=False)[0, 0])


def visualize_sample(
    sample: dict[str, torch.Tensor],
    vocab: SparseDriveVocab,
    pred: dict[str, torch.Tensor],
    target_path_index: int,
    target_velocity_index: int,
    pred_path_index: int,
    pred_velocity_index: int,
    output_path: Path,
    title_prefix: str,
) -> None:
    route_path = sample["route_path"].detach().cpu().numpy()
    target_trajectory = sample["target_trajectory"].detach().cpu().numpy()
    target_mask = sample["target_trajectory_mask"].detach().cpu().numpy()
    target_trajectory = target_trajectory[target_mask > 0.5]
    pred_trajectory = pred["trajectory"][0].detach().cpu().numpy()

    target_path = vocab.path[target_path_index].detach().cpu().numpy()
    pred_path = vocab.path[pred_path_index].detach().cpu().numpy()
    target_velocity = vocab.velocity[target_velocity_index].detach().cpu().numpy()
    pred_velocity = vocab.velocity[pred_velocity_index].detach().cpu().numpy()
    current_speed = float(sample["ego_state"][0])

    velocity_times = np.concatenate(
        [np.array([0.0], dtype=np.float32), FUTURE_TIMES.astype(np.float32)]
    )
    target_velocity_with_current = np.concatenate(
        [np.array([current_speed], dtype=np.float32), target_velocity.astype(np.float32)]
    )
    pred_velocity_with_current = np.concatenate(
        [np.array([current_speed], dtype=np.float32), pred_velocity.astype(np.float32)]
    )

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.0, 1.2),
        height_ratios=(1.0, 1.0),
    )
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_path = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])

    obstacle_centers = sample["obstacle_centers"].detach().cpu().numpy()
    obstacle_sizes = sample["obstacle_sizes"].detach().cpu().numpy()
    obstacle_velocities = sample["obstacle_velocities"].detach().cpu().numpy()
    obstacle_mask = sample["obstacle_mask"].detach().cpu().numpy()
    for obstacle_index in np.where(obstacle_mask > 0.5)[0]:
        center_xy = obstacle_centers[obstacle_index]
        size_xy = obstacle_sizes[obstacle_index]
        velocity_xy = obstacle_velocities[obstacle_index]
        draw_rectangle_world(
            ax_scene,
            center_xy=center_xy,
            size_xy=size_xy,
            color="#666666",
            alpha=0.45,
        )
        ax_scene.annotate(
            f"obs{obstacle_index}",
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
            zorder=10,
        )
        for time_s in FUTURE_TIMES:
            future_center = center_xy + velocity_xy * float(time_s)
            draw_rectangle_world(
                ax_scene,
                center_xy=future_center,
                size_xy=size_xy,
                color="#ff9900",
                alpha=0.07,
            )

    ax_scene.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.75,
        label="route path",
    )
    ax_scene.plot(
        target_trajectory[:, 1],
        target_trajectory[:, 0],
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="human future",
    )
    ax_scene.plot(
        pred_trajectory[:, 1],
        pred_trajectory[:, 0],
        color="#d62728",
        linewidth=1.8,
        marker="x",
        markersize=5,
        label="pred trajectory",
    )
    ax_scene.scatter(
        [float(sample["goal_xy"][1])],
        [float(sample["goal_xy"][0])],
        color="#d62728",
        marker="*",
        s=120,
        label="goal",
        zorder=6,
    )
    setup_scene_axis(
        ax_scene,
        (
            f"{title_prefix} | 4s p{target_path_index} v{target_velocity_index} "
            f"| pred p{pred_path_index} v{pred_velocity_index}"
        ),
    )
    ax_scene.legend(loc="upper right")

    ax_path.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linewidth=1.1,
        linestyle="--",
        alpha=0.75,
        label="route",
    )
    ax_path.plot(
        target_path[:, 1],
        target_path[:, 0],
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=2.5,
        label=f"4s matched path p{target_path_index}",
    )
    ax_path.plot(
        pred_path[:, 1],
        pred_path[:, 0],
        color="#d62728",
        linewidth=1.8,
        marker="x",
        markersize=2.5,
        label=f"pred path p{pred_path_index}",
    )
    ax_path.scatter([0.0], [0.0], color="#2ca02c", s=45, label="start", zorder=5)
    setup_local_axis(ax_path, "selected path vocab shape")
    ax_path.legend(loc="best")

    ax_velocity.plot(
        velocity_times,
        target_velocity_with_current,
        color="#2ca02c",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label=f"4s matched velocity v{target_velocity_index}",
    )
    ax_velocity.plot(
        velocity_times,
        pred_velocity_with_current,
        color="#d62728",
        linewidth=1.8,
        marker="x",
        markersize=4,
        label=f"pred velocity v{pred_velocity_index}",
    )
    ax_velocity.scatter(
        [0.0],
        [current_speed],
        color="#1f77b4",
        s=45,
        label="current ego speed",
        zorder=5,
    )
    ax_velocity.set_title("selected velocity vocab shape | current + 8 future steps")
    ax_velocity.set_xlabel("time [s]")
    ax_velocity.set_ylabel("speed [m/s]")
    ax_velocity.set_xticks(velocity_times)
    ax_velocity.grid(True, alpha=0.25)
    ax_velocity.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_episode_diagnostic(
    dataset: HumanDriveDataset,
    sample_index: int,
    sample: dict[str, torch.Tensor],
    vocab: SparseDriveVocab,
    short_path_index: int,
    long_path_index: int,
    long_path_indices: np.ndarray,
    long_path_weights: np.ndarray,
    output_path: Path,
) -> None:
    raw_sample = dataset.samples[sample_index]
    meta = dataset.metas[sample_index]
    data = raw_sample["episode"]
    ego_history = np.asarray(raw_sample["ego_history"], dtype=np.float32)
    times = np.asarray(raw_sample["times"], dtype=np.float32)
    step_index = int(raw_sample["step_index"])
    current_state = ego_history[step_index]
    current_time = float(times[step_index])
    current_xy = current_state[:2]

    route_path = np.asarray(data["route_path"], dtype=np.float32)
    goal_xy = np.asarray(data["goal_xy"], dtype=np.float32)
    ego_size_xy = np.asarray(data["ego_size_xy"], dtype=np.float32)
    short_end_time = current_time + float(FUTURE_TIMES[-1])
    long_end_time = min(current_time + dataset.long_path_horizon, float(times[-1]))
    short_mask = (times >= current_time - 1.0e-6) & (times <= short_end_time + 1.0e-6)
    long_mask = (times >= current_time - 1.0e-6) & (times <= long_end_time + 1.0e-6)

    short_world = ego_history[short_mask, :2]
    long_world = ego_history[long_mask, :2]
    short_local, _ = dataset.build_target_trajectory(sample_index)
    long_local, long_valid = dataset.build_long_path_target(sample_index)
    short_path = vocab.path[short_path_index].detach().cpu().numpy()
    long_path = vocab.path[long_path_index].detach().cpu().numpy()
    top_long_paths = [
        (
            int(path_index),
            float(weight),
            vocab.path[int(path_index)].detach().cpu().numpy(),
        )
        for path_index, weight in zip(long_path_indices[:8], long_path_weights[:8])
    ]

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.15, 1.0))
    ax_world = fig.add_subplot(gs[0, 0])
    ax_local = fig.add_subplot(gs[0, 1])

    for raw_obstacle in data["obstacles"]:
        center_xy = np.asarray(raw_obstacle["center_xy"], dtype=np.float32)
        velocity_xy = np.asarray(raw_obstacle["velocity_xy"], dtype=np.float32)
        size_xy = np.asarray(raw_obstacle["size_xy"], dtype=np.float32)
        current_center = center_xy + velocity_xy * current_time
        draw_rectangle_world(
            ax_world,
            center_xy=current_center,
            size_xy=size_xy,
            color="#666666",
            alpha=0.45,
        )
        obstacle_id = raw_obstacle.get("id", "")
        ax_world.annotate(
            str(obstacle_id),
            xy=(float(current_center[1]), float(current_center[0])),
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
            zorder=10,
        )
        for time_s in FUTURE_TIMES:
            future_center = current_center + velocity_xy * float(time_s)
            draw_rectangle_world(
                ax_world,
                center_xy=future_center,
                size_xy=size_xy,
                color="#ff9900",
                alpha=0.06,
            )

    draw_rectangle_world(
        ax_world,
        center_xy=current_xy,
        size_xy=ego_size_xy,
        color="#2ca02c",
        alpha=0.20,
    )
    ax_world.plot(
        route_path[:, 1],
        route_path[:, 0],
        color="#1f77b4",
        linewidth=1.4,
        label="route",
    )
    ax_world.plot(
        ego_history[:, 1],
        ego_history[:, 0],
        color="#999999",
        linewidth=1.3,
        linestyle="--",
        label="full human episode",
    )
    ax_world.plot(
        long_world[:, 1],
        long_world[:, 0],
        color="#2ca02c",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=f"human future <= {dataset.long_path_horizon:g}s",
    )
    ax_world.plot(
        short_world[:, 1],
        short_world[:, 0],
        color="#d62728",
        linewidth=2.0,
        marker="x",
        markersize=4,
        label=f"raw future <= {float(FUTURE_TIMES[-1]):g}s",
    )
    ax_world.scatter(
        [float(current_xy[1])],
        [float(current_xy[0])],
        color="#2ca02c",
        s=70,
        label="current ego",
        zorder=6,
    )
    ax_world.scatter(
        [float(goal_xy[1])],
        [float(goal_xy[0])],
        color="#d62728",
        marker="*",
        s=130,
        label="goal",
        zorder=6,
    )
    ax_world.set_title(
        f"episode diagnostic | sample {sample_index} | step {step_index} | t={current_time:.2f}s"
    )
    ax_world.set_xlabel("y left [m]")
    ax_world.set_ylabel("x forward [m]")
    ax_world.grid(True, alpha=0.25)
    ax_world.set_aspect("equal", adjustable="box")
    ax_world.legend(loc="best")

    route_local = sample["route_path"].detach().cpu().numpy()
    goal_local = sample["goal_xy"].detach().cpu().numpy()
    obstacle_centers = sample["obstacle_centers"].detach().cpu().numpy()
    obstacle_sizes = sample["obstacle_sizes"].detach().cpu().numpy()
    obstacle_mask = sample["obstacle_mask"].detach().cpu().numpy()
    for obstacle_index in np.where(obstacle_mask > 0.5)[0]:
        draw_rectangle_world(
            ax_local,
            center_xy=obstacle_centers[obstacle_index],
            size_xy=obstacle_sizes[obstacle_index],
            color="#666666",
            alpha=0.28,
        )

    ax_local.plot(
        route_local[:, 1],
        route_local[:, 0],
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label="local route",
    )
    ax_local.plot(
        short_local[:, 1],
        short_local[:, 0],
        color="#d62728",
        linewidth=2.0,
        marker="x",
        markersize=4,
        label="4s target trajectory",
    )
    if long_valid:
        ax_local.plot(
            long_local[:, 1],
            long_local[:, 0],
            color="#2ca02c",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=f"{dataset.long_path_horizon:g}s long target",
        )
    ax_local.plot(
        short_path[:, 1],
        short_path[:, 0],
        color="#9467bd",
        linewidth=1.3,
        alpha=0.60,
        label=f"4s matched path p{short_path_index}",
    )
    max_long_weight = max([weight for _, weight, _ in top_long_paths] + [1.0e-6])
    for rank, (path_index, weight, path) in enumerate(top_long_paths, start=1):
        if path_index == long_path_index:
            continue
        alpha = 0.15 + 0.35 * weight / max_long_weight
        ax_local.plot(
            path[:, 1],
            path[:, 0],
            color="#ff7f0e",
            linewidth=1.0,
            alpha=alpha,
            label=(
                f"long top-8 paths"
                if rank == 2 or (rank == 1 and long_path_index != path_index)
                else None
            ),
        )
    ax_local.plot(
        long_path[:, 1],
        long_path[:, 0],
        color="#ff7f0e",
        linewidth=2.2,
        alpha=0.95,
        label=f"long top1 path p{long_path_index}",
    )
    ax_local.scatter([0.0], [0.0], color="#2ca02c", s=60, label="local ego", zorder=6)
    ax_local.scatter(
        [float(goal_local[1])],
        [float(goal_local[0])],
        color="#d62728",
        marker="*",
        s=120,
        label="local goal",
        zorder=6,
    )
    setup_local_axis(ax_local, "current ego-frame targets")
    ax_local.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(checkpoint)

    seed = int(choose_arg(args.seed, ckpt_args.get("seed"), 7))
    val_ratio = float(choose_arg(args.val_ratio, ckpt_args.get("val_ratio"), 0.2))
    collision_penalty = float(
        choose_arg(args.collision_penalty, ckpt_args.get("collision_penalty"), 8.0)
    )
    max_episodes = choose_arg(
        args.max_episodes,
        ckpt_args.get("max_episodes"),
        None,
    )

    episode_paths = sorted(args.episode_dir.glob("*.json"))
    if max_episodes is not None:
        episode_paths = episode_paths[: int(max_episodes)]
    if len(episode_paths) < 2:
        raise RuntimeError(f"Need at least two human episodes in {args.episode_dir}")

    checkpoint_val_paths = checkpoint_path_list(checkpoint, "val_episode_paths")
    if checkpoint_val_paths is not None:
        val_paths = checkpoint_val_paths
    else:
        print(
            "warning: checkpoint does not store val_episode_paths; "
            "rebuilding val split from current episode directory.",
            flush=True,
        )
        _, val_paths = split_episode_paths(
            episode_paths=episode_paths,
            val_ratio=val_ratio,
            seed=seed,
        )

    vocab_cpu = SparseDriveVocab.load()
    try:
        val_dataset = HumanDriveDataset(
            episode_paths=val_paths,
            vocab=vocab_cpu,
            cache_path=args.cache_dir / "val_labels.npz",
            rebuild_cache=args.rebuild_cache,
            path_chunk_size=args.path_chunk_size,
            label_batch_size=args.label_batch_size,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Human label cache does not match this validation split. "
            "Re-run with --rebuild-cache."
        ) from exc
    if len(val_dataset) == 0:
        raise RuntimeError("Validation HumanDriveDataset has no samples.")

    rng = random.Random(seed + 1009)
    sample_index = (
        int(args.sample_index)
        if args.sample_index is not None
        else rng.randrange(len(val_dataset))
    )
    if sample_index < 0 or sample_index >= len(val_dataset):
        raise IndexError(
            f"--sample-index {sample_index} outside val dataset length {len(val_dataset)}"
        )

    sample = val_dataset[sample_index]
    meta = val_dataset.metas[sample_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.no_grad():
        output = model(
            input_grid=batch["input_grid"],
            path_vocab=vocab.path,
            velocity_vocab=vocab.velocity,
            trajectory_vocab=vocab.trajectory,
            ego_state=batch["ego_state"],
        )
        pred = select_prediction(
            output,
            use_safety_score=args.use_safety_score,
            collision_penalty=collision_penalty,
        )
        traj_l2, endpoint_l2 = compute_errors(
            pred_trajectory=pred["trajectory"],
            target_trajectory=batch["target_trajectory"],
            target_mask=batch["target_trajectory_mask"],
        )

    target_path_index = int(sample["path_index"])
    target_velocity_index = int(sample["velocity_index"])
    long_path_indices = sample["long_path_indices"].detach().cpu().numpy().astype(np.int64)
    long_path_weights = sample["long_path_weights"].detach().cpu().numpy().astype(np.float32)
    long_path_valid = bool(sample["long_path_valid"].item())
    long_path_index = int(long_path_indices[0])
    audit_valid, audit_progress, audit_rows, audit_ranks = compute_long_path_label_audit(
        dataset=val_dataset,
        sample_index=sample_index,
        vocab=vocab_cpu,
    )
    pred_path_index = int(pred["path_index"][0].detach().cpu())
    pred_velocity_index = int(pred["velocity_index"][0].detach().cpu())
    target_candidate_index = find_target_candidate_index(
        output,
        target_path_index=target_path_index,
        target_velocity_index=target_velocity_index,
    )

    path_scores = output["path_scores"][0].detach().cpu()
    velocity_scores = output["velocity_scores"][0].detach().cpu()
    pred_candidate_index = int(pred["candidate_index"][0].detach().cpu())
    pred_trajectory_score = float(
        output["trajectory_scores"][0, pred_candidate_index].detach().cpu()
    )
    pred_no_collision_logit = float(
        output["no_collision_logits"][0, pred_candidate_index].detach().cpu()
    )
    target_trajectory_score = None
    target_no_collision_logit = None
    if target_candidate_index is not None:
        target_trajectory_score = float(
            output["trajectory_scores"][0, target_candidate_index].detach().cpu()
        )
        target_no_collision_logit = float(
            output["no_collision_logits"][0, target_candidate_index].detach().cpu()
        )

    path_match = pred_path_index == target_path_index
    velocity_match = pred_velocity_index == target_velocity_index
    pair_match = path_match and velocity_match

    output_path = OUTPUT_DIR / f"human_sample_{sample_index:06d}.png"
    diagnostic_path = OUTPUT_DIR / f"human_sample_{sample_index:06d}_episode_debug.png"
    visualize_sample(
        sample=sample,
        vocab=vocab_cpu,
        pred={key: value.detach().cpu() for key, value in pred.items()},
        target_path_index=target_path_index,
        target_velocity_index=target_velocity_index,
        pred_path_index=pred_path_index,
        pred_velocity_index=pred_velocity_index,
        output_path=output_path,
        title_prefix=f"val sample {sample_index}",
    )

    print(f"checkpoint epoch: {checkpoint.get('epoch', 'n/a')}")
    print(f"checkpoint val_metrics: {checkpoint.get('val_metrics', 'n/a')}")
    print(f"device: {device}")
    print(f"episodes: total={len(episode_paths)} val={len(val_paths)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"sample_index: {sample_index}")
    print(f"episode: {meta.episode_path.name}")
    print(f"step_index: {meta.step_index}")
    print(f"current_time: {meta.current_time:.2f}s")
    print(f"ego speed: {float(sample['ego_state'][0]):.3f} m/s")
    print(f"goal_xy local: {sample['goal_xy'].tolist()}")
    print(f"human 4s target path: {target_path_index}")
    print(f"human 4s target velocity: {target_velocity_index}")
    print(f"human long target valid: {long_path_valid}")
    print(
        "human long target top paths: "
        + ", ".join(
            f"p{int(path)}:{float(weight):.3f}"
            for path, weight in zip(long_path_indices[:8], long_path_weights[:8])
        )
    )
    print(f"human long target progress: {audit_progress:.3f}m")
    if audit_valid:
        print(
            "recomputed long path cost top paths: "
            + ", ".join(
                f"p{int(row['path_index'])}:mean{float(row['mean_l2']):.3f}/"
                f"end{float(row['endpoint_l2']):.3f}"
                for row in audit_rows[:8]
            )
        )
        print(
            "recomputed long path ranks: "
            f"4s_path=p{target_path_index}/rank{audit_ranks[target_path_index]} "
            f"long_path=p{long_path_index}/rank{audit_ranks[long_path_index]}"
        )
    print(f"pred path: {pred_path_index}")
    print(f"pred velocity: {pred_velocity_index}")
    print(f"pred candidate index: {pred_candidate_index}")
    print(f"traj_l2_to_human: {traj_l2:.4f}")
    print(f"endpoint_l2_to_human: {endpoint_l2:.4f}")
    print(f"path_match: {path_match}")
    print(f"velocity_match: {velocity_match}")
    print(f"pair_match: {pair_match}")
    print(
        "path_score/rank "
        f"human={float(path_scores[target_path_index]):.4f}/"
        f"{rank_of_score(path_scores, target_path_index)} "
        f"pred={float(path_scores[pred_path_index]):.4f}/"
        f"{rank_of_score(path_scores, pred_path_index)}"
    )
    print(
        "velocity_score/rank "
        f"human={float(velocity_scores[target_velocity_index]):.4f}/"
        f"{rank_of_score(velocity_scores, target_velocity_index)} "
        f"pred={float(velocity_scores[pred_velocity_index]):.4f}/"
        f"{rank_of_score(velocity_scores, pred_velocity_index)}"
    )
    print(
        "trajectory_score "
        f"human={target_trajectory_score if target_trajectory_score is not None else 'n/a'} "
        f"pred={pred_trajectory_score:.4f}"
    )
    print(
        "no_collision_logit "
        f"human={target_no_collision_logit if target_no_collision_logit is not None else 'n/a'} "
        f"pred={pred_no_collision_logit:.4f}"
    )
    print(f"human_in_model_candidate_set: {target_candidate_index is not None}")
    print(f"selection_uses_safety_score: {args.use_safety_score}")
    visualize_episode_diagnostic(
        dataset=val_dataset,
        sample_index=sample_index,
        sample=sample,
        vocab=vocab_cpu,
        short_path_index=target_path_index,
        long_path_index=long_path_index,
        long_path_indices=long_path_indices,
        long_path_weights=long_path_weights,
        output_path=diagnostic_path,
    )

    print(f"collision_penalty: {collision_penalty}")
    print(f"saved visualization to: {output_path}")
    print(f"saved episode diagnostic to: {diagnostic_path}")


if __name__ == "__main__":
    main()
