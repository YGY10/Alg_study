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

from human_dataset import FUTURE_TIMES, HumanDriveDataset
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
        help="Add no_collision_logits when selecting candidates.",
    )
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


def select_prediction(
    output: dict[str, torch.Tensor],
    use_safety_score: bool,
) -> dict[str, torch.Tensor]:
    scores = output["trajectory_scores"]
    if use_safety_score:
        scores = scores + output["no_collision_logits"]

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
            f"{title_prefix} | human p{target_path_index} v{target_velocity_index} "
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
        label=f"human matched path p{target_path_index}",
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
        label=f"human matched velocity v{target_velocity_index}",
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


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(checkpoint)

    seed = int(choose_arg(args.seed, ckpt_args.get("seed"), 7))
    val_ratio = float(choose_arg(args.val_ratio, ckpt_args.get("val_ratio"), 0.2))
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
        pred = select_prediction(output, use_safety_score=args.use_safety_score)
        traj_l2, endpoint_l2 = compute_errors(
            pred_trajectory=pred["trajectory"],
            target_trajectory=batch["target_trajectory"],
            target_mask=batch["target_trajectory_mask"],
        )

    target_path_index = int(sample["path_index"])
    target_velocity_index = int(sample["velocity_index"])
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
    print(f"human target path: {target_path_index}")
    print(f"human target velocity: {target_velocity_index}")
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
    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
