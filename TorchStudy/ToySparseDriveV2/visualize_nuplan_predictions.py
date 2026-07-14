from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_human import (
    compute_candidate_collision,
    compute_planning_scores,
    trajectory_distance_to_human,
)

DATASET_MODULE_PATH = PROJECT_ROOT / "dataset"
if str(DATASET_MODULE_PATH) in sys.path:
    sys.path.remove(str(DATASET_MODULE_PATH))
dataset_module = sys.modules.get("dataset")
if dataset_module is not None and not hasattr(dataset_module, "__path__"):
    del sys.modules["dataset"]

from dataset.nuplan_canonical_dataset import NUM_RASTER_CHANNELS, RASTER_CHANNELS
from model import ToySparseDriveV2Model
from train_nuplan_canonical import (
    DEFAULT_CHECKPOINT_DIR,
    balanced_sample_indices,
    build_dataset,
    collect_episode_paths,
    collate_nuplan_batch,
    split_episode_paths,
)
from vocab import SparseDriveVocab


DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "best_nuplan_canonical_model.pt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "nuplan_canonical" / "prediction_preview"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize nuPlan canonical model predictions against GT and oracle candidates."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--episode-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--scan-samples", type=int, default=128)
    parser.add_argument("--num-plots", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-route-points", type=int, default=None)
    parser.add_argument("--model-topk-path", type=int, default=None)
    parser.add_argument("--val-forced-long-path-topk", type=int, default=None)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--safety-margin", type=float, default=None)
    parser.add_argument("--endpoint-weight", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--select",
        choices=("worst-gap", "best-gap", "random", "first"),
        default="worst-gap",
    )
    parser.add_argument(
        "--diagnostic-topk",
        type=int,
        default=20,
        help="Number of selected records to include in the printed diagnostic table.",
    )
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> argparse.Namespace:
    saved = checkpoint.get("args", {})
    args = argparse.Namespace(**saved)
    if not hasattr(args, "episode_dir"):
        args.episode_dir = None
    return args


def arg_or_saved(
    args: argparse.Namespace, saved: argparse.Namespace, name: str, default
):
    value = getattr(args, name)
    if value is not None:
        return value
    return getattr(saved, name, default)


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


def make_forced_paths(
    batch: dict[str, torch.Tensor],
    val_forced_long_path_topk: int,
) -> torch.Tensor | None:
    forced_long_topk = min(
        max(int(val_forced_long_path_topk), 0),
        batch["long_path_indices"].shape[1],
    )
    forced_paths = []
    if forced_long_topk > 0:
        forced_paths.append(batch["path_index"][:, None])
        forced_paths.append(batch["long_path_indices"][:, :forced_long_topk])
    if not forced_paths:
        return None
    return torch.cat(forced_paths, dim=1)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_batch(
    model: ToySparseDriveV2Model,
    batch: dict[str, torch.Tensor],
    vocab: SparseDriveVocab,
    device: torch.device,
    *,
    val_forced_long_path_topk: int,
    endpoint_weight: float,
    safety_margin: float,
    collision_penalty: float,
    metric_endpoint_score_weight: float = 5.0,
    metric_progress_score_weight: float = 5.0,
    metric_route_score_weight: float = 2.0,
    metric_imitation_score_weight: float = 0.0,
    metric_no_collision_score_weight: float = 1.0,
    use_metric_heads: bool = True,
) -> dict[str, torch.Tensor]:
    batch = move_batch_to_device(batch, device)
    forced_paths = make_forced_paths(batch, val_forced_long_path_topk)

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
        extra_path_indices=forced_paths,
    )
    candidate_collision = compute_candidate_collision(
        candidate_trajectories=output["candidate_trajectories"],
        obstacle_centers=batch["obstacle_centers"],
        obstacle_sizes=batch["obstacle_sizes"],
        obstacle_velocities=batch["obstacle_velocities"],
        obstacle_mask=batch["obstacle_mask"],
        ego_state=batch["ego_state"],
        safety_margin=safety_margin,
    )
    _, candidate_l2, endpoint_l2 = trajectory_distance_to_human(
        candidate_trajectories=output["candidate_trajectories"],
        target_trajectory=batch["target_trajectory"],
        target_mask=batch["target_trajectory_mask"],
        endpoint_weight=endpoint_weight,
    )
    planning_scores = compute_planning_scores(
        trajectory_scores=output["trajectory_scores"],
        no_collision_logits=output["no_collision_logits"],
        collision_penalty=collision_penalty,
        endpoint_metric_logits=(
            output.get("endpoint_metric_logits") if use_metric_heads else None
        ),
        progress_metric_logits=(
            output.get("progress_metric_logits") if use_metric_heads else None
        ),
        route_metric_logits=(
            output.get("route_metric_logits") if use_metric_heads else None
        ),
        endpoint_weight=metric_endpoint_score_weight,
        progress_weight=metric_progress_score_weight,
        route_weight=metric_route_score_weight,
        imitation_weight=metric_imitation_score_weight,
        no_collision_weight=metric_no_collision_score_weight,
    )

    best_index = planning_scores.argmax(dim=1)
    oracle_index = candidate_l2.argmin(dim=1)
    batch_index = torch.arange(planning_scores.shape[0], device=device)
    selected_l2 = candidate_l2[batch_index, best_index]
    selected_endpoint_l2 = endpoint_l2[batch_index, best_index]
    oracle_l2 = candidate_l2[batch_index, oracle_index]
    oracle_endpoint_l2 = endpoint_l2[batch_index, oracle_index]

    return {
        "output": output,
        "planning_scores": planning_scores,
        "candidate_collision": candidate_collision,
        "best_index": best_index,
        "oracle_index": oracle_index,
        "selected_l2": selected_l2,
        "selected_endpoint_l2": selected_endpoint_l2,
        "oracle_l2": oracle_l2,
        "oracle_endpoint_l2": oracle_endpoint_l2,
        "gap": selected_l2 - oracle_l2,
    }


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def finite_route(route_path: np.ndarray) -> np.ndarray:
    if route_path.size == 0:
        return route_path.reshape(0, 2)
    valid = np.isfinite(route_path).all(axis=1)
    return route_path[valid]


def summarize_record(record: dict) -> dict[str, float | int | bool]:
    result = record["result"]
    batch = record["batch"]
    row = int(record["row"])
    output = result["output"]
    best = int(result["best_index"][row].detach().cpu())
    oracle = int(result["oracle_index"][row].detach().cpu())

    pred_path = int(output["candidate_path_indices"][row, best].detach().cpu())
    oracle_path = int(output["candidate_path_indices"][row, oracle].detach().cpu())
    target_path = int(batch["path_index"][row].detach().cpu())
    pred_velocity = int(output["candidate_velocity_indices"][row, best].detach().cpu())
    oracle_velocity = int(
        output["candidate_velocity_indices"][row, oracle].detach().cpu()
    )
    target_velocity = int(batch["velocity_index"][row].detach().cpu())

    candidates = result["output"]["candidate_trajectories"]
    pred_traj = candidates[row, best].detach().cpu()
    oracle_traj = candidates[row, oracle].detach().cpu()
    pred_progress = float(pred_traj[-1, 0])
    oracle_progress = float(oracle_traj[-1, 0])
    pred_lateral = float(pred_traj[-1, 1])
    oracle_lateral = float(oracle_traj[-1, 1])
    pred_length = float(
        (pred_traj[1:, :2] - pred_traj[:-1, :2]).pow(2).sum(dim=-1).sqrt().sum()
    )
    oracle_length = float(
        (oracle_traj[1:, :2] - oracle_traj[:-1, :2]).pow(2).sum(dim=-1).sqrt().sum()
    )

    scores = result["planning_scores"][row].detach().cpu()
    pred_score = float(scores[best])
    oracle_score = float(scores[oracle])

    return {
        "global_index": int(record["global_index"]),
        "best": best,
        "oracle": oracle,
        "pred_path": pred_path,
        "oracle_path": oracle_path,
        "target_path": target_path,
        "pred_velocity": pred_velocity,
        "oracle_velocity": oracle_velocity,
        "target_velocity": target_velocity,
        "path_same": pred_path == oracle_path,
        "velocity_same": pred_velocity == oracle_velocity,
        "both_same": pred_path == oracle_path and pred_velocity == oracle_velocity,
        "target_pair_hit": pred_path == target_path
        and pred_velocity == target_velocity,
        "gap": float(record["gap"]),
        "selected_l2": float(record["selected_l2"]),
        "oracle_l2": float(result["oracle_l2"][row].detach().cpu()),
        "selected_endpoint_l2": float(
            result["selected_endpoint_l2"][row].detach().cpu()
        ),
        "oracle_endpoint_l2": float(result["oracle_endpoint_l2"][row].detach().cpu()),
        "pred_progress": pred_progress,
        "oracle_progress": oracle_progress,
        "progress_delta": pred_progress - oracle_progress,
        "lateral_delta": pred_lateral - oracle_lateral,
        "length_delta": pred_length - oracle_length,
        "pred_score": pred_score,
        "oracle_score": oracle_score,
        "score_delta": pred_score - oracle_score,
        "pred_collision": bool(result["candidate_collision"][row, best].detach().cpu()),
        "oracle_collision": bool(
            result["candidate_collision"][row, oracle].detach().cpu()
        ),
    }


def print_diagnostics(records: list[dict], selected: list[dict], topk: int) -> None:
    def aggregate(name: str, subset: list[dict]) -> None:
        summaries = [summarize_record(record) for record in subset]
        if not summaries:
            print(f"diagnostic {name}: empty", flush=True)
            return
        n = len(summaries)
        path_same = sum(int(item["path_same"]) for item in summaries) / n
        velocity_same = sum(int(item["velocity_same"]) for item in summaries) / n
        both_same = sum(int(item["both_same"]) for item in summaries) / n
        target_pair_hit = sum(int(item["target_pair_hit"]) for item in summaries) / n
        progress_delta = sum(float(item["progress_delta"]) for item in summaries) / n
        abs_progress_delta = (
            sum(abs(float(item["progress_delta"])) for item in summaries) / n
        )
        length_delta = sum(float(item["length_delta"]) for item in summaries) / n
        score_delta = sum(float(item["score_delta"]) for item in summaries) / n
        pred_faster = (
            sum(int(float(item["progress_delta"]) > 1.0) for item in summaries) / n
        )
        pred_slower = (
            sum(int(float(item["progress_delta"]) < -1.0) for item in summaries) / n
        )
        print(
            f"diagnostic {name}: n={n} "
            f"path_same={path_same:.3f} velocity_same={velocity_same:.3f} "
            f"both_same={both_same:.3f} target_pair_hit={target_pair_hit:.3f} "
            f"progress_delta={progress_delta:.2f} abs_progress_delta={abs_progress_delta:.2f} "
            f"length_delta={length_delta:.2f} score_delta={score_delta:.2f} "
            f"pred_farther={pred_faster:.3f} pred_shorter={pred_slower:.3f}",
            flush=True,
        )

    aggregate("scan", records)
    aggregate("selected", selected)

    table = [summarize_record(record) for record in selected[: max(0, int(topk))]]
    if not table:
        return
    print("diagnostic selected table:", flush=True)
    print(
        "sample gap pred_l2 oracle_l2 path[p/o/t] vel[p/o/t] "
        "same[p/v] progress_delta length_delta score_delta collision[p/o]",
        flush=True,
    )
    for item in table:
        print(
            f"{item['global_index']} {item['gap']:.2f} "
            f"{item['selected_l2']:.2f} {item['oracle_l2']:.2f} "
            f"{item['pred_path']}/{item['oracle_path']}/{item['target_path']} "
            f"{item['pred_velocity']}/{item['oracle_velocity']}/{item['target_velocity']} "
            f"{int(item['path_same'])}/{int(item['velocity_same'])} "
            f"{item['progress_delta']:.2f} {item['length_delta']:.2f} "
            f"{item['score_delta']:.2f} "
            f"{int(item['pred_collision'])}/{int(item['oracle_collision'])}",
            flush=True,
        )


def plot_prediction(
    sample: dict[str, torch.Tensor],
    result: dict[str, torch.Tensor],
    row: int,
    global_index: int,
    output_path: Path,
) -> None:
    output = result["output"]
    best = int(result["best_index"][row].detach().cpu())
    oracle = int(result["oracle_index"][row].detach().cpu())
    pred_path = int(output["candidate_path_indices"][row, best].detach().cpu())
    oracle_path = int(output["candidate_path_indices"][row, oracle].detach().cpu())
    pred_velocity = int(output["candidate_velocity_indices"][row, best].detach().cpu())
    oracle_velocity = int(
        output["candidate_velocity_indices"][row, oracle].detach().cpu()
    )
    target_path = int(sample["path_index"].detach().cpu())
    target_velocity = int(sample["velocity_index"].detach().cpu())

    input_grid = to_numpy(sample["input_grid"])
    target = to_numpy(sample["target_trajectory"])
    target_mask = to_numpy(sample["target_trajectory_mask"]).astype(bool)
    target = target[target_mask]
    route_path = finite_route(to_numpy(sample["route_path"]))
    goal_xy = to_numpy(sample["goal_xy"])
    candidates = to_numpy(output["candidate_trajectories"][row])
    scores = to_numpy(result["planning_scores"][row])
    collisions = to_numpy(result["candidate_collision"][row]).astype(bool)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax_traj, ax_overlay, ax_scores = axes[0]
    ax_route, ax_lane, ax_agents = axes[1]

    order = np.argsort(scores)[::-1]
    for candidate_index in order[: min(16, len(order))]:
        traj = candidates[candidate_index]
        color = "#d62728" if candidate_index == best else "#999999"
        alpha = 0.35 if candidate_index == best else 0.12
        linewidth = 2.0 if candidate_index == best else 0.8
        ax_traj.plot(
            traj[:, 1],
            traj[:, 0],
            color=color,
            alpha=alpha,
            lw=linewidth,
            zorder=2,
        )

    if len(route_path) > 0:
        ax_traj.plot(
            route_path[:, 1],
            route_path[:, 0],
            color="#1f77b4",
            lw=2.0,
            linestyle="--",
            alpha=0.8,
            label="route path",
            zorder=1,
        )
    ax_traj.plot(
        target[:, 1],
        target[:, 0],
        color="#2ca02c",
        lw=3.0,
        label="GT",
        zorder=5,
    )
    ax_traj.plot(
        candidates[best, :, 1],
        candidates[best, :, 0],
        color="#d62728",
        lw=3.0,
        label="model top1",
        zorder=6,
    )
    ax_traj.plot(
        candidates[oracle, :, 1],
        candidates[oracle, :, 0],
        color="#9467bd",
        lw=2.8,
        linestyle=":",
        label="oracle candidate",
        zorder=7,
    )
    ax_traj.scatter([0.0], [0.0], color="#111111", s=50, label="ego now", zorder=8)
    ax_traj.scatter(
        [goal_xy[1]],
        [goal_xy[0]],
        color="#ff7f0e",
        marker="*",
        s=130,
        label="goal",
        zorder=8,
    )
    ax_traj.set_xlim(-45, 45)
    ax_traj.set_ylim(-15, 85)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.set_xlabel("y left [m]")
    ax_traj.set_ylabel("x forward [m]")
    ax_traj.set_title("ego-frame trajectories")
    ax_traj.legend(loc="best")

    ax_overlay.imshow(raster_overlay(input_grid))
    ax_overlay.set_title("raster overlay")
    ax_overlay.set_axis_off()

    top_count = min(12, len(order))
    top_indices = order[:top_count][::-1]
    colors = [
        "#d62728" if index == best else "#9467bd" if index == oracle else "#888888"
        for index in top_indices
    ]
    ax_scores.barh(np.arange(top_count), scores[top_indices], color=colors)
    labels = []
    for index in top_indices:
        suffix = ""
        if index == best:
            suffix += " pred"
        if index == oracle:
            suffix += " oracle"
        if collisions[index]:
            suffix += " col"
        path_idx = int(output["candidate_path_indices"][row, index].detach().cpu())
        velocity_idx = int(
            output["candidate_velocity_indices"][row, index].detach().cpu()
        )
        labels.append(f"{index}{suffix} p{path_idx} v{velocity_idx}")
    ax_scores.set_yticks(np.arange(top_count), labels)
    ax_scores.set_title("top planning scores")
    ax_scores.grid(True, axis="x", alpha=0.25)

    route_goal = np.maximum(
        input_grid[RASTER_CHANNELS["route"]],
        input_grid[RASTER_CHANNELS["goal"]],
    )
    ax_route.imshow(route_goal, cmap="Greens", vmin=0.0, vmax=1.0)
    ax_route.set_title("route area + goal")
    ax_route.set_axis_off()

    ax_lane.imshow(
        input_grid[RASTER_CHANNELS["lane_centerline"]],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    ax_lane.set_title("route-road lane centerlines")
    ax_lane.set_axis_off()

    ax_agents.imshow(raster_agents(input_grid))
    ax_agents.set_title("agents")
    ax_agents.set_axis_off()

    selected_l2 = float(result["selected_l2"][row].detach().cpu())
    selected_end = float(result["selected_endpoint_l2"][row].detach().cpu())
    oracle_l2 = float(result["oracle_l2"][row].detach().cpu())
    oracle_end = float(result["oracle_endpoint_l2"][row].detach().cpu())
    gap = float(result["gap"][row].detach().cpu())
    pred_col = bool(result["candidate_collision"][row, best].detach().cpu())
    pred_progress = float(candidates[best, -1, 0])
    oracle_progress = float(candidates[oracle, -1, 0])
    fig.suptitle(
        f"sample={global_index} | pred_l2={selected_l2:.2f} end={selected_end:.2f} "
        f"| oracle_l2={oracle_l2:.2f} end={oracle_end:.2f} | gap={gap:.2f} "
        f"| p/o/t path={pred_path}/{oracle_path}/{target_path} "
        f"vel={pred_velocity}/{oracle_velocity}/{target_velocity} "
        f"| progress_delta={pred_progress - oracle_progress:.2f} "
        f"| pred_collision={pred_col}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint_args(checkpoint)
    print(
        f"checkpoint loaded: epoch={checkpoint.get('epoch', 'unknown')} "
        f"score={checkpoint.get('val_score', 'unknown')}",
        flush=True,
    )

    seed = int(arg_or_saved(args, saved_args, "seed", 7))
    val_ratio = float(arg_or_saved(args, saved_args, "val_ratio", 0.2))
    max_episodes = arg_or_saved(args, saved_args, "max_episodes", None)
    max_val_samples = arg_or_saved(args, saved_args, "max_val_samples", None)
    max_route_points = int(arg_or_saved(args, saved_args, "max_route_points", 256))
    model_topk_path = int(arg_or_saved(args, saved_args, "model_topk_path", 20))
    val_forced_long_path_topk = int(
        arg_or_saved(args, saved_args, "val_forced_long_path_topk", 0)
    )
    endpoint_weight = float(arg_or_saved(args, saved_args, "endpoint_weight", 0.5))
    safety_margin = float(arg_or_saved(args, saved_args, "safety_margin", 0.3))
    collision_penalty = float(arg_or_saved(args, saved_args, "collision_penalty", 0.0))
    metric_endpoint_score_weight = float(
        getattr(saved_args, "metric_endpoint_score_weight", 5.0)
    )
    metric_progress_score_weight = float(
        getattr(saved_args, "metric_progress_score_weight", 5.0)
    )
    metric_route_score_weight = float(
        getattr(saved_args, "metric_route_score_weight", 2.0)
    )
    metric_imitation_score_weight = float(
        getattr(saved_args, "metric_imitation_score_weight", 0.0)
    )
    metric_no_collision_score_weight = float(
        getattr(saved_args, "metric_no_collision_score_weight", -1.0)
    )
    if metric_no_collision_score_weight < 0.0:
        metric_no_collision_score_weight = float(
            getattr(saved_args, "no_collision_loss_weight", 0.0) > 0.0
        )

    random.seed(seed)
    torch.manual_seed(seed)

    episode_dirs = args.episode_dir
    if episode_dirs is None and getattr(saved_args, "episode_dir", None):
        episode_dirs = [Path(path) for path in saved_args.episode_dir]
    if episode_dirs is None and checkpoint.get("val_episode_paths"):
        val_paths = [Path(path) for path in checkpoint["val_episode_paths"]]
        if max_episodes is not None:
            # Keep the checkpoint split when possible; this only caps visualized val episodes.
            val_keep = max(1, int(round(int(max_episodes) * val_ratio)))
            val_paths = val_paths[:val_keep]
        print(f"using checkpoint val episodes: {len(val_paths)}", flush=True)
    else:
        episode_paths = collect_episode_paths(episode_dirs)
        if max_episodes is not None:
            episode_paths = episode_paths[: int(max_episodes)]
        _, val_paths = split_episode_paths(
            episode_paths, val_ratio=val_ratio, seed=seed
        )
        print(f"split val episodes: {len(val_paths)}", flush=True)

    print("loading vocab", flush=True)
    vocab_cpu = SparseDriveVocab.load()
    print(
        "building validation dataset; this is the slow step with the current eager dataset",
        flush=True,
    )
    dataset = build_dataset(val_paths, vocab_cpu, saved_args)
    print(f"dataset built: samples={len(dataset)}", flush=True)
    val_indices = balanced_sample_indices(dataset, max_val_samples, seed=seed + 1)
    if val_indices is None:
        val_indices = list(range(len(dataset)))
    scan_count = min(max(int(args.scan_samples), 1), len(val_indices))
    scan_indices = val_indices[:scan_count]
    loader_dataset = Subset(dataset, scan_indices)
    print(f"scan samples: {scan_count}", flush=True)
    loader = DataLoader(
        loader_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_nuplan_batch, max_route_points=max_route_points),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model(
        input_channels=NUM_RASTER_CHANNELS,
        ego_state_dim=7,
        topk_path=model_topk_path,
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    use_metric_heads = not any("metric_head" in key for key in load_result.missing_keys)
    if not use_metric_heads:
        print(
            "checkpoint has no trained metric heads; using legacy trajectory scoring",
            flush=True,
        )
    model.eval()

    records = []
    print("scanning model predictions", flush=True)
    with torch.no_grad():
        seen = 0
        for batch in loader:
            result = evaluate_batch(
                model=model,
                batch=batch,
                vocab=vocab,
                device=device,
                val_forced_long_path_topk=val_forced_long_path_topk,
                endpoint_weight=endpoint_weight,
                safety_margin=safety_margin,
                collision_penalty=collision_penalty,
                metric_endpoint_score_weight=metric_endpoint_score_weight,
                metric_progress_score_weight=metric_progress_score_weight,
                metric_route_score_weight=metric_route_score_weight,
                metric_imitation_score_weight=metric_imitation_score_weight,
                metric_no_collision_score_weight=metric_no_collision_score_weight,
                use_metric_heads=use_metric_heads,
            )
            batch_size = batch["input_grid"].shape[0]
            for row in range(batch_size):
                global_index = scan_indices[seen + row]
                records.append(
                    {
                        "global_index": global_index,
                        "row": row,
                        "batch": batch,
                        "result": result,
                        "gap": float(result["gap"][row].detach().cpu()),
                        "selected_l2": float(result["selected_l2"][row].detach().cpu()),
                    }
                )
            seen += batch_size
            print(f"scanned {seen}/{scan_count}", flush=True)

    if args.select == "worst-gap":
        records.sort(key=lambda record: record["gap"], reverse=True)
    elif args.select == "best-gap":
        records.sort(key=lambda record: record["gap"])
    elif args.select == "random":
        random.shuffle(records)

    selected = records[: max(1, int(args.num_plots))]
    print_diagnostics(records, selected, args.diagnostic_topk)
    saved_paths = []
    print(f"saving {len(selected)} plots to {args.output_dir}", flush=True)
    for rank, record in enumerate(selected):
        row = int(record["row"])
        sample = {
            key: value[row].detach().cpu() for key, value in record["batch"].items()
        }
        output_path = (
            args.output_dir
            / f"pred_{rank:02d}_sample_{record['global_index']:06d}_gap_{record['gap']:.2f}.png"
        )
        plot_prediction(
            sample=sample,
            result=record["result"],
            row=row,
            global_index=int(record["global_index"]),
            output_path=output_path,
        )
        saved_paths.append(output_path)

    mean_l2 = sum(record["selected_l2"] for record in records) / max(len(records), 1)
    mean_gap = sum(record["gap"] for record in records) / max(len(records), 1)
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(
        f"val episodes={len(val_paths)} dataset_samples={len(dataset)} "
        f"scanned={len(records)} select={args.select}"
    )
    print(
        f"settings: topk_path={model_topk_path} val_forced_long_path_topk={val_forced_long_path_topk} "
        f"max_route_points={max_route_points} collision_penalty={collision_penalty} "
        f"metric_score_w={metric_endpoint_score_weight}/"
        f"{metric_progress_score_weight}/{metric_route_score_weight}/"
        f"{metric_imitation_score_weight} "
        f"no_col_score_w={metric_no_collision_score_weight}"
    )
    print(f"scan mean: pred_l2={mean_l2:.3f} gap={mean_gap:.3f}")
    for path in saved_paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
