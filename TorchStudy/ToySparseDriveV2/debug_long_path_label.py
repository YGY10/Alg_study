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
from vocab import SparseDriveVocab

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints_human" / "best_human_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_long_path_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
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
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--path-chunk-size", type=int, default=32)
    parser.add_argument("--label-batch-size", type=int, default=64)
    parser.add_argument("--long-path-horizon", type=float, default=None)
    parser.add_argument("--long-path-compare-points", type=int, default=None)
    parser.add_argument("--long-path-positive-topk", type=int, default=None)
    parser.add_argument("--long-path-temperature", type=float, default=None)
    parser.add_argument("--long-path-min-progress", type=float, default=None)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--cost-mode",
        choices=["mean", "weighted", "weighted-endpoint-heading"],
        default="mean",
    )
    parser.add_argument("--endpoint-weight", type=float, default=0.3)
    parser.add_argument("--late-weight", type=float, default=0.3)
    parser.add_argument("--heading-weight", type=float, default=0.5)
    parser.add_argument("--heading-meter-scale", type=float, default=5.0)
    parser.add_argument(
        "--path-indices",
        type=int,
        nargs="*",
        default=[479, 833, 804, 1017, 476, 264, 1004],
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Call HumanDriveDataset._build_long_path_labels() and compare this sample with cache.",
    )
    parser.add_argument("--no-plot", action="store_true")
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


def choose_arg(cli_value: Any, checkpoint_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if checkpoint_value is not None:
        return checkpoint_value
    return default_value


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1.0e-6 or norm_b <= 1.0e-6:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_single_sample_long_costs(
    dataset: HumanDriveDataset,
    sample_index: int,
    vocab: SparseDriveVocab,
    cost_mode: str,
    endpoint_weight: float,
    late_weight: float,
    heading_weight: float,
    heading_meter_scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    human_xy, valid = dataset.build_long_path_target(sample_index)
    if not valid:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty, empty, empty, empty, empty, False

    human_progress = float(cumulative_distance(human_xy)[-1])
    query_distance = np.linspace(
        0.0,
        human_progress,
        dataset.long_path_compare_points,
        dtype=np.float32,
    )
    human_points = interp_points_by_distance(human_xy, query_distance)
    path_xy = vocab.path[..., :2].detach().cpu().numpy().astype(np.float32)

    costs = np.zeros((vocab.num_path,), dtype=np.float32)
    mean_l2 = np.zeros((vocab.num_path,), dtype=np.float32)
    endpoint_l2 = np.zeros((vocab.num_path,), dtype=np.float32)
    max_l2 = np.zeros((vocab.num_path,), dtype=np.float32)
    late_l2 = np.zeros((vocab.num_path,), dtype=np.float32)
    heading_error = np.zeros((vocab.num_path,), dtype=np.float32)
    path_points_all = np.zeros(
        (vocab.num_path, len(query_distance), 2),
        dtype=np.float32,
    )

    time_weights = np.linspace(0.5, 1.5, len(query_distance), dtype=np.float32)
    time_weights = time_weights / max(float(time_weights.mean()), 1.0e-6)
    late_start = max(0, int(round(len(query_distance) * 0.6)))
    human_heading_vector = human_points[-1] - human_points[max(0, len(human_points) - 4)]

    for path_index in range(vocab.num_path):
        path_points = interp_points_by_distance(path_xy[path_index], query_distance)
        point_l2 = np.linalg.norm(path_points - human_points, axis=-1)
        mean_l2[path_index] = float(point_l2.mean())
        endpoint_l2[path_index] = float(point_l2[-1])
        max_l2[path_index] = float(point_l2.max())
        late_l2[path_index] = float(point_l2[late_start:].mean())
        path_heading_vector = path_points[-1] - path_points[max(0, len(path_points) - 4)]
        heading_error[path_index] = angle_between_vectors(
            human_heading_vector,
            path_heading_vector,
        )

        if cost_mode == "mean":
            costs[path_index] = mean_l2[path_index]
        elif cost_mode == "weighted":
            costs[path_index] = float((point_l2 * time_weights).mean())
        elif cost_mode == "weighted-endpoint-heading":
            weighted_mean = float((point_l2 * time_weights).mean())
            costs[path_index] = (
                weighted_mean
                + float(endpoint_weight) * endpoint_l2[path_index]
                + float(late_weight) * late_l2[path_index]
                + float(heading_weight)
                * float(heading_meter_scale)
                * heading_error[path_index]
            )
        else:
            raise ValueError(f"unknown cost_mode: {cost_mode}")
        path_points_all[path_index] = path_points

    return (
        costs,
        mean_l2,
        endpoint_l2,
        max_l2,
        late_l2,
        heading_error,
        query_distance,
        path_points_all,
        True,
    )


def print_path_row(
    prefix: str,
    path_index: int,
    costs: np.ndarray,
    mean_l2: np.ndarray,
    endpoint_l2: np.ndarray,
    max_l2: np.ndarray,
    late_l2: np.ndarray,
    heading_error: np.ndarray,
    ranks: np.ndarray,
    vocab: SparseDriveVocab,
) -> None:
    path_xy = vocab.path[path_index, :, :2].detach().cpu().numpy().astype(np.float32)
    path_length = float(cumulative_distance(path_xy)[-1])
    endpoint = path_xy[-1]
    print(
        f"{prefix} p{path_index:4d} "
        f"rank={int(ranks[path_index]):4d} "
        f"cost={float(costs[path_index]):7.3f} "
        f"mean={float(mean_l2[path_index]):7.3f} "
        f"late={float(late_l2[path_index]):7.3f} "
        f"end={float(endpoint_l2[path_index]):7.3f} "
        f"head={float(np.degrees(heading_error[path_index])):6.1f}deg "
        f"max={float(max_l2[path_index]):7.3f} "
        f"len={path_length:7.3f} "
        f"endpoint=[{float(endpoint[0]):7.3f}, {float(endpoint[1]):7.3f}]"
    )


def plot_debug(
    dataset: HumanDriveDataset,
    sample_index: int,
    vocab: SparseDriveVocab,
    top_indices: np.ndarray,
    path_indices: list[int],
    query_distance: np.ndarray,
    path_points_all: np.ndarray,
    output_path: Path,
) -> None:
    human_xy, _ = dataset.build_long_path_target(sample_index)
    human_points = interp_points_by_distance(human_xy, query_distance)
    short_xy, _ = dataset.build_target_trajectory(sample_index)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(human_xy[:, 1], human_xy[:, 0], color="#2ca02c", linewidth=2.2, marker="o", markersize=3, label="raw long human")
    ax.scatter(human_points[:, 1], human_points[:, 0], color="#006400", s=16, alpha=0.65, label="long query points")
    ax.plot(short_xy[:, 1], short_xy[:, 0], color="#d62728", linewidth=1.8, marker="x", markersize=4, label="4s target")

    top_colors = [
        "#ff7f0e",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]
    check_colors = ["#111111", "#444444", "#777777", "#aaaaaa"]

    drawn: set[int] = set()
    for rank, path_index in enumerate(top_indices[:8], start=1):
        path = path_points_all[int(path_index)]
        color = top_colors[(rank - 1) % len(top_colors)]
        ax.plot(
            path[:, 1],
            path[:, 0],
            color=color,
            linewidth=2.6 if rank == 1 else 1.8,
            alpha=0.95 if rank == 1 else 0.82,
            label=f"top{rank} p{int(path_index)}",
        )
        drawn.add(int(path_index))

    check_rank = 0
    for path_index in path_indices:
        if path_index in drawn or path_index < 0 or path_index >= vocab.num_path:
            continue
        path = path_points_all[int(path_index)]
        color = check_colors[check_rank % len(check_colors)]
        check_rank += 1
        ax.plot(
            path[:, 1],
            path[:, 0],
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            label=f"check p{path_index}",
        )

    ax.scatter([0.0], [0.0], color="#2ca02c", s=60, label="ego")
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_title(f"long path label debug | sample {sample_index}")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
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
    max_episodes = choose_arg(args.max_episodes, ckpt_args.get("max_episodes"), None)
    long_path_horizon = float(
        choose_arg(args.long_path_horizon, ckpt_args.get("long_path_horizon"), 8.0)
    )
    long_path_compare_points = int(
        choose_arg(
            args.long_path_compare_points,
            ckpt_args.get("long_path_compare_points"),
            32,
        )
    )
    long_path_positive_topk = int(
        choose_arg(
            args.long_path_positive_topk,
            ckpt_args.get("long_path_positive_topk"),
            8,
        )
    )
    long_path_temperature = float(
        choose_arg(
            args.long_path_temperature,
            ckpt_args.get("long_path_temperature"),
            2.0,
        )
    )
    long_path_min_progress = float(
        choose_arg(
            args.long_path_min_progress,
            ckpt_args.get("long_path_min_progress"),
            5.0,
        )
    )

    episode_paths = sorted(args.episode_dir.glob("*.json"))
    if max_episodes is not None:
        episode_paths = episode_paths[: int(max_episodes)]

    val_paths = checkpoint_path_list(checkpoint, "val_episode_paths")
    if val_paths is None:
        _, val_paths = split_episode_paths(episode_paths, val_ratio=val_ratio, seed=seed)

    vocab = SparseDriveVocab.load()
    dataset = HumanDriveDataset(
        episode_paths=val_paths,
        vocab=vocab,
        cache_path=args.cache_dir / "val_labels.npz",
        rebuild_cache=False,
        path_chunk_size=args.path_chunk_size,
        label_batch_size=args.label_batch_size,
        long_path_horizon=long_path_horizon,
        long_path_compare_points=long_path_compare_points,
        long_path_positive_topk=long_path_positive_topk,
        long_path_temperature=long_path_temperature,
        long_path_min_progress=long_path_min_progress,
    )

    sample_index = int(args.sample_index)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample {sample_index} outside dataset length {len(dataset)}")

    meta = dataset.metas[sample_index]
    human_xy, valid = dataset.build_long_path_target(sample_index)
    short_xy, _ = dataset.build_target_trajectory(sample_index)
    (
        costs,
        mean_l2,
        endpoint_l2,
        max_l2,
        late_l2,
        heading_error,
        query_distance,
        path_points_all,
        valid_cost,
    ) = compute_single_sample_long_costs(
        dataset=dataset,
        sample_index=sample_index,
        vocab=vocab,
        cost_mode=args.cost_mode,
        endpoint_weight=args.endpoint_weight,
        late_weight=args.late_weight,
        heading_weight=args.heading_weight,
        heading_meter_scale=args.heading_meter_scale,
    )
    if not valid or not valid_cost:
        print(f"sample {sample_index} has invalid long path target")
        return

    order = np.argsort(costs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    print(f"sample_index: {sample_index}")
    print(f"episode: {meta.episode_path.name}")
    print(f"step_index: {meta.step_index}")
    print(f"current_time: {meta.current_time:.2f}s")
    print(f"long_path_horizon: {dataset.long_path_horizon:g}s")
    print(f"long_path_compare_points: {dataset.long_path_compare_points}")
    print(f"cost_mode: {args.cost_mode}")
    if args.cost_mode == "weighted-endpoint-heading":
        print(
            "cost_weights: "
            f"endpoint={args.endpoint_weight:g} "
            f"late={args.late_weight:g} "
            f"heading={args.heading_weight:g} "
            f"heading_meter_scale={args.heading_meter_scale:g}"
        )
    print(f"human_long_num_points: {len(human_xy)}")
    print(f"human_long_progress: {float(cumulative_distance(human_xy)[-1]):.3f}m")
    print(f"human_long_start: {human_xy[0].tolist()}")
    print(f"human_long_end: {human_xy[-1].tolist()}")
    print(f"human_4s_end: {short_xy[-1, :2].tolist()}")
    print(
        "cache long top paths: "
        + ", ".join(
            f"p{int(path)}:{float(weight):.3f}"
            for path, weight in zip(
                dataset.long_path_indices[sample_index],
                dataset.long_path_weights[sample_index],
            )
        )
    )
    print("single-sample recomputed top paths:")
    for rank, path_index in enumerate(order[: args.topk], start=1):
        print_path_row(
            f"top{rank:02d}",
            int(path_index),
            costs,
            mean_l2,
            endpoint_l2,
            max_l2,
            late_l2,
            heading_error,
            ranks,
            vocab,
        )

    print("selected path checks:")
    for path_index in args.path_indices:
        if 0 <= path_index < vocab.num_path:
            print_path_row(
                "check",
                int(path_index),
                costs,
                mean_l2,
                endpoint_l2,
                max_l2,
                late_l2,
                heading_error,
                ranks,
                vocab,
            )

    if args.recompute_all:
        print("calling HumanDriveDataset._build_long_path_labels() ...", flush=True)
        all_indices, all_weights, all_valid = dataset._build_long_path_labels()
        print(f"_build_long_path_labels valid: {bool(all_valid[sample_index])}")
        print(
            "_build_long_path_labels sample top paths: "
            + ", ".join(
                f"p{int(path)}:{float(weight):.3f}"
                for path, weight in zip(
                    all_indices[sample_index],
                    all_weights[sample_index],
                )
            )
        )
        print(
            "matches cache: "
            f"indices={bool(np.array_equal(all_indices[sample_index], dataset.long_path_indices[sample_index]))} "
            f"weights={bool(np.allclose(all_weights[sample_index], dataset.long_path_weights[sample_index]))}"
        )

    if not args.no_plot:
        output_path = OUTPUT_DIR / f"sample_{sample_index:06d}_long_path_label_debug.png"
        plot_debug(
            dataset=dataset,
            sample_index=sample_index,
            vocab=vocab,
            top_indices=order[: args.topk],
            path_indices=args.path_indices,
            query_distance=query_distance,
            path_points_all=path_points_all,
            output_path=output_path,
        )
        print(f"saved debug plot to: {output_path}")


if __name__ == "__main__":
    main()
