from __future__ import annotations

from pathlib import Path
import argparse
from functools import partial
import random
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_human import run_one_epoch  # noqa: E402

DATASET_MODULE_PATH = PROJECT_ROOT / "dataset"
if str(DATASET_MODULE_PATH) in sys.path:
    sys.path.remove(str(DATASET_MODULE_PATH))
dataset_module = sys.modules.get("dataset")
if dataset_module is not None and not hasattr(dataset_module, "__path__"):
    del sys.modules["dataset"]

from dataset.nuplan_canonical_dataset import (  # noqa: E402
    FUTURE_TIMES,
    NUM_RASTER_CHANNELS,
    NuPlanCanonicalDataset,
)
from model import ToySparseDriveV2Model  # noqa: E402
from vocab import SparseDriveVocab  # noqa: E402

DEFAULT_EPISODE_DIR = PROJECT_ROOT / "outputs" / "nuplan_canonical" / "episodes"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints_nuplan_canonical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing canonical nuPlan episode json files. Can be repeated.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--max-route-points",
        type=int,
        default=256,
        help="Downsample padded route_path points for explicit path features. Use <=0 to keep all points.",
    )
    parser.add_argument("--model-topk-path", type=int, default=20)
    parser.add_argument("--long-path-horizon", type=float, default=8.0)
    parser.add_argument("--long-path-compare-points", type=int, default=32)
    parser.add_argument("--long-path-positive-topk", type=int, default=8)
    parser.add_argument("--long-path-temperature", type=float, default=2.0)
    parser.add_argument(
        "--forced-long-path-topk",
        type=int,
        default=4,
        help="Add this many long-path teacher anchors to trajectory candidates during training.",
    )
    parser.add_argument(
        "--val-forced-long-path-topk",
        type=int,
        default=0,
        help="Add teacher anchors during validation. Default 0 means free-run validation.",
    )
    parser.add_argument(
        "--path-recall-target-topk",
        type=int,
        default=4,
        help="Measure model path top-k recall against this many long-path labels.",
    )
    parser.add_argument("--path-loss-weight", type=float, default=0.2)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-collision-loss-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--progress-metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--route-metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-metric-scale", type=float, default=2.0)
    parser.add_argument("--progress-metric-scale", type=float, default=2.0)
    parser.add_argument("--route-metric-scale", type=float, default=2.0)
    parser.add_argument("--metric-endpoint-score-weight", type=float, default=5.0)
    parser.add_argument("--metric-progress-score-weight", type=float, default=5.0)
    parser.add_argument("--metric-route-score-weight", type=float, default=2.0)
    parser.add_argument("--metric-imitation-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--metric-no-collision-score-weight",
        type=float,
        default=-1.0,
        help="Weight for no-collision in metric planning score. <0 means auto: enabled only when no-collision loss is trained.",
    )
    parser.add_argument("--unsafe-margin-loss-weight", type=float, default=0.5)
    parser.add_argument("--human-temperature", type=float, default=2.0)
    parser.add_argument("--human-positive-topk", type=int, default=8)
    parser.add_argument("--endpoint-weight", type=float, default=0.5)
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--unsafe-margin", type=float, default=2.0)
    parser.add_argument("--collision-penalty", type=float, default=8.0)
    parser.add_argument(
        "--checkpoint-score",
        choices=("auto", "imitation", "safety"),
        default="auto",
        help=(
            "Metric used to select the best checkpoint. Auto uses imitation when all "
            "safety weights are zero, otherwise safety-aware scoring."
        ),
    )
    return parser.parse_args()


def collect_episode_paths(episode_dirs: list[Path] | None) -> list[Path]:
    dirs = episode_dirs or [DEFAULT_EPISODE_DIR]
    paths: list[Path] = []
    for episode_dir in dirs:
        paths.extend(sorted(Path(episode_dir).glob("*.json")))
    return sorted({path.resolve(): path for path in paths}.values())


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


def summarize_labels(name: str, dataset: NuPlanCanonicalDataset) -> None:
    path_unique = len(set(dataset.path_indices.tolist()))
    velocity_unique = len(set(dataset.velocity_indices.tolist()))
    long_top1_unique = len(set(dataset.long_path_indices[:, 0].tolist()))
    print(
        f"{name} labels: "
        f"path_unique={path_unique} "
        f"velocity_unique={velocity_unique} "
        f"long_top1_unique={long_top1_unique}",
        flush=True,
    )


def balanced_sample_indices(
    dataset: NuPlanCanonicalDataset,
    max_samples: int | None,
    seed: int,
) -> list[int] | None:
    if max_samples is None or max_samples >= len(dataset):
        return None

    episode_groups: dict[Path, list[int]] = {}
    for sample_index, meta in enumerate(dataset.metas):
        episode_groups.setdefault(meta.episode_path, []).append(sample_index)

    rng = random.Random(seed)
    groups = list(episode_groups.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)

    selected: list[int] = []
    group_offsets = [0] * len(groups)
    while len(selected) < max_samples:
        added = False
        for group_index, group in enumerate(groups):
            offset = group_offsets[group_index]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            group_offsets[group_index] += 1
            added = True
            if len(selected) >= max_samples:
                break
        if not added:
            break

    return selected


def resolve_checkpoint_score(args: argparse.Namespace) -> str:
    if args.checkpoint_score != "auto":
        return str(args.checkpoint_score)
    safety_enabled = (
        args.no_collision_loss_weight > 0.0
        or args.unsafe_margin_loss_weight > 0.0
        or args.collision_penalty > 0.0
    )
    return "safety" if safety_enabled else "imitation"


def compute_checkpoint_score(
    metrics: dict[str, float],
    score_mode: str,
) -> float:
    score = metrics["traj_l2_error"] + 0.5 * metrics["traj_endpoint_error"]
    if score_mode == "safety":
        score += 30.0 * metrics["pred_collision_rate"]
    return float(score)


def collate_nuplan_batch(
    batch: list[dict[str, torch.Tensor]],
    max_route_points: int = 256,
) -> dict[str, torch.Tensor]:
    route_paths = [item["route_path"] for item in batch]
    if max_route_points > 0:
        downsampled = []
        for route_path in route_paths:
            if route_path.shape[0] <= max_route_points:
                downsampled.append(route_path)
                continue
            indices = torch.linspace(
                0,
                route_path.shape[0] - 1,
                max_route_points,
                dtype=torch.long,
            )
            downsampled.append(route_path[indices])
        route_paths = downsampled

    max_route_len = max(route_path.shape[0] for route_path in route_paths)
    padded_route_paths = []
    for route_path in route_paths:
        if route_path.shape[0] == max_route_len:
            padded_route_paths.append(route_path)
            continue
        padding = torch.full(
            (max_route_len - route_path.shape[0], route_path.shape[1]),
            float("nan"),
            dtype=route_path.dtype,
        )
        padded_route_paths.append(torch.cat([route_path, padding], dim=0))

    collated = {
        key: default_collate([item[key] for item in batch])
        for key in batch[0].keys()
        if key != "route_path"
    }
    collated["route_path"] = torch.stack(padded_route_paths, dim=0)
    return collated


def build_dataset(
    episode_paths: list[Path],
    vocab: SparseDriveVocab,
    args: argparse.Namespace,
) -> NuPlanCanonicalDataset:
    return NuPlanCanonicalDataset(
        episode_paths=episode_paths,
        vocab=vocab,
        future_times=FUTURE_TIMES,
        long_path_horizon=args.long_path_horizon,
        long_path_compare_points=args.long_path_compare_points,
        long_path_positive_topk=args.long_path_positive_topk,
        long_path_temperature=args.long_path_temperature,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    episode_paths = collect_episode_paths(args.episode_dir)
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if len(episode_paths) < 2:
        raise RuntimeError(
            f"Need at least two canonical nuPlan episodes in {args.episode_dir}"
        )

    train_paths, val_paths = split_episode_paths(
        episode_paths,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"episodes: train={len(train_paths)} val={len(val_paths)}")

    vocab_cpu = SparseDriveVocab.load()
    train_dataset = build_dataset(train_paths, vocab_cpu, args)
    val_dataset = build_dataset(val_paths, vocab_cpu, args)
    print(f"samples: train={len(train_dataset)} val={len(val_dataset)}")
    summarize_labels("train", train_dataset)
    summarize_labels("val", val_dataset)
    train_indices = balanced_sample_indices(
        train_dataset,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_indices = balanced_sample_indices(
        val_dataset,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )
    train_loader_dataset = (
        Subset(train_dataset, train_indices)
        if train_indices is not None
        else train_dataset
    )
    val_loader_dataset = (
        Subset(val_dataset, val_indices) if val_indices is not None else val_dataset
    )
    train_sampled_episodes = len(
        {
            train_dataset.metas[index].episode_path
            for index in (
                train_indices
                if train_indices is not None
                else range(len(train_dataset))
            )
        }
    )
    val_sampled_episodes = len(
        {
            val_dataset.metas[index].episode_path
            for index in (
                val_indices if val_indices is not None else range(len(val_dataset))
            )
        }
    )
    print(
        f"loader samples: train={len(train_loader_dataset)} "
        f"val={len(val_loader_dataset)} sampling=episode_balanced "
        f"sampled_episodes={train_sampled_episodes}/{val_sampled_episodes}",
        flush=True,
    )
    print(
        "candidate forcing: "
        f"model_topk_path={args.model_topk_path} "
        f"train_long_topk={args.forced_long_path_topk} "
        f"val_long_topk={args.val_forced_long_path_topk} "
        f"path_recall_target_topk={args.path_recall_target_topk} "
        f"max_route_points={args.max_route_points}",
        flush=True,
    )
    print(
        "metric heads: "
        f"loss_w(endpoint/progress/route)="
        f"{args.endpoint_metric_loss_weight}/"
        f"{args.progress_metric_loss_weight}/"
        f"{args.route_metric_loss_weight} "
        f"score_w(endpoint/progress/route/imitation)="
        f"{args.metric_endpoint_score_weight}/"
        f"{args.metric_progress_score_weight}/"
        f"{args.metric_route_score_weight}/"
        f"{args.metric_imitation_score_weight} "
        f"no_col_score_w={args.metric_no_collision_score_weight}",
        flush=True,
    )
    collate_fn = partial(
        collate_nuplan_batch,
        max_route_points=args.max_route_points,
    )

    train_loader = DataLoader(
        train_loader_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_loader_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model(
        input_channels=NUM_RASTER_CHANNELS,
        ego_state_dim=7,
        topk_path=args.model_topk_path,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.checkpoint_dir / "best_nuplan_canonical_model.pt"
    checkpoint_score_mode = resolve_checkpoint_score(args)
    print(f"checkpoint score: {checkpoint_score_mode}", flush=True)
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            vocab=vocab,
            device=device,
            optimizer=optimizer,
            args=args,
        )
        with torch.no_grad():
            val_metrics = run_one_epoch(
                model=model,
                dataloader=val_loader,
                vocab=vocab,
                device=device,
                optimizer=None,
                args=args,
            )

        val_score = compute_checkpoint_score(val_metrics, checkpoint_score_mode)
        is_best = val_score < best_score
        if is_best:
            best_score = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_score": val_score,
                    "checkpoint_score_mode": checkpoint_score_mode,
                    "train_episode_paths": [str(path) for path in train_paths],
                    "val_episode_paths": [str(path) for path in val_paths],
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

        suffix = " *best*" if is_best else ""
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"traj_l2 {train_metrics['traj_l2_error']:.3f} "
            f"col {train_metrics['pred_collision_rate']:.3f} "
            f"path_rec {train_metrics['long_path_topk_recall']:.3f} "
            f"pair {train_metrics['trajectory_pair_acc']:.3f} "
            f"hit {train_metrics['target_candidate_hit']:.3f} "
            f"oracle {train_metrics['oracle_candidate_l2']:.3f} "
            f"gap {train_metrics['oracle_gap']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"traj_l2 {val_metrics['traj_l2_error']:.3f} "
            f"end_l2 {val_metrics['traj_endpoint_error']:.3f} "
            f"col {val_metrics['pred_collision_rate']:.3f} "
            f"safe {val_metrics['candidate_safe_rate']:.3f} "
            f"no_col {val_metrics['no_collision_acc']:.3f} "
            f"path_rec {val_metrics['long_path_topk_recall']:.3f} "
            f"pair {val_metrics['trajectory_pair_acc']:.3f} "
            f"hit {val_metrics['target_candidate_hit']:.3f} "
            f"oracle {val_metrics['oracle_candidate_l2']:.3f} "
            f"gap {val_metrics['oracle_gap']:.3f} "
            f"score {val_score:.3f}"
            f"{suffix}",
            flush=True,
        )

    print("best checkpoint:", best_checkpoint_path)


if __name__ == "__main__":
    main()
