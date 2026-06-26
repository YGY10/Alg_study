from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "human_drive",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from human_dataset import HumanDriveDataset
from model import ToySparseDriveV2Model
from vocab import SparseDriveVocab

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints_human"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_human_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--path-chunk-size", type=int, default=32)
    parser.add_argument("--label-batch-size", type=int, default=64)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.2)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--rebuild-cache", action="store_true")
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


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def find_target_candidate_index(
    candidate_path_indices: torch.Tensor,
    candidate_velocity_indices: torch.Tensor,
    target_path_index: torch.Tensor,
    target_velocity_index: torch.Tensor,
) -> torch.Tensor:
    match = (
        (candidate_path_indices == target_path_index[:, None])
        & (candidate_velocity_indices == target_velocity_index[:, None])
    )
    if not bool(match.any(dim=1).all()):
        missing = (~match.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Missing target trajectory in candidates: batch rows {missing}")
    return match.float().argmax(dim=1).long()


def compute_trajectory_errors(
    candidate_trajectories: torch.Tensor,
    trajectory_scores: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_candidate = trajectory_scores.argmax(dim=-1)
    batch_indices = torch.arange(
        trajectory_scores.shape[0],
        device=trajectory_scores.device,
    )
    pred_trajectory = candidate_trajectories[batch_indices, best_candidate]
    mask = target_mask.float()

    diff = pred_trajectory[..., :2] - target_trajectory[..., :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    traj_l2 = (point_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    endpoint_index = mask.sum(dim=-1).long().clamp(min=1) - 1
    pred_endpoint = pred_trajectory[batch_indices, endpoint_index, :2]
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    endpoint_l2 = (pred_endpoint - target_endpoint).pow(2).sum(dim=-1).sqrt()
    return traj_l2.mean(), endpoint_l2.mean()


def run_one_epoch(
    model: ToySparseDriveV2Model,
    dataloader: DataLoader,
    vocab: SparseDriveVocab,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    velocity_loss_weight: float,
    trajectory_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    metric_sums = {
        "loss": 0.0,
        "path_loss": 0.0,
        "velocity_loss": 0.0,
        "trajectory_loss": 0.0,
        "path_acc": 0.0,
        "velocity_acc": 0.0,
        "trajectory_pair_acc": 0.0,
        "traj_l2_error": 0.0,
        "traj_endpoint_error": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        forced_extra_path = batch["path_index"][:, None]

        with torch.set_grad_enabled(is_train):
            output = model(
                input_grid=batch["input_grid"],
                path_vocab=vocab.path,
                velocity_vocab=vocab.velocity,
                trajectory_vocab=vocab.trajectory,
                ego_state=batch["ego_state"],
                extra_path_indices=forced_extra_path,
            )
            target_candidate = find_target_candidate_index(
                candidate_path_indices=output["candidate_path_indices"],
                candidate_velocity_indices=output["candidate_velocity_indices"],
                target_path_index=batch["path_index"],
                target_velocity_index=batch["velocity_index"],
            )

            path_loss = F.cross_entropy(output["path_scores"], batch["path_index"])
            velocity_loss = F.cross_entropy(
                output["velocity_scores"],
                batch["velocity_index"],
            )
            trajectory_loss = F.cross_entropy(
                output["trajectory_scores"],
                target_candidate,
            )
            loss = (
                path_loss
                + trajectory_loss * trajectory_loss_weight
                + velocity_loss * velocity_loss_weight
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        path_pred = output["path_scores"].argmax(dim=-1)
        velocity_pred = output["velocity_scores"].argmax(dim=-1)
        best_candidate = output["trajectory_scores"].argmax(dim=-1)
        batch_indices = torch.arange(
            output["trajectory_scores"].shape[0],
            device=device,
        )
        pred_path = output["candidate_path_indices"][batch_indices, best_candidate]
        pred_velocity = output["candidate_velocity_indices"][
            batch_indices,
            best_candidate,
        ]
        traj_l2, endpoint_l2 = compute_trajectory_errors(
            candidate_trajectories=output["candidate_trajectories"],
            trajectory_scores=output["trajectory_scores"],
            target_trajectory=batch["target_trajectory"],
            target_mask=batch["target_trajectory_mask"],
        )

        metric_sums["loss"] += float(loss.detach())
        metric_sums["path_loss"] += float(path_loss.detach())
        metric_sums["velocity_loss"] += float(velocity_loss.detach())
        metric_sums["trajectory_loss"] += float(trajectory_loss.detach())
        metric_sums["path_acc"] += float((path_pred == batch["path_index"]).float().mean())
        metric_sums["velocity_acc"] += float(
            (velocity_pred == batch["velocity_index"]).float().mean()
        )
        metric_sums["trajectory_pair_acc"] += float(
            (
                (pred_path == batch["path_index"])
                & (pred_velocity == batch["velocity_index"])
            )
            .float()
            .mean()
        )
        metric_sums["traj_l2_error"] += float(traj_l2.detach())
        metric_sums["traj_endpoint_error"] += float(endpoint_l2.detach())
        num_batches += 1

    return {
        key: value / max(num_batches, 1)
        for key, value in metric_sums.items()
    }


def main() -> None:
    args = parse_args()
    episode_paths = sorted(args.episode_dir.glob("*.json"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if len(episode_paths) < 2:
        raise RuntimeError(f"Need at least two human episodes in {args.episode_dir}")

    train_paths, val_paths = split_episode_paths(
        episode_paths,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"episodes: train={len(train_paths)} val={len(val_paths)}")

    vocab_cpu = SparseDriveVocab.load()
    train_dataset = HumanDriveDataset(
        episode_paths=train_paths,
        vocab=vocab_cpu,
        cache_path=args.cache_dir / "train_labels.npz",
        rebuild_cache=args.rebuild_cache,
        path_chunk_size=args.path_chunk_size,
        label_batch_size=args.label_batch_size,
    )
    val_dataset = HumanDriveDataset(
        episode_paths=val_paths,
        vocab=vocab_cpu,
        cache_path=args.cache_dir / "val_labels.npz",
        rebuild_cache=args.rebuild_cache,
        path_chunk_size=args.path_chunk_size,
        label_batch_size=args.label_batch_size,
    )
    print(f"samples: train={len(train_dataset)} val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_l2 = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            vocab=vocab,
            device=device,
            optimizer=optimizer,
            velocity_loss_weight=args.velocity_loss_weight,
            trajectory_loss_weight=args.trajectory_loss_weight,
        )
        with torch.no_grad():
            val_metrics = run_one_epoch(
                model=model,
                dataloader=val_loader,
                vocab=vocab,
                device=device,
                optimizer=None,
                velocity_loss_weight=args.velocity_loss_weight,
                trajectory_loss_weight=args.trajectory_loss_weight,
            )

        is_best = val_metrics["traj_l2_error"] < best_val_l2
        if is_best:
            best_val_l2 = val_metrics["traj_l2_error"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                BEST_CHECKPOINT_PATH,
            )

        suffix = " *best*" if is_best else ""
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"traj_l2 {train_metrics['traj_l2_error']:.3f} "
            f"pair {train_metrics['trajectory_pair_acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"traj_l2 {val_metrics['traj_l2_error']:.3f} "
            f"end_l2 {val_metrics['traj_endpoint_error']:.3f} "
            f"pair {val_metrics['trajectory_pair_acc']:.3f}"
            f"{suffix}",
            flush=True,
        )

    print("best checkpoint:", BEST_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
