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

from human_dataset import FUTURE_TIMES, HumanDriveDataset
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
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
    parser.add_argument("--path-loss-weight", type=float, default=0.2)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-collision-loss-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-margin-loss-weight", type=float, default=0.5)
    parser.add_argument("--human-temperature", type=float, default=2.0)
    parser.add_argument("--human-positive-topk", type=int, default=8)
    parser.add_argument("--endpoint-weight", type=float, default=0.5)
    parser.add_argument("--safety-margin", type=float, default=0.3)
    parser.add_argument("--unsafe-margin", type=float, default=2.0)
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


def trajectory_distance_to_human(
    candidate_trajectories: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
    endpoint_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = target_mask.float()
    diff = candidate_trajectories[..., :2] - target_trajectory[:, None, :, :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    traj_l2 = (point_l2 * mask[:, None, :]).sum(dim=-1) / mask.sum(dim=-1)[:, None].clamp(min=1.0)

    batch_size = candidate_trajectories.shape[0]
    num_candidates = candidate_trajectories.shape[1]
    batch_indices = torch.arange(batch_size, device=candidate_trajectories.device)
    endpoint_index = mask.sum(dim=-1).long().clamp(min=1) - 1
    gather_index = endpoint_index[:, None, None, None].expand(batch_size, num_candidates, 1, 3)
    candidate_endpoint = torch.gather(candidate_trajectories, dim=2, index=gather_index).squeeze(2)[..., :2]
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    endpoint_l2 = (candidate_endpoint - target_endpoint[:, None]).pow(2).sum(dim=-1).sqrt()
    combined = traj_l2 + endpoint_l2 * float(endpoint_weight)
    return combined, traj_l2, endpoint_l2


def soft_path_human_loss(
    path_scores: torch.Tensor,
    path_vocab: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
    positive_topk: int,
    temperature: float,
    endpoint_weight: float,
) -> torch.Tensor:
    target_len = target_trajectory.shape[1]
    path_xy = path_vocab[:, :target_len, :2]
    mask = target_mask.float()
    diff = path_xy[None, :, :, :] - target_trajectory[:, None, :, :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    path_l2 = (point_l2 * mask[:, None, :]).sum(dim=-1) / mask.sum(dim=-1)[:, None].clamp(min=1.0)

    endpoint_index = mask.sum(dim=-1).long().clamp(min=1) - 1
    batch_size = target_trajectory.shape[0]
    num_path = path_xy.shape[0]
    batch_indices = torch.arange(batch_size, device=target_trajectory.device)
    expanded_path_xy = path_xy[None, :, :, :].expand(batch_size, -1, -1, -1)
    gather_index = endpoint_index[:, None, None, None].expand(batch_size, num_path, 1, 2)
    path_endpoint = torch.gather(expanded_path_xy, dim=2, index=gather_index).squeeze(2)
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    endpoint_l2 = (path_endpoint - target_endpoint[:, None]).pow(2).sum(dim=-1).sqrt()
    cost = path_l2 + endpoint_l2 * float(endpoint_weight)

    k = min(int(positive_topk), cost.shape[1])
    top_cost, top_indices = torch.topk(cost, k=k, dim=1, largest=False)
    weights = torch.softmax(-top_cost / max(float(temperature), 1.0e-6), dim=1)
    log_probs = F.log_softmax(path_scores, dim=1)
    selected_log_probs = torch.gather(log_probs, 1, top_indices)
    return -(weights.detach() * selected_log_probs).sum(dim=1).mean()


def compute_candidate_collision(
    candidate_trajectories: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_sizes: torch.Tensor,
    obstacle_velocities: torch.Tensor,
    obstacle_mask: torch.Tensor,
    ego_state: torch.Tensor,
    safety_margin: float,
) -> torch.Tensor:
    device = candidate_trajectories.device
    times = torch.as_tensor(FUTURE_TIMES, dtype=torch.float32, device=device)
    obstacle_centers_t = obstacle_centers[:, None, None, :, :] + (
        obstacle_velocities[:, None, None, :, :] * times[None, None, :, None, None]
    )
    ego_centers = candidate_trajectories[..., :2][:, :, :, None, :]
    ego_size = ego_state[:, 1:3]
    half_extent = (
        0.5 * (ego_size[:, None, None, None, :] + obstacle_sizes[:, None, None, :, :])
        + float(safety_margin)
    )
    overlap_xy = (ego_centers - obstacle_centers_t).abs() <= half_extent
    overlap = overlap_xy.all(dim=-1) & obstacle_mask[:, None, None, :].bool()
    return overlap.any(dim=-1).any(dim=-1)


def soft_human_trajectory_loss(
    trajectory_scores: torch.Tensor,
    candidate_cost: torch.Tensor,
    candidate_collision: torch.Tensor,
    positive_topk: int,
    temperature: float,
) -> torch.Tensor:
    safe_mask = ~candidate_collision.bool()
    has_safe = safe_mask.any(dim=1)
    masked_cost = torch.where(safe_mask, candidate_cost, torch.full_like(candidate_cost, float("inf")))
    cost_for_topk = torch.where(has_safe[:, None], masked_cost, candidate_cost)

    k = min(int(positive_topk), cost_for_topk.shape[1])
    top_cost, top_indices = torch.topk(cost_for_topk, k=k, dim=1, largest=False)
    weights = torch.softmax(-top_cost / max(float(temperature), 1.0e-6), dim=1)
    log_probs = F.log_softmax(trajectory_scores, dim=1)
    selected_log_probs = torch.gather(log_probs, 1, top_indices)
    return -(weights.detach() * selected_log_probs).sum(dim=1).mean()


def unsafe_margin_loss(
    planning_scores: torch.Tensor,
    candidate_collision: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    unsafe_mask = candidate_collision.bool()
    safe_mask = ~unsafe_mask
    has_safe = safe_mask.any(dim=1)
    has_unsafe = unsafe_mask.any(dim=1)
    valid = has_safe & has_unsafe
    if not bool(valid.any()):
        return planning_scores.sum() * 0.0

    very_negative = torch.full_like(planning_scores, -1.0e9)
    safe_scores = torch.where(safe_mask, planning_scores, very_negative)
    unsafe_scores = torch.where(unsafe_mask, planning_scores, very_negative)
    best_safe = safe_scores.max(dim=1).values
    best_unsafe = unsafe_scores.max(dim=1).values
    return F.relu(best_unsafe[valid] + float(margin) - best_safe[valid]).mean()


def compute_trajectory_errors(
    candidate_trajectories: torch.Tensor,
    planning_scores: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_candidate = planning_scores.argmax(dim=-1)
    batch_indices = torch.arange(
        planning_scores.shape[0],
        device=planning_scores.device,
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
    args: argparse.Namespace,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    metric_sums = {
        "loss": 0.0,
        "path_loss": 0.0,
        "velocity_loss": 0.0,
        "trajectory_loss": 0.0,
        "no_collision_loss": 0.0,
        "unsafe_margin_loss": 0.0,
        "path_acc": 0.0,
        "velocity_acc": 0.0,
        "trajectory_pair_acc": 0.0,
        "traj_l2_error": 0.0,
        "traj_endpoint_error": 0.0,
        "pred_collision_rate": 0.0,
        "candidate_safe_rate": 0.0,
        "no_collision_acc": 0.0,
        "human_candidate_min_l2": 0.0,
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
            candidate_collision = compute_candidate_collision(
                candidate_trajectories=output["candidate_trajectories"],
                obstacle_centers=batch["obstacle_centers"],
                obstacle_sizes=batch["obstacle_sizes"],
                obstacle_velocities=batch["obstacle_velocities"],
                obstacle_mask=batch["obstacle_mask"],
                ego_state=batch["ego_state"],
                safety_margin=args.safety_margin,
            )
            candidate_cost, candidate_l2, _ = trajectory_distance_to_human(
                candidate_trajectories=output["candidate_trajectories"],
                target_trajectory=batch["target_trajectory"],
                target_mask=batch["target_trajectory_mask"],
                endpoint_weight=args.endpoint_weight,
            )

            path_loss = soft_path_human_loss(
                path_scores=output["path_scores"],
                path_vocab=vocab.path,
                target_trajectory=batch["target_trajectory"],
                target_mask=batch["target_trajectory_mask"],
                positive_topk=args.human_positive_topk,
                temperature=args.human_temperature,
                endpoint_weight=args.endpoint_weight,
            )
            velocity_loss = F.cross_entropy(
                output["velocity_scores"],
                batch["velocity_index"],
            )
            trajectory_loss = soft_human_trajectory_loss(
                trajectory_scores=output["trajectory_scores"],
                candidate_cost=candidate_cost,
                candidate_collision=candidate_collision,
                positive_topk=args.human_positive_topk,
                temperature=args.human_temperature,
            )
            no_collision_target = (~candidate_collision).float()
            no_collision_loss = F.binary_cross_entropy_with_logits(
                output["no_collision_logits"],
                no_collision_target,
            )
            planning_scores = output["trajectory_scores"] + output["no_collision_logits"]
            margin_loss = unsafe_margin_loss(
                planning_scores=planning_scores,
                candidate_collision=candidate_collision,
                margin=args.unsafe_margin,
            )
            loss = (
                path_loss * args.path_loss_weight
                + trajectory_loss * args.trajectory_loss_weight
                + velocity_loss * args.velocity_loss_weight
                + no_collision_loss * args.no_collision_loss_weight
                + margin_loss * args.unsafe_margin_loss_weight
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        path_pred = output["path_scores"].argmax(dim=-1)
        velocity_pred = output["velocity_scores"].argmax(dim=-1)
        best_candidate = planning_scores.argmax(dim=-1)
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
            planning_scores=planning_scores,
            target_trajectory=batch["target_trajectory"],
            target_mask=batch["target_trajectory_mask"],
        )
        pred_collision = candidate_collision[batch_indices, best_candidate]
        no_collision_prediction = output["no_collision_logits"] >= 0.0
        no_collision_acc = (no_collision_prediction == (~candidate_collision)).float().mean()

        metric_sums["loss"] += float(loss.detach())
        metric_sums["path_loss"] += float(path_loss.detach())
        metric_sums["velocity_loss"] += float(velocity_loss.detach())
        metric_sums["trajectory_loss"] += float(trajectory_loss.detach())
        metric_sums["no_collision_loss"] += float(no_collision_loss.detach())
        metric_sums["unsafe_margin_loss"] += float(margin_loss.detach())
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
        metric_sums["pred_collision_rate"] += float(pred_collision.float().mean())
        metric_sums["candidate_safe_rate"] += float((~candidate_collision).float().mean())
        metric_sums["no_collision_acc"] += float(no_collision_acc.detach())
        metric_sums["human_candidate_min_l2"] += float(candidate_l2.min(dim=1).values.mean().detach())
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

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.checkpoint_dir / "best_human_model.pt"
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

        # Safety first, then human-like geometry. A collision is much worse than a few meters of imitation error.
        val_score = val_metrics["traj_l2_error"] + 50.0 * val_metrics["pred_collision_rate"]
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
            f"pair {train_metrics['trajectory_pair_acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"traj_l2 {val_metrics['traj_l2_error']:.3f} "
            f"end_l2 {val_metrics['traj_endpoint_error']:.3f} "
            f"col {val_metrics['pred_collision_rate']:.3f} "
            f"safe {val_metrics['candidate_safe_rate']:.3f} "
            f"no_col {val_metrics['no_collision_acc']:.3f} "
            f"pair {val_metrics['trajectory_pair_acc']:.3f}"
            f"{suffix}",
            flush=True,
        )

    print("best checkpoint:", best_checkpoint_path)


if __name__ == "__main__":
    main()
