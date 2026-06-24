from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "losses",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dataset.dataset import ToySparseDriveV2Dataset
except ModuleNotFoundError:
    from dataset import ToySparseDriveV2Dataset
from losses import compute_model_candidate_losses
from model import ToySparseDriveV2Model
from teacher import (
    DynamicObstacle,
    EgoState,
    TeacherConfig,
    score_trajectories,
    softmax_from_cost,
)
from vocab import SparseDriveVocab

BATCH_SIZE = 18
NUM_SAMPLES = 1024
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
TEACHER_CACHE_DIR = PROJECT_ROOT / "cache" / "teacher_v1"
SAFETY_SCORE_WEIGHT = 1.0


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_accuracy(
    scores: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    prediction = scores.argmax(dim=-1)
    return (prediction == target_index).float().mean()


def compute_planning_scores(
    trajectory_scores: torch.Tensor,
    no_collision_logits: torch.Tensor,
    safety_score_weight: float = SAFETY_SCORE_WEIGHT,
) -> torch.Tensor:
    return trajectory_scores + no_collision_logits * safety_score_weight


def compute_no_collision_accuracy(
    no_collision_logits: torch.Tensor,
    candidate_collision: torch.Tensor,
) -> torch.Tensor:
    prediction_is_safe = no_collision_logits >= 0.0
    target_is_safe = ~candidate_collision.bool()
    return (prediction_is_safe == target_is_safe).float().mean()


def compute_candidate_metrics(
    trajectory_scores: torch.Tensor,
    candidate_path_indices: torch.Tensor,
    candidate_velocity_indices: torch.Tensor,
    candidate_trajectories: torch.Tensor,
    model_topk_path_indices: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_trajectory_mask: torch.Tensor,
    target_path_index: torch.Tensor,
    target_velocity_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    best_candidate = trajectory_scores.argmax(dim=-1)
    batch_indices = torch.arange(
        trajectory_scores.shape[0],
        device=trajectory_scores.device,
    )

    pred_path = candidate_path_indices[batch_indices, best_candidate]
    pred_velocity = candidate_velocity_indices[batch_indices, best_candidate]
    pred_trajectory = candidate_trajectories[batch_indices, best_candidate]

    target_path_in_candidates = (
        candidate_path_indices == target_path_index[:, None]
    ).any(dim=1)
    target_path_in_model_topk = (
        model_topk_path_indices == target_path_index[:, None]
    ).any(dim=1)
    pred_path_acc = (pred_path == target_path_index).float().mean()
    pred_velocity_acc = (pred_velocity == target_velocity_index).float().mean()
    pred_pair_acc = (
        ((pred_path == target_path_index) & (pred_velocity == target_velocity_index))
        .float()
        .mean()
    )

    mask = target_trajectory_mask.float()
    diff = pred_trajectory[..., :2] - target_trajectory[..., :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    traj_l2_error = (point_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    traj_l2_error = traj_l2_error.mean()

    valid_count = mask.sum(dim=-1).long().clamp(min=1)
    endpoint_index = valid_count - 1
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    pred_endpoint = pred_trajectory[batch_indices, endpoint_index, :2]
    traj_endpoint_error = (
        (pred_endpoint - target_endpoint).pow(2).sum(dim=-1).sqrt().mean()
    )

    return {
        "topk_path_recall": target_path_in_candidates.float().mean(),
        "model_topk_path_recall": target_path_in_model_topk.float().mean(),
        "trajectory_path_acc": pred_path_acc,
        "trajectory_velocity_acc": pred_velocity_acc,
        "trajectory_pair_acc": pred_pair_acc,
        "traj_l2_error": traj_l2_error,
        "traj_endpoint_error": traj_endpoint_error,
    }


def build_teacher_candidate_targets_for_batch(
    vocab: SparseDriveVocab,
    candidate_path_indices: torch.Tensor,
    candidate_velocity_indices: torch.Tensor,
    batch: dict[str, torch.Tensor],
    teacher_config: TeacherConfig,
) -> dict[str, torch.Tensor]:
    device = candidate_path_indices.device
    path_indices_np = candidate_path_indices.detach().cpu().numpy()
    velocity_indices_np = candidate_velocity_indices.detach().cpu().numpy()

    goal_xy_np = batch["goal_xy"].detach().cpu().numpy()
    route_path_np = batch["route_path"].detach().cpu().numpy()[..., :2]
    ego_state_np = batch["ego_state"].detach().cpu().numpy()
    obstacle_centers_np = batch["obstacle_centers"].detach().cpu().numpy()
    obstacle_sizes_np = batch["obstacle_sizes"].detach().cpu().numpy()
    obstacle_velocities_np = batch["obstacle_velocities"].detach().cpu().numpy()
    obstacle_mask_np = batch["obstacle_mask"].detach().cpu().numpy() > 0.5

    batch_size, num_candidates = path_indices_np.shape
    teacher_probs = np.zeros((batch_size, num_candidates), dtype=np.float32)
    teacher_costs = np.zeros((batch_size, num_candidates), dtype=np.float32)
    teacher_collision = np.zeros((batch_size, num_candidates), dtype=bool)
    teacher_clearance = np.zeros((batch_size, num_candidates), dtype=np.float32)

    for batch_index in range(batch_size):
        obstacles = []
        for obstacle_index in np.flatnonzero(obstacle_mask_np[batch_index]):
            obstacles.append(
                DynamicObstacle(
                    center_xy=obstacle_centers_np[batch_index, obstacle_index],
                    size_xy=(
                        float(obstacle_sizes_np[batch_index, obstacle_index, 0]),
                        float(obstacle_sizes_np[batch_index, obstacle_index, 1]),
                    ),
                    velocity_xy=obstacle_velocities_np[batch_index, obstacle_index],
                    id=f"batch{batch_index}_obs{obstacle_index}",
                )
            )

        ego_vector = ego_state_np[batch_index]
        ego_state = EgoState(
            speed=float(ego_vector[0]),
            size_xy=(float(ego_vector[1]), float(ego_vector[2])),
        )

        unique_path_indices = np.unique(path_indices_np[batch_index]).astype(np.int64)
        scored = score_trajectories(
            vocab=vocab,
            path_indices=unique_path_indices,
            goal_xy=goal_xy_np[batch_index],
            obstacles=obstacles,
            ego_state=ego_state,
            route_path=route_path_np[batch_index],
            config=teacher_config,
        )

        scored_flat_indices = scored["candidate_flat_indices"]
        scored_costs = scored["cost"]
        scored_collision = scored["collision"]
        scored_clearance = scored["clearance"]
        cost_by_flat_index = {
            int(flat_index): float(cost)
            for flat_index, cost in zip(scored_flat_indices, scored_costs)
        }
        collision_by_flat_index = {
            int(flat_index): bool(collision)
            for flat_index, collision in zip(scored_flat_indices, scored_collision)
        }
        clearance_by_flat_index = {
            int(flat_index): float(clearance)
            for flat_index, clearance in zip(scored_flat_indices, scored_clearance)
        }

        candidate_flat_indices = (
            path_indices_np[batch_index] * vocab.num_velocity
            + velocity_indices_np[batch_index]
        )
        candidate_costs = np.array(
            [
                cost_by_flat_index[int(flat_index)]
                for flat_index in candidate_flat_indices
            ],
            dtype=np.float32,
        )
        candidate_collision = np.array(
            [
                collision_by_flat_index[int(flat_index)]
                for flat_index in candidate_flat_indices
            ],
            dtype=bool,
        )
        candidate_clearance = np.array(
            [
                clearance_by_flat_index[int(flat_index)]
                for flat_index in candidate_flat_indices
            ],
            dtype=np.float32,
        )
        teacher_probs[batch_index] = softmax_from_cost(
            candidate_costs,
            temperature=teacher_config.temperature,
        ).astype(np.float32)
        teacher_costs[batch_index] = candidate_costs
        teacher_collision[batch_index] = candidate_collision
        teacher_clearance[batch_index] = candidate_clearance

    return {
        "probs": torch.from_numpy(teacher_probs).to(device=device),
        "costs": torch.from_numpy(teacher_costs).to(device=device),
        "collision": torch.from_numpy(teacher_collision).to(device=device),
        "clearance": torch.from_numpy(teacher_clearance).to(device=device),
    }


def train_one_epoch(
    model: ToySparseDriveV2Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab: SparseDriveVocab,
    teacher_vocab: SparseDriveVocab,
    device: torch.device,
    teacher_config: TeacherConfig,
) -> dict[str, float]:
    model.train()

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    metric_sums = {
        "loss": 0.0,
        "path_loss": 0.0,
        "velocity_loss": 0.0,
        "trajectory_loss": 0.0,
        "no_collision_loss": 0.0,
        "collision_margin_loss": 0.0,
        "path_acc": 0.0,
        "velocity_acc": 0.0,
        "topk_path_recall": 0.0,
        "model_topk_path_recall": 0.0,
        "trajectory_path_acc": 0.0,
        "trajectory_velocity_acc": 0.0,
        "trajectory_pair_acc": 0.0,
        "traj_l2_error": 0.0,
        "traj_endpoint_error": 0.0,
        "pred_collision_rate": 0.0,
        "candidate_safe_rate": 0.0,
        "no_collision_acc": 0.0,
        "pred_teacher_cost": 0.0,
        "raw_pred_collision_rate": 0.0,
        "raw_trajectory_pair_acc": 0.0,
        "raw_traj_l2_error": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(
                input_grid=batch["input_grid"],
                path_vocab=vocab.path,
                velocity_vocab=vocab.velocity,
                trajectory_vocab=vocab.trajectory,
                ego_state=batch["ego_state"],
                extra_path_indices=batch["teacher_path_indices"],
            )

            teacher_candidate_targets = build_teacher_candidate_targets_for_batch(
                vocab=teacher_vocab,
                candidate_path_indices=output["candidate_path_indices"],
                candidate_velocity_indices=output["candidate_velocity_indices"],
                batch=batch,
                teacher_config=teacher_config,
            )

            loss_dict = compute_model_candidate_losses(
                path_scores=output["path_scores"],
                velocity_scores=output["velocity_scores"],
                trajectory_scores=output["trajectory_scores"],
                no_collision_logits=output["no_collision_logits"],
                teacher_path_probs=batch["teacher_path_probs"],
                teacher_candidate_probs=teacher_candidate_targets["probs"],
                teacher_velocity_index=batch["velocity_index"],
                candidate_collision=teacher_candidate_targets["collision"],
                safety_score_weight=SAFETY_SCORE_WEIGHT,
            )

        optimizer.zero_grad()

        # 混合精度的反向传播
        if scaler is not None:
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict["loss"].backward()
            optimizer.step()

        path_acc = compute_accuracy(output["path_scores"], batch["path_index"])
        velocity_acc = compute_accuracy(
            output["velocity_scores"], batch["velocity_index"]
        )
        planning_scores = compute_planning_scores(
            trajectory_scores=output["trajectory_scores"],
            no_collision_logits=output["no_collision_logits"],
        )
        best_candidate = planning_scores.argmax(dim=-1)
        batch_indices = torch.arange(
            planning_scores.shape[0],
            device=planning_scores.device,
        )
        pred_collision_rate = (
            teacher_candidate_targets["collision"][
                batch_indices,
                best_candidate,
            ]
            .float()
            .mean()
        )
        candidate_safe_rate = (~teacher_candidate_targets["collision"]).float().mean()
        pred_teacher_cost = teacher_candidate_targets["costs"][
            batch_indices,
            best_candidate,
        ].mean()
        no_collision_acc = compute_no_collision_accuracy(
            no_collision_logits=output["no_collision_logits"],
            candidate_collision=teacher_candidate_targets["collision"],
        )

        candidate_metrics = compute_candidate_metrics(
            trajectory_scores=planning_scores,
            candidate_path_indices=output["candidate_path_indices"],
            candidate_velocity_indices=output["candidate_velocity_indices"],
            candidate_trajectories=output["candidate_trajectories"],
            model_topk_path_indices=output["model_topk_path_indices"],
            target_trajectory=batch["target_trajectory"],
            target_trajectory_mask=batch["target_trajectory_mask"],
            target_path_index=batch["path_index"],
            target_velocity_index=batch["velocity_index"],
        )
        raw_candidate_metrics = compute_candidate_metrics(
            trajectory_scores=output["trajectory_scores"],
            candidate_path_indices=output["candidate_path_indices"],
            candidate_velocity_indices=output["candidate_velocity_indices"],
            candidate_trajectories=output["candidate_trajectories"],
            model_topk_path_indices=output["model_topk_path_indices"],
            target_trajectory=batch["target_trajectory"],
            target_trajectory_mask=batch["target_trajectory_mask"],
            target_path_index=batch["path_index"],
            target_velocity_index=batch["velocity_index"],
        )
        raw_best_candidate = output["trajectory_scores"].argmax(dim=-1)
        raw_pred_collision_rate = (
            teacher_candidate_targets["collision"][
                batch_indices,
                raw_best_candidate,
            ]
            .float()
            .mean()
        )

        batch_metrics = {
            "loss": loss_dict["loss"],
            "path_loss": loss_dict["path_loss"],
            "velocity_loss": loss_dict["velocity_loss"],
            "trajectory_loss": loss_dict["trajectory_loss"],
            "no_collision_loss": loss_dict["no_collision_loss"],
            "collision_margin_loss": loss_dict["collision_margin_loss"],
            "path_acc": path_acc,
            "velocity_acc": velocity_acc,
            **candidate_metrics,
            "pred_collision_rate": pred_collision_rate,
            "candidate_safe_rate": candidate_safe_rate,
            "no_collision_acc": no_collision_acc,
            "pred_teacher_cost": pred_teacher_cost,
            "raw_pred_collision_rate": raw_pred_collision_rate,
            "raw_trajectory_pair_acc": raw_candidate_metrics["trajectory_pair_acc"],
            "raw_traj_l2_error": raw_candidate_metrics["traj_l2_error"],
        }

        for key, value in batch_metrics.items():
            metric_sums[key] += float(value.detach().cpu())
        num_batches += 1

    return {key: value / num_batches for key, value in metric_sums.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: ToySparseDriveV2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": {
                "batch_size": BATCH_SIZE,
                "num_samples": NUM_SAMPLES,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        path,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = ToySparseDriveV2Dataset(
        num_samples=NUM_SAMPLES,
        seed_offset=0,
        teacher_cache_dir=TEACHER_CACHE_DIR,
        require_teacher_cache=True,
    )
    missing_cache_paths = [
        dataset.teacher_cache_path(index)
        for index in range(len(dataset))
        if not dataset.teacher_cache_path(index).is_file()
    ]
    if missing_cache_paths:
        raise FileNotFoundError(
            f"Teacher cache is incomplete: {len(missing_cache_paths)} missing files. "
            f"First missing file: {missing_cache_paths[0]}. "
            "Run: cd dataset && python build_teacher_cache.py "
            "--num-samples 1024 --output-dir ../cache/teacher_v1"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    vocab = SparseDriveVocab.load().to(device)
    model = ToySparseDriveV2Model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-5,
    )

    best_collision_rate = float("inf")
    best_traj_l2 = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            vocab=vocab,
            teacher_vocab=dataset.vocab,
            device=device,
            teacher_config=dataset.teacher_config,
        )
        scheduler.step()

        collision_improved = (
            metrics["pred_collision_rate"] < best_collision_rate - 1.0e-6
        )
        same_collision = (
            abs(metrics["pred_collision_rate"] - best_collision_rate) <= 1.0e-6
        )
        l2_improved = metrics["traj_l2_error"] < best_traj_l2
        is_best = collision_improved or (same_collision and l2_improved)
        if is_best:
            best_collision_rate = metrics["pred_collision_rate"]
            best_traj_l2 = metrics["traj_l2_error"]
            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )

        lr = optimizer.param_groups[0]["lr"]
        best_mark = " *best*" if is_best else ""
        print(
            f"epoch {epoch:03d} | "
            f"lr {lr:.6f} | "
            f"loss {metrics['loss']:.4f} | "
            f"path {metrics['path_loss']:.4f} | "
            f"vel {metrics['velocity_loss']:.4f} | "
            f"traj {metrics['trajectory_loss']:.4f} | "
            f"no_col {metrics['no_collision_loss']:.4f} | "
            f"margin {metrics['collision_margin_loss']:.4f} | "
            f"path_acc {metrics['path_acc']:.3f} | "
            f"vel_acc {metrics['velocity_acc']:.3f} | "
            f"topk_path {metrics['topk_path_recall']:.3f} | "
            f"model_topk {metrics['model_topk_path_recall']:.3f} | "
            f"traj_vel {metrics['trajectory_velocity_acc']:.3f} | "
            f"traj_pair {metrics['trajectory_pair_acc']:.3f} | "
            f"traj_l2 {metrics['traj_l2_error']:.3f} | "
            f"end_l2 {metrics['traj_endpoint_error']:.3f} | "
            f"raw_l2 {metrics['raw_traj_l2_error']:.3f} | "
            f"collision {metrics['pred_collision_rate']:.3f} | "
            f"raw_col {metrics['raw_pred_collision_rate']:.3f} | "
            f"raw_pair {metrics['raw_trajectory_pair_acc']:.3f} | "
            f"safe {metrics['candidate_safe_rate']:.3f} | "
            f"no_col_acc {metrics['no_collision_acc']:.3f} | "
            f"pred_cost {metrics['pred_teacher_cost']:.1f}"
            f"{best_mark}"
        )

    print(f"best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"best collision_rate: {best_collision_rate:.6f}")
    print(f"best traj_l2: {best_traj_l2:.6f}")


if __name__ == "__main__":
    main()
