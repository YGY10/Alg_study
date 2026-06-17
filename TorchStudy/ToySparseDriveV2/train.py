from __future__ import annotations

from pathlib import Path
import sys

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

from dataset import ToySparseDriveV2Dataset
from losses import compute_model_candidate_losses
from model import ToySparseDriveV2Model
from vocab import SparseDriveVocab

BATCH_SIZE = 64
NUM_SAMPLES = 256
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"


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


def train_one_epoch(
    model: ToySparseDriveV2Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab: SparseDriveVocab,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    metric_sums = {
        "loss": 0.0,
        "path_loss": 0.0,
        "velocity_loss": 0.0,
        "trajectory_loss": 0.0,
        "path_acc": 0.0,
        "velocity_acc": 0.0,
        "topk_path_recall": 0.0,
        "model_topk_path_recall": 0.0,
        "trajectory_path_acc": 0.0,
        "trajectory_velocity_acc": 0.0,
        "trajectory_pair_acc": 0.0,
        "traj_l2_error": 0.0,
        "traj_endpoint_error": 0.0,
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

            loss_dict = compute_model_candidate_losses(
                path_scores=output["path_scores"],
                velocity_scores=output["velocity_scores"],
                trajectory_scores=output["trajectory_scores"],
                candidate_trajectories=output["candidate_trajectories"],
                teacher_path_probs=batch["teacher_path_probs"],
                target_trajectory=batch["target_trajectory"],
                target_trajectory_mask=batch["target_trajectory_mask"],
                teacher_velocity_index=batch["velocity_index"],
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
        candidate_metrics = compute_candidate_metrics(
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

        batch_metrics = {
            "loss": loss_dict["loss"],
            "path_loss": loss_dict["path_loss"],
            "velocity_loss": loss_dict["velocity_loss"],
            "trajectory_loss": loss_dict["trajectory_loss"],
            "path_acc": path_acc,
            "velocity_acc": velocity_acc,
            **candidate_metrics,
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

    best_traj_l2 = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            vocab=vocab,
            device=device,
        )
        scheduler.step()

        is_best = metrics["traj_l2_error"] < best_traj_l2
        if is_best:
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
            f"path_acc {metrics['path_acc']:.3f} | "
            f"vel_acc {metrics['velocity_acc']:.3f} | "
            f"topk_path {metrics['topk_path_recall']:.3f} | "
            f"model_topk {metrics['model_topk_path_recall']:.3f} | "
            f"traj_vel {metrics['trajectory_velocity_acc']:.3f} | "
            f"traj_pair {metrics['trajectory_pair_acc']:.3f} | "
            f"traj_l2 {metrics['traj_l2_error']:.3f} | "
            f"end_l2 {metrics['traj_endpoint_error']:.3f}"
            f"{best_mark}"
        )

    print(f"best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"best traj_l2: {best_traj_l2:.6f}")


if __name__ == "__main__":
    main()
