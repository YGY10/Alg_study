from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()


def build_path_soft_targets(
    target_path: torch.Tensor,
    target_path_mask: torch.Tensor,
    path_vocab: torch.Tensor,
    sigma: float = 4.0,
) -> torch.Tensor:
    diff = path_vocab[None, :, :, :2] - target_path[:, None, :, :2]
    dist = diff.pow(2).sum(dim=-1)

    mask = target_path_mask[:, None, :].float()
    dist = dist * mask

    valid_count = mask.sum(dim=-1).clamp(min=1.0)
    dist = dist.sum(dim=-1) / valid_count

    dist = dist * sigma * target_path.shape[1]
    return F.softmax(-dist, dim=-1)


def build_velocity_soft_targets(
    target_velocity: torch.Tensor,
    velocity_vocab: torch.Tensor,
    sigma: float = 4.0,
) -> torch.Tensor:
    dist = (velocity_vocab[None, :, :] - target_velocity[:, None, :]).abs()
    dist = dist.sum(dim=-1) * sigma
    return F.softmax(-dist, dim=-1)


def build_trajectory_soft_targets(
    candidate_trajectories: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_trajectory_mask: torch.Tensor,
    sigma: float = 4.0,
) -> torch.Tensor:
    diff = candidate_trajectories[..., :2] - target_trajectory[:, None, :, :2]
    dist = diff.pow(2).sum(dim=-1)

    mask = target_trajectory_mask[:, None, :].float()
    dist = dist * mask

    valid_count = mask.sum(dim=-1).clamp(min=1.0)
    dist = dist.sum(dim=-1) / valid_count

    dist = dist * sigma * target_trajectory.shape[1]
    return F.softmax(-dist, dim=-1)


def compute_sparse_drive_losses(
    path_scores: torch.Tensor,
    velocity_scores: torch.Tensor,
    trajectory_scores: torch.Tensor,
    candidate_trajectories: torch.Tensor,
    target_path: torch.Tensor,
    target_path_mask: torch.Tensor,
    target_velocity: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_trajectory_mask: torch.Tensor,
    path_vocab: torch.Tensor,
    velocity_vocab: torch.Tensor,
    path_sigma: float = 4.0,
    velocity_sigma: float = 4.0,
    trajectory_sigma: float = 4.0,
    trajectory_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    path_targets = build_path_soft_targets(
        target_path=target_path,
        target_path_mask=target_path_mask,
        path_vocab=path_vocab,
        sigma=path_sigma,
    )
    velocity_targets = build_velocity_soft_targets(
        target_velocity=target_velocity,
        velocity_vocab=velocity_vocab,
        sigma=velocity_sigma,
    )
    trajectory_targets = build_trajectory_soft_targets(
        candidate_trajectories=candidate_trajectories,
        target_trajectory=target_trajectory,
        target_trajectory_mask=target_trajectory_mask,
        sigma=trajectory_sigma,
    )

    path_loss = soft_cross_entropy(path_scores, path_targets)
    velocity_loss = soft_cross_entropy(velocity_scores, velocity_targets)
    trajectory_loss = soft_cross_entropy(trajectory_scores, trajectory_targets)

    total_loss = path_loss + trajectory_loss * trajectory_loss_weight

    return {
        "loss": total_loss,
        "path_loss": path_loss,
        "velocity_loss": velocity_loss,
        "trajectory_loss": trajectory_loss,
    }


def compute_path_velocity_loss(
    path_scores: torch.Tensor,
    velocity_scores: torch.Tensor,
    target_path: torch.Tensor,
    target_path_mask: torch.Tensor,
    target_velocity: torch.Tensor,
    path_vocab: torch.Tensor,
    velocity_vocab: torch.Tensor,
    path_sigma: float = 4.0,
    velocity_sigma: float = 4.0,
) -> dict[str, torch.Tensor]:
    path_targets = build_path_soft_targets(
        target_path=target_path,
        target_path_mask=target_path_mask,
        path_vocab=path_vocab,
        sigma=path_sigma,
    )
    velocity_targets = build_velocity_soft_targets(
        target_velocity=target_velocity,
        velocity_vocab=velocity_vocab,
        sigma=velocity_sigma,
    )

    path_loss = soft_cross_entropy(path_scores, path_targets)
    velocity_loss = soft_cross_entropy(velocity_scores, velocity_targets)

    return {
        "loss": path_loss + velocity_loss,
        "path_loss": path_loss,
        "velocity_loss": velocity_loss,
    }


def compute_teacher_losses(
    path_scores: torch.Tensor,
    velocity_scores: torch.Tensor,
    trajectory_scores: torch.Tensor,
    teacher_path_probs: torch.Tensor,
    teacher_candidate_probs: torch.Tensor,
    teacher_velocity_index: torch.Tensor,
    velocity_loss_weight: float = 0.0,
    trajectory_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    path_loss = soft_cross_entropy(path_scores, teacher_path_probs)
    velocity_loss = F.cross_entropy(velocity_scores, teacher_velocity_index)
    trajectory_loss = soft_cross_entropy(trajectory_scores, teacher_candidate_probs)

    total_loss = (
        path_loss
        + trajectory_loss * trajectory_loss_weight
        + velocity_loss * velocity_loss_weight
    )

    return {
        "loss": total_loss,
        "path_loss": path_loss,
        "velocity_loss": velocity_loss,
        "trajectory_loss": trajectory_loss,
    }


if __name__ == "__main__":
    batch_size = 4
    num_path = 1024
    len_path = 50
    num_velocity = 256
    len_velocity = 8
    num_trajectory = 2048

    loss_dict = compute_sparse_drive_losses(
        path_scores=torch.randn(batch_size, num_path),
        velocity_scores=torch.randn(batch_size, num_velocity),
        trajectory_scores=torch.randn(batch_size, num_trajectory),
        candidate_trajectories=torch.randn(batch_size, num_trajectory, len_velocity, 3),
        target_path=torch.randn(batch_size, len_path, 3),
        target_path_mask=torch.ones(batch_size, len_path),
        target_velocity=torch.rand(batch_size, len_velocity),
        target_trajectory=torch.randn(batch_size, len_velocity, 3),
        target_trajectory_mask=torch.ones(batch_size, len_velocity),
        path_vocab=torch.randn(num_path, len_path, 3),
        velocity_vocab=torch.rand(num_velocity, len_velocity),
    )

    for key, value in loss_dict.items():
        print(key, float(value))
