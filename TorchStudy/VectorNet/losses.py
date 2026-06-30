from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    轨迹 MSE loss。

    pred:
        [B, T, 2]

    target:
        [B, T, 2]

    return:
        标量 loss
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

    return F.mse_loss(pred, target)


def average_displacement_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    ADE: Average Displacement Error。

    对每个未来时间点计算欧氏距离，然后对 batch 和时间取平均。

    pred:
        [B, T, 2]

    target:
        [B, T, 2]

    return:
        标量 ADE
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

    displacement = torch.norm(pred - target, dim=-1)
    return displacement.mean()
