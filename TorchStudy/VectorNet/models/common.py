from __future__ import annotations
import torch
import torch.nn as nn


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    带 mask 的 max pooling。

    x:
      任意 shape 的 tensor。

    mask:
      与 x 在 pooling 维度前的 shape 对齐。
      常见情况：
        x:    [B, P, V, D]
        mask: [B, P, V]
        dim:  2

      或者：
        x:    [B, P, D]
        mask: [B, P]
        dim:  1

    dim:
      要做 max pooling 的维度。

    return:
      去掉 dim 维度后的 tensor。
    """
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    masked_x = x.masked_fill(mask == 0, -1e9)
    pooled = masked_x.max(dim=dim).values

    pooled = torch.where(
        pooled < -1e8,
        torch.zeros_like(pooled),
        pooled,
    )
    return pooled


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gather_target_feature(
    global_feature: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    """
    从 global graph 输出中取出目标 agent 的 polyline feature。

    global_feature:
      [B, P, D]

    target_index:
      [B]
      每个 batch sample 里 target agent history polyline 的 index。

    return:
      target_feature: [B, D]
    """
    batch_size = global_feature.shape[0]
    batch_index = torch.arange(batch_size, device=global_feature.device)

    return global_feature[batch_index, target_index]
