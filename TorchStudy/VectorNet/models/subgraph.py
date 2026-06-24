from __future__ import annotations
import torch
from torch import nn
from models.common import MLPBlock, masked_max_pool


class SubGraphLayer(nn.Module):
    """
    VectorNet的一层 local subgraph
    输入：
        x: [B, P, V, C]
        vector_mask: [B, P, B]
    输出：
        out: [B, P, V, 2 * hidden_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = MLPBlock(
            in_dim=in_dim,
            out_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        vector_mask: torch.Tensor,
    ) -> torch.Tensor:
        local_feature = self.mlp(x)
