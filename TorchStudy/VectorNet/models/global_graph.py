"""
Global Graph把每条poly line当作一个节点，通过self attention建模不同polyline之间的关系
"""

from __future__ import annotations
import math
import torch
from torch import nn


class GlobalGraphLayer(nn.Module):
    """
    VectorNet 的全连接Global Graph
    输入：
        x:             [B, P, D]
        polyline_mask: [B, P]
    输出：
        output:        [B, P, D]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, polyline_mask: torch.Tensor) -> torch.Tensor:
        # [B, P, D]
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        # query @ key^T
        # [B, P, D] @ [B, D, P] -> [B, P, P]
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        attention_scores = attention_scores / math.sqrt(self.hidden_dim)

        # key_mask: [B, 1, P]
        # 最后一个 P 表示被关注的 polyline。
        key_mask = polyline_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(
            key_mask == 0,
            float("-inf"),
        )
        # 对每个query, 计算它对所有key的关注权重
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )  # [B, P, P]
        # [B, P, P] @ [B, P, D] -> [B, P, D]
        output = torch.matmul(
            attention_weights,
            value,
        )
        # padding query 的输出归零。
        output = output * polyline_mask.unsqueeze(-1)
        return output


class GlobalGraphEncoder(nn.Module):
    """
    VectorNet Global Graph Encoder

    原始VectorNet 使用fully-connected graph, 通过self-attention 让不同polyline node交换信息
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.global_graph = GlobalGraphLayer(
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        polyline_feature: torch.Tensor,
        polyline_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.global_graph(polyline_feature, polyline_mask)
