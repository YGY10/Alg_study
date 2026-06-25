from __future__ import annotations
import torch
from torch import nn
from models.common import MLPBlock, masked_max_pool
import numpy as np


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
        # 每个vector node独立经过MLP
        local_feature = self.mlp(x)

        # 对同一条polyline中所有vector的MLP特征做masked max pooling
        polyline_feature = masked_max_pool(
            local_feature,
            vector_mask,
            dim=2,
        )  # [B, P, D]
        num_vectors = local_feature.shape[2]
        polyline_feature = polyline_feature.unsqueeze(2).expand(
            -1,
            -1,
            num_vectors,
            -1,
        )
        # 拼接 vector 自身特征和所属polyline的局部特征
        output = torch.cat(
            [local_feature, polyline_feature],
            dim=-1,
        )  # [B, P, V, 2D]
        # padding vector 保持为0
        output = output * vector_mask.unsqueeze(-1)
        return output


class SubGraphEncoder(nn.Module):
    """
    VectorNet Local SubGraph Encoder
    输入：
        x: [B, P, V, inpue_dim]
        vector_mask: [B, P, V]
    输出：
        polyline_feature: [B, P, hidden_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(
                SubGraphLayer(
                    in_dim=current_dim,
                    hidden_dim=hidden_dim,
                )
            )
            # 每层concat后，输出维度变成 2 * hidden_dim
            current_dim = hidden_dim * 2

        self.layers = nn.ModuleList(layers)

        # 最后一层输出是2D，投影回D，方便GlobalGraph使用
        self.output_projection = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        vector_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, vector_mask)
        # 对最后一层的vector node再做一次 max pooling
        polyline_feature = masked_max_pool(
            h,
            vector_mask,
            dim=2,
        )  # [B, P, 2D]
        polyline_feature = self.output_projection(polyline_feature)  # [B, P, D]

        return polyline_feature
