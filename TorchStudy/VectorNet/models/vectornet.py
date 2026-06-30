from __future__ import annotations

import torch
from torch import nn

from data.vectorize import VECTOR_FEATURE_DIM
from models.common import gather_target_feature
from models.decoder import TrajectoryDecoder
from models.global_graph import GlobalGraphEncoder
from models.subgraph import SubGraphEncoder


class VectorNet(nn.Module):
    """
    VectorNet 主模型。

    输入:
        x: [B, P, V, F]
            B: batch size
            P: polyline 数量
            V: 每条 polyline 内 vector 数量
            F: 每个 vector 的特征维度

        vector_mask: [B, P, V]
            1 表示有效 vector，0 表示 padding vector。

        polyline_mask: [B, P]
            1 表示有效 polyline，0 表示 padding polyline。

        target_index: [B]
            每个 sample 中目标 agent history polyline 的 index。

    输出:
        future_pred: [B, T, 2]
            目标 agent 未来轨迹。
    """

    def __init__(
        self,
        input_dim: int = VECTOR_FEATURE_DIM,
        hidden_dim: int = 64,
        subgraph_layers: int = 3,
        future_steps: int = 30,
    ) -> None:
        super().__init__()

        self.subgraph = SubGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=subgraph_layers,
        )

        self.global_graph = GlobalGraphEncoder(
            hidden_dim=hidden_dim,
        )

        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            future_steps=future_steps,
        )

    def forward(
        self,
        x: torch.Tensor,
        vector_mask: torch.Tensor,
        polyline_mask: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        polyline_feature = self.subgraph(
            x,
            vector_mask,
        )  # [B, P, D]

        global_feature = self.global_graph(
            polyline_feature,
            polyline_mask,
        )  # [B, P, D]

        target_feature = gather_target_feature(
            global_feature,
            target_index,
        )  # [B, D]

        future_pred = self.decoder(target_feature)  # [B, T, 2]

        return future_pred
