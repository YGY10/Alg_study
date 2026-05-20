from __future__ import annotations
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention
    输入： x[B, C, H, W]
    输出： x * weight, 其中weight[B, C, 1, 1]

    作用：
    判断哪些通道更重要
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x：[B, C， H, W]
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        # avg_pool:[B, C, 1, 1]
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        # max_pool:[B, C, 1, 1]

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        weight = self.sigmoid(avg_out + max_out)

        return x * weight


class SpatialAttention(nn.Module):
    """
    Spatial Attention
    输入 x:[B, C, H, W]
    输出 x * weight, 其中weight:[B, 1, H, W]
    作用：
    判断特征图上哪些空间位置更重要
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError(f"kernel_size must be 3 or 7, got {kernel_size}")

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg_map = torch.mean(x, dim=1, keepdim=True)
        # avg_map: [B, 1, H, W]
        max_map = torch.amax(x, dim=1, keepdim=True)
        # max_map: [B, 1, H, W]
        spatial = torch.cat([avg_map, max_map], dim=1)
        # spatial: [B, 2, H, W]
        weight = self.sigmoid(self.conv(spatial))
        # weight: [B, 1, H, W]
        return x * weight


class CBAM(nn.Module):
    """
    CBAM = Channel Attention + Spatial Attention

    输入输出shape不变:
        [B, C, H, W] -> [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
