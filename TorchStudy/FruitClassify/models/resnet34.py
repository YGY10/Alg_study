from __future__ import annotations
import torch
import torch.nn as nn
import timm
from models.cbam import CBAM


class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


def build_resnet34_classifier(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    return ResNet34Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
    )


class ResNet34CBAMClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # 取消原来ResNet34的分类头，只要feature map
        self.backbone = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.attention = CBAM(
            channels=channels,
            reduction=16,
            spatial_kernel=7,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]

        feat = self.backbone(x)[0]
        # feat: [B, 512, H/32, W/32]

        feat = self.attention(feat)
        # feat: [B, 512, H/32, W/32]

        pooled = self.pool(feat)
        # pooled: [B, 512, 1, 1]

        pooled = torch.flatten(pooled, 1)
        # pooled: [B, 512]

        logits = self.classifier(pooled)
        # logits: [B, num_classes]

        return logits


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    if model_name == "resnet34":
        return ResNet34Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "resnet34_cbam":
        return ResNet34CBAMClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
