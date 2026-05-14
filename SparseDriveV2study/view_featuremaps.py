import torch
from pathlib import Path

path = Path(
    "/home/yihang/Documents/Alg_study/SparseDriveV2study/outputs/sparse_backbone_three_cams/feature_maps.pt"
)

feature_maps = torch.load(path, map_location="cpu")

for level, feat in enumerate(feature_maps):
    print(f"level {level}: {feat.shape}")
    print("min:", feat.min().item())
    print("max:", feat.max().item())
    print("mean:", feat.mean().item())
    print("std:", feat.std().item())
    print()
