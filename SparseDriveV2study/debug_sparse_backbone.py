import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import timm
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 你的目录结构：
# Alg_study/
#   SparseDriveV2/
#   SparseDriveV2study/debug_sparse_backbone.py
STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"

sys.path.insert(0, str(PROJECT_ROOT))


# SparseBackbone 文件里 import 了 navsim.agents.sparsedrive.ops.deformable_format，
# 但 SparseBackbone.forward 本身不用这个 CUDA op。
# 为了单独测试 backbone，这里临时 mock 掉 ops，避免没编译 CUDA op 时 import 失败。
mock_ops = types.ModuleType("navsim.agents.sparsedrive.ops")
mock_ops.deformable_format = None
sys.modules["navsim.agents.sparsedrive.ops"] = mock_ops


from navsim.agents.sparsedrive.sparsedrive_backbone import SparseBackbone


def patch_timm_no_pretrained():
    """
    SparseBackbone 默认会加载 ckpt/resnet34.bin。
    我们只是想看网络输入输出，所以先关闭 pretrained，避免缺权重文件报错。
    """
    original_create_model = timm.create_model

    def create_model_without_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        kwargs.pop("pretrained_cfg_overlay", None)
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_without_pretrained


def visualize_feature_map(feature_maps, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    for level, feat in enumerate(feature_maps):
        # feat shape: [B, N_cam, C, H, W]
        print(f"[visualize] level={level}, feat shape={tuple(feat.shape)}")

        # 取 batch=0, cam=0，然后对 256 个 channel 求均值
        heatmap = feat[0, 0].mean(dim=0).detach().cpu()

        plt.figure(figsize=(8, 4))
        plt.imshow(heatmap)
        plt.title(f"SparseBackbone feature mean heatmap - level {level}, cam 0")
        plt.colorbar()
        plt.tight_layout()

        out_path = save_dir / f"sparse_backbone_level_{level}_cam0.png"
        plt.savefig(out_path)
        plt.close()

        print(f"[visualize] saved: {out_path}")


def main():
    patch_timm_no_pretrained()

    config = SimpleNamespace(
        d_model=256,
        use_grid_mask=False,
        with_img_neck=True,
        image_architecture="resnet34",
        bkb_path=str(PROJECT_ROOT / "ckpt/resnet34.bin"),
        num_levels=4,
    )

    model = SparseBackbone(config)
    model.eval()

    # 模拟输入：
    # B=1
    # N_cam=3
    # C=3
    # H=256
    # W=512
    img = torch.randn(1, 3, 3, 256, 512)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("input img:", tuple(img.shape))

    with torch.no_grad():
        feature_maps = model(img)

    print("number of feature levels:", len(feature_maps))

    for i, feat in enumerate(feature_maps):
        print(f"feature_maps[{i}]: {tuple(feat.shape)}")

    visualize_feature_map(
        feature_maps,
        save_dir=STUDY_ROOT / "outputs" / "sparse_backbone",
    )


if __name__ == "__main__":
    main()
