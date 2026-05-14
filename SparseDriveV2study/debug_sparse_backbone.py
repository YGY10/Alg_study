import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

# 目录结构：
# Alg_study/
#   SparseDriveV2/
#   SparseDriveV2study/debug_sparse_backbone.py
STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"

sys.path.insert(0, str(PROJECT_ROOT))


# mock 掉不需要的 CUDA op
mock_ops = types.ModuleType("navsim.agents.sparsedrive.ops")
mock_ops.deformable_format = None
sys.modules["navsim.agents.sparsedrive.ops"] = mock_ops


from navsim.agents.sparsedrive.sparsedrive_backbone import SparseBackbone

CKPT_PATH = PROJECT_ROOT / "ckpt" / "resnet34.bin"

LEFT_IMAGE_PATH = STUDY_ROOT / "assets" / "left.png"
FRONT_IMAGE_PATH = STUDY_ROOT / "assets" / "front.png"
RIGHT_IMAGE_PATH = STUDY_ROOT / "assets" / "right.png"

OUT_DIR = STUDY_ROOT / "outputs" / "sparse_backbone_three_cams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_road_image(image_path: Path, save_name: str):
    """
    读取真实道路图片：
    1. 读取 RGB
    2. 按 2:1 宽高比中心裁剪
    3. resize 到 512x256
    4. 转成 [3, 256, 512]
    5. 数值范围 [0, 1]
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    print(f"{save_name} original size:", img.size)

    target_ratio = 512 / 256  # 2.0
    current_ratio = w / h

    if current_ratio > target_ratio:
        # 图太宽，裁左右
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        right = left + new_w
        img = img.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        # 图太高，裁上下
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        bottom = top + new_h
        img = img.crop((0, top, w, bottom))

    print(f"{save_name} cropped size:", img.size)

    img = img.resize((512, 256))
    img.save(OUT_DIR / f"{save_name}_cropped_resized.jpg")

    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]

    return img_tensor


def normalize_imagenet(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor - mean) / std


def save_input_image(img_tensor, save_name: str):
    img_np = img_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(img_np)
    plt.title(save_name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{save_name}.png")
    plt.close()


def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def visualize_feature_map(feature_maps, save_dir: Path):
    """
    feature_maps[i]: [B, N_cam, 256, H_i, W_i]
    为每个 level、每个 camera 保存 mean heatmap
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    cam_names = ["left", "front", "right"]

    for level, feat in enumerate(feature_maps):
        print(f"[visualize] level={level}, feat shape={tuple(feat.shape)}")

        for cam_idx, cam_name in enumerate(cam_names):
            heatmap = feat[0, cam_idx].mean(dim=0).detach().cpu()

            plt.figure(figsize=(8, 4))
            plt.imshow(heatmap)
            plt.title(f"SparseBackbone level {level} - {cam_name}")
            plt.colorbar()
            plt.tight_layout()

            out_path = save_dir / f"sparse_backbone_level_{level}_{cam_name}.png"
            plt.savefig(out_path)
            plt.close()

            print(f"[visualize] saved: {out_path}")


def visualize_feature_overlay(feature_maps, raw_imgs, save_dir: Path, alpha=0.45):
    """
    raw_imgs:
        list of 3 tensors, each [3, 256, 512]
    feature_maps[i]:
        [B, N_cam, 256, H_i, W_i]
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    cam_names = ["left", "front", "right"]

    for level, feat in enumerate(feature_maps):
        for cam_idx, cam_name in enumerate(cam_names):
            img_raw = raw_imgs[cam_idx]
            img_np = img_raw.detach().cpu().permute(1, 2, 0).numpy()

            heatmap = feat[0, cam_idx].mean(dim=0).detach().cpu()
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = F.interpolate(
                heatmap,
                size=img_raw.shape[1:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            heatmap = normalize_heatmap(heatmap)
            heatmap_np = heatmap.numpy()

            heatmap_color = matplotlib.colormaps["jet"](heatmap_np)[..., :3]

            overlay = (1.0 - alpha) * img_np + alpha * heatmap_color
            overlay = np.clip(overlay, 0.0, 1.0)

            plt.figure(figsize=(10, 5))
            plt.imshow(overlay)
            plt.title(f"Overlay level {level} - {cam_name}")
            plt.axis("off")
            plt.tight_layout()

            out_path = save_dir / f"overlay_level_{level}_{cam_name}.png"
            plt.savefig(out_path)
            plt.close()

            print(f"[overlay] saved: {out_path}")


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

    for p in [LEFT_IMAGE_PATH, FRONT_IMAGE_PATH, RIGHT_IMAGE_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")

    config = SimpleNamespace(
        d_model=256,
        use_grid_mask=False,
        with_img_neck=True,
        image_architecture="resnet34",
        bkb_path=str(CKPT_PATH),
        num_levels=4,
    )

    model = SparseBackbone(config)
    model.eval()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CKPT_PATH:", CKPT_PATH)

    # 读取三张真实图
    img_left_raw = load_real_road_image(LEFT_IMAGE_PATH, "left")
    img_front_raw = load_real_road_image(FRONT_IMAGE_PATH, "front")
    img_right_raw = load_real_road_image(RIGHT_IMAGE_PATH, "right")

    save_input_image(img_left_raw, "input_left")
    save_input_image(img_front_raw, "input_front")
    save_input_image(img_right_raw, "input_right")

    # normalize
    img_left = normalize_imagenet(img_left_raw)
    img_front = normalize_imagenet(img_front_raw)
    img_right = normalize_imagenet(img_right_raw)

    # stack 成 [N_cam, C, H, W]
    imgs = torch.stack([img_left, img_front, img_right], dim=0)

    # 再加 batch 维度 -> [1, 3, 3, 256, 512]
    img = imgs.unsqueeze(0)

    print("input img:", tuple(img.shape))

    with torch.no_grad():
        feature_maps = model(img)
        torch.save(feature_maps, OUT_DIR / "feature_maps.pt")
        print("saved raw feature_maps:", OUT_DIR / "feature_maps.pt")

    print("number of feature levels:", len(feature_maps))

    for i, feat in enumerate(feature_maps):
        print(f"feature_maps[{i}]: {tuple(feat.shape)}")

    visualize_feature_map(feature_maps, OUT_DIR)

    visualize_feature_overlay(
        feature_maps,
        raw_imgs=[img_left_raw, img_front_raw, img_right_raw],
        save_dir=OUT_DIR,
        alpha=0.45,
    )

    print(f"saved images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
