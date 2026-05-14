import torch
import timm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm

# 当前脚本假设放在：
# ~/Documents/Alg_study/SparseDriveV2study/debug_resnet34_example.py
STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"

CKPT_PATH = PROJECT_ROOT / "ckpt" / "resnet34.bin"

# 你自己的真实道路图片
IMAGE_PATH = STUDY_ROOT / "assets" / "my_road.png"

OUT_DIR = STUDY_ROOT / "outputs" / "resnet34_my_road_image"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_road_image(image_path: Path):
    """
    读取真实道路图片。

    处理流程：
    1. 读取 RGB 图片
    2. 按 2:1 宽高比中心裁剪
    3. resize 到 512x256
    4. 转成 PyTorch 格式 [3, 256, 512]
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    print("original image size:", img.size)

    target_ratio = 512 / 256  # 2.0
    current_ratio = w / h

    if current_ratio > target_ratio:
        # 图太宽，裁掉左右
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        right = left + new_w
        img = img.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        # 图太高，裁掉上下
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        bottom = top + new_h
        img = img.crop((0, top, w, bottom))

    print("cropped image size:", img.size)

    # SparseDriveV2 常用输入尺寸：W=512, H=256
    img = img.resize((512, 256))

    # 保存裁剪 resize 后的输入图，方便检查
    img.save(OUT_DIR / "input_real_road_cropped_resized.jpg")

    img_np = np.asarray(img).astype(np.float32) / 255.0

    # [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    return img_tensor


def normalize_heatmap(heatmap):
    """
    heatmap: [H, W]
    归一化到 [0, 1]
    """
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def save_feature_overlay_on_image(img_tensor, feat, name, alpha=0.45):
    """
    把 feature map 的平均响应叠加到原图上。

    img_tensor:
        [3, 256, 512]，未 normalize，数值范围 [0, 1]

    feat:
        [B, C, H, W]

    name:
        输出文件名
    """
    # 原图: [3, H, W] -> [H, W, 3]
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()

    # 对所有 channel 求平均: [C, Hf, Wf] -> [Hf, Wf]
    heatmap = feat[0].mean(dim=0).detach().cpu()

    # resize 到原图大小: [1, 1, Hf, Wf] -> [1, 1, 256, 512]
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=img_tensor.shape[1:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heatmap = normalize_heatmap(heatmap)

    # 转成彩色热力图
    heatmap_np = heatmap.numpy()
    heatmap_color = cm.get_cmap("jet")(heatmap_np)[..., :3]

    # 叠加
    overlay = (1 - alpha) * img_np + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title(name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png")
    plt.close()


def normalize_imagenet(img_tensor):
    """
    ResNet34 ImageNet 预训练权重要求输入做 ImageNet normalization。

    输入:
        img_tensor: [C, H, W], 数值范围 [0, 1]

    输出:
        normalized_img: [C, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (img_tensor - mean) / std


def save_input_image(img_tensor):
    """
    保存原始输入图像。
    注意：这里保存的是未 normalize 的版本，方便人眼查看。
    """
    img_np = img_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(img_np)
    plt.title("Real road image resized to 256x512")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "input_real_road.png")
    plt.close()


def save_feature_mean(feat, name):
    """
    feat shape: [B, C, H, W]

    对 C 个通道求平均，得到一张热力图。
    这只能粗略观察整体响应，不代表某个具体语义。
    """
    heatmap = feat[0].mean(dim=0).detach().cpu()

    plt.figure(figsize=(8, 4))
    plt.imshow(heatmap)
    plt.title(name)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png")
    plt.close()


def save_feature_channels(feat, level, channels):
    """
    单独保存某些 channel 的响应图。
    这样比 mean 更容易看出“不同通道关注不同模式”。
    """
    for ch in channels:
        if ch >= feat.shape[1]:
            continue

        heatmap = feat[0, ch].detach().cpu()

        plt.figure(figsize=(8, 4))
        plt.imshow(heatmap)
        plt.title(f"resnet_feature_{level}_channel_{ch}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"resnet_feature_{level}_channel_{ch}.png")
        plt.close()


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing ResNet34 checkpoint: {CKPT_PATH}\n"
            f"Please download it to SparseDriveV2/ckpt/resnet34.bin"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Missing input image: {IMAGE_PATH}\n"
            f"Please put your road image under SparseDriveV2study/assets/my_road.png"
        )

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CKPT_PATH:", CKPT_PATH)
    print("IMAGE_PATH:", IMAGE_PATH)

    model = timm.create_model(
        "resnet34",
        pretrained=True,
        features_only=True,
        pretrained_cfg_overlay={"file": str(CKPT_PATH)},
        out_indices=(1, 2, 3, 4),
    )
    model.eval()

    # 读取真实道路图像
    img = load_real_road_image(IMAGE_PATH)
    save_input_image(img)

    # 给预训练 ResNet 用的输入需要 normalize
    img_norm = normalize_imagenet(img)

    # 加 batch 维度：[3, 256, 512] -> [1, 3, 256, 512]
    x = img_norm.unsqueeze(0)

    print("input:", tuple(x.shape))

    with torch.no_grad():
        features = model(x)

    for i, feat in enumerate(features):
        print(f"resnet feature[{i}]:", tuple(feat.shape))

        save_feature_mean(feat, f"resnet_feature_{i}_mean")

        save_feature_overlay_on_image(
            img_tensor=img,
            feat=feat,
            name=f"overlay_resnet_feature_{i}_mean",
            alpha=0.45,
        )

        save_feature_channels(
            feat,
            level=i,
            channels=[0, 1, 2, 3, 8, 16, 32],
        )

    print(f"saved images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
