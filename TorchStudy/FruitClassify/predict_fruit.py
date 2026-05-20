from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict fruit class with trained ResNet34 / ResNet34-CBAM."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image.",
    )

    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/resnet34_fruit/best_resnet34_fruit.pth"),
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "resnet34", "resnet34_cbam"],
        help="Model architecture. Use auto to read model_name from checkpoint.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size. Default: read from checkpoint, fallback to 224.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def resolve_model_name(args_model: str, ckpt: dict) -> str:
    if args_model != "auto":
        return args_model

    model_name = ckpt.get("model_name", None)

    if model_name is None:
        # 兼容旧 checkpoint：旧版只保存了普通 ResNet34，没有保存 model_name
        return "resnet34"

    if model_name not in ["resnet34", "resnet34_cbam"]:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")

    return model_name


def resolve_image_size(args_image_size: int | None, ckpt: dict) -> int:
    if args_image_size is not None:
        return args_image_size

    image_size = ckpt.get("image_size", None)

    if image_size is None:
        return 224

    return int(image_size)


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    device = get_device(args.device)

    # 先加载到 CPU，更稳；模型构建完成后再整体移动到 device
    ckpt = torch.load(args.ckpt, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    num_classes = len(class_to_idx)

    model_name = resolve_model_name(args.model, ckpt)
    image_size = resolve_image_size(args.image_size, ckpt)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = build_transform(image_size)

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]

    pred_idx = int(prob.argmax().item())
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(prob[pred_idx].item())

    print("image:", args.image)
    print("checkpoint:", args.ckpt)
    print("model:", model_name)
    print("image_size:", image_size)
    print("prediction:", pred_class)
    print("confidence:", f"{pred_prob:.4f}")
    print()

    print("all probabilities:")
    for idx in range(num_classes):
        cls = idx_to_class[idx]
        p = float(prob[idx].item())
        print(f"  {cls:10s}: {p:.4f}")


if __name__ == "__main__":
    main()
