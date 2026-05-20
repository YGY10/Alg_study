from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from models import build_resnet34_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict fruit class with trained ResNet34."
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
        "--image-size",
        type=int,
        default=224,
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


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    device = get_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    num_classes = len(class_to_idx)

    model = build_resnet34_classifier(num_classes, pretrained=False)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = build_transform(args.image_size)

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]

    pred_idx = int(prob.argmax().item())
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(prob[pred_idx].item())

    print("image:", args.image)
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
