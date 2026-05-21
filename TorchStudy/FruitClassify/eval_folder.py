from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import build_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained fruit classifier on a folder dataset."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder dataset root, e.g. dataset_real/raw or dataset_real/crop.",
    )

    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to checkpoint.",
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
        "--batch-size",
        type=int,
        default=64,
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


def resolve_model_name(args_model: str, ckpt: dict) -> str:
    if args_model != "auto":
        return args_model

    model_name = ckpt.get("model_name", None)

    if model_name is None:
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


def collect_images(data_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name

        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((p, class_name))

    return samples


@torch.no_grad()
def evaluate_samples(
    samples: list[tuple[Path, str]],
    model,
    transform,
    class_to_idx: dict[str, int],
    device: torch.device,
):
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    total = 0
    correct = 0
    wrong_cases = []

    for image_path, true_class in samples:
        if true_class not in class_to_idx:
            print(
                f"[SKIP] class '{true_class}' not in checkpoint classes: {image_path}"
            )
            continue

        true_idx = class_to_idx[true_class]

        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]

        pred_idx = int(prob.argmax().item())
        pred_class = idx_to_class[pred_idx]
        pred_prob = float(prob[pred_idx].item())

        confusion[true_idx, pred_idx] += 1

        total += 1
        if pred_idx == true_idx:
            correct += 1
        else:
            wrong_cases.append(
                {
                    "image": str(image_path),
                    "true": true_class,
                    "pred": pred_class,
                    "confidence": pred_prob,
                    "probs": {
                        idx_to_class[i]: float(prob[i].item())
                        for i in range(num_classes)
                    },
                }
            )

    acc = correct / total if total > 0 else 0.0

    return acc, total, correct, confusion, wrong_cases


def print_confusion(confusion: torch.Tensor, class_to_idx: dict[str, int]):
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nconfusion matrix:")
    print("rows=true, cols=pred")
    print("classes:", classes)
    print(confusion.numpy())


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Missing data root: {args.data_root}")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    device = get_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    model_name = resolve_model_name(args.model, ckpt)
    image_size = resolve_image_size(args.image_size, ckpt)

    print("data_root :", args.data_root.resolve())
    print("ckpt      :", args.ckpt.resolve())
    print("model     :", model_name)
    print("image_size:", image_size)
    print("device    :", device)
    print("classes   :", class_to_idx)

    samples = collect_images(args.data_root)

    print("num samples:", len(samples))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under {args.data_root}")

    model = build_model(
        model_name=model_name,
        num_classes=len(class_to_idx),
        pretrained=False,
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    transform = build_transform(image_size)

    acc, total, correct, confusion, wrong_cases = evaluate_samples(
        samples=samples,
        model=model,
        transform=transform,
        class_to_idx=class_to_idx,
        device=device,
    )

    print("\n========== Result ==========")
    print(f"accuracy: {acc:.4f}")
    print(f"correct : {correct}/{total}")

    print_confusion(confusion, class_to_idx)

    print("\nwrong cases:", len(wrong_cases))
    for item in wrong_cases[:30]:
        print(
            f"{item['image']} | true={item['true']} | "
            f"pred={item['pred']} | conf={item['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
