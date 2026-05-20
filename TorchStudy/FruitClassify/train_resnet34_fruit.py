from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet34 for apple/pear/pineapple classification."
    )

    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/resnet34_fruit")
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        choices=["resnet34", "resnet34_cbam"],
        help="Model architecture.",
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# 构建训练集和测试集
def build_dataloaders(
    dataset_root: Path, image_size: int, batch_size: int, num_workers: int
):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
            transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.ColorJitter(  # 随机改变亮度，饱和度和色调
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),  # 用inmagenet标准归一化
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0  # 累计的loss
    total_correct = 0  # 累计预测正确数量
    total_num = 0  # 累计样本数量

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_num += batch_size

    avg_loss = total_loss / total_num
    acc = total_correct / total_num

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    num_classes = len(loader.dataset.classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_num += batch_size

        for t, p in zip(labels.cpu(), pred.cpu()):
            confusion[t.long(), p.long()] += 1

    avg_loss = total_loss / total_num
    acc = total_correct / total_num

    return avg_loss, acc, confusion


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    epoch,
    best_acc,
    class_to_idx,
    model_name: str,
    image_size: int,
):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
        "model_name": model_name,
        "image_size": image_size,
    }
    torch.save(ckpt, output_dir / "best_resnet34_fruit.pth")


def main():
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    print("dataset_root:", args.dataset_root.resolve())
    print("output_dir:", output_dir.resolve())
    print("device:", device)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("classes:", train_dataset.classes)
    print("class_to_idx:", train_dataset.class_to_idx)
    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"Train classes {train_dataset.classes} != val classes {val_dataset.classes}"
        )
    print("model:", args.model)
    model = build_model(
        model_name=args.model,
        num_classes=len(train_dataset.classes),
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, confusion = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        print("confusion matrix:")
        print(confusion.numpy())

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                class_to_idx=train_dataset.class_to_idx,
                model_name=args.model,
                image_size=args.image_size,
            )
            print(f"[SAVE] best model updated, best_acc={best_acc:.4f}")

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    meta = {
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "best_acc": best_acc,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "model_name": args.model,
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("training done")
    print("best_acc:", best_acc)
    print("best model:", output_dir / "best_resnet34_fruit.pth")


if __name__ == "__main__":
    main()
