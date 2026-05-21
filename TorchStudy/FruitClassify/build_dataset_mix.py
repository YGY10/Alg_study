from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["apple", "pear", "pineapple"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mixed fruit classification dataset from Fruits-360 and real crop datasets."
    )

    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("dataset"),
        help="Original Fruits-360 classification dataset root.",
    )

    parser.add_argument(
        "--old-real-crop-root",
        type=Path,
        default=None,
        help="Old Fruit Object Detection crop root. Expected: crop/apple, crop/pineapple with train_/valid_ filename prefixes.",
    )

    parser.add_argument(
        "--roboflow-crop-root",
        type=Path,
        default=None,
        help="Roboflow crop root. Expected: crop/train/apple, crop/val/apple, etc.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_mix"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output root before building.",
    )

    return parser.parse_args()


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES


def safe_copy(src: Path, dst_dir: Path, prefix: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_name = f"{prefix}_{src.name}"
    dst = dst_dir / dst_name

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = src.stem
    suffix = src.suffix.lower()

    idx = 1
    while True:
        dst = dst_dir / f"{prefix}_{stem}_{idx:04d}{suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
            return dst
        idx += 1


def init_dirs(output_root: Path):
    for split in ["train", "val"]:
        for cls in CLASSES:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)


def copy_base_fruits360(base_root: Path, output_root: Path, stats: dict):
    for split in ["train", "val"]:
        for cls in CLASSES:
            src_dir = base_root / split / cls
            if not src_dir.exists():
                raise FileNotFoundError(f"Missing base dir: {src_dir}")

            dst_dir = output_root / split / cls

            for p in sorted(src_dir.rglob("*")):
                if not is_image(p):
                    continue

                safe_copy(p, dst_dir, prefix=f"fruits360_{split}")
                stats[split][cls] += 1


def copy_old_real_crop(old_root: Path, output_root: Path, stats: dict):
    """
    旧 Fruit Object Detection 转出来的结构是：
        dataset_real/crop/apple/*.jpg
        dataset_real/crop/pineapple/*.jpg

    文件名前缀里有：
        train_xxx.jpg
        valid_xxx.jpg

    所以这里根据文件名前缀分 train / val。
    """
    for cls in ["apple", "pineapple"]:
        src_dir = old_root / cls
        if not src_dir.exists():
            print(f"[WARN] skip missing old real dir: {src_dir}")
            continue

        for p in sorted(src_dir.rglob("*")):
            if not is_image(p):
                continue

            if p.name.startswith("train_"):
                split = "train"
            elif p.name.startswith("valid_"):
                split = "val"
            else:
                print(f"[WARN] skip old real file without train_/valid_ prefix: {p}")
                continue

            dst_dir = output_root / split / cls
            safe_copy(p, dst_dir, prefix=f"oldreal_{split}")
            stats[split][cls] += 1


def copy_roboflow_crop(roboflow_root: Path, output_root: Path, stats: dict):
    """
    Roboflow 转出来的结构是：
        dataset_roboflow_real/crop/train/apple
        dataset_roboflow_real/crop/train/pear
        dataset_roboflow_real/crop/train/pineapple
        dataset_roboflow_real/crop/val/...
    """
    for split in ["train", "val"]:
        for cls in CLASSES:
            src_dir = roboflow_root / split / cls
            if not src_dir.exists():
                print(f"[WARN] skip missing roboflow dir: {src_dir}")
                continue

            dst_dir = output_root / split / cls

            for p in sorted(src_dir.rglob("*")):
                if not is_image(p):
                    continue

                safe_copy(p, dst_dir, prefix=f"roboflow_{split}")
                stats[split][cls] += 1


def count_images(output_root: Path) -> dict:
    counts = {split: {cls: 0 for cls in CLASSES} for split in ["train", "val"]}

    for split in ["train", "val"]:
        for cls in CLASSES:
            d = output_root / split / cls
            counts[split][cls] = sum(1 for p in d.rglob("*") if is_image(p))

    return counts


def print_counts(title: str, counts: dict):
    print(f"\n========== {title} ==========")
    for split in ["train", "val"]:
        print(f"\n[{split}]")
        for cls in CLASSES:
            print(f"  {cls:10s}: {counts[split][cls]}")


def main():
    args = parse_args()

    output_root = args.output_root.resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    init_dirs(output_root)

    stats = {split: {cls: 0 for cls in CLASSES} for split in ["train", "val"]}

    print("base_root          :", args.base_root.resolve())
    print(
        "old_real_crop_root :",
        args.old_real_crop_root.resolve() if args.old_real_crop_root else None,
    )
    print(
        "roboflow_crop_root :",
        args.roboflow_crop_root.resolve() if args.roboflow_crop_root else None,
    )
    print("output_root        :", output_root)

    copy_base_fruits360(
        base_root=args.base_root.resolve(),
        output_root=output_root,
        stats=stats,
    )

    if args.old_real_crop_root is not None:
        copy_old_real_crop(
            old_root=args.old_real_crop_root.resolve(),
            output_root=output_root,
            stats=stats,
        )

    if args.roboflow_crop_root is not None:
        copy_roboflow_crop(
            roboflow_root=args.roboflow_crop_root.resolve(),
            output_root=output_root,
            stats=stats,
        )

    print_counts("Copied counts", stats)

    final_counts = count_images(output_root)
    print_counts("Final dataset_mix counts", final_counts)

    print("\nDone.")


if __name__ == "__main__":
    main()
