from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(class_dir: Path) -> list[Path]:
    files = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
    return sorted(files)


def remove_empty_dirs(root: Path) -> None:
    # 从深层目录往上删空目录
    dirs = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for d in dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


def balance_split(
    dataset_root: Path,
    split: str,
    classes: list[str],
    target_count: int | None,
    seed: int,
    apply: bool,
    cleanup_empty_dirs: bool,
) -> None:
    split_dir = dataset_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    class_to_files: dict[str, list[Path]] = {}

    print(f"\n========== split: {split} ==========")

    for cls in classes:
        class_dir = split_dir / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        files = collect_images(class_dir)
        class_to_files[cls] = files
        print(f"{cls:10s}: {len(files)} images")

    min_count = min(len(files) for files in class_to_files.values())

    if target_count is None:
        keep_count = min_count
    else:
        keep_count = target_count
        if keep_count > min_count:
            raise ValueError(
                f"target_count={keep_count} is larger than min class count={min_count} "
                f"in split={split}"
            )

    print(f"\nTarget keep count for split '{split}': {keep_count}")

    rng = random.Random(seed)

    total_delete = 0

    for cls, files in class_to_files.items():
        num_images = len(files)
        num_delete = num_images - keep_count

        if num_delete <= 0:
            print(f"{cls:10s}: keep all {num_images}")
            continue

        delete_files = files.copy()
        rng.shuffle(delete_files)
        delete_files = delete_files[:num_delete]

        print(f"{cls:10s}: delete {num_delete}, keep {keep_count}")

        total_delete += num_delete

        if apply:
            for p in delete_files:
                p.unlink()

    if apply and cleanup_empty_dirs:
        remove_empty_dirs(split_dir)

    if apply:
        print(f"\n[APPLIED] Deleted {total_delete} images from split '{split}'.")
    else:
        print(f"\n[DRY RUN] Would delete {total_delete} images from split '{split}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance apple/pear/pineapple dataset by deleting extra images in-place."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to balance, e.g. train val.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["apple", "pear", "pineapple"],
        help="Class names.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Images to keep per class. Default: use min class count per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting files to delete.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print what would happen.",
    )
    parser.add_argument(
        "--no-cleanup-empty-dirs",
        action="store_true",
        help="Do not remove empty subdirectories after deleting images.",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    print(f"dataset_root: {dataset_root}")
    print(f"splits: {args.splits}")
    print(f"classes: {args.classes}")
    print(f"target_count: {args.target_count}")
    print(f"apply: {args.apply}")

    for split in args.splits:
        balance_split(
            dataset_root=dataset_root,
            split=split,
            classes=args.classes,
            target_count=args.target_count,
            seed=args.seed,
            apply=args.apply,
            cleanup_empty_dirs=not args.no_cleanup_empty_dirs,
        )


if __name__ == "__main__":
    main()
