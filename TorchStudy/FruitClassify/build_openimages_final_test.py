from __future__ import annotations

from pathlib import Path

from PIL import Image

import fiftyone.zoo as foz

TARGET_NAMES = {
    "Apple": "apple",
    "Pear": "pear",
    "Pineapple": "pineapple",
}


def clamp(v, low, high):
    return max(low, min(high, v))


def main():
    output_root = Path("openimages_final_test/crop")
    for cls in TARGET_NAMES.values():
        (output_root / cls).mkdir(parents=True, exist_ok=True)

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=list(TARGET_NAMES.keys()),
        only_matching=True,
        max_samples=300,
        shuffle=True,
        seed=42,
        dataset_name="openimages_fruit_final_test_tmp",
    )

    counts = {v: 0 for v in TARGET_NAMES.values()}

    for sample in dataset:
        image_path = Path(sample.filepath)
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        detections = sample["ground_truth"].detections

        for det_idx, det in enumerate(detections):
            label = det.label

            if label not in TARGET_NAMES:
                continue

            class_name = TARGET_NAMES[label]

            # FiftyOne bbox format: [x, y, width, height], normalized
            x, y, w, h = det.bounding_box

            x1 = int(round(x * img_w))
            y1 = int(round(y * img_h))
            x2 = int(round((x + w) * img_w))
            y2 = int(round((y + h) * img_h))

            # 加一点 padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(round(box_w * 0.08))
            pad_y = int(round(box_h * 0.08))

            x1 = clamp(x1 - pad_x, 0, img_w - 1)
            y1 = clamp(y1 - pad_y, 0, img_h - 1)
            x2 = clamp(x2 + pad_x, 0, img_w)
            y2 = clamp(y2 + pad_y, 0, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            if crop.size[0] < 24 or crop.size[1] < 24:
                continue

            out_name = f"{image_path.stem}_{det_idx:03d}.jpg"
            out_path = output_root / class_name / out_name

            crop.save(out_path, quality=95)
            counts[class_name] += 1

    print("output_root:", output_root.resolve())
    print("counts:")
    for cls, n in counts.items():
        print(f"  {cls:10s}: {n}")


if __name__ == "__main__":
    main()
