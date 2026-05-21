from __future__ import annotations

import fiftyone as fo
import fiftyone.zoo as foz


def main():
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Apple", "Pear", "Pineapple"],
        only_matching=True,
        max_samples=300,
        shuffle=True,
        seed=42,
        dataset_name="openimages_fruit_val_test",
    )

    print(dataset)
    print("num samples:", len(dataset))

    # 打开 FiftyOne 可视化界面，确认类别和 bbox 是否正确
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
