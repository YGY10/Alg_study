from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "outputs" / "anchor_library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SparseDrive path, velocity, and trajectory anchors."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="SparseDriveV2 project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures and stats.",
    )
    return parser.parse_args()


def load_anchors(project_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_dir = project_root / "ckpt" / "kmeans"
    path_vocab = np.load(anchor_dir / "path_1024.npy")
    velocity_vocab = np.load(anchor_dir / "velocity_256.npy")
    traj_data = np.load(anchor_dir / "trajectory_1024_256.npz")
    traj_vocab = traj_data["trajectory"]
    traj_mask = traj_data["trajectory_mask"]
    return path_vocab, velocity_vocab, traj_vocab, traj_mask


def setup_bev_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")


def plot_all_paths(path_vocab: np.ndarray, output_dir: Path) -> None:
    final_y = path_vocab[:, -1, 1]

    fig, ax = plt.subplots(figsize=(8, 9))
    for path in path_vocab:
        ax.plot(path[:, 1], path[:, 0], color="#1f77b4", alpha=0.07, linewidth=0.7)

    setup_bev_axis(ax, "All 1024 path anchors")
    ax.set_xlim(np.percentile(path_vocab[:, :, 1], [0.2, 99.8]))
    ax.set_ylim(np.percentile(path_vocab[:, :, 0], [0.2, 99.8]))
    text = (
        f"final y min/median/max: "
        f"{final_y.min():.1f} / {np.median(final_y):.1f} / {final_y.max():.1f} m"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "path_all_density.png", dpi=180)
    plt.close(fig)


def plot_paths_by_endpoint(path_vocab: np.ndarray, output_dir: Path) -> None:
    final_y = path_vocab[:, -1, 1]
    buckets = [
        ("negative final y", final_y < -2.0, "#d62728"),
        ("near center", np.abs(final_y) <= 2.0, "#2ca02c"),
        ("positive final y", final_y > 2.0, "#1f77b4"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (title, mask, color) in zip(axes, buckets):
        indices = np.where(mask)[0]
        for idx in indices:
            path = path_vocab[idx]
            ax.plot(path[:, 1], path[:, 0], color=color, alpha=0.11, linewidth=0.8)
        setup_bev_axis(ax, f"{title} ({len(indices)})")

    fig.tight_layout()
    fig.savefig(output_dir / "path_by_lateral_endpoint.png", dpi=180)
    plt.close(fig)


def pick_representative_indices(values: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values)
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return np.unique(order[positions])


def plot_representative_paths(path_vocab: np.ndarray, output_dir: Path) -> np.ndarray:
    final_y = path_vocab[:, -1, 1]
    indices = pick_representative_indices(final_y, 48)
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(indices)))

    fig, ax = plt.subplots(figsize=(8, 9))
    for color, idx in zip(colors, indices):
        path = path_vocab[idx]
        ax.plot(path[:, 1], path[:, 0], color=color, linewidth=1.4, alpha=0.9)
        ax.scatter(path[-1, 1], path[-1, 0], color=color, s=8)

    setup_bev_axis(ax, "48 representative path anchors by final lateral offset")
    fig.tight_layout()
    fig.savefig(output_dir / "path_representatives.png", dpi=180)
    plt.close(fig)
    return indices


def plot_velocity_curves(velocity_vocab: np.ndarray, output_dir: Path) -> np.ndarray:
    t = np.arange(velocity_vocab.shape[1])
    mean_speed = velocity_vocab.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for velocity in velocity_vocab:
        ax.plot(t, velocity, color="#9467bd", alpha=0.12, linewidth=0.8)
    ax.set_title("All 256 velocity anchors")
    ax.set_xlabel("future step")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_all_curves.png", dpi=180)
    plt.close(fig)

    order = np.argsort(mean_speed)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(
        velocity_vocab[order],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_title("Velocity anchors sorted by mean speed")
    ax.set_xlabel("future step")
    ax.set_ylabel("velocity anchor rank")
    fig.colorbar(image, ax=ax, label="speed [m/s]")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_sorted_heatmap.png", dpi=180)
    plt.close(fig)

    representative_indices = pick_representative_indices(mean_speed, 20)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0.0, 1.0, len(representative_indices)))
    for color, idx in zip(colors, representative_indices):
        ax.plot(t, velocity_vocab[idx], color=color, linewidth=1.7, label=f"v{idx}")
    ax.set_title("20 representative velocity anchors")
    ax.set_xlabel("future step")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_representatives.png", dpi=180)
    plt.close(fig)

    return representative_indices


def plot_trajectory_examples(
    path_vocab: np.ndarray,
    velocity_vocab: np.ndarray,
    traj_vocab: np.ndarray,
    output_dir: Path,
) -> tuple[list[int], list[int]]:
    final_y = path_vocab[:, -1, 1]
    mean_speed = velocity_vocab.mean(axis=1)
    path_indices = [
        int(np.argmin(final_y)),
        int(np.argmin(np.abs(final_y))),
        int(np.argmax(final_y)),
    ]
    velocity_indices = [
        int(np.argmin(mean_speed)),
        int(np.argsort(mean_speed)[len(mean_speed) // 2]),
        int(np.argmax(mean_speed)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    for row, path_idx in enumerate(path_indices):
        for col, velocity_idx in enumerate(velocity_indices):
            ax = axes[row, col]
            traj = traj_vocab[path_idx, velocity_idx]
            ax.plot(traj[:, 1], traj[:, 0], "-o", color="#ff7f0e", linewidth=2, markersize=4)
            ax.plot(
                path_vocab[path_idx, :, 1],
                path_vocab[path_idx, :, 0],
                color="#1f77b4",
                alpha=0.3,
                linewidth=1.3,
                label="path anchor",
            )
            setup_bev_axis(
                ax,
                f"path {path_idx}, vel {velocity_idx}\nmean v={mean_speed[velocity_idx]:.1f} m/s",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_examples.png", dpi=180)
    plt.close(fig)

    return path_indices, velocity_indices


def write_stats(
    output_dir: Path,
    path_vocab: np.ndarray,
    velocity_vocab: np.ndarray,
    traj_vocab: np.ndarray,
    traj_mask: np.ndarray,
    representative_path_indices: np.ndarray,
    representative_velocity_indices: np.ndarray,
    example_path_indices: list[int],
    example_velocity_indices: list[int],
) -> None:
    final_y = path_vocab[:, -1, 1]
    final_x = path_vocab[:, -1, 0]
    mean_speed = velocity_vocab.mean(axis=1)
    stats = {
        "path_shape": list(path_vocab.shape),
        "velocity_shape": list(velocity_vocab.shape),
        "trajectory_shape": list(traj_vocab.shape),
        "trajectory_mask_shape": list(traj_mask.shape),
        "path_final_x_min_median_max": [
            float(final_x.min()),
            float(np.median(final_x)),
            float(final_x.max()),
        ],
        "path_final_y_min_median_max": [
            float(final_y.min()),
            float(np.median(final_y)),
            float(final_y.max()),
        ],
        "velocity_mean_min_median_max": [
            float(mean_speed.min()),
            float(np.median(mean_speed)),
            float(mean_speed.max()),
        ],
        "velocity_value_min_max": [
            float(velocity_vocab.min()),
            float(velocity_vocab.max()),
        ],
        "trajectory_mask_valid_ratio": float(traj_mask.mean()),
        "representative_path_indices": representative_path_indices.astype(int).tolist(),
        "representative_velocity_indices": representative_velocity_indices.astype(int).tolist(),
        "example_path_indices": example_path_indices,
        "example_velocity_indices": example_velocity_indices,
    }

    with (output_dir / "anchor_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    lines = [
        "SparseDrive anchor library stats",
        "",
        f"path_vocab: {path_vocab.shape}",
        f"velocity_vocab: {velocity_vocab.shape}",
        f"traj_vocab: {traj_vocab.shape}",
        f"traj_mask: {traj_mask.shape}, valid ratio={traj_mask.mean():.4f}",
        "",
        "path final x min/median/max: "
        f"{final_x.min():.3f}, {np.median(final_x):.3f}, {final_x.max():.3f}",
        "path final y min/median/max: "
        f"{final_y.min():.3f}, {np.median(final_y):.3f}, {final_y.max():.3f}",
        "velocity mean min/median/max: "
        f"{mean_speed.min():.3f}, {np.median(mean_speed):.3f}, {mean_speed.max():.3f}",
        "velocity value min/max: "
        f"{velocity_vocab.min():.3f}, {velocity_vocab.max():.3f}",
        "",
        f"representative path indices: {representative_path_indices.astype(int).tolist()}",
        f"representative velocity indices: {representative_velocity_indices.astype(int).tolist()}",
        f"example path indices: {example_path_indices}",
        f"example velocity indices: {example_velocity_indices}",
    ]
    (output_dir / "anchor_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path_vocab, velocity_vocab, traj_vocab, traj_mask = load_anchors(args.project_root)

    plot_all_paths(path_vocab, output_dir)
    plot_paths_by_endpoint(path_vocab, output_dir)
    representative_path_indices = plot_representative_paths(path_vocab, output_dir)
    representative_velocity_indices = plot_velocity_curves(velocity_vocab, output_dir)
    example_path_indices, example_velocity_indices = plot_trajectory_examples(
        path_vocab, velocity_vocab, traj_vocab, output_dir
    )
    write_stats(
        output_dir,
        path_vocab,
        velocity_vocab,
        traj_vocab,
        traj_mask,
        representative_path_indices,
        representative_velocity_indices,
        example_path_indices,
        example_velocity_indices,
    )

    print(f"saved anchor visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
