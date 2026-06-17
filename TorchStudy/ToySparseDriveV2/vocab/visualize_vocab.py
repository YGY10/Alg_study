from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
if str(TOY_ROOT) not in sys.path:
    sys.path.insert(0, str(TOY_ROOT))

from vocab.vocab import SparseDriveVocab

OUTPUT_DIR = TOY_ROOT / "outputs" / "vocab"


def setup_bev_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def plot_all_paths(vocab: SparseDriveVocab, output_dir: Path) -> None:
    path = vocab.path.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    for one_path in path:
        ax.plot(
            one_path[:, 1],
            one_path[:, 0],
            color="#1f77b4",
            alpha=0.06,
            linewidth=0.7,
        )

    setup_bev_axis(ax, "All SparseDriveV2 Path Vocab")
    ax.set_xlim(65, -65)
    ax.set_ylim(-55, 70)
    fig.tight_layout()
    fig.savefig(output_dir / "path_all.png", dpi=180)
    plt.close(fig)


def plot_representative_paths(vocab: SparseDriveVocab, output_dir: Path) -> None:
    path = vocab.path.cpu().numpy()
    final_y = path[:, -1, 1]
    order = np.argsort(final_y)
    indices = order[np.linspace(0, len(order) - 1, 48, dtype=int)]

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        one_path = path[idx]
        ax.plot(
            one_path[:, 1],
            one_path[:, 0],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )
        ax.scatter(one_path[-1, 1], one_path[-1, 0], color=color, s=8)

    setup_bev_axis(ax, "Representative Path Vocab by Final Lateral Offset")
    ax.set_xlim(65, -65)
    ax.set_ylim(-55, 70)
    fig.tight_layout()
    fig.savefig(output_dir / "path_representatives.png", dpi=180)
    plt.close(fig)


def plot_velocity_curves(vocab: SparseDriveVocab, output_dir: Path) -> None:
    velocity = vocab.velocity.cpu().numpy()
    t = np.arange(velocity.shape[1]) * 0.5 + 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    for one_velocity in velocity:
        ax.plot(t, one_velocity, color="#9467bd", alpha=0.12, linewidth=0.8)

    ax.set_title("All SparseDriveV2 Velocity Vocab")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_all.png", dpi=180)
    plt.close(fig)


def plot_velocity_heatmap(vocab: SparseDriveVocab, output_dir: Path) -> None:
    velocity = vocab.velocity.cpu().numpy()
    order = np.argsort(velocity.mean(axis=1))
    sorted_velocity = velocity[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        sorted_velocity,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    ax.set_title("Velocity Vocab Sorted by Mean Speed")
    ax.set_xlabel("time step")
    ax.set_ylabel("velocity anchor rank")
    fig.colorbar(im, ax=ax, label="speed [m/s]")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_heatmap.png", dpi=180)
    plt.close(fig)


def plot_trajectory_examples(vocab: SparseDriveVocab, output_dir: Path) -> None:
    path = vocab.path.cpu().numpy()
    velocity = vocab.velocity.cpu().numpy()
    trajectory = vocab.trajectory.cpu().numpy()

    final_y = path[:, -1, 1]
    path_order = np.argsort(final_y)
    path_indices = [
        int(path_order[80]),
        int(path_order[len(path_order) // 2]),
        int(path_order[-80]),
    ]

    mean_speed = velocity.mean(axis=1)
    velocity_order = np.argsort(mean_speed)
    velocity_indices = [
        int(velocity_order[20]),
        int(velocity_order[len(velocity_order) // 2]),
        int(velocity_order[-20]),
    ]

    fig, axes = plt.subplots(
        len(path_indices),
        len(velocity_indices),
        figsize=(11, 10),
        sharex=True,
        sharey=True,
    )

    for row, path_idx in enumerate(path_indices):
        for col, velocity_idx in enumerate(velocity_indices):
            ax = axes[row, col]
            one_path = path[path_idx]
            one_traj = trajectory[path_idx, velocity_idx]

            ax.plot(
                one_path[:, 1],
                one_path[:, 0],
                color="#999999",
                linewidth=1.0,
                linestyle="--",
                label="path",
            )
            ax.plot(
                one_traj[:, 1],
                one_traj[:, 0],
                color="#d62728",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label="trajectory",
            )

            setup_bev_axis(
                ax,
                f"path {path_idx}, velocity {velocity_idx}",
            )
            ax.set_xlim(65, -65)
            ax.set_ylim(-55, 70)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "trajectory_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = SparseDriveVocab.load()

    for key, value in vocab.summary().items():
        print(f"{key}: {value}")

    coverage = vocab.check_grid_coverage(
        x_min=-50.0,
        x_max=65.0,
        y_min=-60.0,
        y_max=60.0,
    )
    for key, value in coverage.items():
        print(f"{key}: {value}")

    plot_all_paths(vocab, OUTPUT_DIR)
    plot_representative_paths(vocab, OUTPUT_DIR)
    plot_velocity_curves(vocab, OUTPUT_DIR)
    plot_velocity_heatmap(vocab, OUTPUT_DIR)
    plot_trajectory_examples(vocab, OUTPUT_DIR)

    print(f"saved visualizations to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
