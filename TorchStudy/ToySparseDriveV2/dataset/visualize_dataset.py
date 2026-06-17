from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import TIME_STEPS, ToySparseDriveV2Dataset
from grid import GridConfig

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = TOY_ROOT / "outputs" / "dataset"


def setup_axis(ax: plt.Axes, config: GridConfig, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_xlim(config.y_max, config.y_min)
    ax.set_ylim(config.x_min, config.x_max)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def draw_rectangle_world(
    ax: plt.Axes,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    size_xy = np.asarray(size_xy, dtype=np.float32)

    x = float(center_xy[0])
    y = float(center_xy[1])
    length_x = float(size_xy[0])
    width_y = float(size_xy[1])

    lower_left_y = y - width_y / 2.0
    lower_left_x = x - length_x / 2.0

    rect = plt.Rectangle(
        (lower_left_y, lower_left_x),
        width_y,
        length_x,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.8,
    )
    ax.add_patch(rect)


def plot_sample(
    ax: plt.Axes,
    sample: dict[str, object],
    config: GridConfig,
    sample_index: int,
) -> None:
    target_path = sample["target_path"].numpy()
    target_trajectory = sample["target_trajectory"].numpy()
    target_trajectory_mask = sample["target_trajectory_mask"].numpy()

    obstacle_centers = sample["obstacle_centers"].numpy()
    obstacle_sizes = sample["obstacle_sizes"].numpy()
    obstacle_velocities = sample["obstacle_velocities"].numpy()
    obstacle_mask = sample["obstacle_mask"].numpy()

    valid_obstacle_indices = np.where(obstacle_mask > 0.5)[0]

    for obstacle_index in valid_obstacle_indices:
        center_xy = obstacle_centers[obstacle_index]
        size_xy = obstacle_sizes[obstacle_index]
        velocity_xy = obstacle_velocities[obstacle_index]

        draw_rectangle_world(
            ax,
            center_xy,
            size_xy,
            color="#666666",
            alpha=0.45,
        )

        for time_s in TIME_STEPS:
            future_center = center_xy + velocity_xy * float(time_s)
            draw_rectangle_world(
                ax,
                future_center,
                size_xy,
                color="#ff9900",
                alpha=0.08,
            )

        end_center = center_xy + velocity_xy * float(TIME_STEPS[-1])
        ax.arrow(
            center_xy[1],
            center_xy[0],
            end_center[1] - center_xy[1],
            end_center[0] - center_xy[0],
            color="#cc6600",
            alpha=0.55,
            width=0.05,
            head_width=1.0,
            length_includes_head=True,
        )

    ax.plot(
        target_path[:, 1],
        target_path[:, 0],
        color="#1f77b4",
        linewidth=1.5,
        label="target path",
    )

    valid_traj = target_trajectory[target_trajectory_mask > 0.5]
    ax.plot(
        valid_traj[:, 1],
        valid_traj[:, 0],
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="target trajectory",
    )

    if len(valid_traj) > 0:
        goal_xy = valid_traj[-1, :2]
    else:
        goal_xy = target_trajectory[-1, :2]

    ax.scatter(
        goal_xy[1],
        goal_xy[0],
        color="#2ca02c",
        s=35,
        zorder=5,
        label="goal",
    )

    path_index = int(sample["path_index"])
    velocity_index = int(sample["velocity_index"])
    setup_axis(
        ax,
        config,
        f"sample {sample_index} | p{path_index} v{velocity_index}",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = GridConfig()
    dataset = ToySparseDriveV2Dataset(
        num_samples=8,
        seed_offset=0,
        grid_config=config,
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for i, ax in enumerate(axes):
        sample = dataset[i]
        plot_sample(ax, sample, config, i)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = OUTPUT_DIR / "samples.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
