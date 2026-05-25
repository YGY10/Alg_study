from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset import X_RANGE, Y_RANGE, generate_sample

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def plot_sample(seed: int, save_path: Path) -> None:
    sample = generate_sample(seed=seed)

    fig, ax = plt.subplots(figsize=(7, 8))

    for obstacle in sample.obstacles:
        rectangle = Rectangle(
            (
                obstacle.center_y - obstacle.width / 2.0,
                obstacle.center_x - obstacle.length / 2.0,
            ),
            obstacle.width,
            obstacle.length,
            facecolor="black",
            edgecolor="black",
            alpha=0.8,
        )
        ax.add_patch(rectangle)

    for index, trajectory in enumerate(sample.anchors):
        if sample.collision_flags[index]:
            color = "#e7a0a0"
            alpha = 0.55
        else:
            color = "lightgray"
            alpha = 0.9

        ax.plot(
            trajectory[:, 1],
            trajectory[:, 0],
            color=color,
            linewidth=1.0,
            alpha=alpha,
        )

    best_trajectory = sample.anchors[sample.best_index]
    ax.plot(
        best_trajectory[:, 1],
        best_trajectory[:, 0],
        color="green",
        linewidth=3.0,
        marker="o",
        markersize=3,
        label=f"best anchor {sample.best_index}",
    )

    ax.scatter(
        [0.0],
        [0.0],
        color="red",
        s=70,
        zorder=5,
        label="ego",
    )

    ax.scatter(
        [sample.goal[1]],
        [sample.goal[0]],
        color="blue",
        marker="*",
        s=180,
        zorder=5,
        label="goal",
    )

    ax.set_xlim(Y_RANGE[1], Y_RANGE[0])
    ax.set_ylim(X_RANGE)
    ax.set_xlabel("lateral y [m], left is positive")
    ax.set_ylabel("forward x [m]")
    ax.set_title(f"Toy trajectory selector, seed={seed}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in range(6):
        plot_sample(
            seed=seed,
            save_path=OUTPUT_DIR / f"sample_{seed}.png",
        )

    print(f"saved visualizations to: {OUTPUT_DIR}")
