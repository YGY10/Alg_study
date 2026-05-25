from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ToyTrajectoryDataset, X_RANGE, Y_RANGE
from models.model_attention import AttentionTrajectorySelector

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "attention"
EVAL_DIR = OUTPUT_DIR / "evaluation"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"

NUM_TEST_SAMPLES = 2000
TEST_SEED_OFFSET = 20000
BATCH_SIZE = 64


def evaluate_model(
    model: AttentionTrajectorySelector,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    """Evaluate model selection quality on the test set."""
    model.eval()

    all_predictions = []
    all_best_indices = []
    all_predicted_costs = []
    all_best_costs = []
    all_predicted_collisions = []

    with torch.no_grad():
        for batch in dataloader:
            input_grid = batch["input_grid"].to(device)
            anchors = batch["anchors"].to(device)

            scores = model(input_grid, anchors)
            predictions = scores.argmax(dim=-1).cpu()

            best_indices = batch["best_index"]
            costs = batch["costs"]
            collision_flags = batch["collision_flags"]

            batch_indices = torch.arange(len(predictions))
            predicted_costs = costs[batch_indices, predictions]
            best_costs = costs[batch_indices, best_indices]
            predicted_collisions = collision_flags[batch_indices, predictions]

            all_predictions.append(predictions)
            all_best_indices.append(best_indices)
            all_predicted_costs.append(predicted_costs)
            all_best_costs.append(best_costs)
            all_predicted_collisions.append(predicted_collisions)

    predictions = torch.cat(all_predictions).numpy()
    best_indices = torch.cat(all_best_indices).numpy()
    predicted_costs = torch.cat(all_predicted_costs).numpy()
    best_costs = torch.cat(all_best_costs).numpy()
    predicted_collisions = torch.cat(all_predicted_collisions).numpy()

    regrets = predicted_costs - best_costs
    exact_accuracy = float((predictions == best_indices).mean())
    collision_rate = float(predicted_collisions.mean())
    safe_rate = 1.0 - collision_rate

    return {
        "predictions": predictions,
        "best_indices": best_indices,
        "predicted_costs": predicted_costs,
        "best_costs": best_costs,
        "predicted_collisions": predicted_collisions,
        "regrets": regrets,
        "exact_accuracy": exact_accuracy,
        "collision_rate": collision_rate,
        "safe_rate": safe_rate,
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "p90_regret": float(np.percentile(regrets, 90)),
    }


def plot_prediction(
    sample: dict[str, torch.Tensor],
    predicted_index: int,
    save_path: Path,
    title: str,
) -> None:
    """Visualize one model prediction against the rule-optimal trajectory."""
    obstacle_grid = sample["input_grid"][0].numpy()
    anchors = sample["anchors"].numpy()
    goal = sample["goal"].numpy()
    best_index = int(sample["best_index"].item())
    collision_flags = sample["collision_flags"].numpy()
    costs = sample["costs"].numpy()

    fig, ax = plt.subplots(figsize=(7, 8))

    ax.imshow(
        obstacle_grid,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        alpha=0.8,
        extent=[Y_RANGE[1], Y_RANGE[0], X_RANGE[0], X_RANGE[1]],
        origin="upper",
    )

    for index, trajectory in enumerate(anchors):
        color = "#e7a0a0" if collision_flags[index] else "lightgray"
        ax.plot(
            trajectory[:, 1],
            trajectory[:, 0],
            color=color,
            linewidth=1.0,
            alpha=0.7,
        )

    optimal_trajectory = anchors[best_index]
    predicted_trajectory = anchors[predicted_index]

    ax.plot(
        optimal_trajectory[:, 1],
        optimal_trajectory[:, 0],
        color="green",
        linewidth=3.0,
        marker="o",
        markersize=3,
        label=f"rule best: {best_index}",
    )
    ax.plot(
        predicted_trajectory[:, 1],
        predicted_trajectory[:, 0],
        color="red",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=3,
        label=f"model: {predicted_index}",
    )

    ax.scatter([0.0], [0.0], color="black", s=70, zorder=5, label="ego")
    ax.scatter(
        [goal[1]],
        [goal[0]],
        color="blue",
        marker="*",
        s=180,
        zorder=5,
        label="goal",
    )

    regret = costs[predicted_index] - costs[best_index]
    collision = bool(collision_flags[predicted_index])

    ax.set_xlim(Y_RANGE[1], Y_RANGE[0])
    ax.set_ylim(X_RANGE)
    ax.set_xlabel("lateral y [m], left is positive")
    ax.set_ylabel("forward x [m]")
    ax.set_title(
        f"{title}\n"
        f"pred={predicted_index}, best={best_index}, "
        f"regret={regret:.2f}, collision={collision}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_regret_histogram(
    regrets: np.ndarray,
    predicted_collisions: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    safe_regrets = regrets[~predicted_collisions]
    ax.hist(safe_regrets, bins=30, color="steelblue", alpha=0.85)

    ax.set_title("Cost regret for non-collision-scale errors")
    ax.set_xlabel("predicted cost - best cost")
    ax.set_ylabel("sample count")
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_example_plots(
    dataset: ToyTrajectoryDataset,
    results: dict[str, np.ndarray | float],
) -> None:
    predictions = results["predictions"]
    best_indices = results["best_indices"]
    predicted_collisions = results["predicted_collisions"]
    regrets = results["regrets"]

    correct_indices = np.where(predictions == best_indices)[0]
    safe_wrong_indices = np.where(
        (predictions != best_indices) & (~predicted_collisions)
    )[0]
    collision_indices = np.where(predicted_collisions)[0]

    categories = [
        ("correct", correct_indices),
        ("safe_wrong", safe_wrong_indices),
        ("collision_failure", collision_indices),
    ]

    for category_name, indices in categories:
        for rank, index in enumerate(indices[:3]):
            sample = dataset[int(index)]
            plot_prediction(
                sample=sample,
                predicted_index=int(predictions[index]),
                save_path=EVAL_DIR / f"{category_name}_{rank}.png",
                title=f"{category_name}, test sample {index}",
            )

    if len(collision_indices) > 0:
        worst_indices = collision_indices[np.argsort(regrets[collision_indices])[::-1]]
    else:
        worst_indices = np.argsort(regrets)[::-1]

    for rank, index in enumerate(worst_indices[:3]):
        sample = dataset[int(index)]
        plot_prediction(
            sample=sample,
            predicted_index=int(predictions[index]),
            save_path=EVAL_DIR / f"worst_{rank}.png",
            title=f"worst case, test sample {index}",
        )


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    test_dataset = ToyTrajectoryDataset(
        num_samples=NUM_TEST_SAMPLES,
        seed_offset=TEST_SEED_OFFSET,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AttentionTrajectorySelector().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("loaded checkpoint epoch:", checkpoint["epoch"])
    print("checkpoint val accuracy:", checkpoint["val_accuracy"])

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print(f"test exact accuracy: {results['exact_accuracy']:.4f}")
    print(f"test collision rate: {results['collision_rate']:.4f}")
    print(f"test safe rate: {results['safe_rate']:.4f}")
    print(f"mean regret: {results['mean_regret']:.4f}")
    print(f"median regret: {results['median_regret']:.4f}")
    print(f"p90 regret: {results['p90_regret']:.4f}")

    summary = [
        f"checkpoint epoch: {checkpoint['epoch']}",
        f"checkpoint val accuracy: {checkpoint['val_accuracy']:.4f}",
        f"test exact accuracy: {results['exact_accuracy']:.4f}",
        f"test collision rate: {results['collision_rate']:.4f}",
        f"test safe rate: {results['safe_rate']:.4f}",
        f"mean regret: {results['mean_regret']:.4f}",
        f"median regret: {results['median_regret']:.4f}",
        f"p90 regret: {results['p90_regret']:.4f}",
    ]
    (EVAL_DIR / "evaluation_summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    plot_regret_histogram(
        regrets=results["regrets"],
        predicted_collisions=results["predicted_collisions"],
        save_path=EVAL_DIR / "regret_histogram.png",
    )
    save_example_plots(test_dataset, results)

    print("saved:", EVAL_DIR)


if __name__ == "__main__":
    main()
