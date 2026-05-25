from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from dataset import ToyTrajectoryDataset
from models.model_attention import AttentionTrajectorySelector

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "attention"

NUM_TRAIN_SAMPLES = 10000
NUM_VAL_SAMPLES = 2000
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_trainning = optimizer is not None
    if is_trainning:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        anchors = batch["anchors"].to(device)
        input_grid = batch["input_grid"].to(device)
        best_index = batch["best_index"].to(device)

        if is_trainning:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_trainning):
            scores = model(input_grid, anchors)
            loss = criterion(scores, best_index)

            if is_trainning:
                loss.backward()
                optimizer.step()
            predictions = scores.argmax(dim=-1)

        batch_size = input_grid.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (predictions == best_index).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def plot_history(
    train_losses: list[float],
    val_losses: list[float],
    train_accuracies: list[float],
    val_accuracies: list[float],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, marker="o", label="train")
    axes[0].plot(epochs, val_losses, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, marker="o", label="train")
    axes[1].plot(epochs, val_accuracies, marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = ToyTrajectoryDataset(
        num_samples=NUM_TRAIN_SAMPLES,
        seed_offset=0,
    )
    val_dataset = ToyTrajectoryDataset(
        num_samples=NUM_VAL_SAMPLES,
        seed_offset=10000,
    )

    labels = [train_dataset[i]["best_index"].item() for i in range(len(train_dataset))]
    counts = np.bincount(labels, minlength=21)

    print("label counts:", counts)
    print("majority baseline accuracy:", counts.max() / counts.sum())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AttentionTrajectorySelector().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                OUTPUT_DIR / "best_model.pt",
            )

    plot_history(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        save_path=OUTPUT_DIR / "training_history.png",
    )

    print("best val accuracy:", best_val_accuracy)
    print("saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
