from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.argoverse_dataset import ArgoverseForecastingDataset
from data.collate import vectornet_collate_fn
from losses import average_displacement_error, trajectory_mse_loss
from models.vectornet import VectorNet


def move_batch_to_device(batch, device: torch.device):
    x = batch.x.to(device)
    vector_mask = batch.vector_mask.to(device)
    polyline_mask = batch.polyline_mask.to(device)
    target_index = batch.target_index.to(device)
    future = batch.future.to(device)

    return x, vector_mask, polyline_mask, target_index, future


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_ade = 0.0
    num_batches = 0

    for batch in loader:
        x, vector_mask, polyline_mask, target_index, future = move_batch_to_device(
            batch,
            device,
        )

        future_pred = model(
            x,
            vector_mask,
            polyline_mask,
            target_index,
        )

        loss = trajectory_mse_loss(
            future_pred,
            future,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ade = average_displacement_error(
                future_pred,
                future,
            )

        total_loss += loss.item()
        total_ade += ade.item()
        num_batches += 1

    return total_loss / num_batches, total_ade / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_ade = 0.0
    num_batches = 0

    for batch in loader:
        x, vector_mask, polyline_mask, target_index, future = move_batch_to_device(
            batch,
            device,
        )

        future_pred = model(
            x,
            vector_mask,
            polyline_mask,
            target_index,
        )

        loss = trajectory_mse_loss(
            future_pred,
            future,
        )

        ade = average_displacement_error(
            future_pred,
            future,
        )

        total_loss += loss.item()
        total_ade += ade.item()
        num_batches += 1

    return total_loss / num_batches, total_ade / num_batches


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_data_dir = (
        "TorchStudy/VectorNet/datasets/argoverse1/train/"
        "forecasting_train_v1.1/train/data"
    )

    val_data_dir = (
        "TorchStudy/VectorNet/datasets/argoverse1/val/" "forecasting_val_v1.1/val/data"
    )

    map_dir = "TorchStudy/VectorNet/datasets/argoverse1/map/hd_maps/map_files"

    train_dataset = ArgoverseForecastingDataset(
        data_dir=train_data_dir,
        history_steps=20,
        future_steps=30,
        max_samples=1000,
        map_dir=map_dir,
        use_map=True,
        lane_radius=60.0,
        max_lanes=64,
    )

    val_dataset = ArgoverseForecastingDataset(
        data_dir=val_data_dir,
        history_steps=20,
        future_steps=30,
        max_samples=200,
        map_dir=map_dir,
        use_map=True,
        lane_radius=60.0,
        max_lanes=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=vectornet_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=vectornet_collate_fn,
    )

    model = VectorNet(
        hidden_dim=64,
        subgraph_layers=3,
        future_steps=30,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    num_epochs = 10
    checkpoint_dir = Path("TorchStudy/VectorNet/outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_argoverse.pt"
    best_val_ade = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_ade = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_ade = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        is_best = val_ade < best_val_ade
        if is_best:
            best_val_ade = val_ade
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ade": val_ade,
                    "train_loss": train_loss,
                    "train_ade": train_ade,
                },
                best_checkpoint_path,
            )

        print(
            f"epoch {epoch + 1:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_ade={train_ade:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_ade={val_ade:.4f}"
            f"{' *best*' if is_best else ''}"
        )


if __name__ == "__main__":
    main()
