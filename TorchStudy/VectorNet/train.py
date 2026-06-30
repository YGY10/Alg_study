from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.collate import vectornet_collate_fn
from data.dataset import VectorNetToyDataset
from losses import average_displacement_error, trajectory_mse_loss
from models.vectornet import VectorNet


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VectorNetToyDataset(
        num_samples=2000,
        history_steps=6,
        future_steps=30,
        seed=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
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

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_ade = 0.0
        num_batches = 0

        for batch in loader:
            x = batch.x.to(device)
            vector_mask = batch.vector_mask.to(device)
            polyline_mask = batch.polyline_mask.to(device)
            target_index = batch.target_index.to(device)
            future = batch.future.to(device)

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

        mean_loss = total_loss / num_batches
        mean_ade = total_ade / num_batches

        print(f"epoch {epoch + 1:02d} " f"loss={mean_loss:.4f} " f"ade={mean_ade:.4f}")


if __name__ == "__main__":
    train()
