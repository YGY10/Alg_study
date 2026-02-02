import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from NNPlanner.Model.NNPlanner import NNPlanner
from NNPlanner.Dataset.normalizate_n_dataset import NNFrenetDataset

# ======================
# 基本配置
# ======================
CSV_PATH = "./nn_dataset.csv"
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3
TRAIN_RATIO = 0.8
SEED = 42
DELTA_LOSS_WEIGHT = 1.0
SAVE_PATH = "./nnplanner_two_head.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
random.seed(SEED)

# ======================
# 1. Dataset
# ======================
dataset = NNFrenetDataset(CSV_PATH)

# ======================
# 2. episode-aware split
# ======================
episode_ids = []
with open(CSV_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)

    for line_idx, row in enumerate(reader, start=2):
        if not row:
            raise RuntimeError(f"[Split][FATAL] empty row at line {line_idx}")

        epi_f = float(row[0])
        if not epi_f.is_integer():
            raise RuntimeError(
                f"[Split][FATAL] episode_id not integer-valued: {row[0]} at line {line_idx}"
            )
        episode_ids.append(int(epi_f))

unique_episodes = list(sorted(set(episode_ids)))
random.shuffle(unique_episodes)

n_train = int(len(unique_episodes) * TRAIN_RATIO)
train_epi = set(unique_episodes[:n_train])
val_epi = set(unique_episodes[n_train:])

train_indices = [i for i, epi in enumerate(episode_ids) if epi in train_epi]
val_indices = [i for i, epi in enumerate(episode_ids) if epi in val_epi]

train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(
    f"[Split] episodes: total={len(unique_episodes)}, "
    f"train={len(train_epi)}, val={len(val_epi)}"
)
print(f"[Split] samples : train={len(train_set)}, val={len(val_set)}")

# ======================
# 3. Model
# ======================
model = NNPlanner(in_dim=10).to(device)

# 类别不平衡处理
POS_WEIGHT = torch.tensor([180.0], device=device)
loss_need_fn = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
loss_delta_fn = nn.SmoothL1Loss(reduction="none")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# 4. Train Loop
# ======================
best_val_loss = float("inf")
best_epoch = -1

for epoch in range(EPOCHS):
    # ---------- train ----------
    model.train()
    total_loss = 0.0
    total_need_loss = 0.0
    total_delta_loss = 0.0

    for x, need, delta in train_loader:
        x = x.to(device)
        need = need.to(device)
        delta = delta.to(device)

        pred_need, pred_delta = model(x)

        loss_need = loss_need_fn(pred_need, need)

        mask = (need > 0.5).float()
        loss_delta_raw = loss_delta_fn(pred_delta, delta)
        loss_delta = (loss_delta_raw * mask).sum() / (mask.sum() + 1e-6)

        loss = loss_need + DELTA_LOSS_WEIGHT * loss_delta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_need_loss += loss_need.item() * bs
        total_delta_loss += loss_delta.item() * bs

    total_loss /= len(train_set)
    total_need_loss /= len(train_set)
    total_delta_loss /= len(train_set)

    # ---------- val ----------
    model.eval()
    val_loss = 0.0
    val_need_loss = 0.0
    val_delta_loss = 0.0

    correct_need = 0
    total_need = 0
    tp = 0
    fn = 0

    with torch.no_grad():
        for x, need, delta in val_loader:
            x = x.to(device)
            need = need.to(device)
            delta = delta.to(device)

            pred_need, pred_delta = model(x)

            loss_need = loss_need_fn(pred_need, need)

            mask = (need > 0.5).float()
            loss_delta_raw = loss_delta_fn(pred_delta, delta)
            loss_delta = (loss_delta_raw * mask).sum() / (mask.sum() + 1e-6)

            loss = loss_need + DELTA_LOSS_WEIGHT * loss_delta

            bs = x.size(0)
            val_loss += loss.item() * bs
            val_need_loss += loss_need.item() * bs
            val_delta_loss += loss_delta.item() * bs

            pred_label = (torch.sigmoid(pred_need) > 0.5).float()
            correct_need += (pred_label == need).sum().item()
            total_need += need.numel()

            tp += ((pred_label == 1) & (need == 1)).sum().item()
            fn += ((pred_label == 0) & (need == 1)).sum().item()

    val_loss /= len(val_set)
    val_need_loss /= len(val_set)
    val_delta_loss /= len(val_set)

    need_acc = correct_need / total_need
    recall = tp / (tp + fn + 1e-6)

    print(
        f"[Epoch {epoch:03d}] "
        f"train_loss={total_loss:.4f} "
        f"(need={total_need_loss:.4f}, delta={total_delta_loss:.4f}) | "
        f"val_loss={val_loss:.4f} "
        f"(need={val_need_loss:.4f}, delta={val_delta_loss:.4f}) | "
        f"need_acc={need_acc*100:.2f}% "
        f"need_recall={recall*100:.2f}%"
    )

    # ---------- save best ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), SAVE_PATH)
        print(
            f"[Checkpoint] NEW BEST model saved at epoch {epoch:03d} "
            f"(val_loss={val_loss:.4f})"
        )

print(f"Training finished. Best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
print("Best model saved to:", SAVE_PATH)
