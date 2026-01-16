import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

from SimulationTool.NNPlanner.Dataset.nn_dataset import NNFrenetDataset

# ======================
# 基本配置
# ======================
CSV_PATH = "./nn_dataset_with_epi.csv"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
TRAIN_RATIO = 0.8
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

# ======================
# 1. 读完整 Dataset
# ======================
full_dataset = NNFrenetDataset(CSV_PATH)

# 每个 sample 的 episode_id 在 CSV 第 0 列
episode_ids = []
with open(CSV_PATH, "r") as f:
    for line in f:
        if not line.strip():
            continue
        epi = int(line.split(",")[0])
        episode_ids.append(epi)

# ======================
# 2. episode-aware split
# ======================
unique_episodes = list(sorted(set(episode_ids)))
random.shuffle(unique_episodes)

n_train = int(len(unique_episodes) * TRAIN_RATIO)
train_epi = set(unique_episodes[:n_train])
val_epi = set(unique_episodes[n_train:])

train_indices = []
val_indices = []

for idx, epi in enumerate(episode_ids):
    if epi in train_epi:
        train_indices.append(idx)
    else:
        val_indices.append(idx)

train_subset = torch.utils.data.Subset(full_dataset, train_indices)
val_subset = torch.utils.data.Subset(full_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"train samples: {len(train_subset)}")
print(f"val samples:   {len(val_subset)}")

# ======================
# 3. NN：11 → 1
# ======================
model = nn.Sequential(
    nn.Linear(11, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# 4. 训练循环
# ======================
for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        y = y.unsqueeze(1)  # (B,) -> (B,1)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_subset)

    # ---- val ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            y = y.unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)

    val_loss /= len(val_subset)

    print(
        f"[Epoch {epoch:03d}] " f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
    )

print("training finished")

# ======================
# 5. Sanity check：pred vs GT
# ======================
print("\n[Sanity Check] pred_l vs gt_l")

model.eval()
num_check = min(20, len(val_subset))
check_indices = random.sample(range(len(val_subset)), num_check)

with torch.no_grad():
    for idx in check_indices:
        x, gt_l = val_subset[idx]
        pred_l = model(x.unsqueeze(0))[0, 0].item()
        print(f"  gt_l = {gt_l.item():>6.3f}   pred_l = {pred_l:>6.3f}")

# ======================
# 6. 保存模型
# ======================
SAVE_PATH = "./nn_l_only.pth"
torch.save(model.state_dict(), SAVE_PATH)
print("model saved to:", SAVE_PATH)
