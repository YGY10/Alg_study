import csv
import random

CSV_PATH = "./Simulation/nn_dataset_with_epi.csv"
TRAIN_RATIO = 0.8

# 读取 CSV 到内存（保留整行）
rows = []
with open(CSV_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            rows.append(row)

# 提取所有 episode_id
# 约定：row[0] 是 episode_id
episode_ids = sorted({int(r[0]) for r in rows})

# episode-aware 随机划分
random.shuffle(episode_ids)
n_train = int(len(episode_ids) * TRAIN_RATIO)
train_episodes = set(episode_ids[:n_train])
val_episodes = set(episode_ids[n_train:])

# 按 episode 过滤行
train_rows = [r for r in rows if int(r[0]) in train_episodes]
val_rows = [r for r in rows if int(r[0]) in val_episodes]

print(
    f"episodes: total={len(episode_ids)}, train={len(train_episodes)}, val={len(val_episodes)}"
)
print(f"rows:     train={len(train_rows)}, val={len(val_rows)}")
