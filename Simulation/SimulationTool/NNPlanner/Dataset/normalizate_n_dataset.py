import csv
import torch
from torch.utils.data import Dataset
import math


# =========================
# 列索引约定(V2):
# =========================
# 0  episode_id   (忽略)
# 1  step_id      (忽略)
# 自车状态：
# 2  ego.v
# 3  ego.l
# 障碍车状态：
# 4  obs1.dx
# 5  obs1.dy
# 6  obs1.dv
# 7  obs1.mask :代表是否真实存在
#
# 8  obs2.dx
# 9  obs2.dy
# 10 obs2.dv
# 11 obs2.mask :代表是否真实存在
# label:
# 12 need_lane_change
# 13 delta_l
# =========================


class NNFrenetDataset(Dataset):
    def __init__(self, csv_path):
        self.x = []
        self.need = []
        self.delta = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for line_idx, row in enumerate(reader, start=2):
                # ===== 强校验 =====
                if len(row) != 14:
                    raise RuntimeError(
                        f"[Dataset][FATAL] line {line_idx}: "
                        f"len={len(row)} (expect 14), row={row}"
                    )

                # ===== episode_id 强校验 =====
                try:
                    epi_f = float(row[0])
                except ValueError:
                    raise RuntimeError(
                        f"[Dataset][FATAL] line {line_idx}: "
                        f"episode_id not float: {row[0]}"
                    )

                if not epi_f.is_integer():
                    raise RuntimeError(
                        f"[Dataset][FATAL] line {line_idx}: "
                        f"episode_id not integer-valued: {row[0]}"
                    )

                # ===== features =====
                ego_v = float(row[2]) / 20.0
                ego_l = float(row[3]) / 4.0

                obs1_dx = float(row[4]) / 80.0
                obs1_dy = float(row[5]) / 10.0
                obs1_dv = float(row[6]) / 20.0
                obs1_mask = float(row[7])

                obs2_dx = float(row[8]) / 80.0
                obs2_dy = float(row[9]) / 10.0
                obs2_dv = float(row[10]) / 20.0
                obs2_mask = float(row[11])

                x = [
                    ego_v,
                    ego_l,
                    obs1_dx,
                    obs1_dy,
                    obs1_dv,
                    obs1_mask,
                    obs2_dx,
                    obs2_dy,
                    obs2_dv,
                    obs2_mask,
                ]

                need_lane_change = float(row[12])
                delta_l = float(row[13])

                self.x.append(torch.tensor(x, dtype=torch.float32))
                self.need.append(torch.tensor([need_lane_change], dtype=torch.float32))
                self.delta.append(torch.tensor([delta_l], dtype=torch.float32))

        print(f"[Dataset] Loaded {len(self.x)} samples from {csv_path}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.need[idx], self.delta[idx]
