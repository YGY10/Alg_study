import csv
import torch
from torch.utils.data import Dataset
import math


class NNFrenetDataset(Dataset):
    """
    用于 NNPlanner 的数据集
    - 输入：归一化后的特征 (11 维)
    - 输出：l_target（不做 scale）
    """

    def __init__(self, csv_path):
        self.samples = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                # =========================
                # 列索引约定（非常重要）
                # =========================
                # 0  episode_id   (忽略)
                # 1  step_id      (忽略)
                #
                # 2  ego.v
                # 3  ego.l
                # 4  ego.dl
                # 5  ego.ddl
                # 6  ref.yaw
                #
                # 7  obs1.dx
                # 8  obs1.dy
                # 9  obs1.v
                #
                # 10 obs2.dx
                # 11 obs2.dy
                # 12 obs2.v
                #
                # 13 label_l
                # 14 label_T (暂不用)

                ego_v = float(row[2]) / 20.0
                ego_l = float(row[3]) / 4.0
                ego_dl = float(row[4]) / 2.0
                ego_ddl = float(row[5]) / 3.0

                ref_yaw = float(row[6]) / math.pi

                obs1_dx = float(row[7]) / 80.0
                obs1_dy = float(row[8]) / 10.0
                obs1_v = float(row[9]) / 20.0

                obs2_dx = float(row[10]) / 80.0
                obs2_dy = float(row[11]) / 10.0
                obs2_v = float(row[12]) / 20.0

                features = [
                    ego_v,
                    ego_l,
                    ego_dl,
                    ego_ddl,
                    ref_yaw,
                    obs1_dx,
                    obs1_dy,
                    obs1_v,
                    obs2_dx,
                    obs2_dy,
                    obs2_v,
                ]

                label_l = float(row[13])  # 不做 scale

                self.samples.append(
                    (
                        torch.tensor(features, dtype=torch.float32),
                        torch.tensor(label_l, dtype=torch.float32),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
