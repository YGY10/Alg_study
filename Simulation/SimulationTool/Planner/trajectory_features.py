from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.Common.common import TrajPoint
from typing import List
import numpy as np
import os
import csv


def extract_nn_features(ego: TrajPoint, obstacles: List[VehicleKModel]):
    ego_feats = [
        ego.v,
        ego.l,
    ]

    obs_feats = []
    max_obs = 2
    for i in range(max_obs):
        if i < len(obstacles):
            o = obstacles[i]
            obs_feats += [
                o.x - ego.x,
                o.y - ego.y,
                o.v - ego.v,
                1.0,  # mask
            ]
        else:
            obs_feats += [0.0, 0.0, 0.0, 0.0]

    feats = ego_feats + obs_feats
    return np.array(feats, dtype=np.float32)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    features: raw physical features from extract_nn_features
    order must match training dataset exactly
    """
    f = features.copy().astype(np.float32)

    # ego
    f[0] /= 20.0  # ego.v
    f[1] /= 4.0  # ego.l

    # obs1
    f[2] /= 80.0  # obs1.dx
    f[3] /= 10.0  # obs1.dy
    f[4] /= 20.0  # obs1.dv
    # f[5] is mask

    # obs2
    f[6] /= 80.0  # obs2.dx
    f[7] /= 10.0  # obs2.dy
    f[8] /= 20.0  # obs2.dv
    # f[9] is mask

    return f


def init_nn_dataset(nn_dataset_path):
    if not os.path.exists(nn_dataset_path):
        with open(nn_dataset_path, "w", newline="") as f:
            writer = csv.writer(f)

            header = [
                # 时间戳
                "epi",
                "sim time",
                # ===== ego =====
                "ego.v",
                "ego.l",
                # ===== obstacle 1 =====
                "obs1.dx",
                "obs1.dy",
                "obs1.dv",
                "obs1.mask",
                # ===== obstacle 2 =====
                "obs2.dx",
                "obs2.dy",
                "obs2.dv",
                "obs2.mask",
                # ===== labels =====
                "target lane danger",
                "l target",
            ]

            writer.writerow(header)


def log_nn_sample(nn_dataset_path, epi, sim_time, features, label):
    feats = np.asarray(features, dtype=float)
    lab = np.asarray(label, dtype=float)

    row = [float(epi), float(sim_time)] + list(feats) + list(lab)
    with open(nn_dataset_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
