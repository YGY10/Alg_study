from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.Common.common import TrajPoint
from typing import List
import numpy as np


def extract_nn_features(
    ego: TrajPoint, ref_path: List[TrajPoint], obstacles: List[VehicleKModel]
):
    ego_feats = [
        ego.v,
        ego.l,
        ego.dl,
        ego.ddl,
    ]

    ref = ref_path[0]

    ref_feats = [
        ref.yaw,
    ]

    obs_feats = []
    for o in obstacles[:2]:
        dx = o.x - ego.x
        dy = o.y - ego.y
        obs_feats += [dx, dy, o.v]

    feats = ego_feats + ref_feats + obs_feats
    return np.array(feats, dtype=np.float32)
