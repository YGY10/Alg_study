# perception/perception_output.py

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BEVPerceptionInput:
    """
    BEV perception input.
    """
    bev_image: np.ndarray
    timestamp: float = 0.0
    ego_speed_kmh: float = 0.0

    # BEV 坐标，用于生成 ego footprint / unknown mask
    bev_grid: Optional[object] = None

    # BEVStitcher 输出的有效观测区域
    # 255 = 有观测，0 = 低可信/无观测
    observed_mask: Optional[np.ndarray] = None


@dataclass
class BEVPerceptionOutput:
    """
    BEV perception output.
    """
    white_mask: np.ndarray
    yellow_mask: np.ndarray
    edge_mask: np.ndarray
    lane_candidate_mask: np.ndarray
    road_marking_mask: np.ndarray
    drivable_candidate_mask: Optional[np.ndarray]
    debug_image: np.ndarray

    road_surface_mask: Optional[np.ndarray] = None
    ego_footprint_mask: Optional[np.ndarray] = None
    near_unknown_mask: Optional[np.ndarray] = None
    observed_mask: Optional[np.ndarray] = None