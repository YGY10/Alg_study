from __future__ import annotations

from dataclasses import dataclass

from sim2d.types import (
    Obstacle,
    VehicleState,
)
from sim2d.world.road_geometry import WorldLaneGeometry
from sim2d.world.traffic_signal import WorldTrafficSignal


@dataclass
class WorldState:
    """仿真世界某一时刻的 ground-truth 状态。"""

    time: float
    ego_state: VehicleState

    obstacles: tuple[Obstacle, ...] = ()
    traffic_signals: tuple[WorldTrafficSignal, ...] = ()
    road_lanes: tuple[WorldLaneGeometry, ...] = ()
