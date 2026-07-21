from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim2d.types import GoalState, VehicleState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PerceptionConfig:
    """局部感知范围、刷新节拍以及预留误差模型。"""

    forward_range: float = 60.0
    rear_range: float = 20.0
    lateral_range: float = 35.0
    field_of_view: float = 2.0 * np.pi
    update_period: float = 0.10
    latency: float = 0.0

    position_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    speed_noise_std: float = 0.0
    object_dropout_probability: float = 0.0
    signal_dropout_probability: float = 0.0
    lane_dropout_probability: float = 0.0
    random_seed: int = 0

    def validate(self) -> None:
        values = np.asarray(
            [
                self.forward_range,
                self.rear_range,
                self.lateral_range,
                self.field_of_view,
                self.update_period,
                self.latency,
                self.position_noise_std,
                self.yaw_noise_std,
                self.speed_noise_std,
                self.object_dropout_probability,
                self.signal_dropout_probability,
                self.lane_dropout_probability,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("PerceptionConfig must contain finite values")
        if self.forward_range <= 0.0 or self.rear_range < 0.0:
            raise ValueError("Invalid longitudinal perception range")
        if self.lateral_range <= 0.0:
            raise ValueError("lateral_range must be positive")
        if not (0.0 < self.field_of_view <= 2.0 * np.pi):
            raise ValueError("field_of_view must be within (0, 2*pi]")
        if self.update_period <= 0.0 or self.latency < 0.0:
            raise ValueError("Invalid perception timing configuration")
        for probability in (
            self.object_dropout_probability,
            self.signal_dropout_probability,
            self.lane_dropout_probability,
        ):
            if not (0.0 <= probability <= 1.0):
                raise ValueError("Dropout probabilities must be within [0, 1]")
        if min(
            self.position_noise_std,
            self.yaw_noise_std,
            self.speed_noise_std,
        ) < 0.0:
            raise ValueError("Noise standard deviations must be non-negative")


@dataclass(frozen=True)
class PerceivedObject:
    object_id: str
    object_type: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float
    confidence: float = 1.0


@dataclass(frozen=True)
class PerceivedTrafficSignal:
    entity_id: str
    map_signal_id: str | None
    x: float
    y: float
    yaw: float
    state: str
    remaining_time: float | None
    confidence: float = 1.0


@dataclass(frozen=True)
class PerceivedLaneSegment:
    map_lane_id: str
    centerline: FloatArray
    left_boundary: FloatArray
    right_boundary: FloatArray
    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        for name in ("centerline", "left_boundary", "right_boundary"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 2:
                raise ValueError(f"{name} must have shape [N, 2], N >= 2")
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class PerceptionSnapshot:
    measurement_time: float
    publish_time: float
    frame: int
    ego: VehicleState
    objects: tuple[PerceivedObject, ...]
    traffic_signals: tuple[PerceivedTrafficSignal, ...]
    road_segments: tuple[PerceivedLaneSegment, ...]
    source: str = "ground_truth_local"
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningInput:
    """规划器输入契约；前五个字段兼容旧 Observation 接口。"""

    time: float
    frame: int
    ego: VehicleState
    obstacles: tuple[Any, ...]
    goal: GoalState

    perception: PerceptionSnapshot
    map_network: Any | None = None
    previous_trajectory: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.previous_trajectory is not None:
            value = np.asarray(self.previous_trajectory, dtype=np.float64)
            object.__setattr__(self, "previous_trajectory", value.copy())
