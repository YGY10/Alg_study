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
    update_period: float = 0.05
    latency: float = 0.0

    position_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    speed_noise_std: float = 0.0
    object_dropout_probability: float = 0.0
    signal_dropout_probability: float = 0.0
    lane_dropout_probability: float = 0.0
    random_seed: int = 0

    # 理想车道几何传感器：近处横向扫描 + 前向扇形射线扫描。
    lane_transverse_spacing: float = 1.0
    lane_ray_count: int = 121
    lane_ray_half_angle: float = np.deg2rad(100.0)
    lane_output_point_count: int = 81

    # 纯几何车道线片段关联参数。只依赖端点、切线和空间距离，
    # 不使用 map lane id、前驱后继或导航路径。
    lane_line_join_distance: float = 2.0
    lane_line_join_heading: float = np.deg2rad(40.0)
    lane_line_duplicate_distance: float = 0.35

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
                self.lane_transverse_spacing,
                self.lane_ray_half_angle,
                self.lane_line_join_distance,
                self.lane_line_join_heading,
                self.lane_line_duplicate_distance,
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
        if self.lane_transverse_spacing <= 0.0:
            raise ValueError("lane_transverse_spacing must be positive")
        if not isinstance(self.lane_ray_count, int) or self.lane_ray_count < 3:
            raise ValueError("lane_ray_count must be an integer >= 3")
        if not 0.0 < self.lane_ray_half_angle <= np.pi:
            raise ValueError("lane_ray_half_angle must be within (0, pi]")
        if (
            not isinstance(self.lane_output_point_count, int)
            or self.lane_output_point_count < 2
        ):
            raise ValueError("lane_output_point_count must be an integer >= 2")
        if self.lane_line_join_distance <= 0.0:
            raise ValueError("lane_line_join_distance must be positive")
        if not 0.0 < self.lane_line_join_heading <= np.pi:
            raise ValueError("lane_line_join_heading must be within (0, pi]")
        if self.lane_line_duplicate_distance < 0.0:
            raise ValueError("lane_line_duplicate_distance must be non-negative")


@dataclass(frozen=True)
class PerceivedObject:
    """车辆坐标系中的感知目标。"""

    object_id: str
    object_type: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float
    confidence: float = 1.0
    semantic_type: str = "unknown"


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
class PerceivedLaneLine:
    """纯感知发布的一条局部车道线。

    ``points`` 是车辆坐标系中的有序点列。line_id 只在当前感知帧内用于
    区分实例，不是地图 lane id，也不表达任何拓扑或导航关系。
    """

    line_id: str
    points: FloatArray
    confidence: float = 1.0
    line_type: str = "unknown"

    def __post_init__(self) -> None:
        value = np.asarray(self.points, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 2:
            raise ValueError("points must have shape [N, 2], N >= 2")
        if not np.all(np.isfinite(value)):
            raise ValueError("points contain non-finite values")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "points", value.copy())


@dataclass(frozen=True)
class PerceivedLaneSegment:
    """旧版走廊兼容类型；新 PNC 链路不再由感知模块生成。"""

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
    road_segments: tuple[PerceivedLaneSegment, ...] = ()
    lane_lines: tuple[PerceivedLaneLine, ...] = ()
    source: str = "ground_truth_local"
    coordinate_frame: str = "vehicle"
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
