from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim2d.perception import (
    PerceivedLaneSegment,
    PerceivedObject,
    PerceivedTrafficSignal,
)
from sim2d.types import GoalState, VehicleState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControlSequence:
    """
    优化器的控制变量。

    controls[k] = [acceleration, steering]
    shape = [horizon_steps, 2]
    """

    controls: FloatArray

    def __post_init__(self) -> None:
        value = np.asarray(self.controls, dtype=np.float64)

        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError("controls must have shape [N, 2], " f"got {value.shape}")

        if not np.all(np.isfinite(value)):
            raise ValueError("controls contain non-finite values")

        object.__setattr__(self, "controls", value.copy())


@dataclass(frozen=True)
class SpatiotemporalTrajectory:
    """
    规划时刻冻结自车坐标系中的时空联合轨迹。

    states[k] = [x_forward, y_left, relative_yaw, speed]
    times[k] = 相对于当前规划时刻的时间
    controls[k] = [acceleration, steering]

    states 和 times 的长度为 N + 1，controls 的长度为 N。
    坐标系在一次优化期间保持固定，不随未来自车状态移动。
    """

    times: FloatArray
    states: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        controls = np.asarray(self.controls, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError("times must have shape [N], " f"got {times.shape}")

        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("states must have shape [N, 4], " f"got {states.shape}")

        if controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError(
                "controls must have shape [N-1, 2], " f"got {controls.shape}"
            )

        if states.shape[0] != times.shape[0]:
            raise ValueError("states and times must have equal length")

        if controls.shape[0] != states.shape[0] - 1:
            raise ValueError("controls length must be states length minus one")

        if times.size == 0:
            raise ValueError("trajectory must contain at least one state")

        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")

        if not np.all(np.isfinite(states)):
            raise ValueError("states contain non-finite values")

        if not np.all(np.isfinite(controls)):
            raise ValueError("controls contain non-finite values")

        if abs(float(times[0])) > 1e-12:
            raise ValueError("times must start at zero")

        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "controls", controls.copy())


@dataclass(frozen=True)
class LocalPlanningContext:
    """时空规划器内部统一使用的冻结自车坐标上下文。

    ``world_origin`` 仅用于最终调试/GUI 输出时恢复世界坐标。
    其余几何量全部位于当前规划周期冻结的自车坐标系：
    +x 向前，+y 向左，当前自车位姿为 (0, 0, 0)。
    """

    world_origin: VehicleState
    ego: VehicleState
    goal: GoalState
    reference_path: FloatArray | None
    objects: tuple[PerceivedObject, ...]
    road_segments: tuple[PerceivedLaneSegment, ...]
    traffic_signals: tuple[PerceivedTrafficSignal, ...]

    def __post_init__(self) -> None:
        if (
            abs(self.ego.x) > 1e-12
            or abs(self.ego.y) > 1e-12
            or abs(self.ego.yaw) > 1e-12
        ):
            raise ValueError("local ego pose must be exactly (0, 0, 0)")

        if self.reference_path is not None:
            path = np.asarray(self.reference_path, dtype=np.float64)
            if path.ndim != 2 or path.shape[1] < 3:
                raise ValueError(
                    "reference_path must have shape [N, M], M >= 3, "
                    f"got {path.shape}"
                )
            if not np.all(np.isfinite(path)):
                raise ValueError("reference_path contains non-finite values")
            object.__setattr__(self, "reference_path", path.copy())


@dataclass(frozen=True)
class OptimizationResult:
    trajectory: SpatiotemporalTrajectory
    success: bool
    total_cost: float
    iterations: int
    status: str
    cost_terms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictedObjectTrajectory:
    object_id: str
    semantic_type: str
    times: FloatArray
    positions: FloatArray
    yaws: FloatArray
    speed: float
    length: float
    width: float
    radius: float | None = None


@dataclass(frozen=True)
class ObjectPredictionSet:
    times: FloatArray
    trajectories: tuple[PredictedObjectTrajectory, ...]
