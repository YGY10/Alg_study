from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim2d.perception import (
    PerceivedLaneLine,
    PerceivedObject,
    PerceivedTrafficSignal,
)
from sim2d.types import GoalState, VehicleState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControlSequence:
    """优化器控制变量，controls[k] = [acceleration, steering]。"""

    controls: FloatArray

    def __post_init__(self) -> None:
        value = np.asarray(self.controls, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(
                "controls must have shape [N, 2], "
                f"got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("controls contain non-finite values")
        object.__setattr__(self, "controls", value.copy())


@dataclass(frozen=True)
class SpatiotemporalTrajectory:
    """规划时刻冻结自车坐标系中的时空联合轨迹。"""

    times: FloatArray
    states: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        controls = np.asarray(self.controls, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(f"times must have shape [N], got {times.shape}")
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError(f"states must have shape [N, 4], got {states.shape}")
        if controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError(
                "controls must have shape [N-1, 2], "
                f"got {controls.shape}"
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
    """PNC 内部统一使用的冻结自车坐标上下文。"""

    world_origin: VehicleState
    ego: VehicleState
    goal: GoalState
    reference_path: FloatArray | None
    objects: tuple[PerceivedObject, ...]
    lane_lines: tuple[PerceivedLaneLine, ...]
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
    """单个交通参与者在冻结自车坐标系中的恒速预测。"""

    object_id: str
    object_type: str
    semantic_type: str
    times: FloatArray
    positions: FloatArray
    yaws: FloatArray
    speed: float
    length: float
    width: float
    confidence: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        yaws = np.asarray(self.yaws, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(f"times must have shape [N], got {times.shape}")
        if times.size == 0:
            raise ValueError("times must not be empty")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "positions must have shape [N, 2], "
                f"got {positions.shape}"
            )
        if yaws.ndim != 1:
            raise ValueError(f"yaws must have shape [N], got {yaws.shape}")
        if positions.shape[0] != times.shape[0]:
            raise ValueError("positions and times must have equal length")
        if yaws.shape[0] != times.shape[0]:
            raise ValueError("yaws and times must have equal length")
        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions contain non-finite values")
        if not np.all(np.isfinite(yaws)):
            raise ValueError("yaws contain non-finite values")
        if abs(float(times[0])) > 1e-12:
            raise ValueError("times must start at zero")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing")

        metadata = np.asarray(
            [self.speed, self.length, self.width, self.confidence],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(metadata)):
            raise ValueError("prediction metadata contains non-finite values")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("prediction dimensions must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "yaws", yaws.copy())


@dataclass(frozen=True)
class ObjectPredictionSet:
    times: FloatArray
    trajectories: tuple[PredictedObjectTrajectory, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must have shape [N], N > 0")
        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")
        if abs(float(times[0])) > 1e-12:
            raise ValueError("times must start at zero")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing")
        for trajectory in self.trajectories:
            if trajectory.times.shape != times.shape or not np.allclose(
                trajectory.times, times, rtol=0.0, atol=1e-12
            ):
                raise ValueError("all predictions must use the common time axis")
        object.__setattr__(self, "times", times.copy())


__all__ = [
    "ControlSequence",
    "LocalPlanningContext",
    "ObjectPredictionSet",
    "OptimizationResult",
    "PredictedObjectTrajectory",
    "SpatiotemporalTrajectory",
]
