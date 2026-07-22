from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
            raise ValueError(
                "controls must have shape [N, 2], "
                f"got {value.shape}"
            )

        if not np.all(np.isfinite(value)):
            raise ValueError("controls contain non-finite values")

        object.__setattr__(self, "controls", value.copy())


@dataclass(frozen=True)
class SpatiotemporalTrajectory:
    """
    时空联合轨迹。

    states[k] = [x, y, yaw, speed]
    times[k] = 相对于当前规划时刻的时间
    controls[k] = [acceleration, steering]

    states 和 times 的长度为 N + 1，controls 的长度为 N。
    """

    times: FloatArray
    states: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        controls = np.asarray(self.controls, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(
                "times must have shape [N], "
                f"got {times.shape}"
            )

        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError(
                "states must have shape [N, 4], "
                f"got {states.shape}"
            )

        if controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError(
                "controls must have shape [N-1, 2], "
                f"got {controls.shape}"
            )

        if states.shape[0] != times.shape[0]:
            raise ValueError(
                "states and times must have equal length"
            )

        if controls.shape[0] != states.shape[0] - 1:
            raise ValueError(
                "controls length must be states length minus one"
            )

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
class OptimizationResult:
    trajectory: SpatiotemporalTrajectory
    success: bool
    total_cost: float
    iterations: int
    status: str
    cost_terms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
