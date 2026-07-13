from __future__ import annotations

import math
from dataclasses import dataclass

from sim2d.types import (
    VehicleConfig,
    VehicleControl,
)


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(value, maximum),
    )


@dataclass(frozen=True)
class ManualInputState:
    """
    驾驶员输入状态。

    steering:
        归一化方向盘输入，范围 [-1, 1]。

        -1：向右打满
         0：方向盘居中
        +1：向左打满

    throttle:
        油门踏板输入，范围 [0, 1]。

    brake:
        制动踏板输入，范围 [0, 1]。
    """

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def validate(self) -> None:
        values = {
            "steering": self.steering,
            "throttle": self.throttle,
            "brake": self.brake,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, " f"got {value}")

        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("steering must be within " f"[-1, 1], got {self.steering}")

        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be within " f"[0, 1], got {self.throttle}")

        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("brake must be within " f"[0, 1], got {self.brake}")


class ManualControlMapper:
    """
    将方向盘、油门和制动输入映射为车辆控制量。

    规则：

    1. 方向盘输入线性映射为前轮转角。
    2. 油门映射为正加速度。
    3. 制动映射为负加速度。
    4. 油门和制动同时存在时，制动优先。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
    ) -> None:
        vehicle_config.validate()

        self.vehicle_config = vehicle_config

    def map(
        self,
        input_state: ManualInputState,
    ) -> VehicleControl:
        input_state.validate()

        steering = self._map_steering(input_state.steering)

        acceleration = self._map_longitudinal(
            throttle=input_state.throttle,
            brake=input_state.brake,
        )

        return VehicleControl(
            acceleration=acceleration,
            steering=steering,
        )

    def _map_steering(
        self,
        steering_input: float,
    ) -> float:
        if steering_input >= 0.0:
            steering = steering_input * self.vehicle_config.steering_max
        else:
            steering = -steering_input * self.vehicle_config.steering_min

        return _clamp(
            steering,
            self.vehicle_config.steering_min,
            self.vehicle_config.steering_max,
        )

    def _map_longitudinal(
        self,
        *,
        throttle: float,
        brake: float,
    ) -> float:
        if brake > 0.0:
            acceleration = brake * self.vehicle_config.acceleration_min
        else:
            acceleration = throttle * self.vehicle_config.acceleration_max

        return _clamp(
            acceleration,
            self.vehicle_config.acceleration_min,
            self.vehicle_config.acceleration_max,
        )
