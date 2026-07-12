from __future__ import annotations

import math
from dataclasses import dataclass

from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


@dataclass(frozen=True)
class KinematicBicycleModel:
    """
    运动学自行车模型。

    状态：
        x, y, yaw, speed

    控制：
        acceleration, steering

    坐标约定：
        yaw = 0 时车头朝 +x
        yaw 正方向为逆时针
    """

    config: VehicleConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def step(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        """
        将车辆状态推进一个固定时间步。

        参数：
            state:
                当前车辆状态。

            control:
                当前控制输入。

            dt:
                仿真步长，单位秒。

        返回：
            下一时刻车辆状态。
        """
        if dt <= 0.0:
            raise ValueError(
                f"dt must be positive, got {dt}"
            )

        acceleration = self._clamp(
            control.acceleration,
            self.config.acceleration_min,
            self.config.acceleration_max,
        )

        steering = self._clamp(
            control.steering,
            self.config.steering_min,
            self.config.steering_max,
        )

        speed = self._clamp(
            state.speed,
            self.config.speed_min,
            self.config.speed_max,
        )

        next_x = (
            state.x
            + speed * math.cos(state.yaw) * dt
        )

        next_y = (
            state.y
            + speed * math.sin(state.yaw) * dt
        )

        yaw_rate = (
            speed
            / self.config.wheel_base
            * math.tan(steering)
        )

        next_yaw = state.yaw + yaw_rate * dt

        next_speed = speed + acceleration * dt
        next_speed = self._clamp(
            next_speed,
            self.config.speed_min,
            self.config.speed_max,
        )

        next_yaw = self.normalize_angle(next_yaw)

        return VehicleState(
            x=next_x,
            y=next_y,
            yaw=next_yaw,
            speed=next_speed,
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        将角度归一化到 [-pi, pi)。
        """
        return (angle + math.pi) % (
            2.0 * math.pi
        ) - math.pi

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(value, upper))