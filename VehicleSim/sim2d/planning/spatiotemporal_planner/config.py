from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatiotemporalPlannerConfig:
    # 离散时间
    dt: float = 0.1
    horizon_steps: int = 50

    # 巡航目标
    target_speed: float = 10.0

    # 控制变化限制
    max_acceleration_rate: float = 3.0
    max_steering_rate: float = 0.5

    # 安全距离
    pedestrian_margin: float = 1.5
    small_car_margin: float = 1.2
    large_vehicle_margin: float = 1.8

    # 代价权重
    weight_reference: float = 2.0
    weight_heading: float = 1.0
    weight_speed: float = 1.0
    weight_acceleration: float = 0.1
    weight_steering: float = 0.1
    weight_acceleration_rate: float = 0.5
    weight_steering_rate: float = 1.0
    weight_collision: float = 1000.0
    weight_terminal: float = 5.0

    # 数值优化
    max_iterations: int = 40
    gradient_epsilon: float = 1e-3
    initial_step_size: float = 0.05
    # 碰撞软约束
    collision_softness: float = 0.5

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

        if not isinstance(self.horizon_steps, int):
            raise TypeError("horizon_steps must be an integer")

        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")

        if self.target_speed < 0.0:
            raise ValueError("target_speed must be non-negative")

        if self.collision_softness <= 0.0:
            raise ValueError("collision_softness must be positive")
