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

    # 参考线 Frenet 跟踪、车身走廊与弯道速度
    corridor_vehicle_margin: float = 0.20
    corridor_max_violation_multiplier: float = 20.0
    corridor_barrier_threshold: float = 0.05
    corridor_barrier_multiplier: float = 200.0
    maximum_lateral_acceleration: float = 3.0
    curve_speed_lateral_acceleration: float = 2.5
    minimum_curve_speed: float = 1.0
    frenet_nonmonotonic_tolerance: float = 0.05

    # 高效参考线前馈与可行性恢复
    feedforward_lookahead_base: float = 2.0
    feedforward_lookahead_speed_gain: float = 0.35
    feedforward_lookahead_max: float = 8.0
    feedforward_curve_preview_distance: float = 12.0
    feedforward_speed_response_time: float = 0.6
    feedforward_pure_pursuit_weight: float = 0.70
    warm_start_max_violation: float = 0.15
    result_max_violation: float = 0.10
    fallback_violation_improvement: float = 0.05

    # 代价权重
    weight_reference: float = 3.0
    weight_heading: float = 1.5
    weight_speed: float = 1.0
    weight_acceleration: float = 0.1
    weight_steering: float = 0.1
    weight_acceleration_rate: float = 0.5
    weight_steering_rate: float = 1.0
    weight_collision: float = 1000.0
    weight_terminal: float = 5.0
    weight_corridor: float = 40.0
    weight_frenet_progress: float = 8.0
    weight_lateral_acceleration: float = 4.0

    # 数值优化
    max_iterations: int = 40
    gradient_epsilon: float = 1e-3
    initial_step_size: float = 0.05

    # 碰撞软约束
    collision_softness: float = 0.5

    # 优化收敛与线搜索
    gradient_tolerance: float = 1e-4
    cost_tolerance: float = 1e-6
    line_search_decay: float = 0.5
    min_step_size: float = 1e-6

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
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.gradient_epsilon <= 0.0:
            raise ValueError("gradient_epsilon must be positive")
        if self.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be positive")
        if self.gradient_tolerance < 0.0:
            raise ValueError("gradient_tolerance must be non-negative")
        if self.cost_tolerance < 0.0:
            raise ValueError("cost_tolerance must be non-negative")
        if not 0.0 < self.line_search_decay < 1.0:
            raise ValueError("line_search_decay must be within (0, 1)")
        if self.min_step_size <= 0.0:
            raise ValueError("min_step_size must be positive")
        if self.min_step_size > self.initial_step_size:
            raise ValueError("min_step_size must not exceed initial_step_size")
        if self.corridor_vehicle_margin < 0.0:
            raise ValueError("corridor_vehicle_margin must be non-negative")
        if self.corridor_max_violation_multiplier < 0.0:
            raise ValueError("corridor_max_violation_multiplier must be non-negative")
        if self.corridor_barrier_threshold < 0.0:
            raise ValueError("corridor_barrier_threshold must be non-negative")
        if self.corridor_barrier_multiplier < 0.0:
            raise ValueError("corridor_barrier_multiplier must be non-negative")
        if self.maximum_lateral_acceleration <= 0.0:
            raise ValueError("maximum_lateral_acceleration must be positive")
        if self.curve_speed_lateral_acceleration <= 0.0:
            raise ValueError("curve_speed_lateral_acceleration must be positive")
        if self.minimum_curve_speed < 0.0:
            raise ValueError("minimum_curve_speed must be non-negative")
        if self.minimum_curve_speed > self.target_speed:
            raise ValueError("minimum_curve_speed must not exceed target_speed")
        if self.frenet_nonmonotonic_tolerance < 0.0:
            raise ValueError("frenet_nonmonotonic_tolerance must be non-negative")

        positive_values = (
            self.feedforward_lookahead_base,
            self.feedforward_lookahead_max,
            self.feedforward_curve_preview_distance,
            self.feedforward_speed_response_time,
        )
        if min(positive_values) <= 0.0:
            raise ValueError("feedforward distances and response time must be positive")
        if self.feedforward_lookahead_speed_gain < 0.0:
            raise ValueError("feedforward_lookahead_speed_gain must be non-negative")
        if self.feedforward_lookahead_max < self.feedforward_lookahead_base:
            raise ValueError("feedforward_lookahead_max must not be smaller than base")
        if not 0.0 <= self.feedforward_pure_pursuit_weight <= 1.0:
            raise ValueError("feedforward_pure_pursuit_weight must be within [0, 1]")
        feasibility_values = (
            self.warm_start_max_violation,
            self.result_max_violation,
            self.fallback_violation_improvement,
        )
        if min(feasibility_values) < 0.0:
            raise ValueError("feasibility thresholds must be non-negative")

        weights = (
            self.weight_reference,
            self.weight_heading,
            self.weight_speed,
            self.weight_acceleration,
            self.weight_steering,
            self.weight_acceleration_rate,
            self.weight_steering_rate,
            self.weight_collision,
            self.weight_terminal,
            self.weight_corridor,
            self.weight_frenet_progress,
            self.weight_lateral_acceleration,
        )
        if min(weights) < 0.0:
            raise ValueError("cost weights must be non-negative")
