from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim2d.models.kinematic_bicycle import (
    KinematicBicyclePredictionModel,
)
from sim2d.models.motion_model import (
    MotionModel,
)
from sim2d.planning.base import Planner
from sim2d.types import (
    GoalState,
    Observation,
    PlanResult,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


def normalize_angle(
    angle: float,
) -> float:
    """将角度归一化到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep(
    value: float,
) -> float:
    """将 [0, 1] 内的值平滑映射到 [0, 1]。"""
    value = max(
        0.0,
        min(value, 1.0),
    )

    return value * value * (3.0 - 2.0 * value)


class SimplePlanner(Planner):
    """
    基础目标状态跟踪规划器。

    主要功能：

    1. 远离目标时：
       朝目标位置行驶。

    2. 接近目标时：
       从目标点方向平滑过渡到目标最终航向。

    3. 根据剩余距离和制动能力计算速度上限：
       避免车辆高速穿过终点位置容差。

    4. 一旦进入终点进近区域：
       锁定终点模式，不再恢复普通追踪模式。

    5. 如果目标已经位于车辆后方：
       继续制动并对齐目标航向，不掉头追逐目标点。

    当前仍不考虑障碍物。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        target_speed: float = 4.0,
        steering_gain: float = 1.5,
        speed_gain: float = 1.0,
        slow_down_distance: float = 5.0,
        heading_transition_distance: float = 6.0,
        alignment_speed: float = 0.0,
        braking_deceleration: float = 2.5,
        stop_target_radius: float = 0.15,
        braking_margin: float = 0.20,
        prediction_dt: float = 0.1,
        prediction_steps: int = 30,
    ) -> None:
        vehicle_config.validate()

        if target_speed < 0.0:
            raise ValueError(
                "target_speed must be non-negative, " f"got {target_speed}"
            )

        if steering_gain <= 0.0:
            raise ValueError("steering_gain must be positive, " f"got {steering_gain}")

        if speed_gain <= 0.0:
            raise ValueError("speed_gain must be positive, " f"got {speed_gain}")

        if slow_down_distance <= 0.0:
            raise ValueError(
                "slow_down_distance must be positive, " f"got {slow_down_distance}"
            )

        if heading_transition_distance <= 0.0:
            raise ValueError(
                "heading_transition_distance must be positive, "
                f"got {heading_transition_distance}"
            )

        if alignment_speed < 0.0:
            raise ValueError(
                "alignment_speed must be non-negative, " f"got {alignment_speed}"
            )

        if braking_deceleration <= 0.0:
            raise ValueError(
                "braking_deceleration must be positive, " f"got {braking_deceleration}"
            )

        if stop_target_radius < 0.0:
            raise ValueError(
                "stop_target_radius must be non-negative, " f"got {stop_target_radius}"
            )

        if stop_target_radius >= slow_down_distance:
            raise ValueError(
                "stop_target_radius must be smaller than " "slow_down_distance"
            )

        if braking_margin < 0.0:
            raise ValueError(
                "braking_margin must be non-negative, " f"got {braking_margin}"
            )

        if prediction_dt <= 0.0:
            raise ValueError("prediction_dt must be positive, " f"got {prediction_dt}")

        if prediction_steps <= 0:
            raise ValueError(
                "prediction_steps must be positive, " f"got {prediction_steps}"
            )

        self.vehicle_config = vehicle_config

        self.target_speed = target_speed
        self.steering_gain = steering_gain
        self.speed_gain = speed_gain

        self.slow_down_distance = slow_down_distance

        self.heading_transition_distance = heading_transition_distance

        self.alignment_speed = alignment_speed

        self.braking_deceleration = braking_deceleration

        self.stop_target_radius = stop_target_radius

        self.braking_margin = braking_margin

        self.prediction_dt = prediction_dt
        self.prediction_steps = prediction_steps

        self._motion_model: MotionModel = KinematicBicyclePredictionModel(
            vehicle_config=vehicle_config,
        )

        # 一旦车辆进入终点减速区域，该状态会一直保持，
        # 直到 reset()。
        self._terminal_approach_active = False

    def reset(self) -> None:
        """重置终点进近状态。"""
        self._terminal_approach_active = False

    def plan(
        self,
        observation: Observation,
    ) -> PlanResult:
        """
        计算当前控制量和闭环预测轨迹。
        """
        action, control_debug = self._compute_action(
            ego=observation.ego,
            goal=observation.goal,
            terminal_approach_active=(self._terminal_approach_active),
        )

        # 只允许真实控制计算更新规划器内部状态。
        # 预测 rollout 使用局部状态，不影响真实规划器。
        self._terminal_approach_active = bool(control_debug["terminal_approach_active"])

        trajectory, controls = self._predict_trajectory(
            initial_state=observation.ego,
            goal=observation.goal,
            terminal_approach_active=(self._terminal_approach_active),
        )

        terminal_hold = bool(control_debug["terminal_hold"])

        position_reached = bool(control_debug["position_reached"])

        braking_active = bool(control_debug["braking_active"])

        if terminal_hold:
            status = "goal_hold"
        elif position_reached:
            status = "stopping_at_goal"
        elif braking_active:
            status = "braking_for_goal"
        elif self._terminal_approach_active:
            status = "terminal_approach"
        else:
            status = "tracking_goal"

        return PlanResult(
            action=action,
            trajectory=trajectory,
            controls=controls,
            debug={
                "planner": "SimplePlanner",
                "status": status,
                **control_debug,
                "prediction_dt": (self.prediction_dt),
                "prediction_steps": (self.prediction_steps),
                "prediction_horizon": (self.prediction_dt * self.prediction_steps),
                "prediction_mode": ("closed_loop"),
            },
        )

    def _compute_action(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
        terminal_approach_active: bool,
    ) -> tuple[
        VehicleControl,
        dict[str, Any],
    ]:
        goal_state = goal.state

        dx = goal_state.x - ego.x

        dy = goal_state.y - ego.y

        distance = math.hypot(
            dx,
            dy,
        )

        cos_yaw = math.cos(ego.yaw)

        sin_yaw = math.sin(ego.yaw)

        # 目标在车辆坐标系中的纵向距离：
        #
        # forward_error > 0：
        #     目标在车辆前方。
        #
        # forward_error < 0：
        #     目标已经位于车辆后方。
        forward_error = dx * cos_yaw + dy * sin_yaw

        lateral_error = -dx * sin_yaw + dy * cos_yaw

        goal_yaw_error = normalize_angle(goal_state.yaw - ego.yaw)

        speed_error_to_goal = goal_state.speed - ego.speed

        position_reached = distance <= goal.position_tolerance

        yaw_reached = abs(goal_yaw_error) <= goal.yaw_tolerance

        speed_reached = abs(speed_error_to_goal) <= goal.speed_tolerance

        terminal_hold = position_reached and yaw_reached and speed_reached

        # 一旦进入慢速终点区域，就锁定终点进近模式。
        terminal_approach_active = (
            terminal_approach_active or distance <= self.slow_down_distance
        )

        if terminal_hold:
            action = VehicleControl(
                acceleration=0.0,
                steering=0.0,
            )

            debug: dict[str, Any] = {
                "distance_to_goal": distance,
                "forward_error": forward_error,
                "lateral_error": lateral_error,
                "position_reached": True,
                "yaw_reached": True,
                "speed_reached": True,
                "terminal_hold": True,
                "terminal_approach_active": True,
                "target_behind": (forward_error < 0.0),
                "position_target_yaw": (goal_state.yaw),
                "target_yaw": (goal_state.yaw),
                "goal_yaw_error": (goal_yaw_error),
                "yaw_error": (goal_yaw_error),
                "desired_speed": (goal_state.speed),
                "remaining_distance": 0.0,
                "stopping_distance": 0.0,
                "braking_speed_limit": (goal_state.speed),
                "braking_active": False,
                "raw_acceleration": 0.0,
                "raw_steering": 0.0,
                "heading_blend": 1.0,
            }

            return action, debug

        target_behind = forward_error <= 0.0

        position_target_yaw = math.atan2(
            dy,
            dx,
        )

        if position_reached or (terminal_approach_active and target_behind):
            # 已进入目标容差，或者已经越过目标时：
            #
            # 不再使用 atan2(dy, dx) 追逐车后的目标点，
            # 否则会产生掉头和绕圈。
            target_yaw = goal_state.yaw
            heading_blend = 1.0
        else:
            heading_blend = self._calculate_heading_blend(distance=distance)

            target_yaw = self._blend_angles(
                start=position_target_yaw,
                end=goal_state.yaw,
                weight=heading_blend,
            )

        yaw_error = normalize_angle(target_yaw - ego.yaw)

        raw_steering = self.steering_gain * yaw_error

        steering = self._clamp(
            raw_steering,
            self.vehicle_config.steering_min,
            self.vehicle_config.steering_max,
        )

        (
            desired_speed,
            remaining_distance,
            stopping_distance,
            braking_speed_limit,
            braking_active,
        ) = self._calculate_speed_control_state(
            ego=ego,
            goal=goal,
            distance=distance,
            position_reached=(position_reached),
            target_behind=target_behind,
            terminal_approach_active=(terminal_approach_active),
        )

        raw_acceleration = self.speed_gain * (desired_speed - ego.speed)

        if braking_active:
            # 到达制动边界后直接使用确定的制动力，
            # 不再只依赖速度比例控制器慢慢减速。
            acceleration = max(
                -self.braking_deceleration,
                self.vehicle_config.acceleration_min,
            )
        else:
            acceleration = self._clamp(
                raw_acceleration,
                self.vehicle_config.acceleration_min,
                self.vehicle_config.acceleration_max,
            )

        # 已经位于终点容差中时，只允许继续减速，
        # 不允许重新加速离开终点。
        if position_reached and acceleration > 0.0:
            acceleration = 0.0

        # 已经越过目标后，同样只允许制动或保持，
        # 禁止重新加速进行掉头追逐。
        if terminal_approach_active and target_behind and acceleration > 0.0:
            acceleration = 0.0

        action = VehicleControl(
            acceleration=acceleration,
            steering=steering,
        )

        debug = {
            "distance_to_goal": distance,
            "forward_error": forward_error,
            "lateral_error": lateral_error,
            "position_reached": (position_reached),
            "yaw_reached": (yaw_reached),
            "speed_reached": (speed_reached),
            "terminal_hold": False,
            "terminal_approach_active": (terminal_approach_active),
            "target_behind": (target_behind),
            "position_target_yaw": (position_target_yaw),
            "target_yaw": (target_yaw),
            "goal_yaw_error": (goal_yaw_error),
            "yaw_error": (yaw_error),
            "desired_speed": (desired_speed),
            "remaining_distance": (remaining_distance),
            "stopping_distance": (stopping_distance),
            "braking_speed_limit": (braking_speed_limit),
            "braking_active": (braking_active),
            "raw_acceleration": (raw_acceleration),
            "raw_steering": (raw_steering),
            "heading_blend": (heading_blend),
        }

        return action, debug

    def _calculate_speed_control_state(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
        distance: float,
        position_reached: bool,
        target_behind: bool,
        terminal_approach_active: bool,
    ) -> tuple[
        float,
        float,
        float,
        float,
        bool,
    ]:
        """
        根据制动距离计算期望速度及制动状态。

        返回：
            desired_speed
            remaining_distance
            stopping_distance
            braking_speed_limit
            braking_active
        """
        goal_speed = max(
            goal.state.speed,
            self.vehicle_config.speed_min,
        )

        # 希望车辆停在目标中心附近，而不是只要进入
        # position_tolerance 边缘就算完成减速规划。
        remaining_distance = max(
            distance - self.stop_target_radius,
            0.0,
        )

        braking_speed_limit = math.sqrt(
            max(
                goal_speed * goal_speed
                + 2.0 * self.braking_deceleration * remaining_distance,
                0.0,
            )
        )

        if ego.speed > goal_speed:
            stopping_distance = (ego.speed * ego.speed - goal_speed * goal_speed) / (
                2.0 * self.braking_deceleration
            )
        else:
            stopping_distance = 0.0

        dynamic_margin = max(
            self.braking_margin,
            ego.speed * self.prediction_dt,
        )

        braking_active = ego.speed > goal_speed and (
            stopping_distance + dynamic_margin >= remaining_distance
        )

        if position_reached:
            # 一旦进入位置容差，目标速度直接采用终端速度。
            desired_speed = goal_speed
            braking_active = ego.speed > goal_speed + 1e-6

        elif terminal_approach_active and target_behind:
            # 车辆已经越过目标后禁止恢复巡航速度。
            desired_speed = goal_speed
            braking_active = ego.speed > goal_speed + 1e-6

        elif distance <= self.slow_down_distance:
            desired_speed = min(
                self.target_speed,
                braking_speed_limit,
            )

        else:
            desired_speed = self.target_speed

        desired_speed = self._clamp(
            desired_speed,
            self.vehicle_config.speed_min,
            self.vehicle_config.speed_max,
        )

        return (
            desired_speed,
            remaining_distance,
            stopping_distance,
            braking_speed_limit,
            braking_active,
        )

    def _calculate_heading_blend(
        self,
        *,
        distance: float,
    ) -> float:
        """
        计算目标点方向与最终目标航向之间的混合权重。

        距离较远：
            权重接近 0，主要追踪目标位置。

        距离较近：
            权重逐渐接近 1，主要对齐目标 yaw。
        """
        ratio = 1.0 - distance / self.heading_transition_distance

        return _smoothstep(ratio)

    def _predict_trajectory(
        self,
        *,
        initial_state: VehicleState,
        goal: GoalState,
        terminal_approach_active: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        使用闭环控制生成预测轨迹。

        预测中的终点进近状态只在当前 rollout 内更新，
        不修改规划器真实内部状态。
        """
        predicted_states = [initial_state.as_array()]

        predicted_controls: list[np.ndarray] = []

        predicted_state = initial_state

        predicted_terminal_mode = terminal_approach_active

        for _ in range(self.prediction_steps):
            (
                predicted_action,
                debug,
            ) = self._compute_action(
                ego=predicted_state,
                goal=goal,
                terminal_approach_active=(predicted_terminal_mode),
            )

            predicted_terminal_mode = bool(debug["terminal_approach_active"])

            predicted_controls.append(predicted_action.as_array())

            if bool(debug["terminal_hold"]):
                predicted_states.append(predicted_state.as_array())
                continue

            predicted_state = self._motion_model.predict_step(
                state=predicted_state,
                control=predicted_action,
                dt=self.prediction_dt,
            )

            predicted_states.append(predicted_state.as_array())

        trajectory = np.asarray(
            predicted_states,
            dtype=np.float64,
        )

        controls = np.asarray(
            predicted_controls,
            dtype=np.float64,
        )

        return trajectory, controls

    @staticmethod
    def _blend_angles(
        *,
        start: float,
        end: float,
        weight: float,
    ) -> float:
        """沿最短角度方向混合两个航向角。"""
        weight = max(
            0.0,
            min(weight, 1.0),
        )

        difference = normalize_angle(end - start)

        return normalize_angle(start + weight * difference)

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                value,
                maximum,
            ),
        )
