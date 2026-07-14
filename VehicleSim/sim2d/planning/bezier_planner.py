from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim2d.core.dynamics import (
    KinematicBicycleModel,
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


class BezierPlanner(Planner):
    """
    三次 Bézier 路径规划器。

    规划流程：

        当前车辆状态
        + 目标位置、目标航向
                ↓
        生成满足起终点位置与航向约束的
        三次 Bézier 参考路径
                ↓
        Pure Pursuit 跟踪参考路径
                ↓
        根据剩余弧长规划期望速度
                ↓
        运动学自行车模型闭环预测

    坐标约定：

        车辆前方：车体 +x
        车辆左侧：车体 +y
        正转角：左转
        正 yaw：逆时针

    当前版本暂不进行障碍物绕行。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        target_speed: float = 4.0,
        speed_gain: float = 1.5,
        braking_deceleration: float = 2.5,
        path_sample_count: int = 201,
        handle_scale: float = 0.35,
        minimum_handle_length: float = 2.0,
        maximum_handle_length: float = 12.0,
        lookahead_base: float = 2.0,
        lookahead_speed_gain: float = 0.45,
        minimum_lookahead: float = 1.5,
        stop_target_radius: float = 0.10,
        braking_margin: float = 0.20,
        prediction_dt: float = 0.1,
        prediction_steps: int = 50,
    ) -> None:
        vehicle_config.validate()

        if target_speed < 0.0:
            raise ValueError(
                "target_speed must be non-negative, " f"got {target_speed}"
            )

        if speed_gain <= 0.0:
            raise ValueError("speed_gain must be positive, " f"got {speed_gain}")

        if braking_deceleration <= 0.0:
            raise ValueError(
                "braking_deceleration must be positive, " f"got {braking_deceleration}"
            )

        if path_sample_count < 2:
            raise ValueError(
                "path_sample_count must be at least 2, " f"got {path_sample_count}"
            )

        if handle_scale <= 0.0:
            raise ValueError("handle_scale must be positive, " f"got {handle_scale}")

        if minimum_handle_length <= 0.0:
            raise ValueError(
                "minimum_handle_length must be positive, "
                f"got {minimum_handle_length}"
            )

        if maximum_handle_length < minimum_handle_length:
            raise ValueError(
                "maximum_handle_length must not be smaller "
                "than minimum_handle_length"
            )

        if lookahead_base <= 0.0:
            raise ValueError(
                "lookahead_base must be positive, " f"got {lookahead_base}"
            )

        if lookahead_speed_gain < 0.0:
            raise ValueError(
                "lookahead_speed_gain must be non-negative, "
                f"got {lookahead_speed_gain}"
            )

        if minimum_lookahead <= 0.0:
            raise ValueError(
                "minimum_lookahead must be positive, " f"got {minimum_lookahead}"
            )

        if stop_target_radius < 0.0:
            raise ValueError(
                "stop_target_radius must be non-negative, " f"got {stop_target_radius}"
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
        self.speed_gain = speed_gain

        self.braking_deceleration = braking_deceleration

        self.path_sample_count = path_sample_count

        self.handle_scale = handle_scale

        self.minimum_handle_length = minimum_handle_length

        self.maximum_handle_length = maximum_handle_length

        self.lookahead_base = lookahead_base

        self.lookahead_speed_gain = lookahead_speed_gain

        self.minimum_lookahead = minimum_lookahead

        self.stop_target_radius = stop_target_radius

        self.braking_margin = braking_margin

        self.prediction_dt = prediction_dt

        self.prediction_steps = prediction_steps

        self._dynamics = KinematicBicycleModel(config=vehicle_config)

        self._reference_path: np.ndarray | None = None

        self._path_goal_signature: tuple[float, ...] | None = None

        self._nearest_index = 0

    def reset(self) -> None:
        """
        清除当前参考路径。

        下一次调用 plan() 时，会根据新的初始状态和
        目标状态重新生成 Bézier 路径。
        """
        self._reference_path = None
        self._path_goal_signature = None
        self._nearest_index = 0

    def plan(
        self,
        observation: Observation,
    ) -> PlanResult:
        """
        规划当前控制量和未来闭环预测轨迹。
        """
        reference_path = self._ensure_reference_path(
            ego=observation.ego,
            goal=observation.goal,
        )

        (
            action,
            control_debug,
            nearest_index,
        ) = self._compute_control(
            ego=observation.ego,
            goal=observation.goal,
            reference_path=reference_path,
            nearest_index_hint=(self._nearest_index),
        )

        self._nearest_index = max(
            self._nearest_index,
            nearest_index,
        )

        (
            predicted_trajectory,
            predicted_controls,
        ) = self._predict_trajectory(
            initial_state=observation.ego,
            goal=observation.goal,
            reference_path=reference_path,
            nearest_index=nearest_index,
        )

        if bool(control_debug["terminal_hold"]):
            status = "goal_hold"
        elif bool(control_debug["braking_active"]):
            status = "braking_for_goal"
        else:
            status = "tracking_bezier_path"

        return PlanResult(
            action=action,
            trajectory=predicted_trajectory,
            controls=predicted_controls,
            # 完整 Bézier 参考路径：
            # [x, y, yaw, arc_length, curvature]
            reference_path=reference_path.copy(),
            debug={
                "planner": "BezierPlanner",
                "status": status,
                **control_debug,
                "prediction_dt": self.prediction_dt,
                "prediction_steps": self.prediction_steps,
                "prediction_horizon": (self.prediction_dt * self.prediction_steps),
                "prediction_mode": ("closed_loop_path_tracking"),
                "reference_path_points": (reference_path.shape[0]),
            },
        )

    def _ensure_reference_path(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
    ) -> np.ndarray:
        """
        确保当前 episode 已经生成参考路径。

        路径只在以下情况重新生成：

        - reset() 后第一次规划；
        - 目标状态发生变化。
        """
        goal_signature = (
            goal.state.x,
            goal.state.y,
            goal.state.yaw,
            goal.state.speed,
            goal.position_tolerance,
            goal.yaw_tolerance,
            goal.speed_tolerance,
        )

        if self._reference_path is None or self._path_goal_signature != goal_signature:
            self._reference_path = self._generate_bezier_path(
                start=ego,
                goal=goal.state,
            )

            self._path_goal_signature = goal_signature

            self._nearest_index = 0

        return self._reference_path

    def _generate_bezier_path(
        self,
        *,
        start: VehicleState,
        goal: VehicleState,
    ) -> np.ndarray:
        """
        生成三次 Bézier 参考路径。

        输出列：

            0: x
            1: y
            2: yaw
            3: 累计弧长 s
            4: 曲率 curvature
        """
        p0 = np.array(
            [
                start.x,
                start.y,
            ],
            dtype=np.float64,
        )

        p3 = np.array(
            [
                goal.x,
                goal.y,
            ],
            dtype=np.float64,
        )

        straight_distance = float(np.linalg.norm(p3 - p0))

        handle_length = self._clamp(
            straight_distance * self.handle_scale,
            self.minimum_handle_length,
            self.maximum_handle_length,
        )

        start_direction = np.array(
            [
                math.cos(start.yaw),
                math.sin(start.yaw),
            ],
            dtype=np.float64,
        )

        goal_direction = np.array(
            [
                math.cos(goal.yaw),
                math.sin(goal.yaw),
            ],
            dtype=np.float64,
        )

        p1 = p0 + handle_length * start_direction

        p2 = p3 - handle_length * goal_direction

        t = np.linspace(
            0.0,
            1.0,
            self.path_sample_count,
            dtype=np.float64,
        )

        one_minus_t = 1.0 - t

        points = (
            one_minus_t[:, None] ** 3 * p0[None, :]
            + 3.0 * one_minus_t[:, None] ** 2 * t[:, None] * p1[None, :]
            + 3.0 * one_minus_t[:, None] * t[:, None] ** 2 * p2[None, :]
            + t[:, None] ** 3 * p3[None, :]
        )

        first_derivative = (
            3.0 * one_minus_t[:, None] ** 2 * (p1 - p0)[None, :]
            + 6.0 * one_minus_t[:, None] * t[:, None] * (p2 - p1)[None, :]
            + 3.0 * t[:, None] ** 2 * (p3 - p2)[None, :]
        )

        second_derivative = (
            6.0 * one_minus_t[:, None] * (p2 - 2.0 * p1 + p0)[None, :]
            + 6.0 * t[:, None] * (p3 - 2.0 * p2 + p1)[None, :]
        )

        yaw = np.arctan2(
            first_derivative[:, 1],
            first_derivative[:, 0],
        )

        segment_length = np.linalg.norm(
            np.diff(
                points,
                axis=0,
            ),
            axis=1,
        )

        arc_length = np.concatenate(
            [
                np.array(
                    [0.0],
                    dtype=np.float64,
                ),
                np.cumsum(segment_length),
            ]
        )

        dx_dt = first_derivative[:, 0]
        dy_dt = first_derivative[:, 1]

        ddx_dt = second_derivative[:, 0]

        ddy_dt = second_derivative[:, 1]

        curvature_denominator = (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

        curvature_numerator = dx_dt * ddy_dt - dy_dt * ddx_dt

        curvature = np.divide(
            curvature_numerator,
            curvature_denominator,
            out=np.zeros_like(curvature_numerator),
            where=(curvature_denominator > 1e-9),
        )

        path = np.column_stack(
            [
                points[:, 0],
                points[:, 1],
                yaw,
                arc_length,
                curvature,
            ]
        )

        return path.astype(
            np.float64,
            copy=False,
        )

    def _compute_control(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
        reference_path: np.ndarray,
        nearest_index_hint: int,
    ) -> tuple[
        VehicleControl,
        dict[str, Any],
        int,
    ]:
        """
        根据参考路径计算横向和纵向控制。
        """
        nearest_index = self._find_nearest_index(
            state=ego,
            reference_path=reference_path,
            start_index=nearest_index_hint,
        )

        lookahead_distance = max(
            self.minimum_lookahead,
            self.lookahead_base + self.lookahead_speed_gain * ego.speed,
        )

        target_index = self._find_lookahead_index(
            reference_path=reference_path,
            nearest_index=nearest_index,
            lookahead_distance=(lookahead_distance),
        )

        target_x = float(
            reference_path[
                target_index,
                0,
            ]
        )

        target_y = float(
            reference_path[
                target_index,
                1,
            ]
        )

        dx = target_x - ego.x
        dy = target_y - ego.y

        actual_lookahead = max(
            math.hypot(
                dx,
                dy,
            ),
            1e-6,
        )

        target_heading = math.atan2(
            dy,
            dx,
        )

        alpha = normalize_angle(target_heading - ego.yaw)

        raw_steering = math.atan2(
            2.0 * self.vehicle_config.wheel_base * math.sin(alpha),
            actual_lookahead,
        )

        steering = self._clamp(
            raw_steering,
            self.vehicle_config.steering_min,
            self.vehicle_config.steering_max,
        )

        (
            desired_speed,
            remaining_length,
            stopping_distance,
            braking_speed_limit,
            braking_active,
        ) = self._calculate_speed_state(
            ego=ego,
            goal=goal,
            reference_path=reference_path,
            nearest_index=nearest_index,
        )

        raw_acceleration = self.speed_gain * (desired_speed - ego.speed)

        if braking_active:
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

        (
            position_error,
            yaw_error,
            speed_error,
            position_reached,
            yaw_reached,
            speed_reached,
        ) = self._goal_error(
            state=ego,
            goal=goal,
        )

        terminal_hold = position_reached and yaw_reached and speed_reached

        if terminal_hold:
            acceleration = 0.0
            steering = 0.0

        action = VehicleControl(
            acceleration=acceleration,
            steering=steering,
        )

        nearest_path_x = float(
            reference_path[
                nearest_index,
                0,
            ]
        )

        nearest_path_y = float(
            reference_path[
                nearest_index,
                1,
            ]
        )

        cross_track_error = math.hypot(
            ego.x - nearest_path_x,
            ego.y - nearest_path_y,
        )

        debug: dict[str, Any] = {
            "nearest_index": (nearest_index),
            "target_index": (target_index),
            "lookahead_distance": (lookahead_distance),
            "actual_lookahead": (actual_lookahead),
            "target_x": target_x,
            "target_y": target_y,
            "target_heading": (target_heading),
            "alpha": alpha,
            "raw_steering": (raw_steering),
            "cross_track_error": (cross_track_error),
            "desired_speed": (desired_speed),
            "remaining_length": (remaining_length),
            "stopping_distance": (stopping_distance),
            "braking_speed_limit": (braking_speed_limit),
            "braking_active": (braking_active),
            "raw_acceleration": (raw_acceleration),
            "position_error": (position_error),
            "yaw_error": (yaw_error),
            "speed_error": (speed_error),
            "position_reached": (position_reached),
            "yaw_reached": (yaw_reached),
            "speed_reached": (speed_reached),
            "terminal_hold": (terminal_hold),
        }

        return (
            action,
            debug,
            nearest_index,
        )

    def _calculate_speed_state(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
        reference_path: np.ndarray,
        nearest_index: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
        bool,
    ]:
        """
        根据剩余路径弧长生成速度目标。

        这里规划的是：

            当前速度
            → 沿参考路径运动
            → 收敛到 goal.speed

        当 goal.speed == 0 时：
            在路径终点附近停车。

        当 goal.speed > 0 时：
            以目标速度通过终点状态。
        """
        total_length = float(
            reference_path[
                -1,
                3,
            ]
        )

        current_arc_length = float(
            reference_path[
                nearest_index,
                3,
            ]
        )

        remaining_length = max(
            total_length - current_arc_length - self.stop_target_radius,
            0.0,
        )

        goal_speed = self._clamp(
            goal.state.speed,
            self.vehicle_config.speed_min,
            self.vehicle_config.speed_max,
        )

        braking_speed_limit = math.sqrt(
            max(
                goal_speed * goal_speed
                + 2.0 * self.braking_deceleration * remaining_length,
                0.0,
            )
        )

        desired_speed = min(
            self.target_speed,
            braking_speed_limit,
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

        braking_active = ego.speed > goal_speed + 1e-6 and (
            stopping_distance + dynamic_margin >= remaining_length
        )

        desired_speed = self._clamp(
            desired_speed,
            self.vehicle_config.speed_min,
            self.vehicle_config.speed_max,
        )

        return (
            desired_speed,
            remaining_length,
            stopping_distance,
            braking_speed_limit,
            braking_active,
        )

    def _find_nearest_index(
        self,
        *,
        state: VehicleState,
        reference_path: np.ndarray,
        start_index: int,
    ) -> int:
        """
        从 start_index 开始向路径终点搜索最近点。

        索引只向前推进，避免控制器在路径上倒退。
        """
        start_index = int(
            self._clamp(
                float(start_index),
                0.0,
                float(reference_path.shape[0] - 1),
            )
        )

        path_xy = reference_path[
            start_index:,
            0:2,
        ]

        dx = path_xy[:, 0] - state.x
        dy = path_xy[:, 1] - state.y

        local_index = int(np.argmin(dx * dx + dy * dy))

        return start_index + local_index

    @staticmethod
    def _find_lookahead_index(
        *,
        reference_path: np.ndarray,
        nearest_index: int,
        lookahead_distance: float,
    ) -> int:
        target_arc_length = (
            float(
                reference_path[
                    nearest_index,
                    3,
                ]
            )
            + lookahead_distance
        )

        target_index = int(
            np.searchsorted(
                reference_path[:, 3],
                target_arc_length,
                side="left",
            )
        )

        return min(
            target_index,
            reference_path.shape[0] - 1,
        )

    def _predict_trajectory(
        self,
        *,
        initial_state: VehicleState,
        goal: GoalState,
        reference_path: np.ndarray,
        nearest_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        使用同一条 Bézier 路径进行闭环预测。
        """
        predicted_states = [initial_state.as_array()]

        predicted_controls: list[np.ndarray] = []

        predicted_state = initial_state

        predicted_nearest_index = nearest_index

        for _ in range(self.prediction_steps):
            (
                predicted_action,
                debug,
                predicted_nearest_index,
            ) = self._compute_control(
                ego=predicted_state,
                goal=goal,
                reference_path=reference_path,
                nearest_index_hint=(predicted_nearest_index),
            )

            predicted_controls.append(predicted_action.as_array())

            if bool(debug["terminal_hold"]):
                predicted_states.append(predicted_state.as_array())
                continue

            predicted_state = self._dynamics.step(
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

        return (
            trajectory,
            controls,
        )

    @staticmethod
    def _goal_error(
        *,
        state: VehicleState,
        goal: GoalState,
    ) -> tuple[
        float,
        float,
        float,
        bool,
        bool,
        bool,
    ]:
        target = goal.state

        position_error = math.hypot(
            state.x - target.x,
            state.y - target.y,
        )

        yaw_error = abs(normalize_angle(state.yaw - target.yaw))

        speed_error = abs(state.speed - target.speed)

        position_reached = position_error <= goal.position_tolerance

        yaw_reached = yaw_error <= goal.yaw_tolerance

        speed_reached = speed_error <= goal.speed_tolerance

        return (
            position_error,
            yaw_error,
            speed_error,
            position_reached,
            yaw_reached,
            speed_reached,
        )

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
