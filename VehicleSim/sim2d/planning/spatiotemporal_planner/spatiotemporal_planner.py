from __future__ import annotations

import math

import numpy as np

from sim2d.perception import PlanningInput
from sim2d.planning.base import Planner
from sim2d.types import (
    GoalState,
    PlanResult,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)

from .config import SpatiotemporalPlannerConfig
from .coordinates import build_local_planning_context, local_trajectory_to_world
from .optimizer import SpatiotemporalOptimizer
from .pnc_map import PNCMap, local_reference_path_to_world
from .prediction import ConstantVelocityPredictor
from .types import OptimizationResult


class SpatiotemporalPlanner(Planner):
    """冻结自车坐标系下的时空联合规划器。

    感知模块只发布车道线点列；PNC Map 负责构造并跨帧跟踪 reference line；
    地图导航路径只在感知几何不可用时作为 fallback。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        config: SpatiotemporalPlannerConfig | None = None,
        fallback_path_sample_count: int = 201,
        fallback_handle_scale: float = 0.35,
        fallback_minimum_handle_length: float = 2.0,
        fallback_maximum_handle_length: float = 12.0,
        perception_lane_minimum_confidence: float = 0.5,
        perception_lane_maximum_distance: float = 6.0,
        pnc_map_switch_confirm_frames: int = 5,
    ) -> None:
        vehicle_config.validate()
        if config is None:
            config = SpatiotemporalPlannerConfig()
        config.validate()

        if fallback_path_sample_count < 2:
            raise ValueError("fallback_path_sample_count must be at least 2")
        if fallback_handle_scale <= 0.0:
            raise ValueError("fallback_handle_scale must be positive")
        if fallback_minimum_handle_length <= 0.0:
            raise ValueError("fallback_minimum_handle_length must be positive")
        if fallback_maximum_handle_length < fallback_minimum_handle_length:
            raise ValueError(
                "fallback_maximum_handle_length must not be smaller "
                "than fallback_minimum_handle_length"
            )
        if not 0.0 <= perception_lane_minimum_confidence <= 1.0:
            raise ValueError(
                "perception_lane_minimum_confidence must be within [0, 1]"
            )
        if perception_lane_maximum_distance <= 0.0:
            raise ValueError("perception_lane_maximum_distance must be positive")

        self.vehicle_config = vehicle_config
        self.config = config
        self.fallback_path_sample_count = fallback_path_sample_count
        self.fallback_handle_scale = fallback_handle_scale
        self.fallback_minimum_handle_length = fallback_minimum_handle_length
        self.fallback_maximum_handle_length = fallback_maximum_handle_length

        self.predictor = ConstantVelocityPredictor()
        self.optimizer = SpatiotemporalOptimizer(
            config=config,
            vehicle_config=vehicle_config,
        )
        self.pnc_map = PNCMap(
            minimum_confidence=perception_lane_minimum_confidence,
            maximum_lateral_distance=perception_lane_maximum_distance,
            switch_confirm_frames=pnc_map_switch_confirm_frames,
        )

        self._external_reference_path: np.ndarray | None = None
        self._fallback_reference_path: np.ndarray | None = None
        self._fallback_goal_signature: tuple[float, ...] | None = None

        self._previous_controls: np.ndarray | None = None
        self._previous_control: VehicleControl | None = None
        self._last_optimization_result: OptimizationResult | None = None
        self._last_reference_source: str | None = None
        self._last_pnc_reference_id: str | None = None

    def reset(self) -> None:
        self._external_reference_path = None
        self._fallback_reference_path = None
        self._fallback_goal_signature = None
        self._clear_warm_start()
        self.pnc_map.reset()
        self._last_optimization_result = None
        self._last_reference_source = None
        self._last_pnc_reference_id = None

    def set_reference_path(self, reference_path: np.ndarray) -> None:
        """设置世界坐标导航路径；仅在感知 reference 不可用时 fallback。"""
        path = np.asarray(reference_path, dtype=np.float64)
        self._validate_reference_path(path)
        self._external_reference_path = path.copy()
        self._fallback_reference_path = None
        self._fallback_goal_signature = None
        self._clear_warm_start()

    def clear_external_reference_path(self) -> None:
        self._external_reference_path = None
        self._fallback_reference_path = None
        self._fallback_goal_signature = None
        self._clear_warm_start()

    @property
    def has_external_reference_path(self) -> bool:
        return self._external_reference_path is not None

    @property
    def last_optimization_result(self) -> OptimizationResult | None:
        return self._last_optimization_result

    def plan(self, planning_input: PlanningInput) -> PlanResult:
        navigation_world_path = self._ensure_navigation_path(
            ego=planning_input.ego,
            goal=planning_input.goal,
        )
        context = build_local_planning_context(
            planning_input=planning_input,
            reference_path=navigation_world_path,
        )

        map_update = self.pnc_map.update(
            context.lane_lines,
            world_origin=context.world_origin,
        )
        pnc_reference = map_update.selected

        if pnc_reference is not None:
            optimization_reference = pnc_reference.reference_path
            output_reference_path = local_reference_path_to_world(
                pnc_reference.reference_path,
                context.world_origin,
            )
            reference_source = "pnc_map_perception"
            pnc_reference_id = pnc_reference.reference_id
            pnc_reference_confidence = pnc_reference.confidence
        else:
            if context.reference_path is None:
                raise RuntimeError(
                    "no PNC Map perception reference and no navigation fallback path"
                )
            optimization_reference = context.reference_path
            output_reference_path = navigation_world_path.copy()
            reference_source = (
                "navigation_fallback"
                if self.has_external_reference_path
                else "bezier_fallback"
            )
            pnc_reference_id = None
            pnc_reference_confidence = None

        source_changed = (
            self._last_reference_source is not None
            and self._last_reference_source != reference_source
        )
        reference_changed = source_changed or map_update.reference_changed
        if reference_changed:
            self._clear_warm_start()

        self._last_reference_source = reference_source
        self._last_pnc_reference_id = pnc_reference_id

        warm_start_used = (
            self._previous_controls is not None
            and self._previous_controls.shape == (self.config.horizon_steps, 2)
        )
        initial_controls = self._make_initial_controls(
            current_speed=context.ego.speed,
        )

        prediction_times = (
            np.arange(self.config.horizon_steps + 1, dtype=np.float64)
            * self.config.dt
        )
        predictions = self.predictor.predict(
            objects=context.objects,
            times=prediction_times,
        )
        optimization_result = self.optimizer.optimize(
            ego_state=context.ego,
            initial_controls=initial_controls,
            predictions=predictions,
            reference_path=optimization_reference,
            previous_control=self._previous_control,
        )

        local_trajectory = optimization_result.trajectory
        if local_trajectory.controls.shape[0] == 0:
            raise RuntimeError("optimizer returned an empty control sequence")

        action = VehicleControl.from_array(local_trajectory.controls[0])
        self._previous_controls = local_trajectory.controls.copy()
        self._previous_control = action
        self._last_optimization_result = optimization_result

        world_trajectory = local_trajectory_to_world(
            trajectory=local_trajectory,
            origin=context.world_origin,
        )

        return PlanResult(
            action=action,
            trajectory=world_trajectory.states.copy(),
            controls=local_trajectory.controls.copy(),
            reference_path=output_reference_path.copy(),
            debug={
                "planner": "SpatiotemporalPlanner",
                "status": optimization_result.status,
                "optimization_success": optimization_result.success,
                "optimization_iterations": optimization_result.iterations,
                "optimization_total_cost": optimization_result.total_cost,
                "optimization_cost_terms": dict(optimization_result.cost_terms),
                "optimization_debug": dict(optimization_result.debug),
                "prediction_dt": self.config.dt,
                "prediction_steps": self.config.horizon_steps,
                "prediction_horizon": self.config.dt * self.config.horizon_steps,
                "prediction_mode": "constant_velocity",
                "coordinate_frame_internal": "frozen_vehicle",
                "coordinate_frame_output": "world",
                "perceived_object_count": len(context.objects),
                "perceived_lane_line_count": len(context.lane_lines),
                "pnc_reference_candidate_count": len(map_update.references),
                "pnc_reference_id": pnc_reference_id,
                "pnc_reference_confidence": pnc_reference_confidence,
                "pnc_history_used": map_update.history_used,
                "pnc_switch_pending_frames": map_update.switch_pending_frames,
                "pnc_continuity_cost": map_update.continuity_cost,
                "reference_path_points": output_reference_path.shape[0],
                "reference_path_source": reference_source,
                "navigation_path_available": navigation_world_path is not None,
                "navigation_path_source": (
                    "external" if self.has_external_reference_path else "fallback_bezier"
                ),
                "warm_start_used": warm_start_used,
                "reference_changed": reference_changed,
            },
        )

    def _make_initial_controls(self, *, current_speed: float) -> np.ndarray:
        if self._previous_controls is not None:
            expected_shape = (self.config.horizon_steps, 2)
            if self._previous_controls.shape == expected_shape:
                return self.optimizer.shift_controls_for_warm_start(
                    self._previous_controls
                )

        controls = np.zeros(
            (self.config.horizon_steps, 2),
            dtype=np.float64,
        )
        speed_error = self.config.target_speed - current_speed
        controls[:, 0] = float(
            np.clip(
                speed_error,
                self.vehicle_config.acceleration_min,
                self.vehicle_config.acceleration_max,
            )
        )
        return controls

    def _ensure_navigation_path(
        self,
        *,
        ego: VehicleState,
        goal: GoalState,
    ) -> np.ndarray:
        if self._external_reference_path is not None:
            return self._external_reference_path

        goal_signature = (
            goal.state.x,
            goal.state.y,
            goal.state.yaw,
            goal.state.speed,
            goal.position_tolerance,
            goal.yaw_tolerance,
            goal.speed_tolerance,
        )
        if (
            self._fallback_reference_path is None
            or self._fallback_goal_signature != goal_signature
        ):
            self._fallback_reference_path = self._generate_fallback_bezier_path(
                start=ego,
                goal=goal.state,
            )
            self._fallback_goal_signature = goal_signature
            self._clear_warm_start()
        return self._fallback_reference_path

    def _generate_fallback_bezier_path(
        self,
        *,
        start: VehicleState,
        goal: VehicleState,
    ) -> np.ndarray:
        p0 = np.array([start.x, start.y], dtype=np.float64)
        p3 = np.array([goal.x, goal.y], dtype=np.float64)
        straight_distance = float(np.linalg.norm(p3 - p0))
        handle_length = float(
            np.clip(
                straight_distance * self.fallback_handle_scale,
                self.fallback_minimum_handle_length,
                self.fallback_maximum_handle_length,
            )
        )
        start_direction = np.array(
            [math.cos(start.yaw), math.sin(start.yaw)],
            dtype=np.float64,
        )
        goal_direction = np.array(
            [math.cos(goal.yaw), math.sin(goal.yaw)],
            dtype=np.float64,
        )
        p1 = p0 + handle_length * start_direction
        p2 = p3 - handle_length * goal_direction

        parameter = np.linspace(
            0.0,
            1.0,
            self.fallback_path_sample_count,
            dtype=np.float64,
        )
        one_minus = 1.0 - parameter
        positions = (
            one_minus[:, None] ** 3 * p0
            + 3.0 * one_minus[:, None] ** 2 * parameter[:, None] * p1
            + 3.0 * one_minus[:, None] * parameter[:, None] ** 2 * p2
            + parameter[:, None] ** 3 * p3
        )
        first_derivative = (
            3.0 * one_minus[:, None] ** 2 * (p1 - p0)
            + 6.0 * one_minus[:, None] * parameter[:, None] * (p2 - p1)
            + 3.0 * parameter[:, None] ** 2 * (p3 - p2)
        )
        second_derivative = (
            6.0 * one_minus[:, None] * (p2 - 2.0 * p1 + p0)
            + 6.0 * parameter[:, None] * (p3 - 2.0 * p2 + p1)
        )
        yaws = np.arctan2(first_derivative[:, 1], first_derivative[:, 0])
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        arc_length = np.concatenate(
            (np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths))
        )
        derivative_norm_squared = np.sum(first_derivative**2, axis=1)
        curvature_numerator = (
            first_derivative[:, 0] * second_derivative[:, 1]
            - first_derivative[:, 1] * second_derivative[:, 0]
        )
        curvature_denominator = np.power(
            np.maximum(derivative_norm_squared, 1e-12),
            1.5,
        )
        curvature = curvature_numerator / curvature_denominator
        yaws[0] = start.yaw
        yaws[-1] = goal.yaw
        return np.column_stack(
            (positions[:, 0], positions[:, 1], yaws, arc_length, curvature)
        )

    def _clear_warm_start(self) -> None:
        self._previous_controls = None
        self._previous_control = None

    @staticmethod
    def _validate_reference_path(reference_path: np.ndarray) -> None:
        if reference_path.ndim != 2 or reference_path.shape[1] != 5:
            raise ValueError(
                "reference_path must have shape [N, 5], "
                f"got {reference_path.shape}"
            )
        if reference_path.shape[0] < 2:
            raise ValueError("reference_path must contain at least two points")
        if not np.all(np.isfinite(reference_path)):
            raise ValueError("reference_path contains non-finite values")
        if abs(float(reference_path[0, 3])) > 1e-9:
            raise ValueError("reference_path arc length must start at zero")
        if not np.all(np.diff(reference_path[:, 3]) > 0.0):
            raise ValueError("reference_path arc length must be strictly increasing")
