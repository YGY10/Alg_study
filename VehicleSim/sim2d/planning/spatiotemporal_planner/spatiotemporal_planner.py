from __future__ import annotations

import math

import numpy as np

from sim2d.perception import PlanningInput
from sim2d.planning.base import Planner
from sim2d.types import GoalState, PlanResult, VehicleConfig, VehicleControl, VehicleState

from .config import SpatiotemporalPlannerConfig
from .coordinates import build_local_planning_context, local_trajectory_to_world
from .optimizer import SpatiotemporalOptimizer
from .pnc_map import PNCMap, local_reference_path_to_world
from .prediction import ConstantVelocityPredictor
from .types import OptimizationResult, SpatiotemporalTrajectory


class SpatiotemporalPlanner(Planner):
    """冻结自车坐标系下的时空联合规划器。"""

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
            raise ValueError("invalid fallback handle length range")
        if not 0.0 <= perception_lane_minimum_confidence <= 1.0:
            raise ValueError("perception_lane_minimum_confidence must be within [0, 1]")
        if perception_lane_maximum_distance <= 0.0:
            raise ValueError("perception_lane_maximum_distance must be positive")

        self.vehicle_config = vehicle_config
        self.config = config
        self.fallback_path_sample_count = fallback_path_sample_count
        self.fallback_handle_scale = fallback_handle_scale
        self.fallback_minimum_handle_length = fallback_minimum_handle_length
        self.fallback_maximum_handle_length = fallback_maximum_handle_length
        self.predictor = ConstantVelocityPredictor()
        self.optimizer = SpatiotemporalOptimizer(config=config, vehicle_config=vehicle_config)
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
            optimization_left_boundary = pnc_reference.left_boundary
            optimization_right_boundary = pnc_reference.right_boundary
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
            optimization_left_boundary = None
            optimization_right_boundary = None
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

        warm_start_available = (
            self._previous_controls is not None
            and self._previous_controls.shape == (self.config.horizon_steps, 2)
        )
        (
            initial_controls,
            initial_source,
            feedforward_controls,
            feedforward_trajectory,
            feedforward_diagnostics,
            warm_start_diagnostics,
        ) = self._select_initial_controls(
            ego_state=context.ego,
            reference_path=optimization_reference,
            left_boundary=optimization_left_boundary,
            right_boundary=optimization_right_boundary,
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
            left_boundary=optimization_left_boundary,
            right_boundary=optimization_right_boundary,
        )
        optimization_result = self._apply_feasibility_fallback(
            optimization_result=optimization_result,
            feedforward_controls=feedforward_controls,
            feedforward_trajectory=feedforward_trajectory,
            feedforward_diagnostics=feedforward_diagnostics,
            predictions=predictions,
            reference_path=optimization_reference,
            left_boundary=optimization_left_boundary,
            right_boundary=optimization_right_boundary,
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
                "warm_start_used": warm_start_available,
                "initial_control_source": initial_source,
                "warm_start_diagnostics": warm_start_diagnostics,
                "feedforward_diagnostics": feedforward_diagnostics,
                "reference_changed": reference_changed,
                "corridor_constraint_used": optimization_left_boundary is not None,
            },
        )

    def _select_initial_controls(
        self,
        *,
        ego_state: VehicleState,
        reference_path: np.ndarray | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        str,
        np.ndarray | None,
        SpatiotemporalTrajectory | None,
        dict[str, float | int | bool] | None,
        dict[str, float | int | bool] | None,
    ]:
        warm_controls = self._make_initial_controls(current_speed=ego_state.speed)
        if reference_path is None or left_boundary is None or right_boundary is None:
            return warm_controls, "warm_or_cruise", None, None, None, None

        feedforward_controls = self._make_reference_feedforward_controls(
            ego_state=ego_state,
            reference_path=reference_path,
        )
        feedforward_trajectory = self.optimizer.rollout.rollout_from_ego(
            ego_state=ego_state,
            controls=feedforward_controls,
        )
        feedforward_diagnostics = self.optimizer._trajectory_diagnostics(
            trajectory=feedforward_trajectory,
            reference_path=reference_path,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

        if self._previous_controls is None:
            return (
                feedforward_controls,
                "reference_feedforward",
                feedforward_controls,
                feedforward_trajectory,
                feedforward_diagnostics,
                None,
            )

        warm_trajectory = self.optimizer.rollout.rollout_from_ego(
            ego_state=ego_state,
            controls=warm_controls,
        )
        warm_diagnostics = self.optimizer._trajectory_diagnostics(
            trajectory=warm_trajectory,
            reference_path=reference_path,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )
        warm_violation = float(warm_diagnostics["max_footprint_violation"])
        feedforward_violation = float(
            feedforward_diagnostics["max_footprint_violation"]
        )
        warm_is_acceptable = (
            warm_violation <= self.config.warm_start_max_violation
            and warm_violation <= feedforward_violation + 0.02
        )
        if warm_is_acceptable:
            return (
                warm_controls,
                "warm_start",
                feedforward_controls,
                feedforward_trajectory,
                feedforward_diagnostics,
                warm_diagnostics,
            )
        return (
            feedforward_controls,
            "reference_feedforward_recovery",
            feedforward_controls,
            feedforward_trajectory,
            feedforward_diagnostics,
            warm_diagnostics,
        )

    def _apply_feasibility_fallback(
        self,
        *,
        optimization_result: OptimizationResult,
        feedforward_controls: np.ndarray | None,
        feedforward_trajectory: SpatiotemporalTrajectory | None,
        feedforward_diagnostics: dict[str, float | int | bool] | None,
        predictions,
        reference_path: np.ndarray | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> OptimizationResult:
        if (
            feedforward_controls is None
            or feedforward_trajectory is None
            or feedforward_diagnostics is None
        ):
            return optimization_result

        optimized_diagnostics = optimization_result.debug.get(
            "trajectory_diagnostics", {}
        )
        optimized_violation = float(
            optimized_diagnostics.get("max_footprint_violation", math.inf)
        )
        feedforward_violation = float(
            feedforward_diagnostics.get("max_footprint_violation", math.inf)
        )
        needs_fallback = (
            optimized_violation > self.config.result_max_violation
            and feedforward_violation
            + self.config.fallback_violation_improvement
            < optimized_violation
        )
        if not needs_fallback:
            debug = dict(optimization_result.debug)
            debug["feasibility_fallback_used"] = False
            debug["feedforward_max_violation"] = feedforward_violation
            return OptimizationResult(
                trajectory=optimization_result.trajectory,
                success=optimization_result.success,
                total_cost=optimization_result.total_cost,
                iterations=optimization_result.iterations,
                status=optimization_result.status,
                cost_terms=dict(optimization_result.cost_terms),
                debug=debug,
            )

        fallback_cost, fallback_terms = self.optimizer.cost.evaluate(
            trajectory=feedforward_trajectory,
            predictions=predictions,
            reference_path=reference_path,
            previous_control=self._previous_control,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )
        optimized_collision = float(optimization_result.cost_terms.get("collision", 0.0))
        fallback_collision = float(fallback_terms.get("collision", 0.0))
        if fallback_collision > optimized_collision + 1e-6:
            debug = dict(optimization_result.debug)
            debug["feasibility_fallback_used"] = False
            debug["feasibility_fallback_rejected_by_collision"] = True
            debug["feedforward_max_violation"] = feedforward_violation
            return OptimizationResult(
                trajectory=optimization_result.trajectory,
                success=optimization_result.success,
                total_cost=optimization_result.total_cost,
                iterations=optimization_result.iterations,
                status=optimization_result.status,
                cost_terms=dict(optimization_result.cost_terms),
                debug=debug,
            )

        debug = dict(optimization_result.debug)
        debug.update(
            {
                "feasibility_fallback_used": True,
                "optimized_status_before_fallback": optimization_result.status,
                "optimized_cost_before_fallback": optimization_result.total_cost,
                "optimized_max_violation_before_fallback": optimized_violation,
                "feedforward_max_violation": feedforward_violation,
                "optimized_controls": feedforward_controls.copy(),
                "trajectory_diagnostics": dict(feedforward_diagnostics),
            }
        )
        return OptimizationResult(
            trajectory=feedforward_trajectory,
            success=feedforward_violation <= self.config.result_max_violation,
            total_cost=float(fallback_cost),
            iterations=optimization_result.iterations,
            status="feasibility_feedforward_fallback",
            cost_terms=dict(fallback_terms),
            debug=debug,
        )

    def _make_reference_feedforward_controls(
        self,
        *,
        ego_state: VehicleState,
        reference_path: np.ndarray,
    ) -> np.ndarray:
        path = np.asarray(reference_path, dtype=np.float64)
        controls = np.zeros((self.config.horizon_steps, 2), dtype=np.float64)
        state = self.optimizer.rollout.local_initial_state(ego_state)
        previous_steering = (
            0.0 if self._previous_control is None else self._previous_control.steering
        )
        pure_pursuit_weight = self.config.feedforward_pure_pursuit_weight
        steering_step_limit = max(
            self.config.max_steering_rate * self.config.dt,
            0.10,
        )

        for index in range(self.config.horizon_steps):
            projection = self._project_state_to_reference(state, path)
            current_s = projection["s"]
            lookahead = float(
                np.clip(
                    self.config.feedforward_lookahead_base
                    + self.config.feedforward_lookahead_speed_gain * state.speed,
                    self.config.feedforward_lookahead_base,
                    self.config.feedforward_lookahead_max,
                )
            )
            target = self._interpolate_reference(path, current_s + lookahead)
            dx = target[0] - state.x
            dy = target[1] - state.y
            target_distance = max(math.hypot(dx, dy), 0.5)
            alpha = self._normalize_angle(math.atan2(dy, dx) - state.yaw)
            pure_pursuit = math.atan2(
                2.0 * self.vehicle_config.wheel_base * math.sin(alpha),
                target_distance,
            )
            curvature_feedforward = math.atan(
                self.vehicle_config.wheel_base * target[4]
            )
            steering = (
                pure_pursuit_weight * pure_pursuit
                + (1.0 - pure_pursuit_weight) * curvature_feedforward
            )
            steering = float(
                np.clip(
                    steering,
                    previous_steering - steering_step_limit,
                    previous_steering + steering_step_limit,
                )
            )
            steering = float(
                np.clip(
                    steering,
                    self.vehicle_config.steering_min,
                    self.vehicle_config.steering_max,
                )
            )

            preview_end = current_s + self.config.feedforward_curve_preview_distance
            preview_mask = (path[:, 3] >= current_s) & (path[:, 3] <= preview_end)
            if np.any(preview_mask):
                preview_curvature = float(np.max(np.abs(path[preview_mask, 4])))
            else:
                preview_curvature = abs(float(target[4]))
            curve_speed = math.sqrt(
                self.config.curve_speed_lateral_acceleration
                / max(preview_curvature, 1e-6)
            )
            target_speed = float(
                np.clip(
                    curve_speed,
                    self.config.minimum_curve_speed,
                    self.config.target_speed,
                )
            )
            acceleration = float(
                np.clip(
                    (target_speed - state.speed)
                    / self.config.feedforward_speed_response_time,
                    self.vehicle_config.acceleration_min,
                    self.vehicle_config.acceleration_max,
                )
            )
            controls[index] = (acceleration, steering)
            control = VehicleControl(acceleration=acceleration, steering=steering)
            state = self.optimizer.rollout.motion_model.predict_step(
                state,
                control,
                self.config.dt,
            )
            previous_steering = steering
        return controls

    @staticmethod
    def _project_state_to_reference(
        state: VehicleState,
        reference_path: np.ndarray,
    ) -> dict[str, float]:
        point = np.array([state.x, state.y], dtype=np.float64)
        starts = reference_path[:-1, :2]
        vectors = reference_path[1:, :2] - starts
        squared = np.sum(vectors * vectors, axis=1)
        usable = squared > 1e-12
        parameters = np.zeros(vectors.shape[0], dtype=np.float64)
        parameters[usable] = (
            np.sum((point - starts[usable]) * vectors[usable], axis=1)
            / squared[usable]
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        projected = starts + parameters[:, None] * vectors
        index = int(np.argmin(np.sum((projected - point) ** 2, axis=1)))
        parameter = float(parameters[index])
        yaw_values = np.unwrap(reference_path[:, 2])
        yaw = float(
            yaw_values[index]
            + parameter * (yaw_values[index + 1] - yaw_values[index])
        )
        s = float(
            reference_path[index, 3]
            + parameter
            * (reference_path[index + 1, 3] - reference_path[index, 3])
        )
        curvature = float(
            reference_path[index, 4]
            + parameter
            * (reference_path[index + 1, 4] - reference_path[index, 4])
        )
        normal = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        lateral = float(np.dot(point - projected[index], normal))
        return {"s": s, "lateral": lateral, "yaw": yaw, "curvature": curvature}

    @staticmethod
    def _interpolate_reference(reference_path: np.ndarray, s: float) -> np.ndarray:
        arc = reference_path[:, 3]
        value = float(np.clip(s, arc[0], arc[-1]))
        yaw = np.unwrap(reference_path[:, 2])
        return np.array(
            [
                np.interp(value, arc, reference_path[:, 0]),
                np.interp(value, arc, reference_path[:, 1]),
                (np.interp(value, arc, yaw) + np.pi) % (2.0 * np.pi) - np.pi,
                value,
                np.interp(value, arc, reference_path[:, 4]),
            ],
            dtype=np.float64,
        )

    def _make_initial_controls(self, *, current_speed: float) -> np.ndarray:
        if self._previous_controls is not None:
            expected_shape = (self.config.horizon_steps, 2)
            if self._previous_controls.shape == expected_shape:
                return self.optimizer.shift_controls_for_warm_start(self._previous_controls)
        controls = np.zeros((self.config.horizon_steps, 2), dtype=np.float64)
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
        parameter = np.linspace(0.0, 1.0, self.fallback_path_sample_count)
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
        arc_length = np.concatenate((np.array([0.0]), np.cumsum(segment_lengths)))
        derivative_norm_squared = np.sum(first_derivative**2, axis=1)
        curvature_numerator = (
            first_derivative[:, 0] * second_derivative[:, 1]
            - first_derivative[:, 1] * second_derivative[:, 0]
        )
        curvature = curvature_numerator / np.power(
            np.maximum(derivative_norm_squared, 1e-12),
            1.5,
        )
        yaws[0] = start.yaw
        yaws[-1] = goal.yaw
        return np.column_stack((positions, yaws, arc_length, curvature))

    def _clear_warm_start(self) -> None:
        self._previous_controls = None
        self._previous_control = None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _validate_reference_path(reference_path: np.ndarray) -> None:
        if reference_path.ndim != 2 or reference_path.shape[1] != 5:
            raise ValueError(
                f"reference_path must have shape [N, 5], got {reference_path.shape}"
            )
        if reference_path.shape[0] < 2:
            raise ValueError("reference_path must contain at least two points")
        if not np.all(np.isfinite(reference_path)):
            raise ValueError("reference_path contains non-finite values")
        if abs(float(reference_path[0, 3])) > 1e-9:
            raise ValueError("reference_path arc length must start at zero")
        if not np.all(np.diff(reference_path[:, 3]) > 0.0):
            raise ValueError("reference_path arc length must be strictly increasing")
