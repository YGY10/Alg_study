from __future__ import annotations

import math

import numpy as np

from sim2d.types import VehicleConfig, VehicleControl

from .config import SpatiotemporalPlannerConfig
from .types import ObjectPredictionSet, SpatiotemporalTrajectory


class SpatiotemporalCost:
    """评价冻结自车坐标系中的候选时空轨迹。"""

    def __init__(
        self,
        config: SpatiotemporalPlannerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        config.validate()
        vehicle_config.validate()
        self.config = config
        self.vehicle_config = vehicle_config
        self._ego_collision_radius = 0.5 * math.hypot(
            vehicle_config.length,
            vehicle_config.width,
        )

    def evaluate(
        self,
        trajectory: SpatiotemporalTrajectory,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None = None,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """返回加权总成本和加权分项。"""
        self._validate_time_alignment(trajectory=trajectory, predictions=predictions)

        (
            reference_cost,
            heading_cost,
            terminal_cost,
            corridor_cost,
            progress_cost,
            curve_speed_cost,
        ) = self._reference_cost(
            trajectory=trajectory,
            reference_path=reference_path,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )
        speed_cost = self._speed_cost(trajectory) if reference_path is None else curve_speed_cost
        (
            acceleration_cost,
            steering_cost,
            acceleration_rate_cost,
            steering_rate_cost,
        ) = self._control_cost(
            trajectory=trajectory,
            previous_control=previous_control,
        )
        collision_cost = self._collision_cost(
            trajectory=trajectory,
            predictions=predictions,
        )
        lateral_acceleration_cost = self._lateral_acceleration_cost(trajectory)

        weighted_terms = {
            "reference": self.config.weight_reference * reference_cost,
            "heading": self.config.weight_heading * heading_cost,
            "speed": self.config.weight_speed * speed_cost,
            "acceleration": self.config.weight_acceleration * acceleration_cost,
            "steering": self.config.weight_steering * steering_cost,
            "acceleration_rate": (
                self.config.weight_acceleration_rate * acceleration_rate_cost
            ),
            "steering_rate": self.config.weight_steering_rate * steering_rate_cost,
            "collision": self.config.weight_collision * collision_cost,
            "terminal": self.config.weight_terminal * terminal_cost,
            "corridor": self.config.weight_corridor * corridor_cost,
            "frenet_progress": self.config.weight_frenet_progress * progress_cost,
            "lateral_acceleration": (
                self.config.weight_lateral_acceleration * lateral_acceleration_cost
            ),
        }
        total_cost = float(sum(weighted_terms.values()))
        if not np.isfinite(total_cost):
            raise ValueError("total cost is non-finite")
        return total_cost, {name: float(value) for name, value in weighted_terms.items()}

    def _reference_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
        reference_path: np.ndarray | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> tuple[float, float, float, float, float, float]:
        """在 reference Frenet 坐标系中计算跟踪、走廊与弯道速度代价。"""
        if reference_path is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        path = np.asarray(reference_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] < 3 or path.shape[0] < 2:
            raise ValueError("reference_path must have shape [M, K], M >= 2, K >= 3")
        if not np.all(np.isfinite(path)):
            raise ValueError("reference_path contains non-finite values")

        projection = self._project_to_reference(
            query_points=trajectory.states[:, :2],
            reference_path=path,
        )
        lateral = projection["lateral"]
        projected_yaw = projection["yaw"]
        projected_s = projection["s"]
        heading_errors = self._normalize_angle(
            trajectory.states[:, 2] - projected_yaw
        )
        reference_cost = float(np.mean(lateral**2))
        heading_cost = float(np.mean(heading_errors**2))
        terminal_cost = float(lateral[-1] ** 2 + heading_errors[-1] ** 2)

        s_differences = np.diff(projected_s)
        backward_progress = np.maximum(
            -(s_differences + self.config.frenet_nonmonotonic_tolerance),
            0.0,
        )
        progress_cost = (
            float(np.mean(backward_progress**2))
            if backward_progress.size
            else 0.0
        )

        corridor_cost = 0.0
        if left_boundary is not None and right_boundary is not None:
            corridor_cost = self._footprint_corridor_cost(
                trajectory=trajectory,
                reference_path=path,
                left_boundary=np.asarray(left_boundary, dtype=np.float64),
                right_boundary=np.asarray(right_boundary, dtype=np.float64),
            )

        projected_curvature = self._interpolate_reference_column(
            reference_path=path,
            projection=projection,
            column=4,
            default=0.0,
        )
        curvature_magnitude = np.maximum(np.abs(projected_curvature), 1e-6)
        curve_speed_limit = np.sqrt(
            self.config.curve_speed_lateral_acceleration / curvature_magnitude
        )
        target_speeds = np.clip(
            curve_speed_limit,
            self.config.minimum_curve_speed,
            self.config.target_speed,
        )
        curve_speed_cost = float(
            np.mean((trajectory.states[:, 3] - target_speeds) ** 2)
        )

        return (
            reference_cost,
            heading_cost,
            terminal_cost,
            corridor_cost,
            progress_cost,
            curve_speed_cost,
        )

    def _footprint_corridor_cost(
        self,
        *,
        trajectory: SpatiotemporalTrajectory,
        reference_path: np.ndarray,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> float:
        expected_shape = reference_path[:, :2].shape
        if left_boundary.shape != expected_shape or right_boundary.shape != expected_shape:
            raise ValueError(
                "left_boundary and right_boundary must match reference point count"
            )

        states = trajectory.states
        yaw = states[:, 2]
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        left_axis = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        half_length = 0.5 * self.vehicle_config.length + self.config.corridor_vehicle_margin
        half_width = 0.5 * self.vehicle_config.width + self.config.corridor_vehicle_margin
        centers = states[:, :2]

        footprint = np.stack(
            (
                centers + half_length * forward + half_width * left_axis,
                centers + half_length * forward - half_width * left_axis,
                centers - half_length * forward + half_width * left_axis,
                centers - half_length * forward - half_width * left_axis,
                centers,
            ),
            axis=1,
        )
        flattened = footprint.reshape(-1, 2)
        projection = self._project_to_reference(
            query_points=flattened,
            reference_path=reference_path,
        )
        segment_indices = projection["segment_indices"]
        parameters = projection["parameters"]
        projected_yaw = projection["yaw"]
        projected_positions = projection["positions"]
        normals = np.column_stack((-np.sin(projected_yaw), np.cos(projected_yaw)))

        left_projected = (
            left_boundary[segment_indices]
            + parameters[:, None]
            * (left_boundary[segment_indices + 1] - left_boundary[segment_indices])
        )
        right_projected = (
            right_boundary[segment_indices]
            + parameters[:, None]
            * (right_boundary[segment_indices + 1] - right_boundary[segment_indices])
        )
        point_l = np.sum((flattened - projected_positions) * normals, axis=1)
        left_l = np.sum((left_projected - projected_positions) * normals, axis=1)
        right_l = np.sum((right_projected - projected_positions) * normals, axis=1)
        lower = np.minimum(left_l, right_l)
        upper = np.maximum(left_l, right_l)
        violation = np.maximum(lower - point_l, 0.0) + np.maximum(
            point_l - upper,
            0.0,
        )

        violation_by_state = violation.reshape(states.shape[0], -1)
        mean_squared = float(np.mean(violation_by_state**2))
        max_violation = float(np.max(violation_by_state))
        barrier_excess = np.maximum(
            violation_by_state - self.config.corridor_barrier_threshold,
            0.0,
        )
        barrier = float(np.mean(barrier_excess**2))
        return (
            mean_squared
            + self.config.corridor_max_violation_multiplier * max_violation**2
            + self.config.corridor_barrier_multiplier * barrier
        )

    def _project_to_reference(
        self,
        *,
        query_points: np.ndarray,
        reference_path: np.ndarray,
    ) -> dict[str, np.ndarray]:
        starts = reference_path[:-1, :2]
        vectors = reference_path[1:, :2] - starts
        squared_lengths = np.sum(vectors**2, axis=1)
        usable = squared_lengths > 1e-12
        if not np.any(usable):
            raise ValueError("reference_path has no usable segment")

        delta = query_points[:, None, :] - starts[None, :, :]
        parameters = np.zeros(
            (query_points.shape[0], vectors.shape[0]),
            dtype=np.float64,
        )
        parameters[:, usable] = (
            np.sum(delta[:, usable, :] * vectors[None, usable, :], axis=2)
            / squared_lengths[usable][None, :]
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        projections = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        squared_distances = np.sum(
            (query_points[:, None, :] - projections) ** 2,
            axis=2,
        )
        segment_indices = np.argmin(squared_distances, axis=1)
        rows = np.arange(query_points.shape[0])
        selected_t = parameters[rows, segment_indices]
        selected_positions = projections[rows, segment_indices]

        if reference_path.shape[1] >= 4:
            reference_s = reference_path[:, 3]
        else:
            reference_s = np.concatenate(
                (
                    np.array([0.0]),
                    np.cumsum(np.linalg.norm(vectors, axis=1)),
                )
            )
        projected_s = reference_s[segment_indices] + selected_t * (
            reference_s[segment_indices + 1] - reference_s[segment_indices]
        )
        yaw_unwrapped = np.unwrap(reference_path[:, 2])
        projected_yaw = yaw_unwrapped[segment_indices] + selected_t * (
            yaw_unwrapped[segment_indices + 1] - yaw_unwrapped[segment_indices]
        )
        projected_yaw = self._normalize_angle(projected_yaw)
        normals = np.column_stack((-np.sin(projected_yaw), np.cos(projected_yaw)))
        lateral = np.sum((query_points - selected_positions) * normals, axis=1)
        return {
            "positions": selected_positions,
            "s": projected_s,
            "lateral": lateral,
            "yaw": projected_yaw,
            "segment_indices": segment_indices,
            "parameters": selected_t,
        }

    @staticmethod
    def _interpolate_reference_column(
        *,
        reference_path: np.ndarray,
        projection: dict[str, np.ndarray],
        column: int,
        default: float,
    ) -> np.ndarray:
        count = projection["segment_indices"].shape[0]
        if reference_path.shape[1] <= column:
            return np.full(count, default, dtype=np.float64)
        indices = projection["segment_indices"]
        parameters = projection["parameters"]
        return reference_path[indices, column] + parameters * (
            reference_path[indices + 1, column] - reference_path[indices, column]
        )

    def _speed_cost(self, trajectory: SpatiotemporalTrajectory) -> float:
        speed_errors = trajectory.states[:, 3] - self.config.target_speed
        return float(np.mean(speed_errors**2))

    def _control_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
        previous_control: VehicleControl | None,
    ) -> tuple[float, float, float, float]:
        controls = trajectory.controls
        if controls.shape[0] == 0:
            return 0.0, 0.0, 0.0, 0.0
        accelerations = controls[:, 0]
        steerings = controls[:, 1]
        acceleration_cost = float(np.mean(accelerations**2))
        steering_cost = float(np.mean(steerings**2))
        previous = controls[0] if previous_control is None else previous_control.as_array()
        extended_controls = np.vstack((previous, controls))
        control_rates = np.diff(extended_controls, axis=0) / self.config.dt
        acceleration_rates = control_rates[:, 0]
        steering_rates = control_rates[:, 1]
        acceleration_rate_cost = float(np.mean(acceleration_rates**2))
        steering_rate_cost = float(np.mean(steering_rates**2))
        acceleration_rate_excess = np.maximum(
            np.abs(acceleration_rates) - self.config.max_acceleration_rate,
            0.0,
        )
        steering_rate_excess = np.maximum(
            np.abs(steering_rates) - self.config.max_steering_rate,
            0.0,
        )
        acceleration_rate_cost += float(10.0 * np.mean(acceleration_rate_excess**2))
        steering_rate_cost += float(10.0 * np.mean(steering_rate_excess**2))
        return (
            acceleration_cost,
            steering_cost,
            acceleration_rate_cost,
            steering_rate_cost,
        )

    def _lateral_acceleration_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
    ) -> float:
        if trajectory.controls.shape[0] == 0:
            return 0.0
        speeds = trajectory.states[:-1, 3]
        steering = trajectory.controls[:, 1]
        curvature = np.tan(steering) / self.vehicle_config.wheel_base
        lateral_acceleration = speeds**2 * curvature
        excess = np.maximum(
            np.abs(lateral_acceleration) - self.config.maximum_lateral_acceleration,
            0.0,
        )
        return float(np.mean(excess**2))

    def _collision_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
        predictions: ObjectPredictionSet,
    ) -> float:
        if not predictions.trajectories:
            return 0.0
        ego_positions = trajectory.states[:, :2]
        softness = self.config.collision_softness
        total = 0.0
        for prediction in predictions.trajectories:
            object_radius = 0.5 * math.hypot(prediction.length, prediction.width)
            safety_margin = self._safety_margin(prediction.semantic_type)
            center_distances = np.linalg.norm(
                ego_positions - prediction.positions,
                axis=1,
            )
            clearance = (
                center_distances
                - self._ego_collision_radius
                - object_radius
                - safety_margin
            )
            soft_violation = softness * np.logaddexp(0.0, -clearance / softness)
            object_cost = np.mean(soft_violation**2)
            geometric_clearance = center_distances - self._ego_collision_radius - object_radius
            overlap = np.maximum(-geometric_clearance, 0.0)
            object_cost += 10.0 * np.mean(overlap**2)
            confidence_scale = 0.25 + 0.75 * prediction.confidence
            total += float(confidence_scale * object_cost)
        return total

    def _safety_margin(self, semantic_type: str) -> float:
        if semantic_type == "pedestrian":
            return self.config.pedestrian_margin
        if semantic_type == "small_car":
            return self.config.small_car_margin
        if semantic_type == "large_vehicle":
            return self.config.large_vehicle_margin
        return max(
            self.config.pedestrian_margin,
            self.config.small_car_margin,
            self.config.large_vehicle_margin,
        )

    @staticmethod
    def _normalize_angle(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _validate_time_alignment(
        trajectory: SpatiotemporalTrajectory,
        predictions: ObjectPredictionSet,
    ) -> None:
        if trajectory.times.shape != predictions.times.shape:
            raise ValueError("trajectory and predictions must have the same time-axis shape")
        if not np.allclose(
            trajectory.times,
            predictions.times,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("trajectory and predictions must use the same time axis")
