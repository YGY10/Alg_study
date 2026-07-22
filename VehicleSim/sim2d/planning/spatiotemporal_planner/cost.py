from __future__ import annotations

import math

import numpy as np

from sim2d.types import VehicleConfig, VehicleControl

from .config import SpatiotemporalPlannerConfig
from .types import (
    ObjectPredictionSet,
    SpatiotemporalTrajectory,
)


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
    ) -> tuple[float, dict[str, float]]:
        """返回加权总成本和未加权分项。

        trajectory、predictions、reference_path 必须全部位于同一个
        规划时刻冻结的自车坐标系。
        """
        self._validate_time_alignment(
            trajectory=trajectory,
            predictions=predictions,
        )

        reference_cost, heading_cost, terminal_cost = self._reference_cost(
            trajectory=trajectory,
            reference_path=reference_path,
        )

        speed_cost = self._speed_cost(trajectory)

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

        weighted_terms = {
            "reference": (self.config.weight_reference * reference_cost),
            "heading": (self.config.weight_heading * heading_cost),
            "speed": (self.config.weight_speed * speed_cost),
            "acceleration": (self.config.weight_acceleration * acceleration_cost),
            "steering": (self.config.weight_steering * steering_cost),
            "acceleration_rate": (
                self.config.weight_acceleration_rate * acceleration_rate_cost
            ),
            "steering_rate": (self.config.weight_steering_rate * steering_rate_cost),
            "collision": (self.config.weight_collision * collision_cost),
            "terminal": (self.config.weight_terminal * terminal_cost),
        }

        total_cost = float(sum(weighted_terms.values()))

        if not np.isfinite(total_cost):
            raise ValueError("total cost is non-finite")

        return total_cost, {
            name: float(value) for name, value in weighted_terms.items()
        }

    def _reference_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
        reference_path: np.ndarray | None,
    ) -> tuple[float, float, float]:
        """计算位置、航向和终端参考线跟踪代价。"""
        if reference_path is None:
            return 0.0, 0.0, 0.0

        path = np.asarray(
            reference_path,
            dtype=np.float64,
        )

        if path.ndim != 2 or path.shape[1] < 3:
            raise ValueError(
                "reference_path must have shape [M, K], " f"K >= 3, got {path.shape}"
            )

        if path.shape[0] == 0:
            return 0.0, 0.0, 0.0

        if not np.all(np.isfinite(path)):
            raise ValueError("reference_path contains non-finite values")

        ego_positions = trajectory.states[:, :2]
        ego_yaws = trajectory.states[:, 2]

        reference_positions = path[:, :2]
        reference_yaws = path[:, 2]

        nearest_indices = self._nearest_reference_indices(
            query_points=ego_positions,
            reference_points=reference_positions,
        )

        nearest_positions = reference_positions[nearest_indices]
        nearest_yaws = reference_yaws[nearest_indices]

        position_errors = ego_positions - nearest_positions
        squared_distances = np.sum(
            position_errors**2,
            axis=1,
        )

        heading_errors = self._normalize_angle(ego_yaws - nearest_yaws)

        reference_cost = float(np.mean(squared_distances))
        heading_cost = float(np.mean(heading_errors**2))

        terminal_cost = float(squared_distances[-1] + heading_errors[-1] ** 2)

        return (
            reference_cost,
            heading_cost,
            terminal_cost,
        )

    def _speed_cost(
        self,
        trajectory: SpatiotemporalTrajectory,
    ) -> float:
        speeds = trajectory.states[:, 3]

        speed_errors = speeds - self.config.target_speed

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

        if previous_control is None:
            previous = controls[0]
        else:
            previous = previous_control.as_array()

        extended_controls = np.vstack(
            (
                previous,
                controls,
            )
        )

        control_rates = (
            np.diff(
                extended_controls,
                axis=0,
            )
            / self.config.dt
        )

        acceleration_rates = control_rates[:, 0]
        steering_rates = control_rates[:, 1]

        acceleration_rate_cost = float(np.mean(acceleration_rates**2))
        steering_rate_cost = float(np.mean(steering_rates**2))

        # 对超过配置变化率上限的部分增加额外二次惩罚。
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
            object_radius = 0.5 * math.hypot(
                prediction.length,
                prediction.width,
            )

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

            soft_violation = softness * np.logaddexp(
                0.0,
                -clearance / softness,
            )

            object_cost = np.mean(soft_violation**2)

            # 真实几何包围圆发生重叠时，再增加明显惩罚。
            geometric_clearance = (
                center_distances - self._ego_collision_radius - object_radius
            )

            overlap = np.maximum(
                -geometric_clearance,
                0.0,
            )

            object_cost += 10.0 * np.mean(overlap**2)

            # 低置信度不能完全忽略，但可以适当降低其影响。
            confidence_scale = 0.25 + (0.75 * prediction.confidence)

            total += float(confidence_scale * object_cost)

        return total / len(predictions.trajectories)

    def _safety_margin(
        self,
        semantic_type: str,
    ) -> float:
        if semantic_type == "pedestrian":
            return self.config.pedestrian_margin

        if semantic_type == "small_car":
            return self.config.small_car_margin

        if semantic_type == "large_vehicle":
            return self.config.large_vehicle_margin

        # 对未知目标使用最保守的已配置距离。
        return max(
            self.config.pedestrian_margin,
            self.config.small_car_margin,
            self.config.large_vehicle_margin,
        )

    @staticmethod
    def _nearest_reference_indices(
        query_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        differences = query_points[:, None, :] - reference_points[None, :, :]

        squared_distances = np.sum(
            differences**2,
            axis=2,
        )

        return np.argmin(
            squared_distances,
            axis=1,
        )

    @staticmethod
    def _normalize_angle(
        angles: np.ndarray,
    ) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _validate_time_alignment(
        trajectory: SpatiotemporalTrajectory,
        predictions: ObjectPredictionSet,
    ) -> None:
        if trajectory.times.shape != predictions.times.shape:
            raise ValueError(
                "trajectory and predictions must have " "the same time-axis shape"
            )

        if not np.allclose(
            trajectory.times,
            predictions.times,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "trajectory and predictions must use " "the same time axis"
            )
