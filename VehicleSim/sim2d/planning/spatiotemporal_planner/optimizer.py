from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2d.types import VehicleConfig, VehicleControl, VehicleState

from .config import SpatiotemporalPlannerConfig
from .cost import SpatiotemporalCost
from .rollout import TrajectoryRollout
from .types import ObjectPredictionSet, OptimizationResult, SpatiotemporalTrajectory


@dataclass(frozen=True)
class _CandidateEvaluation:
    controls: np.ndarray
    trajectory: SpatiotemporalTrajectory
    total_cost: float
    cost_terms: dict[str, float]


class SpatiotemporalOptimizer:
    """使用投影梯度下降优化自车控制序列。"""

    def __init__(
        self,
        config: SpatiotemporalPlannerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        config.validate()
        vehicle_config.validate()
        self.config = config
        self.vehicle_config = vehicle_config
        self.rollout = TrajectoryRollout(vehicle_config=vehicle_config, dt=config.dt)
        self.cost = SpatiotemporalCost(config=config, vehicle_config=vehicle_config)

    def optimize(
        self,
        ego_state: VehicleState,
        initial_controls: np.ndarray,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None = None,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
    ) -> OptimizationResult:
        controls = self._validate_and_project_controls(initial_controls)
        self._validate_prediction_time_axis(predictions)
        self._validate_corridor(reference_path, left_boundary, right_boundary)

        current = self._evaluate(
            ego_state=ego_state,
            controls=controls,
            predictions=predictions,
            reference_path=reference_path,
            previous_control=previous_control,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )
        initial_cost = current.total_cost
        best = current
        cost_history = [current.total_cost]
        gradient_norm_history: list[float] = []
        accepted_step_history: list[float] = []
        status = "max_iterations"
        success = False
        completed_iterations = 0

        for iteration in range(1, self.config.max_iterations + 1):
            gradient = self._finite_difference_gradient(
                ego_state=ego_state,
                evaluation=current,
                predictions=predictions,
                reference_path=reference_path,
                previous_control=previous_control,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
            )
            gradient_norm = float(np.linalg.norm(gradient))
            gradient_norm_history.append(gradient_norm)
            if not np.isfinite(gradient_norm):
                status = "non_finite_gradient"
                break
            if gradient_norm <= self.config.gradient_tolerance:
                status = "gradient_converged"
                success = True
                completed_iterations = iteration - 1
                break

            candidate, accepted_step = self._line_search(
                ego_state=ego_state,
                current=current,
                gradient=gradient,
                predictions=predictions,
                reference_path=reference_path,
                previous_control=previous_control,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
            )
            if candidate is None:
                status = "line_search_failed"
                completed_iterations = iteration - 1
                break

            cost_improvement = current.total_cost - candidate.total_cost
            current = candidate
            completed_iterations = iteration
            cost_history.append(current.total_cost)
            accepted_step_history.append(accepted_step)
            if current.total_cost < best.total_cost:
                best = current
            if cost_improvement <= self.config.cost_tolerance:
                status = "cost_converged"
                success = True
                break
        else:
            completed_iterations = self.config.max_iterations

        if best.total_cost < initial_cost:
            success = True
        return OptimizationResult(
            trajectory=best.trajectory,
            success=success,
            total_cost=best.total_cost,
            iterations=completed_iterations,
            status=status,
            cost_terms=best.cost_terms,
            debug={
                "initial_cost": float(initial_cost),
                "final_cost": float(best.total_cost),
                "cost_history": tuple(float(value) for value in cost_history),
                "gradient_norm_history": tuple(float(value) for value in gradient_norm_history),
                "accepted_step_history": tuple(float(value) for value in accepted_step_history),
                "optimized_controls": best.controls.copy(),
                "corridor_used": left_boundary is not None and right_boundary is not None,
            },
        )

    def _evaluate(
        self,
        *,
        ego_state: VehicleState,
        controls: np.ndarray,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> _CandidateEvaluation:
        projected_controls = self._project_controls(controls)
        trajectory = self.rollout.rollout_from_ego(
            ego_state=ego_state,
            controls=projected_controls,
        )
        total_cost, cost_terms = self.cost.evaluate(
            trajectory=trajectory,
            predictions=predictions,
            reference_path=reference_path,
            previous_control=previous_control,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )
        return _CandidateEvaluation(
            controls=projected_controls,
            trajectory=trajectory,
            total_cost=float(total_cost),
            cost_terms=dict(cost_terms),
        )

    def _finite_difference_gradient(
        self,
        *,
        ego_state: VehicleState,
        evaluation: _CandidateEvaluation,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> np.ndarray:
        base_controls = evaluation.controls
        base_cost = evaluation.total_cost
        gradient = np.zeros_like(base_controls, dtype=np.float64)
        epsilon = self.config.gradient_epsilon

        for row in range(base_controls.shape[0]):
            for column in range(base_controls.shape[1]):
                base_value = float(base_controls[row, column])
                lower_bound, upper_bound = self._control_bounds(column)
                if base_value + epsilon <= upper_bound:
                    perturbed_value = base_value + epsilon
                elif base_value - epsilon >= lower_bound:
                    perturbed_value = base_value - epsilon
                else:
                    distance_to_lower = base_value - lower_bound
                    distance_to_upper = upper_bound - base_value
                    perturbed_value = (
                        upper_bound
                        if distance_to_upper >= distance_to_lower
                        else lower_bound
                    )
                actual_delta = perturbed_value - base_value
                if abs(actual_delta) <= 1e-15:
                    continue
                perturbed_controls = base_controls.copy()
                perturbed_controls[row, column] = perturbed_value
                perturbed = self._evaluate(
                    ego_state=ego_state,
                    controls=perturbed_controls,
                    predictions=predictions,
                    reference_path=reference_path,
                    previous_control=previous_control,
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                )
                gradient[row, column] = (
                    perturbed.total_cost - base_cost
                ) / actual_delta
        return gradient

    def _line_search(
        self,
        *,
        ego_state: VehicleState,
        current: _CandidateEvaluation,
        gradient: np.ndarray,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> tuple[_CandidateEvaluation | None, float]:
        step_size = self.config.initial_step_size
        while step_size >= self.config.min_step_size:
            proposed_controls = self._project_controls(
                current.controls - step_size * gradient
            )
            if np.array_equal(proposed_controls, current.controls):
                return None, 0.0
            candidate = self._evaluate(
                ego_state=ego_state,
                controls=proposed_controls,
                predictions=predictions,
                reference_path=reference_path,
                previous_control=previous_control,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
            )
            if candidate.total_cost < current.total_cost:
                return candidate, float(step_size)
            step_size *= self.config.line_search_decay
        return None, 0.0

    def _validate_and_project_controls(self, controls: np.ndarray) -> np.ndarray:
        value = np.asarray(controls, dtype=np.float64)
        expected_shape = (self.config.horizon_steps, 2)
        if value.shape != expected_shape:
            raise ValueError(
                f"initial_controls must have shape {expected_shape}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("initial_controls contain non-finite values")
        return self._project_controls(value)

    def _project_controls(self, controls: np.ndarray) -> np.ndarray:
        projected = np.asarray(controls, dtype=np.float64).copy()
        projected[:, 0] = np.clip(
            projected[:, 0],
            self.vehicle_config.acceleration_min,
            self.vehicle_config.acceleration_max,
        )
        projected[:, 1] = np.clip(
            projected[:, 1],
            self.vehicle_config.steering_min,
            self.vehicle_config.steering_max,
        )
        return projected

    def _control_bounds(self, column: int) -> tuple[float, float]:
        if column == 0:
            return (
                self.vehicle_config.acceleration_min,
                self.vehicle_config.acceleration_max,
            )
        if column == 1:
            return (
                self.vehicle_config.steering_min,
                self.vehicle_config.steering_max,
            )
        raise ValueError(f"unknown control column: {column}")

    def _validate_prediction_time_axis(self, predictions: ObjectPredictionSet) -> None:
        expected_times = (
            np.arange(self.config.horizon_steps + 1, dtype=np.float64)
            * self.config.dt
        )
        if predictions.times.shape != expected_times.shape:
            raise ValueError(
                f"prediction time axis must have shape {expected_times.shape}, "
                f"got {predictions.times.shape}"
            )
        if not np.allclose(
            predictions.times,
            expected_times,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("prediction time axis does not match optimizer dt and horizon")

    @staticmethod
    def _validate_corridor(
        reference_path: np.ndarray | None,
        left_boundary: np.ndarray | None,
        right_boundary: np.ndarray | None,
    ) -> None:
        if left_boundary is None and right_boundary is None:
            return
        if reference_path is None or left_boundary is None or right_boundary is None:
            raise ValueError("reference_path and both boundaries are required together")
        expected = np.asarray(reference_path).shape[0]
        for name, boundary in (
            ("left_boundary", left_boundary),
            ("right_boundary", right_boundary),
        ):
            value = np.asarray(boundary)
            if value.shape != (expected, 2):
                raise ValueError(f"{name} must have shape {(expected, 2)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def shift_controls_for_warm_start(previous_controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(previous_controls, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] == 0:
            raise ValueError("previous_controls must have shape [N, 2], N > 0")
        if not np.all(np.isfinite(controls)):
            raise ValueError("previous_controls contain non-finite values")
        shifted = np.empty_like(controls)
        shifted[:-1] = controls[1:]
        shifted[-1] = controls[-1]
        return shifted
