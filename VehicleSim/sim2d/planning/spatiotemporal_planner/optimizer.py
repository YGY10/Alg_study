from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)

from .config import SpatiotemporalPlannerConfig
from .cost import SpatiotemporalCost
from .rollout import TrajectoryRollout
from .types import (
    ObjectPredictionSet,
    OptimizationResult,
    SpatiotemporalTrajectory,
)


@dataclass(frozen=True)
class _CandidateEvaluation:
    controls: np.ndarray
    trajectory: SpatiotemporalTrajectory
    total_cost: float
    cost_terms: dict[str, float]


class SpatiotemporalOptimizer:
    """使用投影梯度下降优化自车控制序列。

    优化变量：

        controls[k] = [acceleration, steering]

    自车轨迹、参考路径和目标预测必须全部位于当前规划周期冻结的
    自车坐标系中。
    """

    def __init__(
        self,
        config: SpatiotemporalPlannerConfig,
        vehicle_config: VehicleConfig,
    ) -> None:
        config.validate()
        vehicle_config.validate()

        self.config = config
        self.vehicle_config = vehicle_config

        self.rollout = TrajectoryRollout(
            vehicle_config=vehicle_config,
            dt=config.dt,
        )

        self.cost = SpatiotemporalCost(
            config=config,
            vehicle_config=vehicle_config,
        )

    def optimize(
        self,
        ego_state: VehicleState,
        initial_controls: np.ndarray,
        predictions: ObjectPredictionSet,
        reference_path: np.ndarray | None,
        previous_control: VehicleControl | None = None,
    ) -> OptimizationResult:
        """优化控制序列并返回对应的局部时空轨迹。

        ``ego_state`` 可以是当前自车世界状态，也可以是局部状态；
        rollout 只使用其 speed，并从局部状态 (0, 0, 0, speed) 开始。
        """
        controls = self._validate_and_project_controls(initial_controls)

        self._validate_prediction_time_axis(predictions)

        current = self._evaluate(
            ego_state=ego_state,
            controls=controls,
            predictions=predictions,
            reference_path=reference_path,
            previous_control=previous_control,
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
                "gradient_norm_history": tuple(
                    float(value) for value in gradient_norm_history
                ),
                "accepted_step_history": tuple(
                    float(value) for value in accepted_step_history
                ),
                "optimized_controls": best.controls.copy(),
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
    ) -> np.ndarray:
        """使用带边界处理的单边有限差分估计梯度。"""
        base_controls = evaluation.controls
        base_cost = evaluation.total_cost

        gradient = np.zeros_like(
            base_controls,
            dtype=np.float64,
        )

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
                    # 可行区间小于扰动量时，直接移动到更远的边界。
                    distance_to_lower = base_value - lower_bound
                    distance_to_upper = upper_bound - base_value

                    if distance_to_upper >= distance_to_lower:
                        perturbed_value = upper_bound
                    else:
                        perturbed_value = lower_bound

                actual_delta = perturbed_value - base_value

                if abs(actual_delta) <= 1e-15:
                    gradient[row, column] = 0.0
                    continue

                perturbed_controls = base_controls.copy()
                perturbed_controls[row, column] = perturbed_value

                perturbed = self._evaluate(
                    ego_state=ego_state,
                    controls=perturbed_controls,
                    predictions=predictions,
                    reference_path=reference_path,
                    previous_control=previous_control,
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
    ) -> tuple[_CandidateEvaluation | None, float]:
        """沿负梯度方向执行回溯线搜索。"""
        step_size = self.config.initial_step_size

        while step_size >= self.config.min_step_size:
            proposed_controls = self._project_controls(
                current.controls - step_size * gradient
            )

            # 投影后控制没有变化，继续减小步长没有意义。
            if np.array_equal(
                proposed_controls,
                current.controls,
            ):
                return None, 0.0

            candidate = self._evaluate(
                ego_state=ego_state,
                controls=proposed_controls,
                predictions=predictions,
                reference_path=reference_path,
                previous_control=previous_control,
            )

            if candidate.total_cost < current.total_cost:
                return candidate, float(step_size)

            step_size *= self.config.line_search_decay

        return None, 0.0

    def _validate_and_project_controls(
        self,
        controls: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(
            controls,
            dtype=np.float64,
        )

        expected_shape = (
            self.config.horizon_steps,
            2,
        )

        if value.shape != expected_shape:
            raise ValueError(
                "initial_controls must have shape "
                f"{expected_shape}, got {value.shape}"
            )

        if not np.all(np.isfinite(value)):
            raise ValueError("initial_controls contain non-finite values")

        return self._project_controls(value)

    def _project_controls(
        self,
        controls: np.ndarray,
    ) -> np.ndarray:
        projected = np.asarray(
            controls,
            dtype=np.float64,
        ).copy()

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

    def _control_bounds(
        self,
        column: int,
    ) -> tuple[float, float]:
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

    def _validate_prediction_time_axis(
        self,
        predictions: ObjectPredictionSet,
    ) -> None:
        expected_times = (
            np.arange(
                self.config.horizon_steps + 1,
                dtype=np.float64,
            )
            * self.config.dt
        )

        if predictions.times.shape != expected_times.shape:
            raise ValueError(
                "prediction time axis must have shape "
                f"{expected_times.shape}, "
                f"got {predictions.times.shape}"
            )

        if not np.allclose(
            predictions.times,
            expected_times,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "prediction time axis does not match " "optimizer dt and horizon"
            )

    @staticmethod
    def shift_controls_for_warm_start(
        previous_controls: np.ndarray,
    ) -> np.ndarray:
        """将上一周期控制序列左移一位，构造下一周期 warm start。"""
        controls = np.asarray(
            previous_controls,
            dtype=np.float64,
        )

        if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] == 0:
            raise ValueError("previous_controls must have shape [N, 2], N > 0")

        if not np.all(np.isfinite(controls)):
            raise ValueError("previous_controls contain non-finite values")

        shifted = np.empty_like(controls)
        shifted[:-1] = controls[1:]
        shifted[-1] = controls[-1]

        return shifted
