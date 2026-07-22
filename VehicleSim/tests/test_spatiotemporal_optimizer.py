from __future__ import annotations

import numpy as np
import pytest

from sim2d.planning.spatiotemporal_planner import (
    ObjectPredictionSet,
    PredictedObjectTrajectory,
    SpatiotemporalOptimizer,
    SpatiotemporalPlannerConfig,
)
from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


def make_config(
    **overrides: float | int,
) -> SpatiotemporalPlannerConfig:
    values = {
        "dt": 0.1,
        "horizon_steps": 6,
        "target_speed": 5.0,
        "max_iterations": 8,
        "gradient_epsilon": 1e-3,
        "initial_step_size": 0.02,
        "gradient_tolerance": 1e-5,
        "cost_tolerance": 1e-8,
        "line_search_decay": 0.5,
        "min_step_size": 1e-7,
    }
    values.update(overrides)

    return SpatiotemporalPlannerConfig(**values)


def make_optimizer(
    **config_overrides: float | int,
) -> SpatiotemporalOptimizer:
    return SpatiotemporalOptimizer(
        config=make_config(**config_overrides),
        vehicle_config=VehicleConfig(),
    )


def make_ego(
    speed: float = 0.0,
) -> VehicleState:
    # 故意使用非零世界位姿，确认 optimizer 仍从局部原点 rollout。
    return VehicleState(
        x=100.0,
        y=-30.0,
        yaw=1.2,
        speed=speed,
    )


def make_times(
    config: SpatiotemporalPlannerConfig,
) -> np.ndarray:
    return (
        np.arange(
            config.horizon_steps + 1,
            dtype=np.float64,
        )
        * config.dt
    )


def make_empty_predictions(
    config: SpatiotemporalPlannerConfig,
) -> ObjectPredictionSet:
    return ObjectPredictionSet(
        times=make_times(config),
        trajectories=(),
    )


def make_straight_reference(
    length: float = 20.0,
    count: int = 201,
) -> np.ndarray:
    x = np.linspace(
        0.0,
        length,
        count,
        dtype=np.float64,
    )

    return np.column_stack(
        (
            x,
            np.zeros_like(x),
            np.zeros_like(x),
        )
    )


def test_optimizer_reduces_cost_on_straight_road() -> None:
    optimizer = make_optimizer(
        weight_collision=0.0,
        weight_reference=1.0,
        weight_heading=1.0,
        weight_speed=2.0,
        weight_acceleration=0.01,
        weight_steering=0.01,
        weight_acceleration_rate=0.01,
        weight_steering_rate=0.01,
        weight_terminal=1.0,
    )

    config = optimizer.config

    initial_controls = np.zeros(
        (config.horizon_steps, 2),
        dtype=np.float64,
    )

    result = optimizer.optimize(
        ego_state=make_ego(speed=0.0),
        initial_controls=initial_controls,
        predictions=make_empty_predictions(config),
        reference_path=make_straight_reference(),
    )

    assert np.isfinite(result.total_cost)
    assert result.total_cost <= result.debug["initial_cost"]
    assert result.trajectory.states[0, 0] == pytest.approx(0.0)
    assert result.trajectory.states[0, 1] == pytest.approx(0.0)
    assert result.trajectory.states[0, 2] == pytest.approx(0.0)


def test_optimizer_output_controls_respect_vehicle_limits() -> None:
    optimizer = make_optimizer()

    config = optimizer.config
    vehicle_config = optimizer.vehicle_config

    initial_controls = np.tile(
        np.array(
            [[100.0, -100.0]],
            dtype=np.float64,
        ),
        (config.horizon_steps, 1),
    )

    result = optimizer.optimize(
        ego_state=make_ego(speed=2.0),
        initial_controls=initial_controls,
        predictions=make_empty_predictions(config),
        reference_path=None,
    )

    controls = result.trajectory.controls

    assert np.all(controls[:, 0] >= vehicle_config.acceleration_min)
    assert np.all(controls[:, 0] <= vehicle_config.acceleration_max)
    assert np.all(controls[:, 1] >= vehicle_config.steering_min)
    assert np.all(controls[:, 1] <= vehicle_config.steering_max)


def test_optimizer_uses_previous_control_in_objective() -> None:
    optimizer = make_optimizer(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
        weight_acceleration_rate=1.0,
        weight_steering_rate=1.0,
    )

    config = optimizer.config

    initial_controls = np.tile(
        np.array(
            [[1.0, 0.2]],
            dtype=np.float64,
        ),
        (config.horizon_steps, 1),
    )

    matching = optimizer.optimize(
        ego_state=make_ego(speed=2.0),
        initial_controls=initial_controls,
        predictions=make_empty_predictions(config),
        reference_path=None,
        previous_control=VehicleControl(
            acceleration=1.0,
            steering=0.2,
        ),
    )

    different = optimizer.optimize(
        ego_state=make_ego(speed=2.0),
        initial_controls=initial_controls,
        predictions=make_empty_predictions(config),
        reference_path=None,
        previous_control=VehicleControl(
            acceleration=-1.0,
            steering=-0.2,
        ),
    )

    assert matching.debug["initial_cost"] < different.debug["initial_cost"]


def test_optimizer_handles_stationary_obstacle() -> None:
    optimizer = make_optimizer(
        horizon_steps=8,
        target_speed=4.0,
        max_iterations=10,
        initial_step_size=0.01,
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.2,
        weight_acceleration=0.01,
        weight_steering=0.01,
        weight_acceleration_rate=0.01,
        weight_steering_rate=0.01,
        weight_collision=1000.0,
        weight_terminal=0.0,
    )

    config = optimizer.config
    times = make_times(config)

    initial_controls = np.zeros(
        (config.horizon_steps, 2),
        dtype=np.float64,
    )

    obstacle = PredictedObjectTrajectory(
        object_id="stopped-car",
        object_type="box",
        semantic_type="small_car",
        times=times,
        positions=np.column_stack(
            (
                np.full_like(times, 5.0),
                np.zeros_like(times),
            )
        ),
        yaws=np.zeros_like(times),
        speed=0.0,
        length=4.5,
        width=1.8,
        confidence=1.0,
    )

    predictions = ObjectPredictionSet(
        times=times,
        trajectories=(obstacle,),
    )

    result = optimizer.optimize(
        ego_state=make_ego(speed=4.0),
        initial_controls=initial_controls,
        predictions=predictions,
        reference_path=None,
    )

    assert np.isfinite(result.total_cost)
    assert result.total_cost <= result.debug["initial_cost"]


def test_optimizer_rejects_wrong_control_shape() -> None:
    optimizer = make_optimizer()
    config = optimizer.config

    wrong_controls = np.zeros(
        (config.horizon_steps + 1, 2),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="initial_controls must have shape",
    ):
        optimizer.optimize(
            ego_state=make_ego(),
            initial_controls=wrong_controls,
            predictions=make_empty_predictions(config),
            reference_path=None,
        )


def test_optimizer_rejects_non_finite_controls() -> None:
    optimizer = make_optimizer()
    config = optimizer.config

    controls = np.zeros(
        (config.horizon_steps, 2),
        dtype=np.float64,
    )
    controls[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        optimizer.optimize(
            ego_state=make_ego(),
            initial_controls=controls,
            predictions=make_empty_predictions(config),
            reference_path=None,
        )


def test_optimizer_rejects_wrong_prediction_horizon() -> None:
    optimizer = make_optimizer()
    config = optimizer.config

    predictions = ObjectPredictionSet(
        times=np.arange(
            config.horizon_steps,
            dtype=np.float64,
        )
        * config.dt,
        trajectories=(),
    )

    with pytest.raises(
        ValueError,
        match="prediction time axis",
    ):
        optimizer.optimize(
            ego_state=make_ego(),
            initial_controls=np.zeros(
                (config.horizon_steps, 2),
                dtype=np.float64,
            ),
            predictions=predictions,
            reference_path=None,
        )


def test_shift_controls_for_warm_start() -> None:
    controls = np.array(
        [
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
        ],
        dtype=np.float64,
    )

    shifted = SpatiotemporalOptimizer.shift_controls_for_warm_start(controls)

    expected = np.array(
        [
            [2.0, 0.2],
            [3.0, 0.3],
            [3.0, 0.3],
        ],
        dtype=np.float64,
    )

    assert np.array_equal(shifted, expected)


def test_shift_controls_returns_copy() -> None:
    controls = np.array(
        [
            [1.0, 0.1],
            [2.0, 0.2],
        ],
        dtype=np.float64,
    )

    shifted = SpatiotemporalOptimizer.shift_controls_for_warm_start(controls)

    controls[1, 0] = 100.0

    assert shifted[-1, 0] == pytest.approx(2.0)
