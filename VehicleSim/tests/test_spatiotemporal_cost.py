from __future__ import annotations

import math

import numpy as np
import pytest

from sim2d.planning.spatiotemporal_planner import (
    ObjectPredictionSet,
    PredictedObjectTrajectory,
    SpatiotemporalPlannerConfig,
    SpatiotemporalTrajectory,
)
from sim2d.planning.spatiotemporal_planner.cost import (
    SpatiotemporalCost,
)
from sim2d.types import VehicleConfig, VehicleControl


def make_cost(
    **config_overrides: float,
) -> SpatiotemporalCost:
    config = SpatiotemporalPlannerConfig(
        **config_overrides,
    )

    return SpatiotemporalCost(
        config=config,
        vehicle_config=VehicleConfig(),
    )


def make_trajectory(
    *,
    x: np.ndarray,
    y: np.ndarray | None = None,
    yaw: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    controls: np.ndarray | None = None,
    dt: float = 0.1,
) -> SpatiotemporalTrajectory:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must have shape [N]")

    count = x.shape[0]

    if count == 0:
        raise ValueError("trajectory must contain at least one state")

    if y is None:
        y_array = np.zeros(
            count,
            dtype=np.float64,
        )
    else:
        y_array = np.asarray(
            y,
            dtype=np.float64,
        )

    if yaw is None:
        yaw_array = np.zeros(
            count,
            dtype=np.float64,
        )
    else:
        yaw_array = np.asarray(
            yaw,
            dtype=np.float64,
        )

    if speed is None:
        speed_array = np.zeros(
            count,
            dtype=np.float64,
        )
    else:
        speed_array = np.asarray(
            speed,
            dtype=np.float64,
        )

    for name, value in (
        ("y", y_array),
        ("yaw", yaw_array),
        ("speed", speed_array),
    ):
        if value.shape != (count,):
            raise ValueError(
                f"{name} must have shape {(count,)}, " f"got {value.shape}"
            )

    if controls is None:
        controls_array = np.zeros(
            (count - 1, 2),
            dtype=np.float64,
        )
    else:
        controls_array = np.asarray(
            controls,
            dtype=np.float64,
        )

    states = np.column_stack(
        (
            x,
            y_array,
            yaw_array,
            speed_array,
        )
    )

    times = (
        np.arange(
            count,
            dtype=np.float64,
        )
        * dt
    )

    return SpatiotemporalTrajectory(
        times=times,
        states=states,
        controls=controls_array,
    )


def make_empty_predictions(
    times: np.ndarray,
) -> ObjectPredictionSet:
    return ObjectPredictionSet(
        times=np.asarray(
            times,
            dtype=np.float64,
        ),
        trajectories=(),
    )


def make_prediction(
    *,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray | None = None,
    object_id: str = "object-1",
    object_type: str = "box",
    semantic_type: str = "small_car",
    speed: float = 0.0,
    length: float = 4.5,
    width: float = 1.8,
    confidence: float = 1.0,
) -> PredictedObjectTrajectory:
    times_array = np.asarray(
        times,
        dtype=np.float64,
    )
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )
    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    if yaw is None:
        yaw_array = np.zeros_like(
            times_array,
        )
    else:
        yaw_array = np.asarray(
            yaw,
            dtype=np.float64,
        )

    positions = np.column_stack(
        (
            x_array,
            y_array,
        )
    )

    return PredictedObjectTrajectory(
        object_id=object_id,
        object_type=object_type,
        semantic_type=semantic_type,
        times=times_array,
        positions=positions,
        yaws=yaw_array,
        speed=speed,
        length=length,
        width=width,
        confidence=confidence,
    )


def make_straight_reference(
    *,
    length: float = 20.0,
    point_count: int = 201,
    lateral_offset: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    x = np.linspace(
        0.0,
        length,
        point_count,
        dtype=np.float64,
    )

    return np.column_stack(
        (
            x,
            np.full_like(
                x,
                lateral_offset,
            ),
            np.full_like(
                x,
                yaw,
            ),
        )
    )


def test_reference_cost_prefers_trajectory_on_reference() -> None:
    cost = make_cost()

    x = np.linspace(
        0.0,
        10.0,
        11,
        dtype=np.float64,
    )

    on_reference = make_trajectory(
        x=x,
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    offset = make_trajectory(
        x=x,
        y=np.full(
            11,
            2.0,
            dtype=np.float64,
        ),
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    reference_path = make_straight_reference()

    predictions = make_empty_predictions(
        on_reference.times,
    )

    on_total, on_terms = cost.evaluate(
        trajectory=on_reference,
        predictions=predictions,
        reference_path=reference_path,
    )

    offset_total, offset_terms = cost.evaluate(
        trajectory=offset,
        predictions=predictions,
        reference_path=reference_path,
    )

    assert on_terms["reference"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert offset_terms["reference"] > 0.0
    assert on_total < offset_total


def test_heading_cost_prefers_correct_heading() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        10.0,
        11,
        dtype=np.float64,
    )

    aligned = make_trajectory(
        x=x,
        yaw=np.zeros(
            11,
            dtype=np.float64,
        ),
    )

    wrong_heading = make_trajectory(
        x=x,
        yaw=np.full(
            11,
            math.pi / 4.0,
            dtype=np.float64,
        ),
    )

    reference_path = make_straight_reference()

    predictions = make_empty_predictions(
        aligned.times,
    )

    aligned_total, aligned_terms = cost.evaluate(
        trajectory=aligned,
        predictions=predictions,
        reference_path=reference_path,
    )

    wrong_total, wrong_terms = cost.evaluate(
        trajectory=wrong_heading,
        predictions=predictions,
        reference_path=reference_path,
    )

    assert aligned_terms["heading"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert wrong_terms["heading"] > 0.0
    assert aligned_total < wrong_total


def test_heading_cost_handles_angle_wrapping() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        2.0,
        3,
        dtype=np.float64,
    )

    trajectory = make_trajectory(
        x=x,
        yaw=np.full(
            3,
            -math.pi + 0.01,
            dtype=np.float64,
        ),
    )

    reference_path = make_straight_reference(
        length=2.0,
        yaw=math.pi - 0.01,
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    _, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=reference_path,
    )

    expected_unweighted_error = 0.02**2
    expected_weighted_error = cost.config.weight_heading * expected_unweighted_error

    assert terms["heading"] == pytest.approx(
        expected_weighted_error,
        rel=1e-6,
    )


def test_speed_cost_prefers_target_speed() -> None:
    cost = make_cost(
        target_speed=10.0,
        weight_reference=0.0,
        weight_heading=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        10.0,
        11,
        dtype=np.float64,
    )

    target_speed = make_trajectory(
        x=x,
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    slow = make_trajectory(
        x=x,
        speed=np.full(
            11,
            4.0,
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        target_speed.times,
    )

    target_total, target_terms = cost.evaluate(
        trajectory=target_speed,
        predictions=predictions,
        reference_path=None,
    )

    slow_total, slow_terms = cost.evaluate(
        trajectory=slow,
        predictions=predictions,
        reference_path=None,
    )

    assert target_terms["speed"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert slow_terms["speed"] > 0.0
    assert target_total < slow_total


def test_acceleration_magnitude_cost_penalizes_large_acceleration() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        1.0,
        6,
        dtype=np.float64,
    )

    zero_controls = np.zeros(
        (5, 2),
        dtype=np.float64,
    )

    acceleration_controls = np.zeros(
        (5, 2),
        dtype=np.float64,
    )
    acceleration_controls[:, 0] = 2.0

    zero_trajectory = make_trajectory(
        x=x,
        controls=zero_controls,
    )

    acceleration_trajectory = make_trajectory(
        x=x,
        controls=acceleration_controls,
    )

    predictions = make_empty_predictions(
        zero_trajectory.times,
    )

    zero_total, zero_terms = cost.evaluate(
        trajectory=zero_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    acceleration_total, acceleration_terms = cost.evaluate(
        trajectory=acceleration_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert zero_terms["acceleration"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert acceleration_terms["acceleration"] > 0.0
    assert zero_total < acceleration_total


def test_steering_magnitude_cost_penalizes_large_steering() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        1.0,
        6,
        dtype=np.float64,
    )

    zero_controls = np.zeros(
        (5, 2),
        dtype=np.float64,
    )

    steering_controls = np.zeros(
        (5, 2),
        dtype=np.float64,
    )
    steering_controls[:, 1] = 0.3

    zero_trajectory = make_trajectory(
        x=x,
        controls=zero_controls,
    )

    steering_trajectory = make_trajectory(
        x=x,
        controls=steering_controls,
    )

    predictions = make_empty_predictions(
        zero_trajectory.times,
    )

    zero_total, zero_terms = cost.evaluate(
        trajectory=zero_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    steering_total, steering_terms = cost.evaluate(
        trajectory=steering_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert zero_terms["steering"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert steering_terms["steering"] > 0.0
    assert zero_total < steering_total


def test_control_rate_cost_penalizes_oscillation() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        1.0,
        6,
        dtype=np.float64,
    )

    smooth_controls = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.02],
            [0.2, 0.04],
            [0.3, 0.06],
            [0.4, 0.08],
        ],
        dtype=np.float64,
    )

    oscillating_controls = np.array(
        [
            [1.0, 0.3],
            [-1.0, -0.3],
            [1.0, 0.3],
            [-1.0, -0.3],
            [1.0, 0.3],
        ],
        dtype=np.float64,
    )

    smooth = make_trajectory(
        x=x,
        controls=smooth_controls,
    )

    oscillating = make_trajectory(
        x=x,
        controls=oscillating_controls,
    )

    predictions = make_empty_predictions(
        smooth.times,
    )

    smooth_total, smooth_terms = cost.evaluate(
        trajectory=smooth,
        predictions=predictions,
        reference_path=None,
    )

    oscillating_total, oscillating_terms = cost.evaluate(
        trajectory=oscillating,
        predictions=predictions,
        reference_path=None,
    )

    assert smooth_terms["acceleration_rate"] < oscillating_terms["acceleration_rate"]
    assert smooth_terms["steering_rate"] < oscillating_terms["steering_rate"]
    assert smooth_total < oscillating_total


def test_previous_control_penalizes_first_control_jump() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    x = np.linspace(
        0.0,
        1.0,
        6,
        dtype=np.float64,
    )

    controls = np.tile(
        np.array(
            [[1.0, 0.3]],
            dtype=np.float64,
        ),
        (5, 1),
    )

    trajectory = make_trajectory(
        x=x,
        controls=controls,
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    matching_previous_control = VehicleControl(
        acceleration=1.0,
        steering=0.3,
    )

    different_previous_control = VehicleControl(
        acceleration=-1.0,
        steering=-0.3,
    )

    matching_total, matching_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
        previous_control=matching_previous_control,
    )

    different_total, different_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
        previous_control=different_previous_control,
    )

    assert matching_terms["acceleration_rate"] < different_terms["acceleration_rate"]
    assert matching_terms["steering_rate"] < different_terms["steering_rate"]
    assert matching_total < different_total


def test_no_previous_control_does_not_penalize_first_control() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
    )

    controls = np.tile(
        np.array(
            [[1.0, 0.2]],
            dtype=np.float64,
        ),
        (5, 1),
    )

    trajectory = make_trajectory(
        x=np.linspace(
            0.0,
            1.0,
            6,
            dtype=np.float64,
        ),
        controls=controls,
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    _, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
        previous_control=None,
    )

    assert terms["acceleration_rate"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert terms["steering_rate"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_collision_cost_is_zero_without_objects() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.linspace(
            0.0,
            10.0,
            11,
            dtype=np.float64,
        ),
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    _, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert terms["collision"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_collision_cost_prefers_trajectory_away_from_obstacle() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_terminal=0.0,
    )

    times = (
        np.arange(
            11,
            dtype=np.float64,
        )
        * 0.1
    )

    collision_trajectory = make_trajectory(
        x=np.linspace(
            0.0,
            10.0,
            11,
            dtype=np.float64,
        ),
        y=np.zeros(
            11,
            dtype=np.float64,
        ),
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    safe_trajectory = make_trajectory(
        x=np.linspace(
            0.0,
            10.0,
            11,
            dtype=np.float64,
        ),
        y=np.full(
            11,
            8.0,
            dtype=np.float64,
        ),
        speed=np.full(
            11,
            10.0,
            dtype=np.float64,
        ),
    )

    obstacle = make_prediction(
        times=times,
        x=np.full(
            11,
            8.0,
            dtype=np.float64,
        ),
        y=np.zeros(
            11,
            dtype=np.float64,
        ),
    )

    predictions = ObjectPredictionSet(
        times=times,
        trajectories=(obstacle,),
    )

    collision_total, collision_terms = cost.evaluate(
        trajectory=collision_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    safe_total, safe_terms = cost.evaluate(
        trajectory=safe_trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert collision_terms["collision"] > safe_terms["collision"]
    assert collision_total > safe_total


def test_collision_cost_penalizes_crossing_pedestrian() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_terminal=0.0,
    )

    times = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        dtype=np.float64,
    )

    ego_crossing = make_trajectory(
        x=np.array(
            [0.0, 2.0, 4.0, 6.0, 8.0],
            dtype=np.float64,
        ),
        y=np.zeros(
            5,
            dtype=np.float64,
        ),
        speed=np.full(
            5,
            4.0,
            dtype=np.float64,
        ),
        dt=0.5,
    )

    ego_waiting = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0, 2.5, 3.0],
            dtype=np.float64,
        ),
        y=np.zeros(
            5,
            dtype=np.float64,
        ),
        speed=np.array(
            [4.0, 2.0, 1.0, 0.5, 0.0],
            dtype=np.float64,
        ),
        dt=0.5,
    )

    pedestrian = make_prediction(
        times=times,
        x=np.full(
            5,
            4.0,
            dtype=np.float64,
        ),
        y=np.array(
            [-4.0, -2.0, 0.0, 2.0, 4.0],
            dtype=np.float64,
        ),
        yaw=np.full(
            5,
            math.pi / 2.0,
            dtype=np.float64,
        ),
        object_id="pedestrian-1",
        object_type="circle",
        semantic_type="pedestrian",
        speed=4.0,
        length=0.7,
        width=0.7,
    )

    predictions = ObjectPredictionSet(
        times=times,
        trajectories=(pedestrian,),
    )

    crossing_total, crossing_terms = cost.evaluate(
        trajectory=ego_crossing,
        predictions=predictions,
        reference_path=None,
    )

    waiting_total, waiting_terms = cost.evaluate(
        trajectory=ego_waiting,
        predictions=predictions,
        reference_path=None,
    )

    assert crossing_terms["collision"] > waiting_terms["collision"]
    assert crossing_total > waiting_total


def test_collision_cost_prefers_early_deceleration() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_terminal=0.0,
    )

    times = (
        np.arange(
            11,
            dtype=np.float64,
        )
        * 0.5
    )

    maintain_speed = make_trajectory(
        x=np.linspace(
            0.0,
            20.0,
            11,
            dtype=np.float64,
        ),
        speed=np.full(
            11,
            4.0,
            dtype=np.float64,
        ),
        dt=0.5,
    )

    decelerate = make_trajectory(
        x=np.array(
            [
                0.0,
                2.0,
                4.0,
                5.5,
                6.5,
                7.0,
                7.2,
                7.3,
                7.3,
                7.3,
                7.3,
            ],
            dtype=np.float64,
        ),
        speed=np.array(
            [
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.4,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
        dt=0.5,
    )

    obstacle = make_prediction(
        times=times,
        x=np.full(
            11,
            12.0,
            dtype=np.float64,
        ),
        y=np.zeros(
            11,
            dtype=np.float64,
        ),
        object_id="stopped-car",
        semantic_type="small_car",
        speed=0.0,
    )

    predictions = ObjectPredictionSet(
        times=times,
        trajectories=(obstacle,),
    )

    maintain_total, maintain_terms = cost.evaluate(
        trajectory=maintain_speed,
        predictions=predictions,
        reference_path=None,
    )

    decelerate_total, decelerate_terms = cost.evaluate(
        trajectory=decelerate,
        predictions=predictions,
        reference_path=None,
    )

    assert maintain_terms["collision"] > decelerate_terms["collision"]
    assert maintain_total > decelerate_total


def test_pedestrian_margin_is_more_conservative_than_small_car_margin() -> None:
    cost = make_cost(
        pedestrian_margin=2.0,
        small_car_margin=0.5,
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_terminal=0.0,
    )

    trajectory = make_trajectory(
        x=np.zeros(
            3,
            dtype=np.float64,
        ),
    )

    times = trajectory.times

    common_x = np.full(
        3,
        4.0,
        dtype=np.float64,
    )
    common_y = np.zeros(
        3,
        dtype=np.float64,
    )

    pedestrian = make_prediction(
        times=times,
        x=common_x,
        y=common_y,
        object_id="pedestrian-1",
        object_type="circle",
        semantic_type="pedestrian",
        length=0.7,
        width=0.7,
    )

    small_car = make_prediction(
        times=times,
        x=common_x,
        y=common_y,
        object_id="car-1",
        object_type="box",
        semantic_type="small_car",
        length=0.7,
        width=0.7,
    )

    pedestrian_predictions = ObjectPredictionSet(
        times=times,
        trajectories=(pedestrian,),
    )

    small_car_predictions = ObjectPredictionSet(
        times=times,
        trajectories=(small_car,),
    )

    _, pedestrian_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=pedestrian_predictions,
        reference_path=None,
    )

    _, small_car_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=small_car_predictions,
        reference_path=None,
    )

    assert pedestrian_terms["collision"] > small_car_terms["collision"]


def test_low_confidence_object_still_has_nonzero_collision_cost() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_terminal=0.0,
    )

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 2.0, 4.0],
            dtype=np.float64,
        ),
    )

    times = trajectory.times

    high_confidence = make_prediction(
        times=times,
        x=np.full(
            3,
            4.0,
            dtype=np.float64,
        ),
        y=np.zeros(
            3,
            dtype=np.float64,
        ),
        confidence=1.0,
    )

    low_confidence = make_prediction(
        times=times,
        x=np.full(
            3,
            4.0,
            dtype=np.float64,
        ),
        y=np.zeros(
            3,
            dtype=np.float64,
        ),
        confidence=0.0,
    )

    _, high_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=ObjectPredictionSet(
            times=times,
            trajectories=(high_confidence,),
        ),
        reference_path=None,
    )

    _, low_terms = cost.evaluate(
        trajectory=trajectory,
        predictions=ObjectPredictionSet(
            times=times,
            trajectories=(low_confidence,),
        ),
        reference_path=None,
    )

    assert low_terms["collision"] > 0.0
    assert high_terms["collision"] > low_terms["collision"]


def test_terminal_cost_prefers_better_final_alignment() -> None:
    cost = make_cost(
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
    )

    good_terminal = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
        y=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        yaw=np.zeros(
            3,
            dtype=np.float64,
        ),
    )

    bad_terminal = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
        y=np.array(
            [0.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        yaw=np.array(
            [0.0, 0.0, 0.5],
            dtype=np.float64,
        ),
    )

    reference_path = make_straight_reference(
        length=2.0,
    )

    predictions = make_empty_predictions(
        good_terminal.times,
    )

    good_total, good_terms = cost.evaluate(
        trajectory=good_terminal,
        predictions=predictions,
        reference_path=reference_path,
    )

    bad_total, bad_terms = cost.evaluate(
        trajectory=bad_terminal,
        predictions=predictions,
        reference_path=reference_path,
    )

    assert good_terms["terminal"] < bad_terms["terminal"]
    assert good_total < bad_total


def test_cost_returns_expected_term_names() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    total, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert np.isfinite(total)

    assert set(terms) == {
        "reference",
        "heading",
        "speed",
        "acceleration",
        "steering",
        "acceleration_rate",
        "steering_rate",
        "collision",
        "terminal",
    }

    assert total == pytest.approx(
        sum(terms.values()),
    )


def test_cost_accepts_none_reference_path() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    total, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=None,
    )

    assert np.isfinite(total)
    assert terms["reference"] == pytest.approx(0.0)
    assert terms["heading"] == pytest.approx(0.0)
    assert terms["terminal"] == pytest.approx(0.0)


def test_cost_accepts_empty_reference_path() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    empty_reference = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    total, terms = cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=empty_reference,
    )

    assert np.isfinite(total)
    assert terms["reference"] == pytest.approx(0.0)
    assert terms["heading"] == pytest.approx(0.0)
    assert terms["terminal"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "reference_path",
    [
        np.zeros(
            3,
            dtype=np.float64,
        ),
        np.zeros(
            (4, 2),
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, np.nan, 0.0],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_cost_rejects_invalid_reference_path(
    reference_path: np.ndarray,
) -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
    )

    predictions = make_empty_predictions(
        trajectory.times,
    )

    with pytest.raises(ValueError):
        cost.evaluate(
            trajectory=trajectory,
            predictions=predictions,
            reference_path=reference_path,
        )


def test_cost_rejects_prediction_time_axis_with_different_shape() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )

    predictions = ObjectPredictionSet(
        times=np.array(
            [0.0, 0.1],
            dtype=np.float64,
        ),
        trajectories=(),
    )

    with pytest.raises(
        ValueError,
        match="same time-axis shape",
    ):
        cost.evaluate(
            trajectory=trajectory,
            predictions=predictions,
            reference_path=None,
        )


def test_cost_rejects_misaligned_prediction_time_axis() -> None:
    cost = make_cost()

    trajectory = make_trajectory(
        x=np.array(
            [0.0, 1.0, 2.0],
            dtype=np.float64,
        ),
    )

    predictions = ObjectPredictionSet(
        times=np.array(
            [0.0, 0.2, 0.4],
            dtype=np.float64,
        ),
        trajectories=(),
    )

    with pytest.raises(
        ValueError,
        match="same time axis",
    ):
        cost.evaluate(
            trajectory=trajectory,
            predictions=predictions,
            reference_path=None,
        )
