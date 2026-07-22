from __future__ import annotations

import math

import numpy as np
import pytest

from sim2d.perception import PerceivedObject
from sim2d.planning.spatiotemporal_planner.prediction import (
    ConstantVelocityPredictor,
)


def make_object(
    *,
    object_id: str = "object-1",
    object_type: str = "box",
    semantic_type: str = "small_car",
    x: float = 10.0,
    y: float = 0.0,
    yaw: float = 0.0,
    speed: float = 0.0,
    length: float = 4.5,
    width: float = 1.8,
    confidence: float = 1.0,
) -> PerceivedObject:
    return PerceivedObject(
        object_id=object_id,
        object_type=object_type,
        x=x,
        y=y,
        yaw=yaw,
        speed=speed,
        length=length,
        width=width,
        confidence=confidence,
        semantic_type=semantic_type,
    )


def test_predict_empty_object_collection() -> None:
    predictor = ConstantVelocityPredictor()

    times = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(),
        times=times,
    )

    assert result.times.shape == (3,)
    assert np.allclose(result.times, times)
    assert result.trajectories == ()


def test_predict_stationary_object() -> None:
    predictor = ConstantVelocityPredictor()

    obj = make_object(
        x=10.0,
        y=2.0,
        yaw=0.7,
        speed=0.0,
    )

    times = np.array(
        [0.0, 0.5, 1.0, 1.5],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(obj,),
        times=times,
    )

    assert len(result.trajectories) == 1

    trajectory = result.trajectories[0]

    expected_positions = np.array(
        [
            [10.0, 2.0],
            [10.0, 2.0],
            [10.0, 2.0],
            [10.0, 2.0],
        ],
        dtype=np.float64,
    )

    assert trajectory.positions.shape == (4, 2)
    assert np.allclose(
        trajectory.positions,
        expected_positions,
    )
    assert np.allclose(
        trajectory.yaws,
        0.7,
    )


def test_predict_object_moving_forward() -> None:
    predictor = ConstantVelocityPredictor()

    obj = make_object(
        x=10.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )

    times = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(obj,),
        times=times,
    )

    trajectory = result.trajectories[0]

    expected_positions = np.array(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        trajectory.positions,
        expected_positions,
        atol=1e-12,
    )


def test_predict_pedestrian_crossing_from_right_to_left() -> None:
    predictor = ConstantVelocityPredictor()

    pedestrian = make_object(
        object_id="pedestrian-1",
        object_type="circle",
        semantic_type="pedestrian",
        x=8.0,
        y=-3.0,
        yaw=math.pi / 2.0,
        speed=1.0,
        length=0.7,
        width=0.7,
    )

    times = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(pedestrian,),
        times=times,
    )

    trajectory = result.trajectories[0]

    expected_positions = np.array(
        [
            [8.0, -3.0],
            [8.0, -2.0],
            [8.0, -1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        trajectory.positions,
        expected_positions,
        atol=1e-12,
    )


def test_predict_object_moving_toward_ego() -> None:
    predictor = ConstantVelocityPredictor()

    obj = make_object(
        x=15.0,
        y=0.0,
        yaw=math.pi,
        speed=3.0,
    )

    times = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(obj,),
        times=times,
    )

    trajectory = result.trajectories[0]

    expected_positions = np.array(
        [
            [15.0, 0.0],
            [12.0, 0.0],
            [9.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        trajectory.positions,
        expected_positions,
        atol=1e-12,
    )


def test_predict_diagonal_motion() -> None:
    predictor = ConstantVelocityPredictor()

    obj = make_object(
        x=1.0,
        y=2.0,
        yaw=math.pi / 4.0,
        speed=math.sqrt(2.0),
    )

    times = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(obj,),
        times=times,
    )

    trajectory = result.trajectories[0]

    expected_positions = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        trajectory.positions,
        expected_positions,
        atol=1e-12,
    )


def test_predict_multiple_objects_preserves_input_order() -> None:
    predictor = ConstantVelocityPredictor()

    pedestrian = make_object(
        object_id="pedestrian-1",
        object_type="circle",
        semantic_type="pedestrian",
        x=6.0,
        y=-2.0,
        yaw=math.pi / 2.0,
        speed=1.0,
        length=0.7,
        width=0.7,
    )

    small_car = make_object(
        object_id="small-car-1",
        object_type="box",
        semantic_type="small_car",
        x=15.0,
        y=0.0,
        yaw=0.0,
        speed=3.0,
        length=4.5,
        width=1.8,
    )

    large_vehicle = make_object(
        object_id="large-vehicle-1",
        object_type="box",
        semantic_type="large_vehicle",
        x=25.0,
        y=3.0,
        yaw=math.pi,
        speed=2.0,
        length=10.0,
        width=2.5,
    )

    times = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(
            pedestrian,
            small_car,
            large_vehicle,
        ),
        times=times,
    )

    assert [trajectory.object_id for trajectory in result.trajectories] == [
        "pedestrian-1",
        "small-car-1",
        "large-vehicle-1",
    ]


def test_prediction_preserves_object_metadata() -> None:
    predictor = ConstantVelocityPredictor()

    obj = make_object(
        object_id="truck-7",
        object_type="box",
        semantic_type="large_vehicle",
        x=20.0,
        y=1.0,
        yaw=0.3,
        speed=4.0,
        length=10.0,
        width=2.5,
        confidence=0.85,
    )

    result = predictor.predict(
        objects=(obj,),
        times=np.array(
            [0.0, 0.1],
            dtype=np.float64,
        ),
    )

    trajectory = result.trajectories[0]

    assert trajectory.object_id == "truck-7"
    assert trajectory.object_type == "box"
    assert trajectory.semantic_type == "large_vehicle"
    assert trajectory.speed == pytest.approx(4.0)
    assert trajectory.length == pytest.approx(10.0)
    assert trajectory.width == pytest.approx(2.5)
    assert trajectory.confidence == pytest.approx(0.85)


def test_prediction_copies_time_array() -> None:
    predictor = ConstantVelocityPredictor()

    times = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )

    result = predictor.predict(
        objects=(make_object(),),
        times=times,
    )

    times[1] = 100.0

    assert result.times[1] == pytest.approx(0.1)
    assert result.trajectories[0].times[1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "times",
    [
        np.array([], dtype=np.float64),
        np.array([[0.0, 0.1]], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([0.0, 0.2, 0.1], dtype=np.float64),
        np.array([0.0, 0.1, 0.1], dtype=np.float64),
        np.array([0.0, np.nan], dtype=np.float64),
        np.array([0.0, np.inf], dtype=np.float64),
    ],
)
def test_predict_rejects_invalid_time_axis(
    times: np.ndarray,
) -> None:
    predictor = ConstantVelocityPredictor()

    with pytest.raises(ValueError):
        predictor.predict(
            objects=(make_object(),),
            times=times,
        )


def test_prediction_time_axis_matches_rollout_contract() -> None:
    predictor = ConstantVelocityPredictor()

    times = (
        np.arange(
            51,
            dtype=np.float64,
        )
        * 0.1
    )

    result = predictor.predict(
        objects=(make_object(),),
        times=times,
    )

    assert result.times.shape == (51,)
    assert result.trajectories[0].positions.shape == (51, 2)
    assert result.trajectories[0].yaws.shape == (51,)
