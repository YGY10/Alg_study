from __future__ import annotations

import pytest

from sim2d.core.dynamics import KinematicBicycleModel
from sim2d.models.kinematic_bicycle import (
    KinematicBicyclePredictionModel,
)
from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


def make_vehicle_config() -> VehicleConfig:
    return VehicleConfig(
        length=4.6,
        width=1.85,
        wheel_base=2.7,
        acceleration_min=-3.0,
        acceleration_max=2.0,
        steering_min=-0.45,
        steering_max=0.45,
        speed_min=0.0,
        speed_max=15.0,
    )


def make_state() -> VehicleState:
    return VehicleState(
        x=1.0,
        y=2.0,
        yaw=0.2,
        speed=3.0,
    )


def make_control() -> VehicleControl:
    return VehicleControl(
        acceleration=0.5,
        steering=0.1,
    )


def assert_state_close(
    actual: VehicleState,
    expected: VehicleState,
) -> None:
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)
    assert actual.yaw == pytest.approx(expected.yaw)
    assert actual.speed == pytest.approx(expected.speed)


def test_prediction_model_matches_original_model() -> None:
    config = make_vehicle_config()
    state = make_state()
    control = make_control()
    dt = 0.1

    expected = KinematicBicycleModel(
        config=config,
    ).step(
        state=state,
        control=control,
        dt=dt,
    )

    model = KinematicBicyclePredictionModel(
        vehicle_config=config,
    )

    actual = model.predict_step(
        state=state,
        control=control,
        dt=dt,
    )

    assert_state_close(
        actual,
        expected,
    )


def test_prediction_model_does_not_mutate_input_state() -> None:
    model = KinematicBicyclePredictionModel(
        vehicle_config=make_vehicle_config(),
    )

    state = make_state()
    control = make_control()

    before = VehicleState(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        speed=state.speed,
    )

    result = model.predict_step(
        state=state,
        control=control,
        dt=0.1,
    )

    assert_state_close(
        state,
        before,
    )
    assert result is not state


def test_prediction_model_is_repeatable_from_same_state() -> None:
    model = KinematicBicyclePredictionModel(
        vehicle_config=make_vehicle_config(),
    )

    state = make_state()
    control = make_control()

    first = model.predict_step(
        state=state,
        control=control,
        dt=0.1,
    )

    second = model.predict_step(
        state=state,
        control=control,
        dt=0.1,
    )

    assert_state_close(
        first,
        second,
    )


def test_prediction_rollouts_are_independent() -> None:
    model = KinematicBicyclePredictionModel(
        vehicle_config=make_vehicle_config(),
    )

    initial_state = make_state()

    left_control = VehicleControl(
        acceleration=0.0,
        steering=0.2,
    )
    right_control = VehicleControl(
        acceleration=0.0,
        steering=-0.2,
    )

    left_state = model.predict_step(
        state=initial_state,
        control=left_control,
        dt=0.1,
    )

    right_state = model.predict_step(
        state=initial_state,
        control=right_control,
        dt=0.1,
    )

    assert left_state.yaw > initial_state.yaw
    assert right_state.yaw < initial_state.yaw

    # 两个候选都必须从同一个 initial_state 开始，
    # 第一条预测不能污染第二条预测。
    assert_state_close(
        initial_state,
        make_state(),
    )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
    ],
)
def test_prediction_model_rejects_non_positive_dt(
    dt: float,
) -> None:
    model = KinematicBicyclePredictionModel(
        vehicle_config=make_vehicle_config(),
    )

    with pytest.raises(
        ValueError,
        match="dt must be positive",
    ):
        model.predict_step(
            state=make_state(),
            control=make_control(),
            dt=dt,
        )
