import math

import pytest

from sim2d import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)
from sim2d.core.dynamics import (
    KinematicBicycleModel,
)


@pytest.fixture
def model():
    config = VehicleConfig(
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

    return KinematicBicycleModel(config)


def test_straight_motion(model):
    state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )

    control = VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )

    next_state = model.step(
        state=state,
        control=control,
        dt=0.1,
    )

    assert next_state.x == pytest.approx(0.2)
    assert next_state.y == pytest.approx(0.0)
    assert next_state.yaw == pytest.approx(0.0)
    assert next_state.speed == pytest.approx(2.0)


def test_acceleration(model):
    state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )

    control = VehicleControl(
        acceleration=1.5,
        steering=0.0,
    )

    next_state = model.step(
        state=state,
        control=control,
        dt=0.2,
    )

    assert next_state.speed == pytest.approx(2.3)


def test_left_turn_increases_yaw(model):
    state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=5.0,
    )

    control = VehicleControl(
        acceleration=0.0,
        steering=0.2,
    )

    next_state = model.step(
        state=state,
        control=control,
        dt=0.1,
    )

    assert next_state.yaw > 0.0


def test_control_is_clamped(model):
    state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=14.9,
    )

    control = VehicleControl(
        acceleration=100.0,
        steering=100.0,
    )

    next_state = model.step(
        state=state,
        control=control,
        dt=1.0,
    )

    assert next_state.speed == pytest.approx(
        model.config.speed_max
    )


def test_invalid_dt(model):
    state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=1.0,
    )

    control = VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )

    with pytest.raises(ValueError):
        model.step(
            state=state,
            control=control,
            dt=0.0,
        )


def test_angle_normalization(model):
    angle = model.normalize_angle(
        3.0 * math.pi
    )

    assert angle == pytest.approx(-math.pi)