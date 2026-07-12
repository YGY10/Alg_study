import numpy as np

from sim2d import (
    CircleObstacle,
    VehicleControl,
    VehicleState,
)


def test_vehicle_state_array_conversion():
    state = VehicleState(
        x=1.0,
        y=2.0,
        yaw=0.3,
        speed=4.0,
    )

    array = state.as_array()

    assert array.shape == (4,)
    assert np.allclose(
        array,
        np.array([1.0, 2.0, 0.3, 4.0]),
    )

    restored = VehicleState.from_array(array)

    assert restored == state


def test_vehicle_control_array_conversion():
    control = VehicleControl(
        acceleration=1.2,
        steering=-0.1,
    )

    array = control.as_array()

    assert array.shape == (2,)
    assert np.allclose(
        array,
        np.array([1.2, -0.1]),
    )


def test_circle_obstacle_validation():
    obstacle = CircleObstacle(
        obstacle_id="obs_001",
        x=10.0,
        y=0.0,
        radius=1.0,
    )

    obstacle.validate()