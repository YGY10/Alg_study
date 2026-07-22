from __future__ import annotations

import math

import numpy as np
import pytest

from sim2d.planning.spatiotemporal_planner.rollout import (
    TrajectoryRollout,
)
from sim2d.types import VehicleConfig, VehicleState


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


def test_rollout_constant_speed_straight_line() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )

    controls = np.zeros((10, 2), dtype=np.float64)

    trajectory = rollout.rollout(
        initial_state=initial_state,
        controls=controls,
    )

    assert trajectory.times.shape == (11,)
    assert trajectory.states.shape == (11, 4)
    assert trajectory.controls.shape == (10, 2)

    assert trajectory.times[0] == pytest.approx(0.0)
    assert trajectory.times[-1] == pytest.approx(1.0)

    assert trajectory.states[-1, 0] == pytest.approx(
        2.0,
        abs=1e-6,
    )
    assert trajectory.states[-1, 1] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert trajectory.states[-1, 2] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert trajectory.states[-1, 3] == pytest.approx(
        2.0,
        abs=1e-9,
    )


def test_rollout_applies_acceleration() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=0.0,
    )

    controls = np.zeros((10, 2), dtype=np.float64)
    controls[:, 0] = 1.0

    trajectory = rollout.rollout(
        initial_state=initial_state,
        controls=controls,
    )

    assert trajectory.states[-1, 3] == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert trajectory.states[-1, 0] > 0.0
    assert np.allclose(
        trajectory.controls[:, 0],
        1.0,
    )


def test_rollout_positive_steering_turns_left() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=3.0,
    )

    controls = np.zeros((20, 2), dtype=np.float64)
    controls[:, 1] = 0.2

    trajectory = rollout.rollout(
        initial_state=initial_state,
        controls=controls,
    )

    final_state = trajectory.states[-1]

    assert final_state[0] > 0.0
    assert final_state[1] > 0.0
    assert final_state[2] > 0.0


def test_rollout_negative_steering_turns_right() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=3.0,
    )

    controls = np.zeros((20, 2), dtype=np.float64)
    controls[:, 1] = -0.2

    trajectory = rollout.rollout(
        initial_state=initial_state,
        controls=controls,
    )

    final_state = trajectory.states[-1]

    assert final_state[0] > 0.0
    assert final_state[1] < 0.0
    assert final_state[2] < 0.0


def test_rollout_clips_control_limits() -> None:
    config = make_vehicle_config()

    rollout = TrajectoryRollout(
        vehicle_config=config,
        dt=0.1,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )

    controls = np.array(
        [
            [100.0, 100.0],
            [-100.0, -100.0],
        ],
        dtype=np.float64,
    )

    trajectory = rollout.rollout(
        initial_state=initial_state,
        controls=controls,
    )

    assert trajectory.controls[0, 0] == pytest.approx(config.acceleration_max)
    assert trajectory.controls[0, 1] == pytest.approx(config.steering_max)

    assert trajectory.controls[1, 0] == pytest.approx(config.acceleration_min)
    assert trajectory.controls[1, 1] == pytest.approx(config.steering_min)


def test_rollout_does_not_modify_input_controls() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    controls = np.array(
        [
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    original = controls.copy()

    rollout.rollout(
        initial_state=VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=1.0,
        ),
        controls=controls,
    )

    assert np.array_equal(controls, original)


@pytest.mark.parametrize(
    "controls",
    [
        np.zeros(2, dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
    ],
)
def test_rollout_rejects_invalid_control_shape(
    controls: np.ndarray,
) -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    with pytest.raises(ValueError, match="shape"):
        rollout.rollout(
            initial_state=VehicleState(
                x=0.0,
                y=0.0,
                yaw=0.0,
                speed=1.0,
            ),
            controls=controls,
        )


def test_rollout_rejects_non_finite_controls() -> None:
    rollout = TrajectoryRollout(
        vehicle_config=make_vehicle_config(),
        dt=0.1,
    )

    controls = np.array(
        [
            [0.0, math.nan],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="non-finite"):
        rollout.rollout(
            initial_state=VehicleState(
                x=0.0,
                y=0.0,
                yaw=0.0,
                speed=1.0,
            ),
            controls=controls,
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
        math.nan,
        math.inf,
    ],
)
def test_rollout_rejects_invalid_dt(
    dt: float,
) -> None:
    with pytest.raises(ValueError):
        TrajectoryRollout(
            vehicle_config=make_vehicle_config(),
            dt=dt,
        )
