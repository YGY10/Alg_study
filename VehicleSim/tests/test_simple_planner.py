import math

import numpy as np
import pytest

from sim2d.planning import (
    SimplePlanner,
)
from sim2d.types import (
    GoalState,
    Observation,
    VehicleConfig,
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


def make_planner(
    **kwargs,
) -> SimplePlanner:
    return SimplePlanner(
        vehicle_config=(make_vehicle_config()),
        **kwargs,
    )


def make_observation(
    *,
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    ego_yaw: float = 0.0,
    ego_speed: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    goal_yaw: float = 0.0,
    goal_speed: float = 0.0,
    position_tolerance: float = 0.5,
    yaw_tolerance: float = 0.2,
    speed_tolerance: float = 0.5,
) -> Observation:
    return Observation(
        time=0.0,
        frame=0,
        ego=VehicleState(
            x=ego_x,
            y=ego_y,
            yaw=ego_yaw,
            speed=ego_speed,
        ),
        obstacles=(),
        goal=GoalState(
            state=VehicleState(
                x=goal_x,
                y=goal_y,
                yaw=goal_yaw,
                speed=goal_speed,
            ),
            position_tolerance=position_tolerance,
            yaw_tolerance=yaw_tolerance,
            speed_tolerance=speed_tolerance,
        ),
    )


def test_goal_ahead_accelerates_without_steering() -> None:
    planner = make_planner()

    result = planner.plan(make_observation())

    assert result.action.acceleration > 0.0

    assert abs(result.action.steering) < 1e-9


def test_goal_on_left_produces_positive_steering() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            goal_x=10.0,
            goal_y=5.0,
        )
    )

    assert result.action.steering > 0.0


def test_goal_on_right_produces_negative_steering() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            goal_x=10.0,
            goal_y=-5.0,
        )
    )

    assert result.action.steering < 0.0


def test_steering_is_clamped_to_vehicle_limit() -> None:
    planner = make_planner(
        steering_gain=100.0,
    )

    result = planner.plan(
        make_observation(
            goal_x=0.0,
            goal_y=10.0,
        )
    )

    assert result.action.steering == pytest.approx(0.45)


def test_acceleration_is_clamped_to_vehicle_limit() -> None:
    planner = make_planner(
        target_speed=20.0,
        speed_gain=100.0,
    )

    result = planner.plan(
        make_observation(
            ego_speed=0.0,
        )
    )

    assert result.action.acceleration == pytest.approx(2.0)


def test_close_to_goal_reduces_desired_speed() -> None:
    planner = make_planner(
        target_speed=4.0,
        slow_down_distance=5.0,
    )

    result = planner.plan(
        make_observation(
            ego_speed=4.0,
            goal_x=1.0,
        )
    )

    assert result.action.acceleration < 0.0

    assert result.debug["desired_speed"] < 4.0


def test_at_goal_requests_stop() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_speed=2.0,
            goal_x=0.1,
            goal_y=0.0,
            goal_yaw=0.0,
            goal_speed=0.0,
            position_tolerance=0.5,
            yaw_tolerance=0.2,
            speed_tolerance=0.1,
        )
    )

    assert result.action.acceleration < 0.0
    assert result.debug["position_reached"] is True
    assert result.debug["speed_reached"] is False
    assert result.debug["terminal_hold"] is False
    assert result.debug["status"] == "stopping_at_goal"


def test_terminal_hold_outputs_zero_control() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_x=0.1,
            ego_y=0.0,
            ego_yaw=0.05,
            ego_speed=0.0,
            goal_x=0.0,
            goal_y=0.0,
            goal_yaw=0.0,
            goal_speed=0.0,
            position_tolerance=0.5,
            yaw_tolerance=0.2,
            speed_tolerance=0.5,
        )
    )

    assert result.action.acceleration == pytest.approx(0.0)
    assert result.action.steering == pytest.approx(0.0)

    assert result.debug["position_reached"] is True
    assert result.debug["yaw_reached"] is True
    assert result.debug["speed_reached"] is True
    assert result.debug["terminal_hold"] is True
    assert result.debug["status"] == "goal_hold"


def test_trajectory_and_controls_shapes() -> None:
    planner = make_planner(
        prediction_steps=30,
    )

    result = planner.plan(make_observation())

    assert result.trajectory is not None
    assert result.trajectory.shape == (
        31,
        4,
    )

    assert result.controls is not None
    assert result.controls.shape == (
        30,
        2,
    )

    assert np.all(np.isfinite(result.trajectory))

    assert np.all(np.isfinite(result.controls))


def test_trajectory_starts_at_current_state() -> None:
    planner = make_planner()

    observation = make_observation(
        ego_x=1.0,
        ego_y=2.0,
        ego_yaw=0.3,
        ego_speed=2.0,
    )

    result = planner.plan(observation)

    assert result.trajectory is not None

    np.testing.assert_allclose(
        result.trajectory[0],
        observation.ego.as_array(),
    )


def test_predicted_trajectory_moves_forward() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
        )
    )

    assert result.trajectory is not None

    assert result.trajectory[-1, 0] > result.trajectory[0, 0]


def test_straight_prediction_keeps_y_near_zero() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=20.0,
            goal_y=0.0,
        )
    )

    assert result.trajectory is not None

    np.testing.assert_allclose(
        result.trajectory[:, 1],
        0.0,
        atol=1e-10,
    )


def test_left_prediction_increases_y() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=10.0,
            goal_y=5.0,
        )
    )

    assert result.trajectory is not None

    assert result.trajectory[-1, 1] > result.trajectory[0, 1]


def test_closed_loop_controls_are_finite() -> None:
    planner = make_planner(
        prediction_steps=5,
    )

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=10.0,
            goal_y=5.0,
        )
    )

    assert result.controls is not None

    assert result.controls.shape == (
        5,
        2,
    )

    assert np.all(np.isfinite(result.controls))

    assert result.debug["prediction_mode"] == "closed_loop"


def test_closed_loop_steering_changes_during_prediction() -> None:
    planner = make_planner(
        prediction_steps=20,
    )

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=10.0,
            goal_y=5.0,
        )
    )

    assert result.controls is not None

    steering_sequence = result.controls[:, 1]

    assert not np.allclose(
        steering_sequence,
        steering_sequence[0],
    )


def test_first_predicted_control_matches_executed_action() -> None:
    planner = make_planner(
        prediction_steps=10,
    )

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=10.0,
            goal_y=5.0,
        )
    )

    assert result.controls is not None

    np.testing.assert_allclose(
        result.controls[0],
        result.action.as_array(),
    )


def test_closed_loop_steering_reduces_as_heading_converges() -> None:
    planner = make_planner(
        prediction_steps=30,
    )

    result = planner.plan(
        make_observation(
            ego_speed=2.0,
            goal_x=20.0,
            goal_y=5.0,
        )
    )

    assert result.controls is not None

    initial_steering = abs(result.controls[0, 1])

    final_steering = abs(result.controls[-1, 1])

    assert final_steering < initial_steering


def test_prediction_horizon_debug_value() -> None:
    planner = make_planner(
        prediction_dt=0.2,
        prediction_steps=10,
    )

    result = planner.plan(make_observation())

    assert result.debug["prediction_horizon"] == pytest.approx(2.0)


def test_prediction_debug_mode_is_closed_loop() -> None:
    planner = make_planner()

    result = planner.plan(make_observation())

    assert result.debug["prediction_mode"] == "closed_loop"


def test_angle_normalization_across_pi_boundary() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_yaw=math.radians(179.0),
            goal_x=-10.0,
            goal_y=-0.1,
        )
    )

    assert abs(result.debug["yaw_error"]) < 0.1


def test_simple_planner_reset() -> None:
    planner = make_planner()

    planner.reset()


@pytest.mark.parametrize(
    (
        "argument_name",
        "argument_value",
    ),
    [
        ("target_speed", -1.0),
        ("steering_gain", 0.0),
        ("speed_gain", 0.0),
        ("slow_down_distance", 0.0),
        ("heading_transition_distance", 0.0),
        ("alignment_speed", -0.1),
        ("prediction_dt", 0.0),
        ("prediction_steps", 0),
        ("braking_deceleration", 0.0),
        ("stop_target_radius", -0.1),
        ("braking_margin", -0.1),
    ],
)
def test_invalid_configuration_raises(
    argument_name: str,
    argument_value: float | int,
) -> None:
    kwargs = {argument_name: argument_value}

    with pytest.raises(ValueError):
        make_planner(
            **kwargs,
        )


def test_zero_alignment_speed_is_allowed() -> None:
    planner = make_planner(
        alignment_speed=0.0,
    )

    result = planner.plan(make_observation())

    assert result.action is not None
