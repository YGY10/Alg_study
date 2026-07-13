import math

import numpy as np
import pytest

from sim2d.planning import (
    BezierPlanner,
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
) -> BezierPlanner:
    return BezierPlanner(
        vehicle_config=(make_vehicle_config()),
        **kwargs,
    )


def make_observation(
    *,
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    ego_yaw: float = 0.0,
    ego_speed: float = 2.0,
    goal_x: float = 25.0,
    goal_y: float = 6.0,
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
            position_tolerance=(position_tolerance),
            yaw_tolerance=(yaw_tolerance),
            speed_tolerance=(speed_tolerance),
        ),
    )


def test_goal_on_left_produces_positive_steering() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            goal_y=6.0,
        )
    )

    assert result.action.steering > 0.0


def test_goal_on_right_produces_negative_steering() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            goal_y=-6.0,
        )
    )

    assert result.action.steering < 0.0


def test_straight_goal_produces_near_zero_steering() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            goal_y=0.0,
        )
    )

    assert abs(result.action.steering) < 1e-6


def test_trajectory_and_control_shapes() -> None:
    planner = make_planner(
        prediction_steps=30,
    )

    result = planner.plan(make_observation())

    assert result.trajectory is not None
    assert result.controls is not None

    assert result.trajectory.shape == (
        31,
        4,
    )

    assert result.controls.shape == (
        30,
        2,
    )


def test_prediction_starts_at_current_state() -> None:
    planner = make_planner()

    observation = make_observation(
        ego_x=1.0,
        ego_y=2.0,
        ego_yaw=0.2,
        ego_speed=3.0,
    )

    result = planner.plan(observation)

    assert result.trajectory is not None

    np.testing.assert_allclose(
        result.trajectory[0],
        observation.ego.as_array(),
    )


def test_prediction_is_finite() -> None:
    planner = make_planner()

    result = planner.plan(make_observation())

    assert result.trajectory is not None
    assert result.controls is not None

    assert np.all(np.isfinite(result.trajectory))

    assert np.all(np.isfinite(result.controls))


def test_terminal_state_outputs_zero_control() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_x=25.1,
            ego_y=6.1,
            ego_yaw=0.05,
            ego_speed=0.1,
            goal_x=25.0,
            goal_y=6.0,
            goal_yaw=0.0,
            goal_speed=0.0,
        )
    )

    assert result.action.acceleration == pytest.approx(0.0)

    assert result.action.steering == pytest.approx(0.0)

    assert result.debug["terminal_hold"] is True

    assert result.debug["status"] == "goal_hold"


def test_near_goal_requests_braking() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_x=23.0,
            ego_y=6.0,
            ego_yaw=0.0,
            ego_speed=4.0,
            goal_x=25.0,
            goal_y=6.0,
            goal_speed=0.0,
        )
    )

    assert result.action.acceleration < 0.0


def test_nonzero_goal_speed_is_preserved() -> None:
    planner = make_planner(
        target_speed=4.0,
    )

    result = planner.plan(
        make_observation(
            ego_x=24.0,
            ego_y=6.0,
            ego_yaw=0.0,
            ego_speed=2.0,
            goal_x=25.0,
            goal_y=6.0,
            goal_yaw=0.0,
            goal_speed=2.0,
        )
    )

    assert result.debug["desired_speed"] >= 2.0


def test_path_tracking_debug_fields_exist() -> None:
    planner = make_planner()

    result = planner.plan(make_observation())

    assert result.debug["planner"] == "BezierPlanner"

    assert result.debug["prediction_mode"] == "closed_loop_path_tracking"

    assert result.debug["reference_path_points"] == 201

    assert "nearest_index" in result.debug
    assert "target_index" in result.debug
    assert "cross_track_error" in result.debug
    assert "remaining_length" in result.debug


def test_reset_allows_new_path_generation() -> None:
    planner = make_planner()

    left_result = planner.plan(
        make_observation(
            goal_y=6.0,
        )
    )

    assert left_result.action.steering > 0.0

    planner.reset()

    right_result = planner.plan(
        make_observation(
            goal_y=-6.0,
        )
    )

    assert right_result.action.steering < 0.0


@pytest.mark.parametrize(
    (
        "argument_name",
        "argument_value",
    ),
    [
        ("target_speed", -1.0),
        ("speed_gain", 0.0),
        ("braking_deceleration", 0.0),
        ("path_sample_count", 1),
        ("handle_scale", 0.0),
        ("minimum_handle_length", 0.0),
        ("lookahead_base", 0.0),
        ("lookahead_speed_gain", -0.1),
        ("minimum_lookahead", 0.0),
        ("stop_target_radius", -0.1),
        ("braking_margin", -0.1),
        ("prediction_dt", 0.0),
        ("prediction_steps", 0),
    ],
)
def test_invalid_configuration_raises(
    argument_name: str,
    argument_value: float | int,
) -> None:
    with pytest.raises(ValueError):
        make_planner(**{argument_name: argument_value})


def test_maximum_handle_cannot_be_smaller_than_minimum() -> None:
    with pytest.raises(ValueError):
        make_planner(
            minimum_handle_length=5.0,
            maximum_handle_length=2.0,
        )


def test_angle_normalization() -> None:
    planner = make_planner()

    result = planner.plan(
        make_observation(
            ego_yaw=math.radians(359.0),
            goal_y=0.0,
        )
    )

    assert abs(result.action.steering) < 0.1
