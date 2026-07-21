from __future__ import annotations

import math

import numpy as np

from sim2d.core import DrivingEnv, EnvironmentConfig
from sim2d.perception import PerceptionConfig, PlanningInput
from sim2d.types import (
    CircleObstacle,
    GoalState,
    VehicleConfig,
    VehicleState,
)
from sim2d.world.road_geometry import WorldLaneGeometry
from sim2d.map.types import LaneType


def _make_env() -> DrivingEnv:
    env = DrivingEnv(
        vehicle_config=VehicleConfig(),
        environment_config=EnvironmentConfig(),
    )
    env.reset(
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0),
        goal=GoalState(VehicleState(20.0, 0.0, 0.0, 0.0)),
        obstacles=(
            CircleObstacle("near", 10.0, 0.0, 1.0),
            CircleObstacle("far", 100.0, 0.0, 1.0),
            CircleObstacle("behind", -30.0, 0.0, 1.0),
        ),
    )
    return env


def test_planner_input_is_backward_compatible_and_local():
    env = _make_env()
    planning_input = env.get_observation()

    assert isinstance(planning_input, PlanningInput)
    assert planning_input.time == 0.0
    assert planning_input.frame == 0
    assert planning_input.goal.state.x == 20.0
    assert [item.obstacle_id for item in planning_input.obstacles] == ["near"]
    assert [item.object_id for item in planning_input.perception.objects] == ["near"]


def test_perception_range_is_configurable():
    env = _make_env()
    env.set_perception_config(
        PerceptionConfig(
            forward_range=120.0,
            rear_range=40.0,
            lateral_range=20.0,
            update_period=0.05,
        )
    )

    snapshot = env.get_perception_snapshot()
    assert {item.object_id for item in snapshot.objects} == {
        "near",
        "far",
        "behind",
    }


def test_perception_noise_interface_is_deterministic():
    first = _make_env()
    second = _make_env()
    config = PerceptionConfig(
        position_noise_std=0.1,
        yaw_noise_std=0.01,
        speed_noise_std=0.05,
        random_seed=7,
    )
    first.set_perception_config(config)
    second.set_perception_config(config)

    first_snapshot = first.get_perception_snapshot()
    second_snapshot = second.get_perception_snapshot()

    assert first_snapshot.ego == second_snapshot.ego
    assert first_snapshot.objects == second_snapshot.objects


def test_local_world_road_is_clipped_around_ego():
    env = _make_env()
    points = np.column_stack(
        [
            np.linspace(-100.0, 100.0, 201),
            np.zeros(201),
        ]
    )
    env.world.state.road_lanes = (
        WorldLaneGeometry(
            entity_id="world_lane_test",
            map_lane_id="lane_test",
            lane_type=LaneType.DRIVING,
            centerline=points,
            left_boundary=points + np.array([0.0, 1.75]),
            right_boundary=points - np.array([0.0, 1.75]),
            predecessor_ids=(),
            successor_ids=(),
        ),
    )

    snapshot = env.get_perception_snapshot()
    assert len(snapshot.road_segments) == 1
    segment = snapshot.road_segments[0]
    assert float(segment.centerline[:, 0].min()) >= -21.0
    assert float(segment.centerline[:, 0].max()) <= 61.0


def test_field_of_view_can_hide_rear_objects():
    env = _make_env()
    env.set_perception_config(
        PerceptionConfig(
            forward_range=60.0,
            rear_range=40.0,
            lateral_range=40.0,
            field_of_view=math.pi,
        )
    )
    snapshot = env.get_perception_snapshot()
    assert {item.object_id for item in snapshot.objects} == {"near"}
