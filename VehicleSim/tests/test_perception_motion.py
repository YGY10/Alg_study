from __future__ import annotations

import numpy as np

from sim2d.core import DrivingEnv, EnvironmentConfig
from sim2d.perception import PerceptionConfig
from sim2d.types import (
    CircleObstacle,
    GoalState,
    VehicleConfig,
    VehicleState,
)


def test_static_world_object_moves_in_vehicle_frame_when_ego_moves() -> None:
    env = DrivingEnv(
        vehicle_config=VehicleConfig(),
        environment_config=EnvironmentConfig(dt=0.05),
    )
    env.reset(
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0),
        goal=GoalState(VehicleState(30.0, 0.0, 0.0, 0.0)),
        obstacles=(CircleObstacle("target", 12.0, 0.0, 1.0),),
    )
    env.set_perception_config(PerceptionConfig(update_period=0.05))

    initial = env.get_perception_snapshot()
    assert np.isclose(initial.objects[0].x, 12.0)
    assert np.isclose(initial.objects[0].y, 0.0)

    # 障碍物保持在世界坐标 (12, 0)，自车向前移动到 (2, 0)。
    env.world.step(
        0.05,
        ego_state=VehicleState(2.0, 0.0, 0.0, 0.0),
    )
    moved = env.get_perception_snapshot()

    assert moved.measurement_time == 0.05
    assert np.isclose(moved.objects[0].x, 10.0)
    assert np.isclose(moved.objects[0].y, 0.0)


def test_vehicle_rotation_changes_static_object_local_coordinates() -> None:
    env = DrivingEnv(
        vehicle_config=VehicleConfig(),
        environment_config=EnvironmentConfig(dt=0.05),
    )
    env.reset(
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0),
        goal=GoalState(VehicleState(30.0, 0.0, 0.0, 0.0)),
        obstacles=(CircleObstacle("target", 10.0, 0.0, 1.0),),
    )
    env.set_perception_config(PerceptionConfig(update_period=0.05))
    env.get_perception_snapshot()

    env.world.step(
        0.05,
        ego_state=VehicleState(0.0, 0.0, 0.5 * np.pi, 0.0),
    )
    rotated = env.get_perception_snapshot()

    assert np.isclose(rotated.objects[0].x, 0.0, atol=1e-9)
    assert np.isclose(rotated.objects[0].y, -10.0, atol=1e-9)
