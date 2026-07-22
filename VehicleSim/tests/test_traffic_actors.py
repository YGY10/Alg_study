from __future__ import annotations

import math

import numpy as np

from sim2d.core import DrivingEnv, EnvironmentConfig
from sim2d.perception import PlanningInput
from sim2d.types import GoalState, VehicleConfig, VehicleControl, VehicleState
from sim2d.world import (
    TrafficActorType,
    create_traffic_actor,
)


def test_semantic_actor_factory_uses_expected_geometry():
    pedestrian = create_traffic_actor(
        TrafficActorType.PEDESTRIAN,
        actor_id="pedestrian_001",
        x=1.0,
        y=2.0,
        yaw=0.3,
        speed=1.2,
    )
    car = create_traffic_actor(
        "small_car",
        actor_id="small_car_001",
        x=3.0,
        y=4.0,
        yaw=0.5,
        speed=5.0,
    )
    truck = create_traffic_actor(
        "large_vehicle",
        actor_id="large_vehicle_001",
        x=5.0,
        y=6.0,
    )

    assert pedestrian.semantic_type == "pedestrian"
    assert np.isclose(pedestrian.radius, 0.35)
    assert car.semantic_type == "small_car"
    assert np.isclose(car.length, 4.5)
    assert np.isclose(car.width, 1.8)
    assert truck.semantic_type == "large_vehicle"
    assert np.isclose(truck.length, 10.0)
    assert np.isclose(truck.width, 2.5)


def test_traffic_actor_advances_along_initial_direction():
    env = DrivingEnv(
        vehicle_config=VehicleConfig(),
        environment_config=EnvironmentConfig(dt=0.05),
    )
    actor = create_traffic_actor(
        "pedestrian",
        actor_id="pedestrian_001",
        x=10.0,
        y=0.0,
        yaw=0.5 * math.pi,
        speed=2.0,
    )
    env.reset(
        initial_state=VehicleState(0.0, 0.0, 0.0, 0.0),
        goal=GoalState(VehicleState(50.0, 0.0, 0.0, 0.0)),
        obstacles=(actor,),
    )

    env.step(VehicleControl(acceleration=0.0, steering=0.0))
    moved = env.world.state.obstacles[0]
    assert np.isclose(moved.x, 10.0, atol=1e-9)
    assert np.isclose(moved.y, 0.1, atol=1e-9)


def test_perception_exposes_semantic_type_speed_and_relative_yaw():
    env = DrivingEnv(
        vehicle_config=VehicleConfig(),
        environment_config=EnvironmentConfig(),
    )
    actor = create_traffic_actor(
        "small_car",
        actor_id="small_car_001",
        x=15.0,
        y=2.0,
        yaw=0.4,
        speed=6.0,
    )
    env.reset(
        initial_state=VehicleState(0.0, 0.0, 0.1, 0.0),
        goal=GoalState(VehicleState(50.0, 0.0, 0.0, 0.0)),
        obstacles=(actor,),
    )

    planning_input = env.get_observation()
    assert isinstance(planning_input, PlanningInput)
    perceived = planning_input.perception.objects[0]
    assert perceived.semantic_type == "small_car"
    assert np.isclose(perceived.speed, 6.0)
    assert np.isclose(perceived.yaw, 0.3)
