import math

import numpy as np
import pytest

from sim2d import (
    BoxObstacle,
    CircleObstacle,
    VehicleConfig,
    VehicleState,
)
from sim2d.core.collision import (
    circle_box_clearance,
    circle_box_collision,
    collision_with_obstacle,
    has_collision,
    minimum_clearance,
    obstacle_clearance,
    oriented_box_clearance,
    oriented_box_collision,
    oriented_box_corners,
    vehicle_corners,
)


@pytest.fixture
def vehicle_config():
    return VehicleConfig(
        length=4.0,
        width=2.0,
        wheel_base=2.5,
        acceleration_min=-3.0,
        acceleration_max=2.0,
        steering_min=-0.5,
        steering_max=0.5,
        speed_min=0.0,
        speed_max=15.0,
    )


@pytest.fixture
def ego_state():
    return VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=0.0,
    )


def test_vehicle_corners_at_zero_yaw(
    vehicle_config,
    ego_state,
):
    corners = vehicle_corners(
        ego_state,
        vehicle_config,
    )

    expected = np.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [-2.0, -1.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert corners.shape == (4, 2)
    assert np.allclose(corners, expected)


def test_oriented_box_corners_rotated_90_degrees():
    corners = oriented_box_corners(
        x=0.0,
        y=0.0,
        yaw=math.pi / 2.0,
        length=4.0,
        width=2.0,
    )

    expected = np.array(
        [
            [-1.0, 2.0],
            [1.0, 2.0],
            [1.0, -2.0],
            [-1.0, -2.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        corners,
        expected,
        atol=1e-9,
    )


def test_circle_box_no_collision():
    collided = circle_box_collision(
        box_x=0.0,
        box_y=0.0,
        box_yaw=0.0,
        box_length=4.0,
        box_width=2.0,
        circle_x=5.0,
        circle_y=0.0,
        circle_radius=1.0,
    )

    assert collided is False


def test_circle_box_touching_is_collision():
    # 矩形前边界为 x=2，圆左边界也为 x=2。
    collided = circle_box_collision(
        box_x=0.0,
        box_y=0.0,
        box_yaw=0.0,
        box_length=4.0,
        box_width=2.0,
        circle_x=3.0,
        circle_y=0.0,
        circle_radius=1.0,
    )

    clearance = circle_box_clearance(
        box_x=0.0,
        box_y=0.0,
        box_yaw=0.0,
        box_length=4.0,
        box_width=2.0,
        circle_x=3.0,
        circle_y=0.0,
        circle_radius=1.0,
    )

    assert collided is True
    assert clearance == pytest.approx(0.0)


def test_circle_center_inside_box():
    clearance = circle_box_clearance(
        box_x=0.0,
        box_y=0.0,
        box_yaw=0.0,
        box_length=4.0,
        box_width=2.0,
        circle_x=0.0,
        circle_y=0.0,
        circle_radius=0.5,
    )

    assert clearance < 0.0


def test_axis_aligned_boxes_overlap():
    box_a = oriented_box_corners(
        x=0.0,
        y=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
    )

    box_b = oriented_box_corners(
        x=2.5,
        y=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
    )

    assert oriented_box_collision(
        box_a,
        box_b,
    )


def test_rotated_boxes_overlap():
    box_a = oriented_box_corners(
        x=0.0,
        y=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
    )

    box_b = oriented_box_corners(
        x=1.5,
        y=0.5,
        yaw=math.radians(35.0),
        length=3.0,
        width=1.5,
    )

    assert oriented_box_collision(
        box_a,
        box_b,
    )


def test_boxes_are_separated():
    box_a = oriented_box_corners(
        x=0.0,
        y=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
    )

    box_b = oriented_box_corners(
        x=8.0,
        y=0.0,
        yaw=math.radians(25.0),
        length=4.0,
        width=2.0,
    )

    clearance = oriented_box_clearance(
        box_a,
        box_b,
    )

    assert not oriented_box_collision(
        box_a,
        box_b,
    )
    assert clearance > 0.0


def test_collision_with_circle_obstacle(
    vehicle_config,
    ego_state,
):
    obstacle = CircleObstacle(
        obstacle_id="circle_001",
        x=2.5,
        y=0.0,
        radius=0.75,
    )

    assert collision_with_obstacle(
        state=ego_state,
        config=vehicle_config,
        obstacle=obstacle,
    )


def test_collision_with_box_obstacle(
    vehicle_config,
    ego_state,
):
    obstacle = BoxObstacle(
        obstacle_id="box_001",
        x=2.5,
        y=0.0,
        yaw=math.radians(20.0),
        length=2.0,
        width=1.5,
    )

    assert collision_with_obstacle(
        state=ego_state,
        config=vehicle_config,
        obstacle=obstacle,
    )


def test_minimum_clearance_returns_nearest_obstacle(
    vehicle_config,
    ego_state,
):
    near_obstacle = CircleObstacle(
        obstacle_id="near",
        x=4.0,
        y=0.0,
        radius=0.5,
    )

    far_obstacle = CircleObstacle(
        obstacle_id="far",
        x=10.0,
        y=0.0,
        radius=1.0,
    )

    near_clearance = obstacle_clearance(
        state=ego_state,
        config=vehicle_config,
        obstacle=near_obstacle,
    )

    result = minimum_clearance(
        state=ego_state,
        config=vehicle_config,
        obstacles=(
            near_obstacle,
            far_obstacle,
        ),
    )

    assert result == pytest.approx(
        near_clearance
    )


def test_minimum_clearance_without_obstacles(
    vehicle_config,
    ego_state,
):
    result = minimum_clearance(
        state=ego_state,
        config=vehicle_config,
        obstacles=(),
    )

    assert result is None


def test_has_collision_with_multiple_obstacles(
    vehicle_config,
    ego_state,
):
    obstacles = (
        CircleObstacle(
            obstacle_id="far",
            x=20.0,
            y=0.0,
            radius=1.0,
        ),
        BoxObstacle(
            obstacle_id="collision",
            x=1.5,
            y=0.0,
            yaw=0.0,
            length=1.0,
            width=1.0,
        ),
    )

    assert has_collision(
        state=ego_state,
        config=vehicle_config,
        obstacles=obstacles,
    )


def test_invalid_box_dimension():
    with pytest.raises(ValueError):
        oriented_box_corners(
            x=0.0,
            y=0.0,
            yaw=0.0,
            length=-1.0,
            width=2.0,
        )