from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2d.perception.ideal_lane_sensor import perceive_lane_lines
from sim2d.perception.types import PerceptionConfig
from sim2d.types import VehicleState


@dataclass(frozen=True)
class _Lane:
    left_boundary: np.ndarray
    right_boundary: np.ndarray


def _ego() -> VehicleState:
    return VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.0)


def test_straight_lane_scan_publishes_two_independent_lane_lines() -> None:
    x = np.linspace(-10.0, 40.0, 101)
    lane = _Lane(
        left_boundary=np.column_stack((x, np.full_like(x, 1.8))),
        right_boundary=np.column_stack((x, np.full_like(x, -1.8))),
    )
    config = PerceptionConfig(
        forward_range=30.0,
        rear_range=5.0,
        lateral_range=10.0,
        lane_transverse_spacing=1.0,
        lane_ray_count=61,
        lane_output_point_count=41,
    )

    lines = perceive_lane_lines(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert len(lines) == 2
    assert all(line.points.shape == (41, 2) for line in lines)
    lateral_positions = sorted(
        float(np.median(line.points[:, 1])) for line in lines
    )
    assert lateral_positions == [-1.8, 1.8]
    assert all(line.line_id.startswith("lane_line_") for line in lines)


def test_fan_rays_keep_right_angle_lane_lines_visible() -> None:
    left = np.array(
        [
            [-5.0, 1.8],
            [8.0, 1.8],
            [9.5, 1.2],
            [10.2, 0.0],
            [10.2, -12.0],
        ],
        dtype=np.float64,
    )
    right = np.array(
        [
            [-5.0, -1.8],
            [6.5, -1.8],
            [7.5, -2.5],
            [8.2, -3.8],
            [8.2, -12.0],
        ],
        dtype=np.float64,
    )
    lane = _Lane(left_boundary=left, right_boundary=right)
    config = PerceptionConfig(
        forward_range=25.0,
        rear_range=5.0,
        lateral_range=20.0,
        lane_transverse_spacing=1.0,
        lane_ray_count=181,
        lane_ray_half_angle=np.deg2rad(110.0),
        lane_output_point_count=61,
    )

    lines = perceive_lane_lines(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert len(lines) == 2
    assert all(np.min(line.points[:, 1]) < -8.0 for line in lines)


def test_geometrically_continuous_fragments_are_joined_across_lanes() -> None:
    first = _Lane(
        left_boundary=np.array([[-5.0, 1.8], [8.0, 1.8]]),
        right_boundary=np.array([[-5.0, -1.8], [8.0, -1.8]]),
    )
    second = _Lane(
        left_boundary=np.array([[8.2, 1.8], [10.0, 1.0], [10.0, -10.0]]),
        right_boundary=np.array([[8.2, -1.8], [8.2, -3.0], [8.2, -10.0]]),
    )
    config = PerceptionConfig(
        forward_range=25.0,
        rear_range=5.0,
        lateral_range=20.0,
        lane_ray_count=181,
        lane_ray_half_angle=np.deg2rad(110.0),
        lane_line_join_distance=2.0,
        lane_line_join_heading=np.deg2rad(100.0),
        lane_output_point_count=61,
    )

    lines = perceive_lane_lines(
        (first, second),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert len(lines) == 2
    assert all(np.max(line.points[:, 0]) >= 8.0 for line in lines)
    assert all(np.min(line.points[:, 1]) < -7.0 for line in lines)


def test_lane_dropout_removes_all_scanned_lines() -> None:
    x = np.linspace(-5.0, 20.0, 51)
    lane = _Lane(
        left_boundary=np.column_stack((x, np.full_like(x, 1.5))),
        right_boundary=np.column_stack((x, np.full_like(x, -1.5))),
    )
    config = PerceptionConfig(lane_dropout_probability=1.0)

    lines = perceive_lane_lines(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert lines == ()
