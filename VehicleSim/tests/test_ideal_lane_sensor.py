from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2d.perception.ideal_lane_sensor import perceive_lane_corridors
from sim2d.perception.types import PerceptionConfig
from sim2d.types import VehicleState


@dataclass(frozen=True)
class _Lane:
    left_boundary: np.ndarray
    right_boundary: np.ndarray


def _ego() -> VehicleState:
    return VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.0)


def test_straight_lane_scan_recovers_local_corridor_without_topology() -> None:
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

    corridors = perceive_lane_corridors(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert len(corridors) == 1
    corridor = corridors[0]
    assert corridor.map_lane_id == "scan_corridor_0"
    assert corridor.predecessor_ids == ()
    assert corridor.successor_ids == ()
    assert corridor.centerline.shape == (41, 2)
    assert np.max(np.abs(corridor.centerline[:, 1])) < 1e-9
    assert np.allclose(corridor.left_boundary[:, 1], 1.8)
    assert np.allclose(corridor.right_boundary[:, 1], -1.8)


def test_fan_rays_keep_right_angle_exit_visible() -> None:
    # 车辆沿 +x 接近右转弯，弯后道路沿 -y。横向 x=常量扫描会在
    # 弯后竖直边界处退化，扇形射线应继续提供该部分的观测点。
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

    corridors = perceive_lane_corridors(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert len(corridors) == 1
    corridor = corridors[0]
    assert np.min(corridor.left_boundary[:, 1]) < -8.0
    assert np.min(corridor.right_boundary[:, 1]) < -8.0
    assert np.min(corridor.centerline[:, 1]) < -8.0
    assert corridor.predecessor_ids == ()
    assert corridor.successor_ids == ()


def test_lane_dropout_removes_scanned_corridor() -> None:
    x = np.linspace(-5.0, 20.0, 51)
    lane = _Lane(
        left_boundary=np.column_stack((x, np.full_like(x, 1.5))),
        right_boundary=np.column_stack((x, np.full_like(x, -1.5))),
    )
    config = PerceptionConfig(lane_dropout_probability=1.0)

    corridors = perceive_lane_corridors(
        (lane,),
        _ego(),
        config,
        np.random.default_rng(0),
    )

    assert corridors == ()
