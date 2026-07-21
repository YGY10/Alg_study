from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim2d.map.types import RoadNetwork
from sim2d.world.road_geometry import WorldLaneGeometry

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoadDeformationConfig:
    """
    地图几何到真实世界几何的连续形变参数。

    形变由三部分组成：
        1. 全局平移和旋转；
        2. 沿 deformation_axis_yaw 方向的整体纵横向缩放；
        3. 随空间连续变化的纵横向局部扰动。

    因为所有 lane 都使用同一个连续空间场，所以地图中共享的连接点
    在世界层仍映射到同一个点，不会产生逐 lane 独立偏移造成的断裂。
    """

    offset_x: float = 0.0
    offset_y: float = 0.0
    yaw_offset: float = 0.0

    longitudinal_scale: float = 1.0
    lateral_scale: float = 1.0

    local_longitudinal_amplitude: float = 0.0
    local_lateral_amplitude: float = 0.0
    local_wavelength: float = 80.0
    local_phase: float = 0.0
    deformation_axis_yaw: float = 0.0

    def validate(self) -> None:
        values = np.array(
            [
                self.offset_x,
                self.offset_y,
                self.yaw_offset,
                self.longitudinal_scale,
                self.lateral_scale,
                self.local_longitudinal_amplitude,
                self.local_lateral_amplitude,
                self.local_wavelength,
                self.local_phase,
                self.deformation_axis_yaw,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "RoadDeformationConfig must contain finite values"
            )

        if self.longitudinal_scale <= 0.0:
            raise ValueError(
                "longitudinal_scale must be positive"
            )

        if self.lateral_scale <= 0.0:
            raise ValueError(
                "lateral_scale must be positive"
            )

        if self.local_wavelength <= 0.0:
            raise ValueError(
                "local_wavelength must be positive"
            )


def deform_points(
    points: FloatArray,
    config: RoadDeformationConfig,
) -> FloatArray:
    """使用连续二维形变场转换一组地图点。"""
    config.validate()

    array = np.asarray(
        points,
        dtype=np.float64,
    )

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"points must have shape [N, 2], got {array.shape}"
        )

    axis_cos = math.cos(config.deformation_axis_yaw)
    axis_sin = math.sin(config.deformation_axis_yaw)

    x = array[:, 0]
    y = array[:, 1]

    longitudinal = axis_cos * x + axis_sin * y
    lateral = -axis_sin * x + axis_cos * y

    phase = (
        2.0
        * math.pi
        * longitudinal
        / config.local_wavelength
        + config.local_phase
    )

    deformed_longitudinal = (
        config.longitudinal_scale * longitudinal
        + config.local_longitudinal_amplitude * np.sin(phase)
    )
    deformed_lateral = (
        config.lateral_scale * lateral
        + config.local_lateral_amplitude * np.sin(
            phase + 0.5 * math.pi
        )
    )

    local_x = (
        axis_cos * deformed_longitudinal
        - axis_sin * deformed_lateral
    )
    local_y = (
        axis_sin * deformed_longitudinal
        + axis_cos * deformed_lateral
    )

    yaw_cos = math.cos(config.yaw_offset)
    yaw_sin = math.sin(config.yaw_offset)

    world_x = (
        yaw_cos * local_x
        - yaw_sin * local_y
        + config.offset_x
    )
    world_y = (
        yaw_sin * local_x
        + yaw_cos * local_y
        + config.offset_y
    )

    return np.column_stack(
        [
            world_x,
            world_y,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


def deform_pose(
    *,
    x: float,
    y: float,
    yaw: float,
    config: RoadDeformationConfig,
) -> tuple[float, float, float]:
    """转换一个二维位姿，并由形变后的切向量得到新航向。"""
    origin = np.array(
        [[x, y]],
        dtype=np.float64,
    )

    epsilon = 1.0e-3
    heading_point = np.array(
        [
            [
                x + epsilon * math.cos(yaw),
                y + epsilon * math.sin(yaw),
            ]
        ],
        dtype=np.float64,
    )

    world_origin = deform_points(
        origin,
        config,
    )[0]
    world_heading = deform_points(
        heading_point,
        config,
    )[0]

    direction = world_heading - world_origin

    world_yaw = math.atan2(
        float(direction[1]),
        float(direction[0]),
    )

    return (
        float(world_origin[0]),
        float(world_origin[1]),
        world_yaw,
    )


def deform_road_network(
    road_network: RoadNetwork,
    config: RoadDeformationConfig,
) -> tuple[WorldLaneGeometry, ...]:
    """在继承地图拓扑的前提下生成真实世界车道几何。"""
    config.validate()

    return tuple(
        WorldLaneGeometry(
            entity_id=f"world_lane_{lane.lane_id}",
            map_lane_id=lane.lane_id,
            lane_type=lane.lane_type,
            centerline=deform_points(
                lane.centerline.points,
                config,
            ),
            left_boundary=deform_points(
                lane.left_boundary.points,
                config,
            ),
            right_boundary=deform_points(
                lane.right_boundary.points,
                config,
            ),
            predecessor_ids=lane.predecessor_ids,
            successor_ids=lane.successor_ids,
        )
        for lane in road_network.lanes
    )
