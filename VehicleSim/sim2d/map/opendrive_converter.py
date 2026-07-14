from __future__ import annotations

from pathlib import Path

from sim2d.map.opendrive_geometry import (
    sample_road_reference_line,
)
from sim2d.map.opendrive_lanes import (
    OpenDriveLaneSample,
    sample_road_lanes,
)
from sim2d.map.opendrive_parser import (
    parse_opendrive_file,
)
from sim2d.map.opendrive_types import (
    OpenDriveLaneSide,
    OpenDriveRoad,
)
from sim2d.map.types import (
    Lane,
    LaneType,
    Polyline2D,
    RoadNetwork,
)


def map_lane_type(
    lane_type: str,
) -> LaneType:
    """
    将 OpenDRIVE lane type 转换为 VehicleSim LaneType。

    当前只映射基础类型，未知类型统一转为 UNKNOWN。
    """
    normalized = lane_type.strip().lower()

    mapping = {
        "driving": LaneType.DRIVING,
        "shoulder": LaneType.SHOULDER,
        "parking": LaneType.PARKING,
        "biking": LaneType.BIKE,
        "bike": LaneType.BIKE,
        "sidewalk": LaneType.SIDEWALK,
    }

    return mapping.get(
        normalized,
        LaneType.UNKNOWN,
    )


def make_lane_uid(
    *,
    road_id: str,
    section_index: int,
    lane_id: int,
) -> str:
    """
    生成 RoadNetwork 内全局唯一车道 ID。

    OpenDRIVE lane_id 只在 laneSection 内唯一，
    因此必须组合 road 和 section 信息。
    """
    return f"road_{road_id}" f"/section_{section_index}" f"/lane_{lane_id}"


def _reverse_polyline(
    points,
):
    """
    反转折线点顺序，并返回独立副本。
    """
    return points[::-1].copy()


def convert_lane_sample(
    sample: OpenDriveLaneSample,
) -> Lane:
    """
    将一条 OpenDriveLaneSample 转换为统一 Lane。

    当前假设地图采用右侧通行 RHT。

    对于右侧车道：
        行驶方向与 road s 增大方向一致。

    对于左侧车道：
        行驶方向与 road s 增大方向相反，
        因此需要反转点序，并交换 inner/outer
        对应的最终左右边界语义。
    """
    lane_uid = make_lane_uid(
        road_id=sample.road_id,
        section_index=sample.section_index,
        lane_id=sample.lane_id,
    )

    if sample.side is OpenDriveLaneSide.RIGHT:
        centerline_points = sample.centerline.copy()

        # 右侧车道沿 road s 正方向行驶。
        #
        # 面向行驶方向：
        # inner boundary 位于左侧，
        # outer boundary 位于右侧。
        left_boundary_points = sample.inner_boundary.copy()

        right_boundary_points = sample.outer_boundary.copy()

    elif sample.side is OpenDriveLaneSide.LEFT:
        # 左侧车道在 RHT 下逆 road s 方向行驶。
        centerline_points = _reverse_polyline(sample.centerline)

        # 反向行驶后：
        # 原 outer boundary 成为行驶方向左边界；
        # 原 inner boundary 成为行驶方向右边界。
        left_boundary_points = _reverse_polyline(sample.outer_boundary)

        right_boundary_points = _reverse_polyline(sample.inner_boundary)

    else:
        raise ValueError("Center lane cannot be converted to a " "VehicleSim Lane")

    return Lane(
        lane_id=lane_uid,
        lane_type=map_lane_type(sample.lane_type),
        centerline=Polyline2D(points=centerline_points),
        left_boundary=Polyline2D(points=left_boundary_points),
        right_boundary=Polyline2D(points=right_boundary_points),
        speed_limit=None,
        predecessor_ids=(),
        successor_ids=(),
        metadata={
            "source": "opendrive",
            "road_id": sample.road_id,
            "section_index": (sample.section_index),
            "lane_id": sample.lane_id,
            "lane_side": sample.side.value,
            "lane_type_raw": sample.lane_type,
            "section_s_start": (sample.section_s_start),
            "section_s_end": (sample.section_s_end),
        },
    )


def convert_road(
    road: OpenDriveRoad,
    *,
    sample_step: float = 0.5,
) -> tuple[Lane, ...]:
    """
    将一条 OpenDriveRoad 转换为统一 Lane 集合。

    当前不建立跨 laneSection 的拓扑连接。
    """
    reference_line = sample_road_reference_line(
        road=road,
        step=sample_step,
    )

    lane_samples = sample_road_lanes(
        road=road,
        reference_line=reference_line,
    )

    return tuple(convert_lane_sample(sample) for sample in lane_samples)


def convert_roads(
    roads: tuple[OpenDriveRoad, ...],
    *,
    sample_step: float = 0.5,
    source_name: str | None = None,
) -> RoadNetwork:
    """
    将多条 OpenDriveRoad 转换为统一 RoadNetwork。
    """
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")

    lanes: list[Lane] = []

    for road in roads:
        lanes.extend(
            convert_road(
                road=road,
                sample_step=sample_step,
            )
        )

    return RoadNetwork(
        lanes=tuple(lanes),
        source_type="opendrive",
        source_name=source_name,
        metadata={
            "road_count": len(roads),
            "sample_step": sample_step,
            "traffic_rule": "RHT",
        },
    )


def load_opendrive_road_network(
    path: str | Path,
    *,
    sample_step: float = 0.5,
) -> RoadNetwork:
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")

    file_path = Path(path).expanduser()

    roads = parse_opendrive_file(file_path)

    return convert_roads(
        roads=roads,
        sample_step=sample_step,
        source_name=file_path.name,
    )
