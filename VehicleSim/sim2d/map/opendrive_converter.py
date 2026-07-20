from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from sim2d.map.opendrive_geometry import (
    normalize_angle,
    sample_road_reference_line,
)
from sim2d.map.opendrive_lanes import (
    OpenDriveLaneSample,
    sample_road_lanes,
)
from sim2d.map.opendrive_parser import (
    parse_opendrive_map_file,
)
from sim2d.map.opendrive_types import (
    OpenDriveContactPoint,
    OpenDriveElementType,
    OpenDriveJunction,
    OpenDriveLaneSide,
    OpenDriveMap,
    OpenDriveArcGeometry,
    OpenDriveLineGeometry,
    OpenDriveRoad,
    OpenDriveRoadLink,
    OpenDriveSignal,
)
from sim2d.map.types import (
    Lane,
    LaneType,
    Polyline2D,
    RoadNetwork,
    TrafficSignal,
)


def _road_reference_pose_at_s(
    road: OpenDriveRoad,
    s: float,
    *,
    tolerance: float = 1e-6,
) -> tuple[float, float, float]:
    """
    精确计算 road reference line 在绝对弧长 s 处的二维位姿。

    返回：

        x, y, heading

    这里直接使用 line/arc 的解析表达式，不依赖地图采样步长。
    """
    if not math.isfinite(s):
        raise ValueError("Reference-line query s must be finite")

    if s < -tolerance or s > road.length + tolerance:
        raise ValueError(
            f"Reference-line query s={s} is outside road "
            f"{road.road_id!r} range [0, {road.length}]"
        )

    clamped_s = min(
        max(s, 0.0),
        road.length,
    )

    selected_geometry = None

    for geometry in road.geometries:
        geometry_end = geometry.s + geometry.length

        if (
            clamped_s >= geometry.s - tolerance
            and clamped_s <= geometry_end + tolerance
        ):
            selected_geometry = geometry

    if selected_geometry is None:
        raise ValueError(f"Road {road.road_id!r} has no geometry covering s={s}")

    local_s = min(
        max(clamped_s - selected_geometry.s, 0.0),
        selected_geometry.length,
    )

    if isinstance(selected_geometry, OpenDriveLineGeometry):
        x = selected_geometry.x + local_s * math.cos(selected_geometry.heading)
        y = selected_geometry.y + local_s * math.sin(selected_geometry.heading)
        heading = selected_geometry.heading

    elif isinstance(selected_geometry, OpenDriveArcGeometry):
        raw_heading = selected_geometry.heading + selected_geometry.curvature * local_s

        x = (
            selected_geometry.x
            + (math.sin(raw_heading) - math.sin(selected_geometry.heading))
            / selected_geometry.curvature
        )

        y = (
            selected_geometry.y
            - (math.cos(raw_heading) - math.cos(selected_geometry.heading))
            / selected_geometry.curvature
        )

        heading = raw_heading

    else:
        raise TypeError(
            "Unsupported OpenDRIVE geometry type: "
            f"{type(selected_geometry).__name__}"
        )

    return (
        float(x),
        float(y),
        float(normalize_angle(heading)),
    )


def _build_signal_controller_index(
    opendrive_map: OpenDriveMap,
) -> dict[str, tuple[str, ...]]:
    """
    建立 signal ID → controller ID 列表。
    """
    controller_ids_by_signal: dict[str, set[str]] = {}

    for controller in opendrive_map.controllers:
        for control in controller.controls:
            controller_ids_by_signal.setdefault(
                control.signal_id,
                set(),
            ).add(controller.controller_id)

    return {
        signal_id: tuple(sorted(controller_ids))
        for signal_id, controller_ids in controller_ids_by_signal.items()
    }


def convert_traffic_signal(
    signal: OpenDriveSignal,
    *,
    road: OpenDriveRoad,
    controller_ids: tuple[str, ...] = (),
) -> TrafficSignal:
    """
    将 OpenDRIVE signal 的 road s/t 坐标转换为世界坐标。

    OpenDRIVE 横向坐标 t 的正方向位于 reference line 左侧：

        x = x_ref - t * sin(heading_ref)
        y = y_ref + t * cos(heading_ref)

    signal 的实体朝向为：

        yaw = heading_ref + hOffset
    """
    x_ref, y_ref, heading_ref = _road_reference_pose_at_s(
        road,
        signal.s,
    )

    x = x_ref - signal.t * math.sin(heading_ref)
    y = y_ref + signal.t * math.cos(heading_ref)

    yaw = float(normalize_angle(heading_ref + signal.h_offset))

    valid_lane_ranges = tuple(
        (
            validity.from_lane,
            validity.to_lane,
        )
        for validity in signal.validities
    )

    return TrafficSignal(
        signal_id=signal.signal_id,
        road_id=road.road_id,
        x=x,
        y=y,
        yaw=yaw,
        s=signal.s,
        t=signal.t,
        z_offset=signal.z_offset,
        dynamic=signal.dynamic,
        name=signal.name,
        signal_type=signal.signal_type,
        subtype=signal.subtype,
        orientation=signal.orientation.value,
        controller_ids=controller_ids,
        valid_lane_ranges=valid_lane_ranges,
        metadata={
            "source": "opendrive",
            "country": signal.country,
            "value": signal.value,
            "text": signal.text,
            "height": signal.height,
            "width": signal.width,
            "roll": signal.roll,
            "pitch": signal.pitch,
            "h_offset": signal.h_offset,
        },
    )


def convert_traffic_signals(
    opendrive_map: OpenDriveMap,
) -> tuple[TrafficSignal, ...]:
    """
    转换完整地图中的物理 signal。

    signalReference 不会生成重复的 TrafficSignal；它在后续建立
    “道路/车道受哪个信号控制”的语义关联时使用。
    """
    controller_index = _build_signal_controller_index(opendrive_map)

    converted: list[TrafficSignal] = []

    for road in opendrive_map.roads:
        for signal in road.signals:
            converted.append(
                convert_traffic_signal(
                    signal,
                    road=road,
                    controller_ids=controller_index.get(
                        signal.signal_id,
                        (),
                    ),
                )
            )

    converted.sort(
        key=lambda signal: (
            signal.road_id,
            signal.s,
            signal.signal_id,
        )
    )

    return tuple(converted)


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
    *,
    road: OpenDriveRoad,
) -> Lane:
    """
    将一条 OpenDriveLaneSample 转换为统一 Lane。

    road 用于补充道路与路口语义元数据。

    当前假设地图采用右侧通行 RHT。

    对于右侧车道：
        行驶方向与 road s 增大方向一致。

    对于左侧车道：
        行驶方向与 road s 增大方向相反，
        因此需要反转点序，并交换 inner/outer
        对应的最终左右边界语义。

    注意：
        该函数只转换单条车道的几何和元数据。
        跨 laneSection 拓扑由 convert_road() 在所有
        lane 都创建完后统一建立。
    """
    if sample.road_id != road.road_id:
        raise ValueError(
            "OpenDriveLaneSample road_id does not match "
            "the supplied OpenDriveRoad: "
            f"{sample.road_id!r} != {road.road_id!r}"
        )

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

    is_junction_connector = road.junction_id is not None and road.junction_id != "-1"

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
            "road_id": road.road_id,
            "road_name": road.name,
            "junction_id": road.junction_id,
            "is_junction_connector": (is_junction_connector),
            "section_index": sample.section_index,
            "lane_id": sample.lane_id,
            "lane_side": sample.side.value,
            "lane_type_raw": sample.lane_type,
            "section_s_start": sample.section_s_start,
            "section_s_end": sample.section_s_end,
            "opendrive_predecessor_lane_id": (sample.predecessor_lane_id),
            "opendrive_successor_lane_id": (sample.successor_lane_id),
        },
    )


def _require_adjacent_lane_uid(
    *,
    road: OpenDriveRoad,
    lane_by_uid: dict[str, Lane],
    source_sample: OpenDriveLaneSample,
    target_section_index: int,
    target_lane_id: int,
    relation: str,
) -> str:
    """
    解析同一 road 内相邻 laneSection 的目标 lane UID。

    只有相邻 section 确实存在时才调用该函数。
    显式 lane link 指向不存在的 lane 时视为地图数据错误。
    """
    target_uid = make_lane_uid(
        road_id=road.road_id,
        section_index=target_section_index,
        lane_id=target_lane_id,
    )

    if target_uid not in lane_by_uid:
        source_uid = make_lane_uid(
            road_id=road.road_id,
            section_index=source_sample.section_index,
            lane_id=source_sample.lane_id,
        )

        raise ValueError(
            "OpenDRIVE lane link references an unknown "
            "adjacent lane: "
            f"source={source_uid!r}, "
            f"relation={relation!r}, "
            f"target={target_uid!r}"
        )

    return target_uid


def _build_intra_road_topology(
    *,
    road: OpenDriveRoad,
    lane_samples: tuple[OpenDriveLaneSample, ...],
    lanes: tuple[Lane, ...],
) -> tuple[Lane, ...]:
    """
    建立同一 road 内跨 laneSection 的有向车道拓扑。

    RHT 方向规则：

    RIGHT lane：
        行驶方向与 road s 增大方向一致。

        predecessor:
            section_index - 1
            lane_id = predecessor_lane_id

        successor:
            section_index + 1
            lane_id = successor_lane_id

    LEFT lane：
        行驶方向与 road s 增大方向相反。

        predecessor:
            section_index + 1
            lane_id = successor_lane_id

        successor:
            section_index - 1
            lane_id = predecessor_lane_id

    road 首尾处指向其他 road 的 lane link 暂不处理；
    该部分将在跨 road 拓扑阶段实现。
    """
    if len(lane_samples) != len(lanes):
        raise ValueError("lane_samples and lanes must have the same " "length")

    lane_by_uid = {lane.lane_id: lane for lane in lanes}

    sample_by_uid = {
        make_lane_uid(
            road_id=sample.road_id,
            section_index=sample.section_index,
            lane_id=sample.lane_id,
        ): sample
        for sample in lane_samples
    }

    if set(lane_by_uid) != set(sample_by_uid):
        raise ValueError("Converted lane IDs do not match sampled " "lane IDs")

    section_count = len(road.lane_sections)

    updated_lanes: list[Lane] = []

    for lane in lanes:
        sample = sample_by_uid[lane.lane_id]

        predecessor_ids: list[str] = []
        successor_ids: list[str] = []

        if sample.side is OpenDriveLaneSide.RIGHT:
            previous_section_index = sample.section_index - 1
            next_section_index = sample.section_index + 1

            if previous_section_index >= 0 and sample.predecessor_lane_id is not None:
                predecessor_ids.append(
                    _require_adjacent_lane_uid(
                        road=road,
                        lane_by_uid=lane_by_uid,
                        source_sample=sample,
                        target_section_index=(previous_section_index),
                        target_lane_id=(sample.predecessor_lane_id),
                        relation="predecessor",
                    )
                )

            if (
                next_section_index < section_count
                and sample.successor_lane_id is not None
            ):
                successor_ids.append(
                    _require_adjacent_lane_uid(
                        road=road,
                        lane_by_uid=lane_by_uid,
                        source_sample=sample,
                        target_section_index=(next_section_index),
                        target_lane_id=(sample.successor_lane_id),
                        relation="successor",
                    )
                )

        elif sample.side is OpenDriveLaneSide.LEFT:
            next_section_index = sample.section_index + 1
            previous_section_index = sample.section_index - 1

            # 左侧 lane 的 VehicleSim 行驶方向与 road s 相反。
            #
            # OpenDRIVE successor 指向更高 s 的 section，
            # 在车辆行驶语义中它是 predecessor。
            if (
                next_section_index < section_count
                and sample.successor_lane_id is not None
            ):
                predecessor_ids.append(
                    _require_adjacent_lane_uid(
                        road=road,
                        lane_by_uid=lane_by_uid,
                        source_sample=sample,
                        target_section_index=(next_section_index),
                        target_lane_id=(sample.successor_lane_id),
                        relation="predecessor",
                    )
                )

            # OpenDRIVE predecessor 指向更低 s 的 section，
            # 在车辆行驶语义中它是 successor。
            if previous_section_index >= 0 and sample.predecessor_lane_id is not None:
                successor_ids.append(
                    _require_adjacent_lane_uid(
                        road=road,
                        lane_by_uid=lane_by_uid,
                        source_sample=sample,
                        target_section_index=(previous_section_index),
                        target_lane_id=(sample.predecessor_lane_id),
                        relation="successor",
                    )
                )

        else:
            raise ValueError(
                "Center lane sample cannot participate " "in VehicleSim topology"
            )

        updated_lanes.append(
            replace(
                lane,
                predecessor_ids=tuple(predecessor_ids),
                successor_ids=tuple(successor_ids),
            )
        )

    return tuple(updated_lanes)


def convert_road(
    road: OpenDriveRoad,
    *,
    sample_step: float = 0.5,
) -> tuple[Lane, ...]:
    """
    将一条 OpenDriveRoad 转换为统一 Lane 集合。

    当前建立：

        几何；
        道路/路口元数据；
        同一 road 内跨 laneSection 的拓扑。

    暂不建立：

        跨 road 拓扑；
        junction connection / laneLink 拓扑。
    """
    reference_line = sample_road_reference_line(
        road=road,
        step=sample_step,
    )

    lane_samples = sample_road_lanes(
        road=road,
        reference_line=reference_line,
    )

    lanes = tuple(
        convert_lane_sample(
            sample,
            road=road,
        )
        for sample in lane_samples
    )

    return _build_intra_road_topology(
        road=road,
        lane_samples=lane_samples,
        lanes=lanes,
    )


def _boundary_section_index(
    road: OpenDriveRoad,
    contact_point: OpenDriveContactPoint,
) -> int:
    if contact_point is OpenDriveContactPoint.START:
        return 0

    return len(road.lane_sections) - 1


def _travel_exit_contact_point(
    side: OpenDriveLaneSide,
) -> OpenDriveContactPoint:
    """
    RHT 下 lane 沿行驶方向离开 road 的端点。
    """
    if side is OpenDriveLaneSide.RIGHT:
        return OpenDriveContactPoint.END

    if side is OpenDriveLaneSide.LEFT:
        return OpenDriveContactPoint.START

    raise ValueError("Center lane has no travel exit contact point")


def _road_link_at_contact_point(
    road: OpenDriveRoad,
    contact_point: OpenDriveContactPoint,
) -> OpenDriveRoadLink | None:
    if contact_point is OpenDriveContactPoint.START:
        return road.predecessor

    return road.successor


def _lane_link_id_at_contact_point(
    lane_metadata: dict[str, object],
    contact_point: OpenDriveContactPoint,
) -> int | None:
    if contact_point is OpenDriveContactPoint.START:
        value = lane_metadata.get("opendrive_predecessor_lane_id")
    else:
        value = lane_metadata.get("opendrive_successor_lane_id")

    if value is None:
        return None

    return int(value)


def _add_edge(
    edges: set[tuple[str, str]],
    *,
    source_uid: str,
    target_uid: str,
    lane_by_uid: dict[str, Lane],
    strict: bool,
    context: str,
) -> None:
    if source_uid not in lane_by_uid:
        if strict:
            raise ValueError(f"{context}: unknown source lane {source_uid!r}")
        return

    if target_uid not in lane_by_uid:
        if strict:
            raise ValueError(f"{context}: unknown target lane {target_uid!r}")
        return

    if source_uid == target_uid:
        return

    edges.add(
        (
            source_uid,
            target_uid,
        )
    )


def _add_direct_road_successor_edges(
    *,
    roads: tuple[OpenDriveRoad, ...],
    lane_by_uid: dict[str, Lane],
    edges: set[tuple[str, str]],
) -> None:
    """
    建立 elementType="road" 的跨 road 有向连接。

    仅从 lane 的行驶出口端建立 successor。
    """
    road_by_id = {road.road_id: road for road in roads}

    for lane in tuple(lane_by_uid.values()):
        road_id = str(lane.metadata["road_id"])
        section_index = int(lane.metadata["section_index"])
        lane_id = int(lane.metadata["lane_id"])
        side = OpenDriveLaneSide(str(lane.metadata["lane_side"]))

        source_road = road_by_id[road_id]

        exit_contact = _travel_exit_contact_point(side)

        source_boundary_section = _boundary_section_index(
            source_road,
            exit_contact,
        )

        if section_index != source_boundary_section:
            continue

        road_link = _road_link_at_contact_point(
            source_road,
            exit_contact,
        )

        if road_link is None or road_link.element_type is not OpenDriveElementType.ROAD:
            continue

        if road_link.contact_point is None:
            continue

        target_road = road_by_id.get(road_link.element_id)

        if target_road is None:
            continue

        target_lane_id = _lane_link_id_at_contact_point(
            dict(lane.metadata),
            exit_contact,
        )

        if target_lane_id is None:
            continue

        target_section_index = _boundary_section_index(
            target_road,
            road_link.contact_point,
        )

        target_uid = make_lane_uid(
            road_id=target_road.road_id,
            section_index=target_section_index,
            lane_id=target_lane_id,
        )

        _add_edge(
            edges,
            source_uid=lane.lane_id,
            target_uid=target_uid,
            lane_by_uid=lane_by_uid,
            strict=False,
            context="direct road link",
        )


def _incoming_junction_contact_point(
    *,
    incoming_road: OpenDriveRoad,
    junction_id: str,
) -> OpenDriveContactPoint | None:
    if (
        incoming_road.predecessor is not None
        and incoming_road.predecessor.element_type is OpenDriveElementType.JUNCTION
        and incoming_road.predecessor.element_id == junction_id
    ):
        return OpenDriveContactPoint.START

    if (
        incoming_road.successor is not None
        and incoming_road.successor.element_type is OpenDriveElementType.JUNCTION
        and incoming_road.successor.element_id == junction_id
    ):
        return OpenDriveContactPoint.END

    return None


def _add_junction_edges(
    *,
    roads: tuple[OpenDriveRoad, ...],
    junctions: tuple[OpenDriveJunction, ...],
    lane_by_uid: dict[str, Lane],
    edges: set[tuple[str, str]],
) -> None:
    """
    根据 junction connection/laneLink 建立：

        incoming road lane
        → connecting road lane
    """
    road_by_id = {road.road_id: road for road in roads}

    for junction in junctions:
        for connection in junction.connections:
            incoming_road = road_by_id.get(connection.incoming_road_id)
            connecting_road = road_by_id.get(connection.connecting_road_id)

            if incoming_road is None or connecting_road is None:
                continue

            incoming_contact = _incoming_junction_contact_point(
                incoming_road=incoming_road,
                junction_id=junction.junction_id,
            )

            if incoming_contact is None:
                continue

            incoming_section_index = _boundary_section_index(
                incoming_road,
                incoming_contact,
            )

            connecting_section_index = _boundary_section_index(
                connecting_road,
                connection.contact_point,
            )

            for lane_link in connection.lane_links:
                source_uid = make_lane_uid(
                    road_id=incoming_road.road_id,
                    section_index=incoming_section_index,
                    lane_id=lane_link.from_lane_id,
                )

                target_uid = make_lane_uid(
                    road_id=connecting_road.road_id,
                    section_index=connecting_section_index,
                    lane_id=lane_link.to_lane_id,
                )

                _add_edge(
                    edges,
                    source_uid=source_uid,
                    target_uid=target_uid,
                    lane_by_uid=lane_by_uid,
                    strict=False,
                    context=(
                        "junction "
                        f"{junction.junction_id!r} "
                        f"connection {connection.connection_id!r}"
                    ),
                )


def _rebuild_bidirectional_lane_topology(
    *,
    lanes: tuple[Lane, ...],
    cross_road_edges: set[tuple[str, str]],
) -> tuple[Lane, ...]:
    """
    合并已有同 road 边与跨 road 边，并根据 successor
    统一重建 predecessor，保证二者互相一致。
    """
    lane_by_uid = {lane.lane_id: lane for lane in lanes}

    edges: set[tuple[str, str]] = set(cross_road_edges)

    for lane in lanes:
        for successor_id in lane.successor_ids:
            if successor_id in lane_by_uid:
                edges.add(
                    (
                        lane.lane_id,
                        successor_id,
                    )
                )

    predecessor_by_uid: dict[
        str,
        set[str],
    ] = {lane_id: set() for lane_id in lane_by_uid}

    successor_by_uid: dict[
        str,
        set[str],
    ] = {lane_id: set() for lane_id in lane_by_uid}

    for source_uid, target_uid in edges:
        if source_uid not in lane_by_uid or target_uid not in lane_by_uid:
            continue

        successor_by_uid[source_uid].add(target_uid)

        predecessor_by_uid[target_uid].add(source_uid)

    return tuple(
        replace(
            lane,
            predecessor_ids=tuple(sorted(predecessor_by_uid[lane.lane_id])),
            successor_ids=tuple(sorted(successor_by_uid[lane.lane_id])),
        )
        for lane in lanes
    )


def convert_opendrive_map(
    opendrive_map: OpenDriveMap,
    *,
    sample_step: float = 0.5,
    source_name: str | None = None,
) -> RoadNetwork:
    """
    将完整 OpenDriveMap 转换为统一 RoadNetwork。

    建立：

        同一 road 内跨 laneSection；
        普通 road-to-road；
        incoming road → junction connecting road；
        connecting road → outgoing road
            由 connecting road 自身的 road link 处理。
    """
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")

    lanes: list[Lane] = []

    junction_road_count = 0

    for road in opendrive_map.roads:
        if road.junction_id is not None:
            junction_road_count += 1

        lanes.extend(
            convert_road(
                road=road,
                sample_step=sample_step,
            )
        )

    signal_count = sum(len(road.signals) for road in opendrive_map.roads)
    signal_reference_count = sum(
        len(road.signal_references) for road in opendrive_map.roads
    )
    object_count = sum(len(road.objects) for road in opendrive_map.roads)
    controller_count = len(opendrive_map.controllers)
    controller_control_count = sum(
        len(controller.controls) for controller in opendrive_map.controllers
    )

    traffic_signals = convert_traffic_signals(opendrive_map)

    initial_lanes = tuple(lanes)

    lane_by_uid = {lane.lane_id: lane for lane in initial_lanes}

    cross_road_edges: set[tuple[str, str]] = set()

    _add_direct_road_successor_edges(
        roads=opendrive_map.roads,
        lane_by_uid=lane_by_uid,
        edges=cross_road_edges,
    )

    _add_junction_edges(
        roads=opendrive_map.roads,
        junctions=opendrive_map.junctions,
        lane_by_uid=lane_by_uid,
        edges=cross_road_edges,
    )

    final_lanes = _rebuild_bidirectional_lane_topology(
        lanes=initial_lanes,
        cross_road_edges=cross_road_edges,
    )

    intra_road_edge_count = sum(
        1
        for lane in final_lanes
        for successor_id in lane.successor_ids
        if str(lane.metadata["road_id"])
        == str(lane_by_uid[successor_id].metadata["road_id"])
    )

    cross_road_edge_count = sum(
        1
        for lane in final_lanes
        for successor_id in lane.successor_ids
        if str(lane.metadata["road_id"])
        != str(lane_by_uid[successor_id].metadata["road_id"])
    )

    return RoadNetwork(
        lanes=final_lanes,
        source_type="opendrive",
        source_name=(
            source_name if source_name is not None else opendrive_map.source_name
        ),
        traffic_signals=traffic_signals,
        metadata={
            "road_count": len(opendrive_map.roads),
            "junction_count": len(opendrive_map.junctions),
            "junction_road_count": (junction_road_count),
            "signal_count": signal_count,
            "signal_reference_count": (signal_reference_count),
            "object_count": object_count,
            "controller_count": controller_count,
            "controller_control_count": (controller_control_count),
            "sample_step": sample_step,
            "traffic_rule": "RHT",
            "intra_road_topology_edge_count": (intra_road_edge_count),
            "cross_road_topology_edge_count": (cross_road_edge_count),
            "topology_edge_count": (intra_road_edge_count + cross_road_edge_count),
        },
    )


def convert_roads(
    roads: tuple[OpenDriveRoad, ...],
    *,
    sample_step: float = 0.5,
    source_name: str | None = None,
) -> RoadNetwork:
    """
    兼容旧接口：没有 junction 文档信息时仍可转换 roads。
    """
    return convert_opendrive_map(
        OpenDriveMap(
            roads=roads,
            junctions=(),
            source_name=source_name,
        ),
        sample_step=sample_step,
        source_name=source_name,
    )


def load_opendrive_road_network(
    path: str | Path,
    *,
    sample_step: float = 0.5,
) -> RoadNetwork:
    """
    从 OpenDRIVE 文件加载统一 RoadNetwork。
    """
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")

    file_path = Path(path).expanduser().resolve()

    opendrive_map = parse_opendrive_map_file(file_path)

    return convert_opendrive_map(
        opendrive_map=opendrive_map,
        sample_step=sample_step,
        source_name=file_path.name,
    )
