from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from sim2d.map.types import Lane, RoadNetwork


def _heading_at_start(lane: Lane) -> float:
    vector = lane.centerline.points[1] - lane.centerline.points[0]
    return math.atan2(float(vector[1]), float(vector[0]))


def _heading_at_end(lane: Lane) -> float:
    vector = lane.centerline.points[-1] - lane.centerline.points[-2]
    return math.atan2(float(vector[1]), float(vector[0]))


def _angle_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _explicit_edges(lanes: tuple[Lane, ...]) -> set[tuple[str, str]]:
    lane_ids = {lane.lane_id for lane in lanes}
    return {
        (lane.lane_id, successor_id)
        for lane in lanes
        for successor_id in lane.successor_ids
        if successor_id in lane_ids and successor_id != lane.lane_id
    }


def _infer_geometric_edges(
    lanes: tuple[Lane, ...],
    *,
    endpoint_tolerance: float,
    heading_tolerance: float,
    ambiguity_tolerance: float,
) -> set[tuple[str, str]]:
    """
    从 lane 行驶方向终点到其他 lane 行驶方向起点推断缺失拓扑。

    仅当端点足够接近且航向连续时建立连接。对一个 source lane，保留
    与最佳候选距离接近的候选，从而允许 junction 分叉，同时避免把同一
    路口附近但明显更远的 lane 全部连接起来。
    """
    if endpoint_tolerance <= 0.0:
        raise ValueError("endpoint_tolerance must be positive")
    if heading_tolerance <= 0.0:
        raise ValueError("heading_tolerance must be positive")
    if ambiguity_tolerance < 0.0:
        raise ValueError("ambiguity_tolerance must be non-negative")

    starts = np.vstack([lane.centerline.points[0] for lane in lanes])
    start_headings = np.array([_heading_at_start(lane) for lane in lanes])
    inferred: set[tuple[str, str]] = set()

    for source_index, source in enumerate(lanes):
        source_end = source.centerline.points[-1]
        source_heading = _heading_at_end(source)
        deltas = starts - source_end[None, :]
        distances = np.linalg.norm(deltas, axis=1)

        candidates: list[tuple[float, float, int]] = []
        for target_index, target in enumerate(lanes):
            if target_index == source_index:
                continue

            distance = float(distances[target_index])
            if distance > endpoint_tolerance:
                continue

            heading_error = _angle_difference(
                source_heading,
                float(start_headings[target_index]),
            )
            if heading_error > heading_tolerance:
                continue

            candidates.append((distance, heading_error, target_index))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], item[1], lanes[item[2]].lane_id))
        best_distance = candidates[0][0]

        for distance, _, target_index in candidates:
            if distance > best_distance + ambiguity_tolerance:
                break
            inferred.add((source.lane_id, lanes[target_index].lane_id))

    return inferred


def repair_road_network_topology(
    road_network: RoadNetwork,
    *,
    endpoint_tolerance: float = 0.75,
    heading_tolerance_degrees: float = 35.0,
    ambiguity_tolerance: float = 0.15,
) -> RoadNetwork:
    """
    保留 OpenDRIVE 显式拓扑，并用几何连续性补全缺失 lane 连接。

    该函数不改变 lane 几何和 ID，只重建 predecessor/successor。地图已有
    显式连接时始终优先保留；几何推断只补充缺失信息。
    """
    road_network.validate()
    lanes = road_network.lanes
    if not lanes:
        return road_network

    explicit = _explicit_edges(lanes)
    inferred = _infer_geometric_edges(
        lanes,
        endpoint_tolerance=endpoint_tolerance,
        heading_tolerance=math.radians(heading_tolerance_degrees),
        ambiguity_tolerance=ambiguity_tolerance,
    )
    edges = explicit | inferred

    predecessor_by_id = {lane.lane_id: set() for lane in lanes}
    successor_by_id = {lane.lane_id: set() for lane in lanes}

    for source_id, target_id in edges:
        successor_by_id[source_id].add(target_id)
        predecessor_by_id[target_id].add(source_id)

    repaired_lanes = tuple(
        replace(
            lane,
            predecessor_ids=tuple(sorted(predecessor_by_id[lane.lane_id])),
            successor_ids=tuple(sorted(successor_by_id[lane.lane_id])),
        )
        for lane in lanes
    )

    metadata = dict(road_network.metadata)
    metadata.update(
        {
            "explicit_topology_edge_count": len(explicit),
            "inferred_topology_edge_count": len(inferred - explicit),
            "topology_edge_count": len(edges),
            "topology_repair_endpoint_tolerance": endpoint_tolerance,
            "topology_repair_heading_tolerance_degrees": heading_tolerance_degrees,
        }
    )

    return RoadNetwork(
        lanes=repaired_lanes,
        source_type=road_network.source_type,
        source_name=road_network.source_name,
        traffic_signals=road_network.traffic_signals,
        metadata=metadata,
    )
