from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim2d.map.types import (
    Lane,
    LaneProjection,
    LaneType,
    RoadNetwork,
)

FloatArray = NDArray[np.float64]


class NoRouteError(RuntimeError):
    """无法生成地图车道路由。"""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        if not reason.strip():
            raise ValueError("NoRouteError.reason cannot be empty")
        self.reason = reason


@dataclass(frozen=True)
class LaneRoute:
    """基于 RoadNetwork lane graph 的有向路线。"""

    lane_ids: tuple[str, ...]
    reference_path: FloatArray
    start_projection: LaneProjection
    goal_projection: LaneProjection

    def __post_init__(self) -> None:
        if not isinstance(self.lane_ids, tuple):
            object.__setattr__(self, "lane_ids", tuple(self.lane_ids))
        if not self.lane_ids:
            raise ValueError("LaneRoute.lane_ids cannot be empty")
        if len(set(self.lane_ids)) != len(self.lane_ids):
            raise ValueError("LaneRoute.lane_ids contains duplicates")
        if self.start_projection.lane_id != self.lane_ids[0]:
            raise ValueError(
                "LaneRoute.start_projection.lane_id must match the first lane_id"
            )
        if self.goal_projection.lane_id != self.lane_ids[-1]:
            raise ValueError(
                "LaneRoute.goal_projection.lane_id must match the last lane_id"
            )

        path = np.asarray(self.reference_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 5:
            raise ValueError(
                "LaneRoute.reference_path must have " f"shape [N, 5], got {path.shape}"
            )
        if path.shape[0] < 2:
            raise ValueError(
                "LaneRoute.reference_path must contain at least two points"
            )
        if not np.all(np.isfinite(path)):
            raise ValueError("LaneRoute.reference_path contains non-finite values")
        if abs(float(path[0, 3])) > 1e-9:
            raise ValueError("LaneRoute.reference_path arc length must start at zero")
        if not np.all(np.diff(path[:, 3]) > 0.0):
            raise ValueError(
                "LaneRoute.reference_path arc length must be strictly increasing"
            )
        object.__setattr__(self, "reference_path", path.copy())


def build_lane_route(
    road_network: RoadNetwork,
    *,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    snap_distance: float = 3.0,
    minimum_route_length: float = 1e-3,
    minimum_point_spacing: float = 1e-6,
    maximum_connection_gap: float = 2.0,
) -> LaneRoute:
    """
    在 RoadNetwork 的有向 lane graph 上生成路线。

    同 lane 时直接精确截取；不同 lane 时沿 successor_ids
    执行 Dijkstra，之后裁剪首尾 lane 并拼接中心线。
    """
    _validate_route_arguments(
        road_network=road_network,
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        snap_distance=snap_distance,
        minimum_route_length=minimum_route_length,
        minimum_point_spacing=minimum_point_spacing,
        maximum_connection_gap=maximum_connection_gap,
    )

    start_projection = _project_route_point(
        road_network=road_network,
        x=start_x,
        y=start_y,
        snap_distance=snap_distance,
        point_name="start",
    )
    goal_projection = _project_route_point(
        road_network=road_network,
        x=goal_x,
        y=goal_y,
        snap_distance=snap_distance,
        point_name="goal",
    )

    if start_projection.lane_id == goal_projection.lane_id:
        return _build_single_lane_route(
            road_network=road_network,
            start_projection=start_projection,
            goal_projection=goal_projection,
            minimum_route_length=minimum_route_length,
            minimum_point_spacing=minimum_point_spacing,
        )

    lane_ids = _find_shortest_lane_path(
        road_network=road_network,
        start_projection=start_projection,
        goal_projection=goal_projection,
    )

    route_xy = _assemble_multi_lane_route_xy(
        road_network=road_network,
        lane_ids=lane_ids,
        start_projection=start_projection,
        goal_projection=goal_projection,
        minimum_point_spacing=minimum_point_spacing,
        maximum_connection_gap=maximum_connection_gap,
    )
    reference_path = _build_reference_path(
        route_xy,
        minimum_point_spacing=minimum_point_spacing,
    )

    if float(reference_path[-1, 3]) < minimum_route_length:
        raise NoRouteError("Generated route is too short", reason="route_too_short")

    return LaneRoute(
        lane_ids=lane_ids,
        reference_path=reference_path,
        start_projection=start_projection,
        goal_projection=goal_projection,
    )


def build_same_lane_route(
    road_network: RoadNetwork,
    *,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    snap_distance: float = 3.0,
    minimum_route_length: float = 1e-3,
    minimum_point_spacing: float = 1e-6,
) -> LaneRoute:
    """保留旧接口，只允许同一条有向 driving lane 内的前向路线。"""
    _validate_route_arguments(
        road_network=road_network,
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        snap_distance=snap_distance,
        minimum_route_length=minimum_route_length,
        minimum_point_spacing=minimum_point_spacing,
        maximum_connection_gap=2.0,
    )
    start_projection = _project_route_point(
        road_network=road_network,
        x=start_x,
        y=start_y,
        snap_distance=snap_distance,
        point_name="start",
    )
    goal_projection = _project_route_point(
        road_network=road_network,
        x=goal_x,
        y=goal_y,
        snap_distance=snap_distance,
        point_name="goal",
    )
    if start_projection.lane_id != goal_projection.lane_id:
        raise NoRouteError(
            "Start and goal are on different lanes: "
            f"{start_projection.lane_id!r} != {goal_projection.lane_id!r}",
            reason="different_lane",
        )
    return _build_single_lane_route(
        road_network=road_network,
        start_projection=start_projection,
        goal_projection=goal_projection,
        minimum_route_length=minimum_route_length,
        minimum_point_spacing=minimum_point_spacing,
    )


def _validate_route_arguments(
    *,
    road_network: RoadNetwork,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    snap_distance: float,
    minimum_route_length: float,
    minimum_point_spacing: float,
    maximum_connection_gap: float,
) -> None:
    road_network.validate()
    values = np.array(
        [
            start_x,
            start_y,
            goal_x,
            goal_y,
            snap_distance,
            minimum_route_length,
            minimum_point_spacing,
            maximum_connection_gap,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Route construction arguments must be finite")
    if snap_distance < 0.0:
        raise ValueError("snap_distance must be non-negative")
    if minimum_route_length <= 0.0:
        raise ValueError("minimum_route_length must be positive")
    if minimum_point_spacing <= 0.0:
        raise ValueError("minimum_point_spacing must be positive")
    if maximum_connection_gap < 0.0:
        raise ValueError("maximum_connection_gap must be non-negative")


def _project_route_point(
    *,
    road_network: RoadNetwork,
    x: float,
    y: float,
    snap_distance: float,
    point_name: str,
) -> LaneProjection:
    projection = road_network.nearest_lane_point(
        x=x,
        y=y,
        lane_types=(LaneType.DRIVING,),
        max_distance=snap_distance,
    )
    if projection is None:
        raise NoRouteError(
            f"{point_name.capitalize()} point cannot be projected onto a "
            "driving lane within snap_distance",
            reason=f"{point_name}_projection_failed",
        )
    return projection


def _build_single_lane_route(
    *,
    road_network: RoadNetwork,
    start_projection: LaneProjection,
    goal_projection: LaneProjection,
    minimum_route_length: float,
    minimum_point_spacing: float,
) -> LaneRoute:
    route_length = goal_projection.arc_length - start_projection.arc_length
    if route_length < minimum_route_length:
        if route_length < 0.0:
            reason = "goal_behind_start"
            message = "Goal lies behind start on the same directed lane"
        else:
            reason = "route_too_short"
            message = "Start and goal projections are too close to form a route"
        raise NoRouteError(message, reason=reason)

    lane = road_network.get_lane(start_projection.lane_id)
    route_xy = _extract_projected_subpolyline(
        centerline=lane.centerline.points,
        start_projection=start_projection,
        goal_projection=goal_projection,
        minimum_point_spacing=minimum_point_spacing,
    )
    reference_path = _build_reference_path(
        route_xy,
        minimum_point_spacing=minimum_point_spacing,
    )
    return LaneRoute(
        lane_ids=(lane.lane_id,),
        reference_path=reference_path,
        start_projection=start_projection,
        goal_projection=goal_projection,
    )


def _find_shortest_lane_path(
    *,
    road_network: RoadNetwork,
    start_projection: LaneProjection,
    goal_projection: LaneProjection,
) -> tuple[str, ...]:
    """使用 Dijkstra 搜索 start lane 到 goal lane 的最短有向路径。"""
    start_lane_id = start_projection.lane_id
    goal_lane_id = goal_projection.lane_id
    start_lane = road_network.get_lane(start_lane_id)

    initial_cost = max(
        start_lane.centerline.length - start_projection.arc_length,
        0.0,
    )
    distance_by_lane: dict[str, float] = {start_lane_id: initial_cost}
    predecessor_by_lane: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(initial_cost, start_lane_id)]
    visited: set[str] = set()

    while queue:
        current_cost, current_lane_id = heapq.heappop(queue)
        if current_lane_id in visited:
            continue
        visited.add(current_lane_id)

        if current_lane_id == goal_lane_id:
            return _reconstruct_lane_path(
                predecessor_by_lane=predecessor_by_lane,
                start_lane_id=start_lane_id,
                goal_lane_id=goal_lane_id,
            )

        current_lane = road_network.get_lane(current_lane_id)
        for successor_id in current_lane.successor_ids:
            successor = road_network.find_lane(successor_id)
            if successor is None:
                raise NoRouteError(
                    f"Lane graph contains an unknown successor {successor_id!r}",
                    reason="invalid_lane_graph",
                )
            if successor.lane_type is not LaneType.DRIVING:
                continue

            edge_cost = (
                goal_projection.arc_length
                if successor_id == goal_lane_id
                else successor.centerline.length
            )
            candidate_cost = current_cost + edge_cost
            known_cost = distance_by_lane.get(successor_id, math.inf)
            if candidate_cost >= known_cost:
                continue

            distance_by_lane[successor_id] = candidate_cost
            predecessor_by_lane[successor_id] = current_lane_id
            heapq.heappush(queue, (candidate_cost, successor_id))

    raise NoRouteError(
        "No directed lane path exists from " f"{start_lane_id!r} to {goal_lane_id!r}",
        reason="unreachable",
    )


def _reconstruct_lane_path(
    *,
    predecessor_by_lane: dict[str, str],
    start_lane_id: str,
    goal_lane_id: str,
) -> tuple[str, ...]:
    lane_ids = [goal_lane_id]
    current_lane_id = goal_lane_id
    while current_lane_id != start_lane_id:
        predecessor_id = predecessor_by_lane.get(current_lane_id)
        if predecessor_id is None:
            raise NoRouteError(
                "Failed to reconstruct lane path",
                reason="invalid_lane_graph",
            )
        lane_ids.append(predecessor_id)
        current_lane_id = predecessor_id
    lane_ids.reverse()
    return tuple(lane_ids)


def _assemble_multi_lane_route_xy(
    *,
    road_network: RoadNetwork,
    lane_ids: tuple[str, ...],
    start_projection: LaneProjection,
    goal_projection: LaneProjection,
    minimum_point_spacing: float,
    maximum_connection_gap: float,
) -> FloatArray:
    if len(lane_ids) < 2:
        raise ValueError("Multi-lane route requires at least two lanes")

    points: list[FloatArray] = []
    for lane_index, lane_id in enumerate(lane_ids):
        lane = road_network.get_lane(lane_id)
        if lane_index == 0:
            lane_points = _extract_lane_suffix(
                lane=lane,
                projection=start_projection,
                minimum_point_spacing=minimum_point_spacing,
            )
        elif lane_index == len(lane_ids) - 1:
            lane_points = _extract_lane_prefix(
                lane=lane,
                projection=goal_projection,
                minimum_point_spacing=minimum_point_spacing,
            )
        else:
            lane_points = lane.centerline.points

        _append_polyline(
            points=points,
            polyline=lane_points,
            minimum_point_spacing=minimum_point_spacing,
            maximum_connection_gap=maximum_connection_gap,
            previous_lane_id=None if lane_index == 0 else lane_ids[lane_index - 1],
            current_lane_id=lane_id,
        )

    result = np.asarray(points, dtype=np.float64)
    if result.shape[0] < 2:
        raise NoRouteError(
            "Assembled route contains fewer than two distinct points",
            reason="route_too_short",
        )
    return result


def _extract_lane_suffix(
    *,
    lane: Lane,
    projection: LaneProjection,
    minimum_point_spacing: float,
) -> FloatArray:
    if projection.lane_id != lane.lane_id:
        raise ValueError("Projection lane_id does not match lane")
    points: list[FloatArray] = [projection.point.copy()]
    for point in lane.centerline.points[projection.segment_index + 1 :]:
        _append_if_separated(points, point, minimum_point_spacing)
    if len(points) < 2:
        raise NoRouteError(
            "Start projection is too close to lane end",
            reason="route_too_short",
        )
    return np.asarray(points, dtype=np.float64)


def _extract_lane_prefix(
    *,
    lane: Lane,
    projection: LaneProjection,
    minimum_point_spacing: float,
) -> FloatArray:
    if projection.lane_id != lane.lane_id:
        raise ValueError("Projection lane_id does not match lane")
    points: list[FloatArray] = []
    for point in lane.centerline.points[: projection.segment_index + 1]:
        if not points:
            points.append(np.asarray(point, dtype=np.float64).copy())
        else:
            _append_if_separated(points, point, minimum_point_spacing)
    if not points:
        points.append(lane.centerline.points[0].copy())
    _append_if_separated(points, projection.point, minimum_point_spacing)
    if len(points) < 2:
        raise NoRouteError(
            "Goal projection is too close to lane start",
            reason="route_too_short",
        )
    return np.asarray(points, dtype=np.float64)


def _append_polyline(
    *,
    points: list[FloatArray],
    polyline: FloatArray,
    minimum_point_spacing: float,
    maximum_connection_gap: float,
    previous_lane_id: str | None,
    current_lane_id: str,
) -> None:
    values = np.asarray(polyline, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 1:
        raise ValueError("polyline must have shape [N, 2]")

    if not points:
        points.append(values[0].copy())
        start_index = 1
    else:
        connection_gap = float(np.linalg.norm(values[0] - points[-1]))
        if connection_gap > maximum_connection_gap:
            raise NoRouteError(
                "Adjacent lane centerlines are not geometrically continuous: "
                f"{previous_lane_id!r} -> {current_lane_id!r}, "
                f"gap={connection_gap:.6f} m",
                reason="lane_connection_gap",
            )
        _append_if_separated(points, values[0], minimum_point_spacing)
        start_index = 1

    for point in values[start_index:]:
        _append_if_separated(points, point, minimum_point_spacing)


def _extract_projected_subpolyline(
    *,
    centerline: FloatArray,
    start_projection: LaneProjection,
    goal_projection: LaneProjection,
    minimum_point_spacing: float,
) -> FloatArray:
    points: list[FloatArray] = [start_projection.point.copy()]
    first_vertex_index = start_projection.segment_index + 1
    last_vertex_index = goal_projection.segment_index
    if first_vertex_index <= last_vertex_index:
        for point in centerline[first_vertex_index : last_vertex_index + 1]:
            _append_if_separated(points, point, minimum_point_spacing)
    _append_if_separated(points, goal_projection.point, minimum_point_spacing)
    result = np.asarray(points, dtype=np.float64)
    if result.shape[0] < 2:
        raise NoRouteError(
            "Projected route contains fewer than two distinct points",
            reason="route_too_short",
        )
    return result


def _append_if_separated(
    points: list[FloatArray],
    point: FloatArray,
    minimum_point_spacing: float,
) -> None:
    value = np.asarray(point, dtype=np.float64)
    if np.linalg.norm(value - points[-1]) >= minimum_point_spacing:
        points.append(value.copy())


def _build_reference_path(
    points: FloatArray,
    *,
    minimum_point_spacing: float,
) -> FloatArray:
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths < minimum_point_spacing):
        raise NoRouteError(
            "Route contains duplicate or near-duplicate consecutive points",
            reason="degenerate_route_geometry",
        )

    arc_length = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
    )
    edge_order = 2 if points.shape[0] >= 3 else 1
    dx_ds = np.gradient(points[:, 0], arc_length, edge_order=edge_order)
    dy_ds = np.gradient(points[:, 1], arc_length, edge_order=edge_order)
    raw_yaw = np.unwrap(np.arctan2(dy_ds, dx_ds))

    if points.shape[0] >= 3:
        ddx_ds = np.gradient(dx_ds, arc_length, edge_order=edge_order)
        ddy_ds = np.gradient(dy_ds, arc_length, edge_order=edge_order)
        denominator = np.power(dx_ds * dx_ds + dy_ds * dy_ds, 1.5)
        curvature = np.divide(
            dx_ds * ddy_ds - dy_ds * ddx_ds,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
    else:
        curvature = np.zeros(points.shape[0], dtype=np.float64)

    yaw = (raw_yaw + math.pi) % (2.0 * math.pi) - math.pi
    path = np.column_stack([points[:, 0], points[:, 1], yaw, arc_length, curvature])
    if not np.all(np.isfinite(path)):
        raise NoRouteError(
            "Reference path calculation produced non-finite values",
            reason="invalid_reference_path",
        )
    return path.astype(np.float64, copy=False)
