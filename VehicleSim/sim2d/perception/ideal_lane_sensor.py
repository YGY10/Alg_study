from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim2d.perception.types import PerceivedLaneSegment, PerceptionConfig
from sim2d.types import VehicleState


@dataclass(frozen=True)
class _BoundaryHit:
    progress: float
    point: np.ndarray


def perceive_lane_corridors(
    road_lanes,
    ego: VehicleState,
    config: PerceptionConfig,
    rng: np.random.Generator,
) -> tuple[PerceivedLaneSegment, ...]:
    """使用局部横向扫描线和前向扇形射线感知道路几何。

    输出不携带地图 lane id、前驱或后继拓扑。每个结果只表示当前帧
    在车辆坐标系中恢复出的局部车道走廊。
    """
    corridors: list[PerceivedLaneSegment] = []

    for lane in road_lanes:
        if rng.random() < config.lane_dropout_probability:
            continue

        left_world = np.asarray(lane.left_boundary, dtype=np.float64)
        right_world = np.asarray(lane.right_boundary, dtype=np.float64)

        left_local = _world_polyline_to_vehicle(left_world, ego)
        right_local = _world_polyline_to_vehicle(right_world, ego)

        left_hits = _scan_boundary(left_local, config)
        right_hits = _scan_boundary(right_local, config)

        if len(left_hits) < 2 or len(right_hits) < 2:
            continue

        left_boundary = _resample_hits(left_hits, config.lane_output_point_count)
        right_boundary = _resample_hits(right_hits, config.lane_output_point_count)

        if config.position_noise_std > 0.0:
            left_boundary = left_boundary + rng.normal(
                0.0,
                config.position_noise_std,
                left_boundary.shape,
            )
            right_boundary = right_boundary + rng.normal(
                0.0,
                config.position_noise_std,
                right_boundary.shape,
            )

        centerline = 0.5 * (left_boundary + right_boundary)

        corridors.append(
            PerceivedLaneSegment(
                map_lane_id=f"scan_corridor_{len(corridors)}",
                centerline=centerline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                predecessor_ids=(),
                successor_ids=(),
                confidence=1.0,
            )
        )

    return tuple(corridors)


def _world_polyline_to_vehicle(
    points: np.ndarray,
    ego: VehicleState,
) -> np.ndarray:
    dx = points[:, 0] - float(ego.x)
    dy = points[:, 1] - float(ego.y)
    cosine = math.cos(ego.yaw)
    sine = math.sin(ego.yaw)
    return np.column_stack(
        (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
        )
    )


def _scan_boundary(
    polyline: np.ndarray,
    config: PerceptionConfig,
) -> list[_BoundaryHit]:
    hits: list[_BoundaryHit] = []

    scan_x_values = np.arange(
        -config.rear_range,
        config.forward_range + 0.5 * config.lane_transverse_spacing,
        config.lane_transverse_spacing,
        dtype=np.float64,
    )

    for scan_x in scan_x_values:
        hits.extend(_intersections_with_vertical_line(polyline, float(scan_x)))

    angles = np.linspace(
        -config.lane_ray_half_angle,
        config.lane_ray_half_angle,
        config.lane_ray_count,
        dtype=np.float64,
    )
    max_distance = math.hypot(config.forward_range, config.lateral_range)

    for angle in angles:
        direction = np.array(
            [math.cos(float(angle)), math.sin(float(angle))],
            dtype=np.float64,
        )
        hit = _nearest_ray_intersection(polyline, direction, max_distance)
        if hit is not None:
            hits.append(hit)

    filtered = [
        hit
        for hit in hits
        if -config.rear_range - 1e-9 <= hit.point[0] <= config.forward_range + 1e-9
        and abs(float(hit.point[1])) <= config.lateral_range + 1e-9
        and _inside_field_of_view(hit.point, config.field_of_view)
    ]

    filtered.sort(key=lambda item: item.progress)
    return _deduplicate_hits(filtered)


def _intersections_with_vertical_line(
    polyline: np.ndarray,
    scan_x: float,
) -> list[_BoundaryHit]:
    hits: list[_BoundaryHit] = []
    cumulative = _cumulative_lengths(polyline)

    for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
        dx = float(end[0] - start[0])
        if abs(dx) <= 1e-12:
            continue
        parameter = (scan_x - float(start[0])) / dx
        if -1e-12 <= parameter <= 1.0 + 1e-12:
            parameter = float(np.clip(parameter, 0.0, 1.0))
            point = start + parameter * (end - start)
            segment_length = float(np.linalg.norm(end - start))
            hits.append(
                _BoundaryHit(
                    progress=float(cumulative[index] + parameter * segment_length),
                    point=np.asarray(point, dtype=np.float64),
                )
            )
    return hits


def _nearest_ray_intersection(
    polyline: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
) -> _BoundaryHit | None:
    cumulative = _cumulative_lengths(polyline)
    best: tuple[float, _BoundaryHit] | None = None

    for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
        segment = end - start
        matrix = np.column_stack((direction, -segment))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1e-12:
            continue

        distance, parameter = np.linalg.solve(matrix, start)
        distance = float(-distance)
        parameter = float(-parameter)

        if distance < -1e-9 or distance > max_distance + 1e-9:
            continue
        if parameter < -1e-9 or parameter > 1.0 + 1e-9:
            continue

        distance = max(distance, 0.0)
        parameter = float(np.clip(parameter, 0.0, 1.0))
        point = distance * direction
        segment_length = float(np.linalg.norm(segment))
        hit = _BoundaryHit(
            progress=float(cumulative[index] + parameter * segment_length),
            point=np.asarray(point, dtype=np.float64),
        )
        if best is None or distance < best[0]:
            best = (distance, hit)

    return None if best is None else best[1]


def _resample_hits(hits: list[_BoundaryHit], point_count: int) -> np.ndarray:
    progress = np.asarray([hit.progress for hit in hits], dtype=np.float64)
    points = np.asarray([hit.point for hit in hits], dtype=np.float64)

    if progress[-1] - progress[0] <= 1e-9:
        raise ValueError("lane scan contains insufficient longitudinal extent")

    sample_progress = np.linspace(progress[0], progress[-1], point_count)
    return np.column_stack(
        (
            np.interp(sample_progress, progress, points[:, 0]),
            np.interp(sample_progress, progress, points[:, 1]),
        )
    )


def _deduplicate_hits(hits: list[_BoundaryHit]) -> list[_BoundaryHit]:
    result: list[_BoundaryHit] = []
    for hit in hits:
        if result and (
            abs(hit.progress - result[-1].progress) <= 1e-6
            or np.linalg.norm(hit.point - result[-1].point) <= 1e-5
        ):
            continue
        result.append(hit)
    return result


def _cumulative_lengths(polyline: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.array([0.0], dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)),
        )
    )


def _inside_field_of_view(point: np.ndarray, field_of_view: float) -> bool:
    if field_of_view >= 2.0 * math.pi - 1e-9:
        return True
    angle = abs(math.atan2(float(point[1]), float(point[0])))
    return angle <= 0.5 * field_of_view + 1e-9


__all__ = ["perceive_lane_corridors"]
