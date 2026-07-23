from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim2d.perception.types import PerceivedLaneLine, PerceptionConfig
from sim2d.types import VehicleState


@dataclass(frozen=True)
class _BoundaryHit:
    progress: float
    point: np.ndarray


@dataclass(frozen=True)
class _BoundarySegments:
    starts: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    progress_starts: np.ndarray


def perceive_lane_lines(
    road_lanes,
    ego: VehicleState,
    config: PerceptionConfig,
    rng: np.random.Generator,
) -> tuple[PerceivedLaneLine, ...]:
    """发布车辆坐标系中的纯几何车道线点列。

    传感器先扫描所有可见道路边界，再仅依据端点距离与切线连续性把
    大概率属于同一条车道线的片段连接起来。输出不携带地图 lane id、
    左右边界配对、前驱后继或导航语义。
    """
    fragments: list[np.ndarray] = []

    for lane in road_lanes:
        if rng.random() < config.lane_dropout_probability:
            continue

        for world_boundary in (lane.left_boundary, lane.right_boundary):
            local = _world_polyline_to_vehicle(
                np.asarray(world_boundary, dtype=np.float64),
                ego,
            )
            segments = _prepare_visible_segments(local, config)
            if segments is None:
                continue

            hits = _scan_boundary(segments, config)
            if len(hits) < 2:
                continue

            points = _resample_hits(hits, config.lane_output_point_count)
            if config.position_noise_std > 0.0:
                points = points + rng.normal(
                    0.0,
                    config.position_noise_std,
                    points.shape,
                )
            fragments.append(points)

    fragments = _remove_duplicate_lines(
        fragments,
        maximum_mean_distance=config.lane_line_duplicate_distance,
    )
    connected = _connect_line_fragments(
        fragments,
        maximum_endpoint_distance=config.lane_line_join_distance,
        maximum_heading_error=config.lane_line_join_heading,
        output_point_count=config.lane_output_point_count,
    )
    connected = _remove_duplicate_lines(
        connected,
        maximum_mean_distance=config.lane_line_duplicate_distance,
    )

    return tuple(
        PerceivedLaneLine(
            line_id=f"lane_line_{index}",
            points=points,
            confidence=1.0,
        )
        for index, points in enumerate(connected)
    )


def perceive_lane_corridors(
    road_lanes,
    ego: VehicleState,
    config: PerceptionConfig,
    rng: np.random.Generator,
) -> tuple[PerceivedLaneLine, ...]:
    """兼容旧调用名；返回值已经改为感知车道线，而不是走廊。"""
    return perceive_lane_lines(road_lanes, ego, config, rng)


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


def _prepare_visible_segments(
    polyline: np.ndarray,
    config: PerceptionConfig,
) -> _BoundarySegments | None:
    if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] < 2:
        return None

    starts = polyline[:-1]
    ends = polyline[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    segment_minimum = np.minimum(starts, ends)
    segment_maximum = np.maximum(starts, ends)

    visible = (
        (lengths > 1e-12)
        & (segment_maximum[:, 0] >= -config.rear_range)
        & (segment_minimum[:, 0] <= config.forward_range)
        & (segment_maximum[:, 1] >= -config.lateral_range)
        & (segment_minimum[:, 1] <= config.lateral_range)
    )
    indices = np.flatnonzero(visible)
    if indices.size == 0:
        return None

    cumulative = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(lengths))
    )
    return _BoundarySegments(
        starts=np.asarray(starts[indices], dtype=np.float64),
        vectors=np.asarray(vectors[indices], dtype=np.float64),
        lengths=np.asarray(lengths[indices], dtype=np.float64),
        progress_starts=np.asarray(cumulative[indices], dtype=np.float64),
    )


def _scan_boundary(
    segments: _BoundarySegments,
    config: PerceptionConfig,
) -> list[_BoundaryHit]:
    hits: list[_BoundaryHit] = []

    scan_x_values = np.arange(
        -config.rear_range,
        config.forward_range + 0.5 * config.lane_transverse_spacing,
        config.lane_transverse_spacing,
        dtype=np.float64,
    )
    hits.extend(_vertical_scan_hits(segments, scan_x_values))

    angles = np.linspace(
        -config.lane_ray_half_angle,
        config.lane_ray_half_angle,
        config.lane_ray_count,
        dtype=np.float64,
    )
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    hits.extend(
        _nearest_ray_hits(
            segments,
            directions,
            math.hypot(config.forward_range, config.lateral_range),
        )
    )

    filtered = [
        hit
        for hit in hits
        if -config.rear_range - 1e-9 <= hit.point[0] <= config.forward_range + 1e-9
        and abs(float(hit.point[1])) <= config.lateral_range + 1e-9
        and _inside_field_of_view(hit.point, config.field_of_view)
    ]
    filtered.sort(key=lambda item: item.progress)
    return _deduplicate_hits(filtered)


def _vertical_scan_hits(
    segments: _BoundarySegments,
    scan_x_values: np.ndarray,
) -> list[_BoundaryHit]:
    dx = segments.vectors[:, 0]
    non_vertical = np.abs(dx) > 1e-12
    if not np.any(non_vertical):
        return []

    parameters = np.full(
        (scan_x_values.size, segments.starts.shape[0]),
        np.nan,
        dtype=np.float64,
    )
    parameters[:, non_vertical] = (
        scan_x_values[:, None] - segments.starts[None, non_vertical, 0]
    ) / dx[None, non_vertical]
    valid = (
        np.isfinite(parameters)
        & (parameters >= -1e-12)
        & (parameters <= 1.0 + 1e-12)
    )
    scan_indices, segment_indices = np.nonzero(valid)
    if scan_indices.size == 0:
        return []

    values = np.clip(parameters[scan_indices, segment_indices], 0.0, 1.0)
    points = (
        segments.starts[segment_indices]
        + values[:, None] * segments.vectors[segment_indices]
    )
    progress = (
        segments.progress_starts[segment_indices]
        + values * segments.lengths[segment_indices]
    )
    return [
        _BoundaryHit(float(item_progress), np.asarray(point, dtype=np.float64))
        for item_progress, point in zip(progress, points)
    ]


def _nearest_ray_hits(
    segments: _BoundarySegments,
    directions: np.ndarray,
    max_distance: float,
) -> list[_BoundaryHit]:
    direction_x = directions[:, 0, None]
    direction_y = directions[:, 1, None]
    segment_x = segments.vectors[None, :, 0]
    segment_y = segments.vectors[None, :, 1]
    denominator = direction_x * segment_y - direction_y * segment_x

    start_x = segments.starts[None, :, 0]
    start_y = segments.starts[None, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = (start_x * segment_y - start_y * segment_x) / denominator
        parameters = (start_x * direction_y - start_y * direction_x) / denominator

    valid = (
        (np.abs(denominator) > 1e-12)
        & np.isfinite(distances)
        & np.isfinite(parameters)
        & (distances >= -1e-9)
        & (distances <= max_distance + 1e-9)
        & (parameters >= -1e-9)
        & (parameters <= 1.0 + 1e-9)
    )
    candidate_distances = np.where(valid, distances, np.inf)
    nearest_segment = np.argmin(candidate_distances, axis=1)
    nearest_distance = candidate_distances[
        np.arange(directions.shape[0]), nearest_segment
    ]
    ray_indices = np.flatnonzero(np.isfinite(nearest_distance))
    if ray_indices.size == 0:
        return []

    segment_indices = nearest_segment[ray_indices]
    distance_values = np.maximum(nearest_distance[ray_indices], 0.0)
    parameter_values = np.clip(
        parameters[ray_indices, segment_indices], 0.0, 1.0
    )
    points = distance_values[:, None] * directions[ray_indices]
    progress = (
        segments.progress_starts[segment_indices]
        + parameter_values * segments.lengths[segment_indices]
    )
    return [
        _BoundaryHit(float(item_progress), np.asarray(point, dtype=np.float64))
        for item_progress, point in zip(progress, points)
    ]


def _connect_line_fragments(
    fragments: list[np.ndarray],
    *,
    maximum_endpoint_distance: float,
    maximum_heading_error: float,
    output_point_count: int,
) -> list[np.ndarray]:
    lines = [np.asarray(item, dtype=np.float64).copy() for item in fragments]
    changed = True
    while changed and len(lines) > 1:
        changed = False
        best: tuple[float, int, int, np.ndarray] | None = None
        for first_index in range(len(lines)):
            for second_index in range(first_index + 1, len(lines)):
                merged = _best_geometric_merge(
                    lines[first_index],
                    lines[second_index],
                    maximum_endpoint_distance,
                    maximum_heading_error,
                )
                if merged is None:
                    continue
                score, points = merged
                if best is None or score < best[0]:
                    best = (score, first_index, second_index, points)
        if best is not None:
            _, first_index, second_index, points = best
            lines[first_index] = _resample_polyline(points, output_point_count)
            del lines[second_index]
            changed = True
    return lines


def _best_geometric_merge(
    first: np.ndarray,
    second: np.ndarray,
    maximum_endpoint_distance: float,
    maximum_heading_error: float,
) -> tuple[float, np.ndarray] | None:
    options = (
        (first, second),
        (first, second[::-1]),
        (first[::-1], second),
        (first[::-1], second[::-1]),
    )
    best: tuple[float, np.ndarray] | None = None
    for left, right in options:
        distance = float(np.linalg.norm(left[-1] - right[0]))
        if distance > maximum_endpoint_distance:
            continue
        left_tangent = _unit_tangent(left[-2], left[-1])
        right_tangent = _unit_tangent(right[0], right[1])
        if left_tangent is None or right_tangent is None:
            continue
        heading_error = math.acos(
            float(np.clip(np.dot(left_tangent, right_tangent), -1.0, 1.0))
        )
        if heading_error > maximum_heading_error:
            continue
        score = distance + maximum_endpoint_distance * heading_error
        connector = 0.5 * (left[-1] + right[0])
        merged = np.vstack((left[:-1], connector, right[1:]))
        if best is None or score < best[0]:
            best = (score, merged)
    return best


def _remove_duplicate_lines(
    lines: list[np.ndarray],
    *,
    maximum_mean_distance: float,
) -> list[np.ndarray]:
    if maximum_mean_distance <= 0.0:
        return lines
    unique: list[np.ndarray] = []
    for line in lines:
        sample = _resample_polyline(line, 25)
        duplicate = False
        for existing in unique:
            other = _resample_polyline(existing, 25)
            distance = min(
                float(np.mean(np.linalg.norm(sample - other, axis=1))),
                float(np.mean(np.linalg.norm(sample - other[::-1], axis=1))),
            )
            if distance <= maximum_mean_distance:
                duplicate = True
                break
        if not duplicate:
            unique.append(line)
    return unique


def _unit_tangent(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    value = end - start
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return None
    return value / norm


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


def _resample_polyline(points: np.ndarray, point_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate((np.array([True]), segment_lengths > 1e-9))
    points = points[keep]
    if points.shape[0] < 2:
        return points
    arc = np.concatenate((np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    samples = np.linspace(0.0, float(arc[-1]), point_count)
    return np.column_stack(
        (
            np.interp(samples, arc, points[:, 0]),
            np.interp(samples, arc, points[:, 1]),
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


def _inside_field_of_view(point: np.ndarray, field_of_view: float) -> bool:
    if field_of_view >= 2.0 * math.pi - 1e-9:
        return True
    angle = abs(math.atan2(float(point[1]), float(point[0])))
    return angle <= 0.5 * field_of_view + 1e-9


__all__ = ["perceive_lane_corridors", "perceive_lane_lines"]
