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


@dataclass(frozen=True)
class _BoundarySegments:
    starts: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    progress_starts: np.ndarray


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

        left_local = _world_polyline_to_vehicle(
            np.asarray(lane.left_boundary, dtype=np.float64),
            ego,
        )
        right_local = _world_polyline_to_vehicle(
            np.asarray(lane.right_boundary, dtype=np.float64),
            ego,
        )

        # 先在车辆坐标系中裁剪到感知窗口附近的局部线段。后续扫描只处理
        # 这些候选线段，避免每条扫描线和射线遍历整幅地图中的长边界。
        left_segments = _prepare_visible_segments(left_local, config)
        right_segments = _prepare_visible_segments(right_local, config)
        if left_segments is None or right_segments is None:
            continue

        left_hits = _scan_boundary(left_segments, config)
        right_hits = _scan_boundary(right_segments, config)

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


def _prepare_visible_segments(
    polyline: np.ndarray,
    config: PerceptionConfig,
) -> _BoundarySegments | None:
    """保留包围盒可能与局部感知窗口相交的折线段。"""
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
        (
            np.array([0.0], dtype=np.float64),
            np.cumsum(lengths),
        )
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
    max_distance = math.hypot(config.forward_range, config.lateral_range)
    hits.extend(_nearest_ray_hits(segments, directions, max_distance))

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
    """批量计算所有横向扫描线与候选线段的交点。"""
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

    values = np.clip(
        parameters[scan_indices, segment_indices],
        0.0,
        1.0,
    )
    points = (
        segments.starts[segment_indices]
        + values[:, None] * segments.vectors[segment_indices]
    )
    progress = (
        segments.progress_starts[segment_indices]
        + values * segments.lengths[segment_indices]
    )

    return [
        _BoundaryHit(
            progress=float(item_progress),
            point=np.asarray(point, dtype=np.float64),
        )
        for item_progress, point in zip(progress, points)
    ]


def _nearest_ray_hits(
    segments: _BoundarySegments,
    directions: np.ndarray,
    max_distance: float,
) -> list[_BoundaryHit]:
    """使用二维叉积批量计算每条射线的最近线段交点。"""
    direction_x = directions[:, 0, None]
    direction_y = directions[:, 1, None]
    segment_x = segments.vectors[None, :, 0]
    segment_y = segments.vectors[None, :, 1]

    denominator = direction_x * segment_y - direction_y * segment_x

    start_x = segments.starts[None, :, 0]
    start_y = segments.starts[None, :, 1]
    distance_numerator = start_x * segment_y - start_y * segment_x
    parameter_numerator = start_x * direction_y - start_y * direction_x

    with np.errstate(divide="ignore", invalid="ignore"):
        distances = distance_numerator / denominator
        parameters = parameter_numerator / denominator

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
        np.arange(directions.shape[0]),
        nearest_segment,
    ]
    ray_indices = np.flatnonzero(np.isfinite(nearest_distance))
    if ray_indices.size == 0:
        return []

    segment_indices = nearest_segment[ray_indices]
    distance_values = np.maximum(nearest_distance[ray_indices], 0.0)
    parameter_values = np.clip(
        parameters[ray_indices, segment_indices],
        0.0,
        1.0,
    )
    points = distance_values[:, None] * directions[ray_indices]
    progress = (
        segments.progress_starts[segment_indices]
        + parameter_values * segments.lengths[segment_indices]
    )

    return [
        _BoundaryHit(
            progress=float(item_progress),
            point=np.asarray(point, dtype=np.float64),
        )
        for item_progress, point in zip(progress, points)
    ]


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


def _inside_field_of_view(point: np.ndarray, field_of_view: float) -> bool:
    if field_of_view >= 2.0 * math.pi - 1e-9:
        return True
    angle = abs(math.atan2(float(point[1]), float(point[0])))
    return angle <= 0.5 * field_of_view + 1e-9


__all__ = ["perceive_lane_corridors"]
