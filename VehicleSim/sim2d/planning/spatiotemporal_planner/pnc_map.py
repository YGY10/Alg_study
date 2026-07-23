from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim2d.perception import PerceivedLaneLine
from sim2d.types import VehicleState


@dataclass(frozen=True)
class PNCReferenceLine:
    """PNC map 小模块根据感知车道线构造的候选 reference line。"""

    reference_id: str
    confidence: float
    reference_path: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray

    def __post_init__(self) -> None:
        reference_path = np.asarray(self.reference_path, dtype=np.float64)
        left_boundary = np.asarray(self.left_boundary, dtype=np.float64)
        right_boundary = np.asarray(self.right_boundary, dtype=np.float64)
        if reference_path.ndim != 2 or reference_path.shape[1] != 5:
            raise ValueError("reference_path must have shape [N, 5]")
        for name, value in (
            ("left_boundary", left_boundary),
            ("right_boundary", right_boundary),
        ):
            if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 2:
                raise ValueError(f"{name} must have shape [N, 2], N >= 2")
        object.__setattr__(self, "reference_path", reference_path.copy())
        object.__setattr__(self, "left_boundary", left_boundary.copy())
        object.__setattr__(self, "right_boundary", right_boundary.copy())


def build_reference_lines(
    lane_lines: tuple[PerceivedLaneLine, ...],
    *,
    minimum_confidence: float = 0.5,
    minimum_lane_width: float = 2.0,
    maximum_lane_width: float = 6.0,
    output_point_count: int = 101,
) -> tuple[PNCReferenceLine, ...]:
    """根据一组纯感知车道线构造所有合理的局部 reference line。

    本模块属于 PNC 的局部 map/geometry 层。它负责车道线排序、相邻线配对
    和中心线生成；感知模块本身不再输出 lane 或 corridor。
    """
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within [0, 1]")
    if minimum_lane_width <= 0.0 or maximum_lane_width < minimum_lane_width:
        raise ValueError("invalid lane width range")
    if output_point_count < 2:
        raise ValueError("output_point_count must be at least 2")

    prepared: list[tuple[PerceivedLaneLine, np.ndarray, float, float]] = []
    for line in lane_lines:
        if line.confidence < minimum_confidence:
            continue
        points = _orient_line_forward(line.points)
        nearest_index = int(np.argmin(np.linalg.norm(points, axis=1)))
        tangent = _local_tangent(points, nearest_index)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            continue
        heading_error = abs(
            math.atan2(float(tangent[1]), float(tangent[0]))
        )
        lateral = float(points[nearest_index, 1])
        prepared.append((line, points, lateral, heading_error))

    # 按自车附近横向位置从左到右排列，只在相邻车道线之间构造走廊。
    prepared.sort(key=lambda item: item[2], reverse=True)
    references: list[PNCReferenceLine] = []

    for index in range(len(prepared) - 1):
        left_line, left_points, left_y, left_heading = prepared[index]
        right_line, right_points, right_y, right_heading = prepared[index + 1]
        local_width = left_y - right_y
        if not minimum_lane_width <= local_width <= maximum_lane_width:
            continue
        if max(left_heading, right_heading) > math.radians(100.0):
            continue

        left_sample = _resample_polyline(left_points, output_point_count)
        right_sample = _resample_polyline(right_points, output_point_count)
        if left_sample.shape != right_sample.shape:
            continue

        widths = np.linalg.norm(left_sample - right_sample, axis=1)
        valid_ratio = float(
            np.mean(
                (widths >= 0.75 * minimum_lane_width)
                & (widths <= 1.25 * maximum_lane_width)
            )
        )
        if valid_ratio < 0.65:
            continue

        centerline = 0.5 * (left_sample + right_sample)
        try:
            reference_path = polyline_to_reference_path(centerline)
        except ValueError:
            continue

        confidence = float(
            min(left_line.confidence, right_line.confidence) * valid_ratio
        )
        references.append(
            PNCReferenceLine(
                reference_id=(
                    f"ref_{left_line.line_id}__{right_line.line_id}"
                ),
                confidence=confidence,
                reference_path=reference_path,
                left_boundary=left_sample,
                right_boundary=right_sample,
            )
        )

    return tuple(references)


def select_current_reference_line(
    references: tuple[PNCReferenceLine, ...],
    *,
    maximum_lateral_distance: float = 6.0,
    backward_margin: float = 2.0,
) -> PNCReferenceLine | None:
    """选择当前包围自车或与自车最连续的 reference line。"""
    candidates: list[tuple[int, float, PNCReferenceLine, int]] = []
    for reference in references:
        centerline = reference.reference_path[:, :2]
        distances = np.linalg.norm(centerline, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance > maximum_lateral_distance:
            continue

        left_y = float(reference.left_boundary[nearest_index, 1])
        right_y = float(reference.right_boundary[nearest_index, 1])
        contains_ego = (
            min(left_y, right_y) - 1e-6
            <= 0.0
            <= max(left_y, right_y) + 1e-6
        )
        heading_error = abs(float(reference.reference_path[nearest_index, 2]))
        score = nearest_distance + 2.0 * heading_error + (1.0 - reference.confidence)
        candidates.append(
            (0 if contains_ego else 1, score, reference, nearest_index)
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, selected, nearest_index = candidates[0]

    start_index = nearest_index
    centerline = selected.reference_path[:, :2]
    while start_index > 0 and centerline[start_index - 1, 0] >= -backward_margin:
        start_index -= 1

    if selected.reference_path.shape[0] - start_index < 2:
        return None

    sliced_path = selected.reference_path[start_index:].copy()
    sliced_path[:, 3] -= sliced_path[0, 3]
    return PNCReferenceLine(
        reference_id=selected.reference_id,
        confidence=selected.confidence,
        reference_path=sliced_path,
        left_boundary=selected.left_boundary[start_index:],
        right_boundary=selected.right_boundary[start_index:],
    )


def build_current_reference_line(
    lane_lines: tuple[PerceivedLaneLine, ...],
    *,
    minimum_confidence: float = 0.5,
    maximum_lateral_distance: float = 6.0,
) -> tuple[PNCReferenceLine | None, tuple[PNCReferenceLine, ...]]:
    references = build_reference_lines(
        lane_lines,
        minimum_confidence=minimum_confidence,
    )
    return (
        select_current_reference_line(
            references,
            maximum_lateral_distance=maximum_lateral_distance,
        ),
        references,
    )


def local_reference_path_to_world(
    reference_path: np.ndarray,
    origin: VehicleState,
) -> np.ndarray:
    path = np.asarray(reference_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 5:
        raise ValueError("reference_path must have shape [N, 5]")
    result = path.copy()
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    local_x = path[:, 0]
    local_y = path[:, 1]
    result[:, 0] = origin.x + cosine * local_x - sine * local_y
    result[:, 1] = origin.y + sine * local_x + cosine * local_y
    result[:, 2] = normalize_angle(path[:, 2] + origin.yaw)
    return result


def polyline_to_reference_path(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("points must have shape [N, 2], N >= 2")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate((np.array([True]), segment_lengths > 1e-9))]
    if points.shape[0] < 2:
        raise ValueError("reference line has fewer than two unique points")

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    yaws = np.unwrap(np.arctan2(dy, dx))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths))
    )
    if points.shape[0] >= 3:
        curvature = np.asarray(
            np.gradient(yaws, arc_length, edge_order=1), dtype=np.float64
        )
    else:
        curvature = np.zeros(points.shape[0], dtype=np.float64)
    return np.column_stack(
        (points[:, 0], points[:, 1], normalize_angle(yaws), arc_length, curvature)
    )


def _orient_line_forward(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    nearest_index = int(np.argmin(np.linalg.norm(value, axis=1)))
    tangent = _local_tangent(value, nearest_index)
    if float(tangent[0]) < 0.0:
        return value[::-1].copy()
    return value.copy()


def _resample_polyline(points: np.ndarray, point_count: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate((np.array([True]), segment_lengths > 1e-9))
    points = points[keep]
    if points.shape[0] < 2:
        return points
    arc = np.concatenate(
        (np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    samples = np.linspace(0.0, float(arc[-1]), point_count)
    return np.column_stack(
        (
            np.interp(samples, arc, points[:, 0]),
            np.interp(samples, arc, points[:, 1]),
        )
    )


def _local_tangent(points: np.ndarray, index: int) -> np.ndarray:
    if index <= 0:
        return points[1] - points[0]
    if index >= points.shape[0] - 1:
        return points[-1] - points[-2]
    return points[index + 1] - points[index - 1]


def normalize_angle(angle: np.ndarray | float):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "PNCReferenceLine",
    "build_current_reference_line",
    "build_reference_lines",
    "local_reference_path_to_world",
    "polyline_to_reference_path",
    "select_current_reference_line",
]
