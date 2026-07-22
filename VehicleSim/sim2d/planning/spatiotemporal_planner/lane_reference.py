from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim2d.perception import PerceivedLaneSegment
from sim2d.types import VehicleState


@dataclass(frozen=True)
class PerceptionLaneReference:
    """由纯几何感知走廊生成的冻结自车坐标参考线。"""

    lane_id: str
    confidence: float
    reference_path: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray

    def __post_init__(self) -> None:
        reference_path = np.asarray(self.reference_path, dtype=np.float64)
        left_boundary = np.asarray(self.left_boundary, dtype=np.float64)
        right_boundary = np.asarray(self.right_boundary, dtype=np.float64)

        if reference_path.ndim != 2 or reference_path.shape[1] != 5:
            raise ValueError(
                "reference_path must have shape [N, 5], "
                f"got {reference_path.shape}"
            )
        if reference_path.shape[0] < 2:
            raise ValueError("reference_path must contain at least two points")
        for name, value in (
            ("left_boundary", left_boundary),
            ("right_boundary", right_boundary),
        ):
            if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 2:
                raise ValueError(f"{name} must have shape [N, 2], N >= 2")

        object.__setattr__(self, "reference_path", reference_path.copy())
        object.__setattr__(self, "left_boundary", left_boundary.copy())
        object.__setattr__(self, "right_boundary", right_boundary.copy())


def build_perception_lane_reference(
    road_segments: tuple[PerceivedLaneSegment, ...],
    *,
    minimum_confidence: float = 0.5,
    maximum_lateral_distance: float = 6.0,
    backward_margin: float = 2.0,
) -> PerceptionLaneReference | None:
    """从扫描恢复的几何走廊中找到当前自车实际所在走廊。

    不读取地图 lane id、successor 或导航路线。首选左右边界包围车辆原点的
    走廊；只有扫描不完整时才回退到最近且方向一致的几何走廊。
    """
    if not road_segments:
        return None
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within [0, 1]")
    if maximum_lateral_distance <= 0.0:
        raise ValueError("maximum_lateral_distance must be positive")
    if backward_margin < 0.0:
        raise ValueError("backward_margin must be non-negative")

    candidates: list[
        tuple[
            int,
            float,
            PerceivedLaneSegment,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for segment in road_segments:
        if segment.confidence < minimum_confidence:
            continue

        centerline = np.asarray(segment.centerline, dtype=np.float64)
        left_boundary = np.asarray(segment.left_boundary, dtype=np.float64)
        right_boundary = np.asarray(segment.right_boundary, dtype=np.float64)

        if centerline.shape[0] < 2:
            continue

        distances = np.linalg.norm(centerline, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance > maximum_lateral_distance:
            continue

        tangent = _local_tangent(centerline, nearest_index)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            continue
        tangent = tangent / tangent_norm

        # 扫描恢复曲线可能按道路原始存储方向排列；统一成沿自车前方增长。
        if float(tangent[0]) < 0.0:
            centerline = centerline[::-1].copy()
            left_boundary, right_boundary = (
                right_boundary[::-1].copy(),
                left_boundary[::-1].copy(),
            )
            distances = np.linalg.norm(centerline, axis=1)
            nearest_index = int(np.argmin(distances))
            tangent = _local_tangent(centerline, nearest_index)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-12:
                continue
            tangent = tangent / tangent_norm

        left_y = float(left_boundary[nearest_index, 1])
        right_y = float(right_boundary[nearest_index, 1])
        lower_y = min(left_y, right_y)
        upper_y = max(left_y, right_y)
        contains_ego = lower_y - 1e-6 <= 0.0 <= upper_y + 1e-6

        heading_error = abs(math.atan2(float(tangent[1]), float(tangent[0])))
        score = (
            nearest_distance
            + 2.0 * heading_error
            + 1.5 * (1.0 - float(segment.confidence))
        )
        candidates.append(
            (
                0 if contains_ego else 1,
                score,
                segment,
                centerline,
                left_boundary,
                right_boundary,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, segment, centerline, left_boundary, right_boundary = candidates[0]
    nearest_index = int(np.argmin(np.linalg.norm(centerline, axis=1)))

    start_index = nearest_index
    while start_index > 0 and centerline[start_index - 1, 0] >= -backward_margin:
        start_index -= 1

    centerline = centerline[start_index:].copy()
    left_boundary = left_boundary[start_index:].copy()
    right_boundary = right_boundary[start_index:].copy()
    if centerline.shape[0] < 2:
        return None

    reference_path = _polyline_to_reference_path(centerline)

    return PerceptionLaneReference(
        lane_id=segment.map_lane_id,
        confidence=float(segment.confidence),
        reference_path=reference_path,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
    )


def local_reference_path_to_world(
    reference_path: np.ndarray,
    origin: VehicleState,
) -> np.ndarray:
    """将局部 [x, y, yaw, s, curvature] 参考线恢复到世界坐标。"""
    path = np.asarray(reference_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 5:
        raise ValueError(
            "reference_path must have shape [N, 5], "
            f"got {path.shape}"
        )

    result = path.copy()
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    local_x = path[:, 0]
    local_y = path[:, 1]

    result[:, 0] = origin.x + cosine * local_x - sine * local_y
    result[:, 1] = origin.y + sine * local_x + cosine * local_y
    result[:, 2] = _normalize_angle(path[:, 2] + origin.yaw)
    return result


def _polyline_to_reference_path(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("points must have shape [N, 2], N >= 2")

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid = np.concatenate((np.array([True]), segment_lengths > 1e-9))
    points = points[valid]
    if points.shape[0] < 2:
        raise ValueError("lane centerline contains fewer than two unique points")

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    yaws = np.unwrap(np.arctan2(dy, dx))

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths))
    )

    if points.shape[0] >= 3:
        curvature = np.asarray(
            np.gradient(yaws, arc_length, edge_order=1),
            dtype=np.float64,
        )
    else:
        curvature = np.zeros(points.shape[0], dtype=np.float64)

    return np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            _normalize_angle(yaws),
            arc_length,
            curvature,
        )
    )


def _local_tangent(points: np.ndarray, index: int) -> np.ndarray:
    if index <= 0:
        return points[1] - points[0]
    if index >= points.shape[0] - 1:
        return points[-1] - points[-2]
    return points[index + 1] - points[index - 1]


def _normalize_angle(angle: np.ndarray | float):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "PerceptionLaneReference",
    "build_perception_lane_reference",
    "local_reference_path_to_world",
]
