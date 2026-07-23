from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim2d.perception import PerceivedLaneLine
from sim2d.types import VehicleState


@dataclass(frozen=True)
class PNCReferenceLine:
    """PNC Map 根据感知车道线构造的候选 reference line。"""

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
        if reference_path.shape[0] != left_boundary.shape[0]:
            raise ValueError("reference_path and boundaries must have equal length")
        if reference_path.shape[0] != right_boundary.shape[0]:
            raise ValueError("reference_path and boundaries must have equal length")
        object.__setattr__(self, "reference_path", reference_path.copy())
        object.__setattr__(self, "left_boundary", left_boundary.copy())
        object.__setattr__(self, "right_boundary", right_boundary.copy())


@dataclass(frozen=True)
class PNCMapUpdate:
    selected: PNCReferenceLine | None
    references: tuple[PNCReferenceLine, ...]
    reference_changed: bool
    history_used: bool
    switch_pending_frames: int
    continuity_cost: float | None


class PNCMap:
    """有历史状态的局部 PNC Map。

    单帧几何负责生成候选 reference；跨帧历史负责保持逻辑本车道，避免车辆
    因控制误差靠近相邻车道时立即跳 reference。只有新候选连续出现若干帧，
    或旧 reference 已无法匹配，才确认切换。
    """

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.5,
        maximum_lateral_distance: float = 6.0,
        switch_confirm_frames: int = 5,
        history_retention_cost: float = 1.5,
        history_hard_limit: float = 4.0,
        pending_match_cost: float = 1.0,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within [0, 1]")
        if maximum_lateral_distance <= 0.0:
            raise ValueError("maximum_lateral_distance must be positive")
        if switch_confirm_frames < 1:
            raise ValueError("switch_confirm_frames must be positive")
        if min(history_retention_cost, history_hard_limit, pending_match_cost) <= 0.0:
            raise ValueError("history thresholds must be positive")
        if history_hard_limit < history_retention_cost:
            raise ValueError("history_hard_limit must not be smaller than retention")

        self.minimum_confidence = minimum_confidence
        self.maximum_lateral_distance = maximum_lateral_distance
        self.switch_confirm_frames = switch_confirm_frames
        self.history_retention_cost = history_retention_cost
        self.history_hard_limit = history_hard_limit
        self.pending_match_cost = pending_match_cost

        self._current_world_path: np.ndarray | None = None
        self._pending_world_path: np.ndarray | None = None
        self._pending_frames = 0

    def reset(self) -> None:
        self._current_world_path = None
        self._pending_world_path = None
        self._pending_frames = 0

    def update(
        self,
        lane_lines: tuple[PerceivedLaneLine, ...],
        *,
        world_origin: VehicleState,
    ) -> PNCMapUpdate:
        references = build_reference_lines(
            lane_lines,
            minimum_confidence=self.minimum_confidence,
        )
        geometric = select_current_reference_line(
            references,
            maximum_lateral_distance=self.maximum_lateral_distance,
        )

        if not references:
            self._pending_world_path = None
            self._pending_frames = 0
            return PNCMapUpdate(
                selected=None,
                references=(),
                reference_changed=False,
                history_used=False,
                switch_pending_frames=0,
                continuity_cost=None,
            )

        world_paths = [
            local_reference_path_to_world(reference.reference_path, world_origin)
            for reference in references
        ]

        if self._current_world_path is None:
            selected = geometric or references[0]
            selected_world = local_reference_path_to_world(
                selected.reference_path,
                world_origin,
            )
            self._current_world_path = selected_world
            self._pending_world_path = None
            self._pending_frames = 0
            return PNCMapUpdate(
                selected=selected,
                references=references,
                reference_changed=False,
                history_used=False,
                switch_pending_frames=0,
                continuity_cost=None,
            )

        costs = np.asarray(
            [
                _reference_continuity_cost(path, self._current_world_path)
                for path in world_paths
            ],
            dtype=np.float64,
        )
        history_index = int(np.argmin(costs))
        history_cost = float(costs[history_index])
        history_reference = references[history_index]
        history_world = world_paths[history_index]

        # 正常情况下持续追踪与上一周期几何最连续的 reference。即便自车瞬时
        # 更靠近旁车道，也不会改变逻辑本车道。
        if history_cost <= self.history_retention_cost:
            changed = not _same_reference_geometry(
                self._current_world_path,
                history_world,
                threshold=0.8,
            )
            self._current_world_path = history_world
            self._pending_world_path = None
            self._pending_frames = 0
            return PNCMapUpdate(
                selected=history_reference,
                references=references,
                reference_changed=False if not changed else False,
                history_used=True,
                switch_pending_frames=0,
                continuity_cost=history_cost,
            )

        target = geometric or history_reference
        target_world = local_reference_path_to_world(
            target.reference_path,
            world_origin,
        )

        if self._pending_world_path is not None and _reference_continuity_cost(
            target_world,
            self._pending_world_path,
        ) <= self.pending_match_cost:
            self._pending_frames += 1
            self._pending_world_path = target_world
        else:
            self._pending_world_path = target_world
            self._pending_frames = 1

        # 在确认窗口内，仍优先保持最接近历史车道的当前帧候选。这样控制偏差
        # 不会立刻演变成 reference 跳到旁车道的正反馈。
        if (
            self._pending_frames < self.switch_confirm_frames
            and history_cost <= self.history_hard_limit
        ):
            return PNCMapUpdate(
                selected=history_reference,
                references=references,
                reference_changed=False,
                history_used=True,
                switch_pending_frames=self._pending_frames,
                continuity_cost=history_cost,
            )

        self._current_world_path = target_world
        self._pending_world_path = None
        self._pending_frames = 0
        return PNCMapUpdate(
            selected=target,
            references=references,
            reference_changed=True,
            history_used=False,
            switch_pending_frames=0,
            continuity_cost=history_cost,
        )


def build_reference_lines(
    lane_lines: tuple[PerceivedLaneLine, ...],
    *,
    minimum_confidence: float = 0.5,
    minimum_lane_width: float = 2.0,
    maximum_lane_width: float = 6.0,
    output_point_count: int = 101,
) -> tuple[PNCReferenceLine, ...]:
    """从纯感知车道线构造所有合理的局部 reference line。"""
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
        tangent = _normalized_tangent(points, nearest_index)
        if tangent is None:
            continue
        lateral = float(points[nearest_index, 1])
        heading_error = abs(math.atan2(float(tangent[1]), float(tangent[0])))
        prepared.append((line, points, lateral, heading_error))

    prepared.sort(key=lambda item: item[2], reverse=True)
    references: list[PNCReferenceLine] = []

    for index in range(len(prepared) - 1):
        left_line, left_points, _, left_heading = prepared[index]
        right_line, right_points, _, right_heading = prepared[index + 1]
        if max(left_heading, right_heading) > math.radians(120.0):
            continue

        pairing = _pair_boundaries_by_normal_projection(
            left_points,
            right_points,
            output_point_count=output_point_count,
            minimum_lane_width=minimum_lane_width,
            maximum_lane_width=maximum_lane_width,
        )
        if pairing is None:
            continue
        left_sample, right_sample, valid_ratio = pairing

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
                reference_id=f"ref_{left_line.line_id}__{right_line.line_id}",
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
    """无历史时，根据当前几何选择包围自车的 reference。"""
    candidates: list[tuple[int, float, PNCReferenceLine, int]] = []
    for reference in references:
        centerline = reference.reference_path[:, :2]
        distances = np.linalg.norm(centerline, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance > maximum_lateral_distance:
            continue

        left = reference.left_boundary[nearest_index]
        right = reference.right_boundary[nearest_index]
        tangent = _normalized_tangent(centerline, nearest_index)
        if tangent is None:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        signed_left = float(np.dot(left, normal))
        signed_right = float(np.dot(right, normal))
        contains_ego = (
            min(signed_left, signed_right) - 1e-6
            <= 0.0
            <= max(signed_left, signed_right) + 1e-6
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
    return _slice_reference(selected, nearest_index, backward_margin)


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


def _pair_boundaries_by_normal_projection(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    output_point_count: int,
    minimum_lane_width: float,
    maximum_lane_width: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """沿道路局部法向建立左右边界对应，而不是使用固定 y 差。"""
    left_sample = _resample_polyline(left_points, output_point_count)
    if left_sample.shape[0] < 2:
        return None

    right_sample = np.empty_like(left_sample)
    valid = np.zeros(left_sample.shape[0], dtype=bool)

    for index, left_point in enumerate(left_sample):
        right_point = _closest_point_on_polyline(left_point, right_points)
        tangent = _normalized_tangent(left_sample, index)
        if tangent is None:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        delta = right_point - left_point
        normal_width = abs(float(np.dot(delta, normal)))
        tangent_offset = abs(float(np.dot(delta, tangent)))
        right_sample[index] = right_point
        valid[index] = (
            minimum_lane_width <= normal_width <= maximum_lane_width
            and tangent_offset <= 0.5 * maximum_lane_width
        )

    valid_ratio = float(np.mean(valid))
    if valid_ratio < 0.65:
        return None

    # 少量无效点仍使用最近投影点保持数组连续；候选置信度由 valid_ratio 降低。
    return left_sample, right_sample, valid_ratio


def _closest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared = np.sum(vectors * vectors, axis=1)
    parameters = np.zeros(vectors.shape[0], dtype=np.float64)
    usable = squared > 1e-12
    parameters[usable] = np.sum(
        (point - starts[usable]) * vectors[usable], axis=1
    ) / squared[usable]
    parameters = np.clip(parameters, 0.0, 1.0)
    projected = starts + parameters[:, None] * vectors
    index = int(np.argmin(np.linalg.norm(projected - point, axis=1)))
    return projected[index]


def _orient_line_forward(points: np.ndarray) -> np.ndarray:
    """统一点序，但不按 x 排序；急弯时使用可见弧长和端点延伸判向。"""
    value = np.asarray(points, dtype=np.float64)
    nearest_index = int(np.argmin(np.linalg.norm(value, axis=1)))
    tangent = _normalized_tangent(value, nearest_index)
    if tangent is None:
        return value.copy()

    # 道路在自车附近明显朝前/朝后时，局部切线最可靠。
    if abs(float(tangent[0])) >= 0.15:
        return value[::-1].copy() if tangent[0] < 0.0 else value.copy()

    # 直角弯附近 tangent.x 接近零，不能由微小数值符号决定整条线方向。
    # 此时把离自车最近点两侧视作两个候选延伸，优先选择可见弧长更长、
    # 且端点离自车更远的一侧作为未来方向。通常前向感知范围大于后向范围。
    segment_lengths = np.linalg.norm(np.diff(value, axis=0), axis=1)
    before_length = float(np.sum(segment_lengths[:nearest_index]))
    after_length = float(np.sum(segment_lengths[nearest_index:]))
    before_endpoint = float(np.linalg.norm(value[0]))
    after_endpoint = float(np.linalg.norm(value[-1]))
    before_score = before_length + 0.25 * before_endpoint
    after_score = after_length + 0.25 * after_endpoint
    return value[::-1].copy() if before_score > after_score else value.copy()


def _reference_continuity_cost(
    first_world_path: np.ndarray,
    second_world_path: np.ndarray,
) -> float:
    first = _resample_polyline(first_world_path[:, :2], 41)
    second = _resample_polyline(second_world_path[:, :2], 41)
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    geometric = 0.5 * (
        float(np.mean(np.min(distances, axis=1)))
        + float(np.mean(np.min(distances, axis=0)))
    )

    first_yaw = float(first_world_path[min(3, first_world_path.shape[0] - 1), 2])
    second_yaw = float(second_world_path[min(3, second_world_path.shape[0] - 1), 2])
    heading = abs(float(normalize_angle(first_yaw - second_yaw)))
    return geometric + 0.75 * heading


def _same_reference_geometry(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold: float,
) -> bool:
    return _reference_continuity_cost(first, second) <= threshold


def _slice_reference(
    selected: PNCReferenceLine,
    nearest_index: int,
    backward_margin: float,
) -> PNCReferenceLine | None:
    start_index = nearest_index
    centerline = selected.reference_path[:, :2]
    travelled = 0.0
    while start_index > 0:
        segment = float(np.linalg.norm(centerline[start_index] - centerline[start_index - 1]))
        if travelled + segment > backward_margin:
            break
        travelled += segment
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


def _resample_polyline(points: np.ndarray, point_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
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


def _normalized_tangent(points: np.ndarray, index: int) -> np.ndarray | None:
    if points.shape[0] < 2:
        return None
    if index <= 0:
        tangent = points[1] - points[0]
    elif index >= points.shape[0] - 1:
        tangent = points[-1] - points[-2]
    else:
        tangent = points[index + 1] - points[index - 1]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return None
    return tangent / norm


def normalize_angle(angle: np.ndarray | float):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "PNCMap",
    "PNCMapUpdate",
    "PNCReferenceLine",
    "build_current_reference_line",
    "build_reference_lines",
    "local_reference_path_to_world",
    "polyline_to_reference_path",
    "select_current_reference_line",
]
