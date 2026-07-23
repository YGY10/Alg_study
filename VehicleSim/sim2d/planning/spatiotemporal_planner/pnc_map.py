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
    candidate_costs: tuple[tuple[str, float], ...] = ()
    selected_orientation: str | None = None


@dataclass(frozen=True)
class _LocalReferenceCandidate:
    reference: PNCReferenceLine
    source_id: str
    orientation: str
    nearest_distance: float
    contains_ego: bool
    heading_error: float
    forward_length: float
    forward_progress: float


class PNCMap:
    """有历史状态的局部 PNC Map。

    每条单帧 reference 同时保留正向和反向解释。首次选择依据自车附近航向、
    前向长度和前向位移；有历史后，两种方向都转换到世界坐标与上一帧比较，
    从而避免在直角弯临界位置因 tangent.x 符号变化而整条 reference 翻转。
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
        backward_margin: float = 2.0,
        history_forward_length: float = 30.0,
        minimum_forward_length: float = 8.0,
        minimum_forward_progress: float = 1.0,
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
        if backward_margin < 0.0 or history_forward_length <= 0.0:
            raise ValueError("invalid local history comparison range")
        if minimum_forward_length <= 0.0 or minimum_forward_progress < 0.0:
            raise ValueError("invalid forward validity thresholds")

        self.minimum_confidence = minimum_confidence
        self.maximum_lateral_distance = maximum_lateral_distance
        self.switch_confirm_frames = switch_confirm_frames
        self.history_retention_cost = history_retention_cost
        self.history_hard_limit = history_hard_limit
        self.pending_match_cost = pending_match_cost
        self.backward_margin = backward_margin
        self.history_forward_length = history_forward_length
        self.minimum_forward_length = minimum_forward_length
        self.minimum_forward_progress = minimum_forward_progress

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
        raw_references = build_reference_lines(
            lane_lines,
            minimum_confidence=self.minimum_confidence,
        )
        candidates = _build_bidirectional_local_candidates(
            raw_references,
            maximum_lateral_distance=self.maximum_lateral_distance,
            backward_margin=self.backward_margin,
        )
        candidates = _filter_forward_valid_candidates(
            candidates,
            minimum_forward_length=self.minimum_forward_length,
            minimum_forward_progress=self.minimum_forward_progress,
        )
        geometric = _select_geometric_candidate(candidates)

        if not candidates:
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

        references = tuple(candidate.reference for candidate in candidates)
        world_paths = tuple(
            local_reference_path_to_world(reference.reference_path, world_origin)
            for reference in references
        )

        if self._current_world_path is None:
            selected_candidate = geometric or candidates[0]
            selected_index = candidates.index(selected_candidate)
            self._current_world_path = world_paths[selected_index]
            self._pending_world_path = None
            self._pending_frames = 0
            return PNCMapUpdate(
                selected=selected_candidate.reference,
                references=references,
                reference_changed=False,
                history_used=False,
                switch_pending_frames=0,
                continuity_cost=None,
                selected_orientation=selected_candidate.orientation,
            )

        current_history = _crop_world_reference_near_ego(
            self._current_world_path,
            world_origin,
            backward_margin=self.backward_margin,
            forward_length=self.history_forward_length,
        )
        costs = np.asarray(
            [
                _reference_continuity_cost(
                    _crop_world_reference_near_ego(
                        path,
                        world_origin,
                        backward_margin=self.backward_margin,
                        forward_length=self.history_forward_length,
                    ),
                    current_history,
                )
                for path in world_paths
            ],
            dtype=np.float64,
        )
        history_index = int(np.argmin(costs))
        history_cost = float(costs[history_index])
        history_candidate = candidates[history_index]
        history_world = world_paths[history_index]
        candidate_costs = tuple(
            (
                f"{candidate.source_id}@{candidate.orientation}",
                float(cost),
            )
            for candidate, cost in zip(candidates, costs)
        )

        if history_cost <= self.history_retention_cost:
            self._current_world_path = history_world
            self._pending_world_path = None
            self._pending_frames = 0
            return PNCMapUpdate(
                selected=history_candidate.reference,
                references=references,
                reference_changed=False,
                history_used=True,
                switch_pending_frames=0,
                continuity_cost=history_cost,
                candidate_costs=candidate_costs,
                selected_orientation=history_candidate.orientation,
            )

        target_candidate = geometric or history_candidate
        target_index = candidates.index(target_candidate)
        target_world = world_paths[target_index]
        pending_cost = (
            _reference_continuity_cost(
                _crop_world_reference_near_ego(
                    target_world,
                    world_origin,
                    backward_margin=self.backward_margin,
                    forward_length=self.history_forward_length,
                ),
                _crop_world_reference_near_ego(
                    self._pending_world_path,
                    world_origin,
                    backward_margin=self.backward_margin,
                    forward_length=self.history_forward_length,
                ),
            )
            if self._pending_world_path is not None
            else math.inf
        )
        if pending_cost <= self.pending_match_cost:
            self._pending_frames += 1
            self._pending_world_path = target_world
        else:
            self._pending_world_path = target_world
            self._pending_frames = 1

        if (
            self._pending_frames < self.switch_confirm_frames
            and history_cost <= self.history_hard_limit
        ):
            self._current_world_path = history_world
            return PNCMapUpdate(
                selected=history_candidate.reference,
                references=references,
                reference_changed=False,
                history_used=True,
                switch_pending_frames=self._pending_frames,
                continuity_cost=history_cost,
                candidate_costs=candidate_costs,
                selected_orientation=history_candidate.orientation,
            )

        self._current_world_path = target_world
        self._pending_world_path = None
        self._pending_frames = 0
        return PNCMapUpdate(
            selected=target_candidate.reference,
            references=references,
            reference_changed=True,
            history_used=False,
            switch_pending_frames=0,
            continuity_cost=history_cost,
            candidate_costs=candidate_costs,
            selected_orientation=target_candidate.orientation,
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
        points = _orient_line_for_pairing(line.points)
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
    candidates = _build_bidirectional_local_candidates(
        references,
        maximum_lateral_distance=maximum_lateral_distance,
        backward_margin=backward_margin,
    )
    candidates = _filter_forward_valid_candidates(
        candidates,
        minimum_forward_length=8.0,
        minimum_forward_progress=1.0,
    )
    selected = _select_geometric_candidate(candidates)
    return None if selected is None else selected.reference


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
    candidates = _build_bidirectional_local_candidates(
        references,
        maximum_lateral_distance=maximum_lateral_distance,
        backward_margin=2.0,
    )
    candidates = _filter_forward_valid_candidates(
        candidates,
        minimum_forward_length=8.0,
        minimum_forward_progress=1.0,
    )
    selected = _select_geometric_candidate(candidates)
    localized = tuple(candidate.reference for candidate in candidates)
    return (None if selected is None else selected.reference, localized)


def _build_bidirectional_local_candidates(
    references: tuple[PNCReferenceLine, ...],
    *,
    maximum_lateral_distance: float,
    backward_margin: float,
) -> tuple[_LocalReferenceCandidate, ...]:
    candidates: list[_LocalReferenceCandidate] = []
    for reference in references:
        for orientation, oriented in (
            ("forward", reference),
            ("reverse", _reverse_reference(reference)),
        ):
            centerline = oriented.reference_path[:, :2]
            distances = np.linalg.norm(centerline, axis=1)
            nearest_index = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_index])
            if nearest_distance > maximum_lateral_distance:
                continue
            sliced = _slice_reference(oriented, nearest_index, backward_margin)
            if sliced is None:
                continue
            candidate = _describe_local_candidate(
                sliced,
                source_id=reference.reference_id,
                orientation=orientation,
            )
            if candidate is not None:
                candidates.append(candidate)
    return tuple(candidates)


def _describe_local_candidate(
    reference: PNCReferenceLine,
    *,
    source_id: str,
    orientation: str,
) -> _LocalReferenceCandidate | None:
    centerline = reference.reference_path[:, :2]
    distances = np.linalg.norm(centerline, axis=1)
    nearest_index = int(np.argmin(distances))
    tangent = _normalized_tangent(centerline, nearest_index)
    if tangent is None:
        return None

    left = reference.left_boundary[nearest_index]
    right = reference.right_boundary[nearest_index]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    signed_left = float(np.dot(left, normal))
    signed_right = float(np.dot(right, normal))
    contains_ego = (
        min(signed_left, signed_right) - 1e-6
        <= 0.0
        <= max(signed_left, signed_right) + 1e-6
    )
    heading_error = abs(float(normalize_angle(reference.reference_path[nearest_index, 2])))
    forward_length = max(
        0.0,
        float(reference.reference_path[-1, 3] - reference.reference_path[nearest_index, 3]),
    )
    forward_points = centerline[nearest_index:]
    if forward_points.shape[0] < 2:
        forward_progress = 0.0
    else:
        relative_x = forward_points[:, 0] - forward_points[0, 0]
        forward_progress = float(np.max(relative_x))

    return _LocalReferenceCandidate(
        reference=reference,
        source_id=source_id,
        orientation=orientation,
        nearest_distance=float(distances[nearest_index]),
        contains_ego=contains_ego,
        heading_error=heading_error,
        forward_length=forward_length,
        forward_progress=forward_progress,
    )


def _filter_forward_valid_candidates(
    candidates: tuple[_LocalReferenceCandidate, ...],
    *,
    minimum_forward_length: float,
    minimum_forward_progress: float,
) -> tuple[_LocalReferenceCandidate, ...]:
    strict = tuple(
        candidate
        for candidate in candidates
        if candidate.forward_length >= minimum_forward_length
        and candidate.forward_progress >= minimum_forward_progress
        and candidate.heading_error <= math.radians(100.0)
    )
    if strict:
        return strict

    # 感知范围很短时允许退化，但仍拒绝明显朝后的解释。
    relaxed = tuple(
        candidate
        for candidate in candidates
        if candidate.forward_length >= 2.0
        and candidate.forward_progress >= 0.2
        and candidate.heading_error <= math.radians(120.0)
    )
    return relaxed


def _select_geometric_candidate(
    candidates: tuple[_LocalReferenceCandidate, ...],
) -> _LocalReferenceCandidate | None:
    if not candidates:
        return None

    def score(candidate: _LocalReferenceCandidate) -> tuple[int, float]:
        short_penalty = max(0.0, 12.0 - candidate.forward_length) * 0.3
        progress_penalty = max(0.0, 5.0 - candidate.forward_progress) * 0.4
        value = (
            candidate.nearest_distance
            + 2.0 * candidate.heading_error
            + (1.0 - candidate.reference.confidence)
            + short_penalty
            + progress_penalty
        )
        return (0 if candidate.contains_ego else 1, value)

    return min(candidates, key=score)


def _reverse_reference(reference: PNCReferenceLine) -> PNCReferenceLine:
    centerline = reference.reference_path[::-1, :2].copy()
    reversed_path = polyline_to_reference_path(centerline)
    return PNCReferenceLine(
        reference_id=reference.reference_id,
        confidence=reference.confidence,
        reference_path=reversed_path,
        left_boundary=reference.right_boundary[::-1].copy(),
        right_boundary=reference.left_boundary[::-1].copy(),
    )


def _pair_boundaries_by_normal_projection(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    output_point_count: int,
    minimum_lane_width: float,
    maximum_lane_width: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
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


def _orient_line_for_pairing(points: np.ndarray) -> np.ndarray:
    """仅为左右边界配对提供稳定点序，不再使用 tangent.x 的硬阈值。"""
    value = np.asarray(points, dtype=np.float64)
    if value.shape[0] < 2:
        return value.copy()
    forward_score = _orientation_visibility_score(value)
    reverse_score = _orientation_visibility_score(value[::-1])
    return value.copy() if forward_score >= reverse_score else value[::-1].copy()


def _orientation_visibility_score(points: np.ndarray) -> float:
    nearest = int(np.argmin(np.linalg.norm(points, axis=1)))
    tangent = _normalized_tangent(points, nearest)
    if tangent is None:
        return -math.inf
    forward = points[nearest:]
    if forward.shape[0] < 2:
        return -math.inf
    arc = float(np.sum(np.linalg.norm(np.diff(forward, axis=0), axis=1)))
    relative_x = forward[:, 0] - forward[0, 0]
    progress = float(np.max(relative_x))
    initial_alignment = float(tangent[0])
    return 2.0 * initial_alignment + 0.4 * progress + 0.05 * arc


def _crop_world_reference_near_ego(
    path: np.ndarray,
    ego: VehicleState,
    *,
    backward_margin: float,
    forward_length: float,
) -> np.ndarray:
    if path is None or path.shape[0] < 2:
        return path
    distances = np.linalg.norm(
        path[:, :2] - np.array([ego.x, ego.y], dtype=np.float64),
        axis=1,
    )
    nearest = int(np.argmin(distances))
    arc = path[:, 3]
    center_s = float(arc[nearest])
    keep = (arc >= center_s - backward_margin) & (arc <= center_s + forward_length)
    indices = np.flatnonzero(keep)
    if indices.size < 2:
        start = max(0, nearest - 1)
        end = min(path.shape[0], nearest + 2)
        return path[start:end].copy()
    return path[indices[0] : indices[-1] + 1].copy()


def _reference_continuity_cost(
    first_world_path: np.ndarray,
    second_world_path: np.ndarray,
) -> float:
    first = _resample_polyline(first_world_path[:, :2], 41)
    second = _resample_polyline(second_world_path[:, :2], 41)
    point_distance = float(np.mean(np.linalg.norm(first - second, axis=1)))

    first_yaw = np.unwrap(first_world_path[:, 2])
    second_yaw = np.unwrap(second_world_path[:, 2])
    samples = np.linspace(0.0, 1.0, 41)
    first_yaw_sample = np.interp(
        samples,
        np.linspace(0.0, 1.0, first_yaw.size),
        first_yaw,
    )
    second_yaw_sample = np.interp(
        samples,
        np.linspace(0.0, 1.0, second_yaw.size),
        second_yaw,
    )
    heading = float(
        np.mean(np.abs(normalize_angle(first_yaw_sample - second_yaw_sample)))
    )
    start_distance = float(
        np.linalg.norm(first_world_path[0, :2] - second_world_path[0, :2])
    )
    return point_distance + 1.25 * heading + 0.25 * start_distance


def _slice_reference(
    selected: PNCReferenceLine,
    nearest_index: int,
    backward_margin: float,
) -> PNCReferenceLine | None:
    start_index = nearest_index
    centerline = selected.reference_path[:, :2]
    travelled = 0.0
    while start_index > 0:
        segment = float(
            np.linalg.norm(centerline[start_index] - centerline[start_index - 1])
        )
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
        return np.repeat(points, point_count, axis=0)
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
