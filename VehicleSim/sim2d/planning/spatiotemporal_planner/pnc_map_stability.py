from __future__ import annotations

import math

import numpy as np

from sim2d.types import VehicleState

from . import pnc_map as legacy


_INSTALLED = False
_ORIGINAL_UPDATE = legacy.PNCMap.update


def install_pnc_map_stability() -> None:
    """安装稳定候选配对与切换状态机。"""
    global _INSTALLED
    if _INSTALLED:
        return
    legacy.build_reference_lines = _build_reference_lines_all_pairs
    legacy.PNCMap.update = _stable_update
    _INSTALLED = True


def _build_reference_lines_all_pairs(
    lane_lines,
    *,
    minimum_confidence: float = 0.5,
    minimum_lane_width: float = 2.0,
    maximum_lane_width: float = 6.0,
    output_point_count: int = 101,
):
    """尝试所有合理左右边界组合，避免重复短线阻断正确相邻配对。"""
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within [0, 1]")
    if minimum_lane_width <= 0.0 or maximum_lane_width < minimum_lane_width:
        raise ValueError("invalid lane width range")
    if output_point_count < 2:
        raise ValueError("output_point_count must be at least 2")

    prepared = []
    for line in lane_lines:
        if line.confidence < minimum_confidence:
            continue
        points = legacy._orient_line_for_pairing(line.points)
        nearest_index = int(np.argmin(np.linalg.norm(points, axis=1)))
        tangent = legacy._normalized_tangent(points, nearest_index)
        if tangent is None:
            continue
        lateral = float(points[nearest_index, 1])
        heading_error = abs(math.atan2(float(tangent[1]), float(tangent[0])))
        prepared.append((line, points, lateral, heading_error))

    prepared.sort(key=lambda item: item[2], reverse=True)
    references = []
    seen_geometry: list[np.ndarray] = []

    for left_index in range(len(prepared) - 1):
        for right_index in range(left_index + 1, len(prepared)):
            left_line, left_points, left_lateral, left_heading = prepared[left_index]
            right_line, right_points, right_lateral, right_heading = prepared[right_index]

            local_gap = left_lateral - right_lateral
            if local_gap < 0.6 * minimum_lane_width:
                continue
            if local_gap > 1.5 * maximum_lane_width:
                break
            if max(left_heading, right_heading) > math.radians(120.0):
                continue

            pairing = legacy._pair_boundaries_by_normal_projection(
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
                reference_path = legacy.polyline_to_reference_path(centerline)
            except ValueError:
                continue

            # 多个重复感知片段可能生成几乎相同的 corridor，只保留第一条。
            signature = legacy._resample_polyline(centerline, 31)
            duplicate = False
            for previous in seen_geometry:
                if float(np.mean(np.linalg.norm(signature - previous, axis=1))) < 0.20:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_geometry.append(signature)

            confidence = float(
                min(left_line.confidence, right_line.confidence) * valid_ratio
            )
            references.append(
                legacy.PNCReferenceLine(
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


def _stable_update(self, lane_lines, *, world_origin: VehicleState):
    """所有切换均要求相对优势和连续确认；候选短时丢失时沿用历史。"""
    switch_advantage_margin = 0.5
    coast_max_frames = 3

    raw_references = _build_reference_lines_all_pairs(
        lane_lines,
        minimum_confidence=self.minimum_confidence,
    )
    candidates = legacy._build_bidirectional_local_candidates(
        raw_references,
        maximum_lateral_distance=self.maximum_lateral_distance,
        backward_margin=self.backward_margin,
    )
    candidates = legacy._filter_forward_valid_candidates(
        candidates,
        minimum_forward_length=self.minimum_forward_length,
        minimum_forward_progress=self.minimum_forward_progress,
    )
    geometric = legacy._select_geometric_candidate(candidates)

    references = tuple(candidate.reference for candidate in candidates)
    world_paths = tuple(
        legacy.local_reference_path_to_world(reference.reference_path, world_origin)
        for reference in references
    )

    if self._current_world_path is None:
        if not candidates:
            result = legacy.PNCMapUpdate(
                selected=None,
                references=(),
                reference_changed=False,
                history_used=False,
                switch_pending_frames=0,
                continuity_cost=None,
            )
            self._performance_debug_last_update = result
            return result
        selected_candidate = geometric or candidates[0]
        selected_index = _identity_index(candidates, selected_candidate)
        self._current_world_path = world_paths[selected_index]
        self._pending_world_path = None
        self._pending_frames = 0
        self._coast_frames = 0
        result = legacy.PNCMapUpdate(
            selected=selected_candidate.reference,
            references=references,
            reference_changed=False,
            history_used=False,
            switch_pending_frames=0,
            continuity_cost=None,
            selected_orientation=selected_candidate.orientation,
        )
        self._performance_debug_last_update = result
        return result

    current_history = legacy._crop_world_reference_near_ego(
        self._current_world_path,
        world_origin,
        backward_margin=self.backward_margin,
        forward_length=self.history_forward_length,
    )

    continuity = tuple(
        self._continuity_metrics(
            legacy._crop_world_reference_near_ego(
                path,
                world_origin,
                backward_margin=self.backward_margin,
                forward_length=self.history_forward_length,
            ),
            current_history,
            candidate_id=f"{candidate.source_id}@{candidate.orientation}",
        )
        for candidate, path in zip(candidates, world_paths)
    )
    candidate_costs = tuple(
        (metrics.candidate_id, metrics.cost) for metrics in continuity
    )
    finite_indices = [
        index for index, metrics in enumerate(continuity) if math.isfinite(metrics.cost)
    ]

    if not finite_indices:
        result = _coast_or_reinitialize(
            self,
            candidates=candidates,
            references=references,
            world_paths=world_paths,
            geometric=geometric,
            world_origin=world_origin,
            continuity=continuity,
            candidate_costs=candidate_costs,
            coast_max_frames=coast_max_frames,
        )
        self._performance_debug_last_update = result
        return result

    history_index = min(finite_indices, key=lambda index: continuity[index].cost)
    history_cost = float(continuity[history_index].cost)
    history_candidate = candidates[history_index]
    history_world = world_paths[history_index]
    self._coast_frames = 0

    if history_cost <= self.history_retention_cost:
        self._current_world_path = history_world
        self._pending_world_path = None
        self._pending_frames = 0
        result = legacy.PNCMapUpdate(
            selected=history_candidate.reference,
            references=references,
            reference_changed=False,
            history_used=True,
            switch_pending_frames=0,
            continuity_cost=history_cost,
            candidate_costs=candidate_costs,
            selected_orientation=history_candidate.orientation,
            candidate_continuity=continuity,
        )
        self._performance_debug_last_update = result
        return result

    target_candidate = geometric or history_candidate
    target_index = _identity_index(candidates, target_candidate)
    target_cost = float(continuity[target_index].cost)

    # 历史代价变大并不等于新候选更可信。只有新候选相对历史候选显著占优，
    # 才进入切换确认；否则继续跟踪当前帧中最接近历史车道的候选。
    target_has_advantage = (
        target_index != history_index
        and math.isfinite(target_cost)
        and target_cost + switch_advantage_margin < history_cost
    )
    if not target_has_advantage:
        self._current_world_path = history_world
        self._pending_world_path = None
        self._pending_frames = 0
        result = legacy.PNCMapUpdate(
            selected=history_candidate.reference,
            references=references,
            reference_changed=False,
            history_used=True,
            switch_pending_frames=0,
            continuity_cost=history_cost,
            candidate_costs=candidate_costs,
            selected_orientation=history_candidate.orientation,
            candidate_continuity=continuity,
        )
        self._performance_debug_last_update = result
        return result

    target_world = world_paths[target_index]
    pending_cost = math.inf
    if self._pending_world_path is not None:
        pending_cost = self._continuity_metrics(
            legacy._crop_world_reference_near_ego(
                target_world,
                world_origin,
                backward_margin=self.backward_margin,
                forward_length=self.history_forward_length,
            ),
            legacy._crop_world_reference_near_ego(
                self._pending_world_path,
                world_origin,
                backward_margin=self.backward_margin,
                forward_length=self.history_forward_length,
            ),
            candidate_id="pending",
        ).cost

    if pending_cost <= self.pending_match_cost:
        self._pending_frames += 1
    else:
        self._pending_frames = 1
    self._pending_world_path = target_world

    # 不再使用 history_hard_limit 绕过确认窗口。
    if self._pending_frames < self.switch_confirm_frames:
        self._current_world_path = history_world
        result = legacy.PNCMapUpdate(
            selected=history_candidate.reference,
            references=references,
            reference_changed=False,
            history_used=True,
            switch_pending_frames=self._pending_frames,
            continuity_cost=history_cost,
            candidate_costs=candidate_costs,
            selected_orientation=history_candidate.orientation,
            candidate_continuity=continuity,
        )
        self._performance_debug_last_update = result
        return result

    self._current_world_path = target_world
    self._pending_world_path = None
    self._pending_frames = 0
    result = legacy.PNCMapUpdate(
        selected=target_candidate.reference,
        references=references,
        reference_changed=True,
        history_used=False,
        switch_pending_frames=0,
        continuity_cost=history_cost,
        candidate_costs=candidate_costs,
        selected_orientation=target_candidate.orientation,
        candidate_continuity=continuity,
    )
    self._performance_debug_last_update = result
    return result


def _coast_or_reinitialize(
    self,
    *,
    candidates,
    references,
    world_paths,
    geometric,
    world_origin,
    continuity,
    candidate_costs,
    coast_max_frames: int,
):
    coast_frames = int(getattr(self, "_coast_frames", 0)) + 1
    self._coast_frames = coast_frames
    self._pending_world_path = None
    self._pending_frames = 0

    if coast_frames <= coast_max_frames:
        coast_reference = _world_history_to_local_reference(
            self._current_world_path,
            world_origin,
            backward_margin=self.backward_margin,
            forward_length=self.history_forward_length,
        )
        return legacy.PNCMapUpdate(
            selected=coast_reference,
            references=references,
            reference_changed=False,
            history_used=True,
            switch_pending_frames=0,
            continuity_cost=math.inf,
            candidate_costs=candidate_costs,
            selected_orientation="coast",
            candidate_continuity=continuity,
        )

    if geometric is None:
        return legacy.PNCMapUpdate(
            selected=None,
            references=references,
            reference_changed=False,
            history_used=False,
            switch_pending_frames=0,
            continuity_cost=math.inf,
            candidate_costs=candidate_costs,
            candidate_continuity=continuity,
        )

    target_index = _identity_index(candidates, geometric)
    self._current_world_path = world_paths[target_index]
    self._coast_frames = 0
    return legacy.PNCMapUpdate(
        selected=geometric.reference,
        references=references,
        reference_changed=True,
        history_used=False,
        switch_pending_frames=0,
        continuity_cost=math.inf,
        candidate_costs=candidate_costs,
        selected_orientation=geometric.orientation,
        candidate_continuity=continuity,
    )


def _world_history_to_local_reference(
    world_path: np.ndarray,
    ego: VehicleState,
    *,
    backward_margin: float,
    forward_length: float,
):
    cropped = legacy._crop_world_reference_near_ego(
        world_path,
        ego,
        backward_margin=backward_margin,
        forward_length=forward_length,
    )
    cosine = math.cos(ego.yaw)
    sine = math.sin(ego.yaw)
    dx = cropped[:, 0] - ego.x
    dy = cropped[:, 1] - ego.y
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    local_yaw = legacy.normalize_angle(cropped[:, 2] - ego.yaw)
    points = np.column_stack((local_x, local_y))
    local_path = legacy.polyline_to_reference_path(points)
    local_path[:, 2] = local_yaw
    local_path[:, 4] = cropped[:, 4]

    normals = np.column_stack((-np.sin(local_path[:, 2]), np.cos(local_path[:, 2])))
    half_width = 1.8
    left_boundary = local_path[:, :2] + half_width * normals
    right_boundary = local_path[:, :2] - half_width * normals
    return legacy.PNCReferenceLine(
        reference_id="coast_history",
        confidence=0.5,
        reference_path=local_path,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
    )


def _identity_index(items, target) -> int:
    for index, item in enumerate(items):
        if item is target:
            return index
    raise ValueError("target candidate is not in candidate sequence")


__all__ = ["install_pnc_map_stability"]
