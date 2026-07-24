from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim2d.perception import PerceivedLaneLine

from .pnc_map import PNCMap


@dataclass(frozen=True)
class LaneFragmentStitchingConfig:
    """PNC Map 入口处的纯几何车道线片段拼接参数。"""

    maximum_join_distance: float = 7.0
    maximum_join_heading: float = math.radians(55.0)
    maximum_join_curvature: float = 0.45
    maximum_endpoint_lateral_error: float = 2.0
    minimum_fragment_length: float = 1.0
    bridge_point_spacing: float = 0.5
    maximum_chain_fragments: int = 6

    def validate(self) -> None:
        values = (
            self.maximum_join_distance,
            self.maximum_join_heading,
            self.maximum_join_curvature,
            self.maximum_endpoint_lateral_error,
            self.minimum_fragment_length,
            self.bridge_point_spacing,
        )
        if min(values) <= 0.0:
            raise ValueError("lane fragment stitching thresholds must be positive")
        if self.maximum_join_heading > math.pi:
            raise ValueError("maximum_join_heading must not exceed pi")
        if self.maximum_chain_fragments < 2:
            raise ValueError("maximum_chain_fragments must be at least 2")


@dataclass(frozen=True)
class LaneFragmentStitchingDebug:
    input_count: int
    output_count: int
    stitched_chain_count: int
    chains: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _WorkingLine:
    fragments: tuple[PerceivedLaneLine, ...]
    points: np.ndarray


@dataclass(frozen=True)
class _MergeCandidate:
    first_index: int
    second_index: int
    first_points: np.ndarray
    second_points: np.ndarray
    score: float


_ORIGINAL_PNC_UPDATE = None


def install(config: LaneFragmentStitchingConfig | None = None) -> None:
    """在 PNCMap.update 入口安装片段拼接，不改变感知模块公开输出。"""
    global _ORIGINAL_PNC_UPDATE
    if _ORIGINAL_PNC_UPDATE is not None:
        return
    stitch_config = config or LaneFragmentStitchingConfig()
    stitch_config.validate()
    original_update = PNCMap.update
    _ORIGINAL_PNC_UPDATE = original_update

    def stitched_update(self, lane_lines, *, world_origin):
        stitched_lines, debug = stitch_lane_line_fragments(
            tuple(lane_lines),
            config=stitch_config,
        )
        self._lane_fragment_stitching_debug = debug
        self._lane_fragment_stitched_lines = stitched_lines
        return original_update(
            self,
            stitched_lines,
            world_origin=world_origin,
        )

    PNCMap.update = stitched_update


def stitch_lane_line_fragments(
    lane_lines: tuple[PerceivedLaneLine, ...],
    *,
    config: LaneFragmentStitchingConfig | None = None,
) -> tuple[tuple[PerceivedLaneLine, ...], LaneFragmentStitchingDebug]:
    """把属于同一物理边界的前后片段拼成 boundary chain。

    对每一对 fragment 同时尝试四种点序组合，因此直角弯处即使局部 x 几乎
    不变化，也不依赖单帧按 x 猜方向。每次只合并全局连接代价最低的一对，
    再继续搜索下一段，最终得到 line1→line3 这样的最大边界链。
    """
    stitch_config = config or LaneFragmentStitchingConfig()
    stitch_config.validate()

    working: list[_WorkingLine] = []
    for line in lane_lines:
        points = _remove_duplicate_points(np.asarray(line.points, dtype=np.float64))
        if points.shape[0] < 2:
            continue
        working.append(_WorkingLine((line,), points))

    maximum_merges = max(0, len(working) - 1)
    for _ in range(maximum_merges):
        candidate = _find_best_merge(working, stitch_config)
        if candidate is None:
            break
        first = working[candidate.first_index]
        second = working[candidate.second_index]
        bridge = _hermite_bridge(
            candidate.first_points[-1],
            _endpoint_tangent(candidate.first_points, at_start=False),
            candidate.second_points[0],
            _endpoint_tangent(candidate.second_points, at_start=True),
            spacing=stitch_config.bridge_point_spacing,
        )
        pieces = [candidate.first_points]
        if bridge.shape[0] > 0:
            pieces.append(bridge)
        pieces.append(candidate.second_points)
        merged_points = _remove_duplicate_points(np.vstack(pieces))
        merged_fragments = first.fragments + second.fragments
        merged = _WorkingLine(merged_fragments, merged_points)

        high = max(candidate.first_index, candidate.second_index)
        low = min(candidate.first_index, candidate.second_index)
        working.pop(high)
        working.pop(low)
        working.append(merged)

    output: list[PerceivedLaneLine] = []
    chains: list[tuple[str, ...]] = []
    for item in working:
        if len(item.fragments) == 1:
            output.append(item.fragments[0])
            continue
        ids = tuple(fragment.line_id for fragment in item.fragments)
        chains.append(ids)
        confidence = min(fragment.confidence for fragment in item.fragments)
        confidence *= 0.97 ** (len(item.fragments) - 1)
        line_types = {fragment.line_type for fragment in item.fragments}
        line_type = next(iter(line_types)) if len(line_types) == 1 else "stitched"
        output.append(
            PerceivedLaneLine(
                line_id="stitch[" + "+".join(ids) + "]",
                points=item.points,
                confidence=float(confidence),
                line_type=line_type,
            )
        )

    result = tuple(output)
    debug = LaneFragmentStitchingDebug(
        input_count=len(lane_lines),
        output_count=len(result),
        stitched_chain_count=len(chains),
        chains=tuple(chains),
    )
    return result, debug


def _find_best_merge(
    working: list[_WorkingLine],
    config: LaneFragmentStitchingConfig,
) -> _MergeCandidate | None:
    best: _MergeCandidate | None = None
    for first_index in range(len(working)):
        first = working[first_index]
        if len(first.fragments) >= config.maximum_chain_fragments:
            continue
        for second_index in range(first_index + 1, len(working)):
            second = working[second_index]
            if len(first.fragments) + len(second.fragments) > config.maximum_chain_fragments:
                continue
            for first_points in (first.points, first.points[::-1]):
                for second_points in (second.points, second.points[::-1]):
                    score = _join_score(first_points, second_points, config)
                    if score is None:
                        continue
                    candidate = _MergeCandidate(
                        first_index=first_index,
                        second_index=second_index,
                        first_points=first_points.copy(),
                        second_points=second_points.copy(),
                        score=score,
                    )
                    if best is None or candidate.score < best.score:
                        best = candidate
    return best


def _join_score(
    first: np.ndarray,
    second: np.ndarray,
    config: LaneFragmentStitchingConfig,
) -> float | None:
    first_tangent = _endpoint_tangent(first, at_start=False)
    second_tangent = _endpoint_tangent(second, at_start=True)
    if first_tangent is None or second_tangent is None:
        return None

    delta = second[0] - first[-1]
    gap = float(np.linalg.norm(delta))
    if gap > config.maximum_join_distance:
        return None

    heading_delta = abs(_angle_difference(first_tangent, second_tangent))
    if heading_delta > config.maximum_join_heading:
        return None

    if gap <= 1e-6:
        first_bridge_angle = 0.0
        bridge_second_angle = 0.0
        lateral_error = 0.0
        effective_curvature = heading_delta / 0.5
    else:
        bridge_direction = delta / gap
        first_bridge_angle = abs(_angle_difference(first_tangent, bridge_direction))
        bridge_second_angle = abs(_angle_difference(bridge_direction, second_tangent))
        if max(first_bridge_angle, bridge_second_angle) > config.maximum_join_heading:
            return None
        normal = np.array([-first_tangent[1], first_tangent[0]], dtype=np.float64)
        lateral_error = abs(float(np.dot(delta, normal)))
        if lateral_error > config.maximum_endpoint_lateral_error:
            return None
        effective_curvature = max(first_bridge_angle, bridge_second_angle) / max(gap, 0.5)

    if effective_curvature > config.maximum_join_curvature:
        return None

    forward_projection = float(np.dot(delta, first_tangent))
    if gap > 0.5 and forward_projection < -0.25:
        return None

    return float(
        gap
        + 2.0 * heading_delta
        + first_bridge_angle
        + bridge_second_angle
        + 0.75 * lateral_error
        + 2.0 * effective_curvature
    )


def _hermite_bridge(
    start: np.ndarray,
    start_tangent: np.ndarray | None,
    end: np.ndarray,
    end_tangent: np.ndarray | None,
    *,
    spacing: float,
) -> np.ndarray:
    if start_tangent is None or end_tangent is None:
        return np.empty((0, 2), dtype=np.float64)
    gap = float(np.linalg.norm(end - start))
    if gap <= max(1e-6, 0.25 * spacing):
        return np.empty((0, 2), dtype=np.float64)
    interior_count = max(1, int(math.ceil(gap / spacing)) - 1)
    parameter = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    tangent_scale = max(gap, spacing)
    m0 = tangent_scale * start_tangent
    m1 = tangent_scale * end_tangent
    t = parameter[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * start + h10 * m0 + h01 * end + h11 * m1


def _endpoint_tangent(points: np.ndarray, *, at_start: bool) -> np.ndarray | None:
    if points.shape[0] < 2:
        return None
    window = min(5, points.shape[0] - 1)
    vector = points[window] - points[0] if at_start else points[-1] - points[-1 - window]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _remove_duplicate_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points.copy()
    keep = np.concatenate(
        (
            np.array([True]),
            np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8,
        )
    )
    return points[keep].copy()


def _angle_difference(first: np.ndarray, second: np.ndarray) -> float:
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.atan2(cross, dot)


__all__ = [
    "LaneFragmentStitchingConfig",
    "LaneFragmentStitchingDebug",
    "install",
    "stitch_lane_line_fragments",
]
