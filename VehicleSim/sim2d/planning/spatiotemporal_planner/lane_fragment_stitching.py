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
class _OrientedFragment:
    index: int
    line: PerceivedLaneLine
    points: np.ndarray
    length: float
    start_tangent: np.ndarray
    end_tangent: np.ndarray


@dataclass(frozen=True)
class _JoinCandidate:
    source: int
    target: int
    score: float
    gap: float


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

    该函数只使用车辆坐标系下的点、端点距离和局部切向，不依赖地图 lane id、
    前驱后继或导航路径。参与链的原始 fragment 会被最大链替代，未参与拼接的
    车道线原样保留，避免重复 fragment 插在两条真实边界之间。
    """
    stitch_config = config or LaneFragmentStitchingConfig()
    stitch_config.validate()
    if len(lane_lines) < 2:
        debug = LaneFragmentStitchingDebug(
            input_count=len(lane_lines),
            output_count=len(lane_lines),
            stitched_chain_count=0,
            chains=(),
        )
        return tuple(lane_lines), debug

    fragments = tuple(
        fragment
        for index, line in enumerate(lane_lines)
        if (
            fragment := _prepare_fragment(index, line, stitch_config)
        )
        is not None
    )
    if len(fragments) < 2:
        debug = LaneFragmentStitchingDebug(
            input_count=len(lane_lines),
            output_count=len(lane_lines),
            stitched_chain_count=0,
            chains=(),
        )
        return tuple(lane_lines), debug

    joins = _select_non_conflicting_joins(fragments, stitch_config)
    chains = _build_chains(fragments, joins, stitch_config.maximum_chain_fragments)
    chained_indices = {fragment.index for chain in chains for fragment in chain}

    output: list[PerceivedLaneLine] = []
    chain_ids: list[tuple[str, ...]] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        stitched = _stitch_chain(chain, stitch_config)
        if stitched is None:
            continue
        output.append(stitched)
        chain_ids.append(tuple(fragment.line.line_id for fragment in chain))

    successfully_chained = {
        fragment.index
        for chain, ids in zip(chains, chain_ids)
        for fragment in chain
        if len(ids) >= 2
    }
    if successfully_chained != chained_indices:
        chained_indices = successfully_chained

    output.extend(
        line for index, line in enumerate(lane_lines) if index not in chained_indices
    )
    result = tuple(output)
    debug = LaneFragmentStitchingDebug(
        input_count=len(lane_lines),
        output_count=len(result),
        stitched_chain_count=len(chain_ids),
        chains=tuple(chain_ids),
    )
    return result, debug


def _prepare_fragment(
    index: int,
    line: PerceivedLaneLine,
    config: LaneFragmentStitchingConfig,
) -> _OrientedFragment | None:
    points = _remove_duplicate_points(np.asarray(line.points, dtype=np.float64))
    if points.shape[0] < 2:
        return None
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if length < config.minimum_fragment_length:
        return None

    forward_score = _orientation_score(points)
    reverse_score = _orientation_score(points[::-1])
    if reverse_score > forward_score:
        points = points[::-1].copy()

    start_tangent = _endpoint_tangent(points, at_start=True)
    end_tangent = _endpoint_tangent(points, at_start=False)
    if start_tangent is None or end_tangent is None:
        return None
    return _OrientedFragment(
        index=index,
        line=line,
        points=points,
        length=length,
        start_tangent=start_tangent,
        end_tangent=end_tangent,
    )


def _orientation_score(points: np.ndarray) -> float:
    nearest = int(np.argmin(np.linalg.norm(points, axis=1)))
    future = points[nearest:]
    if future.shape[0] < 2:
        return -1e6
    tangent = _endpoint_tangent(future, at_start=True)
    if tangent is None:
        return -1e6
    future_length = float(np.sum(np.linalg.norm(np.diff(future, axis=0), axis=1)))
    displacement = future[-1] - future[0]
    # +x 是自车前方；弯道中 x 可能不再增长，因此弧长只作为较弱的补充项。
    return (
        3.0 * float(tangent[0])
        + 0.35 * float(displacement[0])
        + 0.05 * future_length
    )


def _select_non_conflicting_joins(
    fragments: tuple[_OrientedFragment, ...],
    config: LaneFragmentStitchingConfig,
) -> dict[int, int]:
    candidates: list[_JoinCandidate] = []
    for source in fragments:
        for target in fragments:
            if source.index == target.index:
                continue
            candidate = _evaluate_join(source, target, config)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.score)

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for candidate in candidates:
        if candidate.source in outgoing or candidate.target in incoming:
            continue
        if _would_create_cycle(outgoing, candidate.source, candidate.target):
            continue
        outgoing[candidate.source] = candidate.target
        incoming[candidate.target] = candidate.source
    return outgoing


def _evaluate_join(
    source: _OrientedFragment,
    target: _OrientedFragment,
    config: LaneFragmentStitchingConfig,
) -> _JoinCandidate | None:
    delta = target.points[0] - source.points[-1]
    gap = float(np.linalg.norm(delta))
    if gap > config.maximum_join_distance:
        return None

    heading_delta = abs(_angle_difference(source.end_tangent, target.start_tangent))
    if heading_delta > config.maximum_join_heading:
        return None

    if gap > 1e-6:
        bridge_direction = delta / gap
        source_bridge_angle = abs(_angle_difference(source.end_tangent, bridge_direction))
        bridge_target_angle = abs(_angle_difference(bridge_direction, target.start_tangent))
        if max(source_bridge_angle, bridge_target_angle) > config.maximum_join_heading:
            return None
        normal = np.array([-source.end_tangent[1], source.end_tangent[0]])
        lateral_error = abs(float(np.dot(delta, normal)))
        if lateral_error > config.maximum_endpoint_lateral_error:
            return None
        effective_curvature = max(source_bridge_angle, bridge_target_angle) / max(gap, 0.5)
        if effective_curvature > config.maximum_join_curvature:
            return None
    else:
        source_bridge_angle = 0.0
        bridge_target_angle = 0.0
        lateral_error = 0.0
        effective_curvature = 0.0

    # 禁止把后方片段接到前方片段之后形成明显回折。
    forward_projection = float(np.dot(delta, source.end_tangent))
    if gap > 0.5 and forward_projection < -0.25:
        return None

    score = (
        gap
        + 2.0 * heading_delta
        + source_bridge_angle
        + bridge_target_angle
        + 0.5 * lateral_error
        + 2.0 * effective_curvature
    )
    return _JoinCandidate(
        source=source.index,
        target=target.index,
        score=float(score),
        gap=gap,
    )


def _would_create_cycle(outgoing: dict[int, int], source: int, target: int) -> bool:
    cursor = target
    visited: set[int] = set()
    while cursor in outgoing and cursor not in visited:
        if cursor == source:
            return True
        visited.add(cursor)
        cursor = outgoing[cursor]
    return cursor == source


def _build_chains(
    fragments: tuple[_OrientedFragment, ...],
    outgoing: dict[int, int],
    maximum_chain_fragments: int,
) -> tuple[tuple[_OrientedFragment, ...], ...]:
    by_index = {fragment.index: fragment for fragment in fragments}
    incoming = {target: source for source, target in outgoing.items()}
    starts = [index for index in outgoing if index not in incoming]
    chains: list[tuple[_OrientedFragment, ...]] = []
    consumed: set[int] = set()
    for start in starts:
        indices = [start]
        cursor = start
        while cursor in outgoing and len(indices) < maximum_chain_fragments:
            cursor = outgoing[cursor]
            if cursor in indices:
                break
            indices.append(cursor)
        if len(indices) >= 2:
            chains.append(tuple(by_index[index] for index in indices))
            consumed.update(indices)
    return tuple(chains)


def _stitch_chain(
    chain: tuple[_OrientedFragment, ...],
    config: LaneFragmentStitchingConfig,
) -> PerceivedLaneLine | None:
    points = chain[0].points.copy()
    for previous, following in zip(chain[:-1], chain[1:]):
        bridge = _hermite_bridge(
            previous.points[-1],
            previous.end_tangent,
            following.points[0],
            following.start_tangent,
            spacing=config.bridge_point_spacing,
        )
        pieces = [points]
        if bridge.shape[0] > 0:
            pieces.append(bridge)
        pieces.append(following.points)
        points = _remove_duplicate_points(np.vstack(pieces))
    if points.shape[0] < 2:
        return None

    ids = tuple(fragment.line.line_id for fragment in chain)
    confidence = min(fragment.line.confidence for fragment in chain)
    # 片段越多，连接不确定性越大，但不让合理的两段拼接被过度降权。
    confidence *= 0.97 ** max(0, len(chain) - 1)
    line_types = {fragment.line.line_type for fragment in chain}
    line_type = line_types.pop() if len(line_types) == 1 else "stitched"
    return PerceivedLaneLine(
        line_id="stitch[" + "+".join(ids) + "]",
        points=points,
        confidence=float(confidence),
        line_type=line_type,
    )


def _hermite_bridge(
    start: np.ndarray,
    start_tangent: np.ndarray,
    end: np.ndarray,
    end_tangent: np.ndarray,
    *,
    spacing: float,
) -> np.ndarray:
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
