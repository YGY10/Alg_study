from __future__ import annotations

from sim2d.perception import PerceivedLaneLine

from .pnc_map import (
    PNCReferenceLine,
    build_current_reference_line,
    local_reference_path_to_world,
)

# 保留旧名称，避免上层规划器接口和已有测试一次性全部失效。
PerceptionLaneReference = PNCReferenceLine


def build_perception_lane_reference(
    lane_lines: tuple[PerceivedLaneLine, ...],
    *,
    minimum_confidence: float = 0.5,
    maximum_lateral_distance: float = 6.0,
    backward_margin: float = 2.0,
) -> PerceptionLaneReference | None:
    """由 PNC map 小模块根据感知车道线构造并选择当前 reference line。"""
    del backward_margin
    selected, _ = build_current_reference_line(
        lane_lines,
        minimum_confidence=minimum_confidence,
        maximum_lateral_distance=maximum_lateral_distance,
    )
    return selected


__all__ = [
    "PerceptionLaneReference",
    "build_perception_lane_reference",
    "local_reference_path_to_world",
]
