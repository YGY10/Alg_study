from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2d.perception import PerceivedLaneLine

from .pnc_map import (
    build_current_reference_line,
    local_reference_path_to_world,
)


@dataclass(frozen=True)
class PerceptionLaneReference:
    """规划器使用的当前参考线兼容视图。"""

    lane_id: str
    confidence: float
    reference_path: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray


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
    if selected is None:
        return None
    return PerceptionLaneReference(
        lane_id=selected.reference_id,
        confidence=selected.confidence,
        reference_path=selected.reference_path,
        left_boundary=selected.left_boundary,
        right_boundary=selected.right_boundary,
    )


__all__ = [
    "PerceptionLaneReference",
    "build_perception_lane_reference",
    "local_reference_path_to_world",
]
