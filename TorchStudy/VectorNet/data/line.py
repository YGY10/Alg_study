from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List

import numpy as np


class LineKind(IntEnum):
    """
    区分三类：车道线/路沿/车辆轨迹
    """

    LANE_LINE = 0
    CURB = 1
    AGENT_HISTORY = 2


@dataclass
class Line:
    """
    一条 polyline，可以是地图线，也可以是车辆历史轨迹。
    """

    points: np.ndarray
    kind: LineKind
    line_id: int
    agent_id: int = -1
    is_target: bool = False

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)

        if self.points.ndim != 2:
            raise ValueError("Line.points must have shape [N, 2].")

        if self.points.shape[1] != 2:
            raise ValueError("Line.points must have shape [N, 2].")

        if self.points.shape[0] < 2:
            raise ValueError("Line must contain at least 2 points.")

        if self.kind != LineKind.AGENT_HISTORY and self.agent_id != -1:
            raise ValueError("Only AGENT_HISTORY lines should have agent_id != -1.")


@dataclass
class SceneSample:
    """
    一个训练样本。
    """

    lines: List[Line]
    target_agent_id: int
    future: np.ndarray

    def __post_init__(self):
        self.future = np.asarray(self.future, dtype=np.float32)

        if self.future.ndim != 2:
            raise ValueError("SceneSample.future must have shape [T, 2].")

        if self.future.shape[1] != 2:
            raise ValueError("SceneSample.future must have shape [T, 2].")

        target_lines = [
            line
            for line in self.lines
            if line.kind == LineKind.AGENT_HISTORY
            and line.agent_id == self.target_agent_id
            and line.is_target
        ]

        if len(target_lines) != 1:
            raise ValueError(
                "SceneSample must contain exactly one target AGENT_HISTORY line."
            )
