from __future__ import annotations
from typing import List, Tuple
import numpy as np
from data.line import Line, LineKind, SceneSample

VECTOR_FEATURE_DIM = 11


def line_to_vectors(line: Line) -> np.ndarray:
    """
    把一条Line转成 vector segments
    输入：
        line.points: [N, 2]
    输出：
        vector:[N - 1, F]
    当前每个 vector feature:
      0: x_start
      1: y_start
      2: x_end
      3: y_end
      4: dx
      5: dy
      6: length
      7: is_lane_line
      8: is_curb
      9: is_agent
      10: is_target
    """

    points = line.points.astype(np.float32)
    vectors = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx * dx + dy * dy)

        is_lane_line = float(line.kind == LineKind.LANE_LINE)
        is_curb = float(line.kind == LineKind.CURB)
        is_agent = float(line.kind == LineKind.AGENT_HISTORY)
        is_target = float(line.is_target)

        vectors.append(
            [
                x0,
                y0,
                x1,
                y1,
                dx,
                dy,
                length,
                is_lane_line,
                is_curb,
                is_agent,
                is_target,
            ]
        )
    return np.asarray(vectors, dtype=np.float32)


def scene_to_polylines(sample: SceneSample) -> Tuple[List[np.ndarray], int]:
    """
    把一个SceneSample转成polyline vector list

    输出：
        polylines:
        list of [Vi, F]
        每个元素对应一条Line转成的vectors

        target_index:
        目标它车历史轨迹在polylines里的index
    """
    polylines = []
    target_index = -1
    for line_idx, line in enumerate(sample.lines):
        vectors = line_to_vectors(line)
        polylines.append(vectors)
        if (
            line.kind == LineKind.AGENT_HISTORY
            and line.agent_id == sample.target_agent_id
            and line.is_target
        ):
            target_index = line_idx

    if target_index < 0:
        raise ValueError("Target agent history line was not found in SceneSample.")

    return polylines, target_index
