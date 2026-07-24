from __future__ import annotations

import numpy as np

from sim2d.perception import PerceivedLaneLine
from sim2d.planning.spatiotemporal_planner.lane_fragment_stitching import (
    LaneFragmentStitchingConfig,
    stitch_lane_line_fragments,
)
from sim2d.planning.spatiotemporal_planner.pnc_map import build_reference_lines


def _line(line_id: str, points: list[list[float]]) -> PerceivedLaneLine:
    return PerceivedLaneLine(
        line_id=line_id,
        points=np.asarray(points, dtype=np.float64),
        confidence=1.0,
    )


def test_stitches_front_and_rear_fragments_before_corridor_pairing() -> None:
    line_1 = _line("line_1", [[-10.0, 2.0], [-4.0, 2.0], [2.0, 2.0]])
    line_2 = _line("line_2", [[-10.0, -2.0], [-4.0, -2.0], [2.0, -2.0]])
    line_3 = _line("line_3", [[4.0, 2.0], [8.0, 3.2], [12.0, 6.0]])
    line_4 = _line("line_4", [[4.0, -2.0], [8.0, -0.8], [12.0, 2.0]])

    stitched, debug = stitch_lane_line_fragments(
        (line_1, line_2, line_3, line_4),
        config=LaneFragmentStitchingConfig(maximum_join_distance=4.0),
    )

    assert debug.stitched_chain_count == 2
    assert len(stitched) == 2
    ids = {line.line_id for line in stitched}
    assert "stitch[line_1+line_3]" in ids
    assert "stitch[line_2+line_4]" in ids

    references = build_reference_lines(
        stitched,
        minimum_lane_width=2.0,
        maximum_lane_width=6.0,
        output_point_count=81,
    )
    assert len(references) == 1
    assert references[0].reference_path[-1, 0] > 10.0
    assert references[0].reference_path[-1, 1] > 3.0


def test_does_not_join_left_and_right_boundaries_across_lane_width() -> None:
    left = _line("left", [[-5.0, 2.0], [0.0, 2.0], [5.0, 2.0]])
    right = _line("right", [[-5.0, -2.0], [0.0, -2.0], [5.0, -2.0]])

    stitched, debug = stitch_lane_line_fragments((left, right))

    assert debug.stitched_chain_count == 0
    assert len(stitched) == 2
    assert {line.line_id for line in stitched} == {"left", "right"}


def test_point_order_does_not_prevent_fragment_connection() -> None:
    rear = _line("rear", [[2.0, 1.8], [-4.0, 1.8], [-10.0, 1.8]])
    front = _line("front", [[12.0, 5.0], [8.0, 2.8], [4.0, 1.8]])

    stitched, debug = stitch_lane_line_fragments(
        (rear, front),
        config=LaneFragmentStitchingConfig(maximum_join_distance=4.0),
    )

    assert debug.stitched_chain_count == 1
    assert len(stitched) == 1
    assert stitched[0].points.shape[0] > rear.points.shape[0] + front.points.shape[0]
    assert min(stitched[0].points[:, 0]) <= -10.0
    assert max(stitched[0].points[:, 0]) >= 12.0
