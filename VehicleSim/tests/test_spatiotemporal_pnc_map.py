from __future__ import annotations

import numpy as np

from sim2d.perception import PerceivedLaneLine
from sim2d.planning.spatiotemporal_planner.pnc_map import (
    build_current_reference_line,
    build_reference_lines,
)


def _line(line_id: str, points: np.ndarray) -> PerceivedLaneLine:
    return PerceivedLaneLine(
        line_id=line_id,
        points=np.asarray(points, dtype=np.float64),
        confidence=1.0,
    )


def test_pnc_map_builds_center_reference_from_two_lane_lines() -> None:
    x = np.linspace(-5.0, 30.0, 71)
    left = _line("left", np.column_stack((x, np.full_like(x, 1.8))))
    right = _line("right", np.column_stack((x, np.full_like(x, -1.8))))

    references = build_reference_lines((left, right))

    assert len(references) == 1
    reference = references[0]
    assert np.max(np.abs(reference.reference_path[:, 1])) < 1e-9
    assert np.allclose(reference.left_boundary[:, 1], 1.8)
    assert np.allclose(reference.right_boundary[:, 1], -1.8)


def test_pnc_map_preserves_turn_geometry_from_connected_lane_lines() -> None:
    left = _line(
        "left",
        np.array(
            [
                [-5.0, 1.8],
                [8.0, 1.8],
                [9.5, 1.0],
                [10.0, 0.0],
                [10.0, -12.0],
            ]
        ),
    )
    right = _line(
        "right",
        np.array(
            [
                [-5.0, -1.8],
                [6.5, -1.8],
                [7.5, -2.6],
                [8.2, -4.0],
                [8.2, -12.0],
            ]
        ),
    )

    selected, references = build_current_reference_line((left, right))

    assert len(references) == 1
    assert selected is not None
    assert np.min(selected.reference_path[:, 1]) < -8.0
    assert np.max(np.abs(selected.reference_path[:, 2])) > 1.0


def test_pnc_map_builds_multiple_reference_candidates_from_three_lines() -> None:
    x = np.linspace(-5.0, 25.0, 61)
    lines = (
        _line("left", np.column_stack((x, np.full_like(x, 5.4)))),
        _line("middle", np.column_stack((x, np.full_like(x, 1.8)))),
        _line("right", np.column_stack((x, np.full_like(x, -1.8)))),
    )

    references = build_reference_lines(lines)

    assert len(references) == 2
    lateral_centers = sorted(
        float(np.median(reference.reference_path[:, 1]))
        for reference in references
    )
    assert np.allclose(lateral_centers, [0.0, 3.6])
