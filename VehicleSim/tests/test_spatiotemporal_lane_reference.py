from __future__ import annotations

import math

import numpy as np
import pytest

from sim2d.perception import PerceivedLaneLine
from sim2d.planning.spatiotemporal_planner import (
    build_perception_lane_reference,
    local_reference_path_to_world,
)
from sim2d.types import VehicleState


def make_boundaries(
    *,
    centerline: np.ndarray,
    confidence: float = 1.0,
) -> tuple[PerceivedLaneLine, PerceivedLaneLine]:
    centerline = np.asarray(centerline, dtype=np.float64)
    left = PerceivedLaneLine(
        line_id="left",
        points=centerline + np.array([0.0, 1.75]),
        confidence=confidence,
    )
    right = PerceivedLaneLine(
        line_id="right",
        points=centerline - np.array([0.0, 1.75]),
        confidence=confidence,
    )
    return left, right


def test_builds_reference_nearest_to_ego() -> None:
    far_left, far_right = make_boundaries(
        centerline=np.array([[-2.0, 4.0], [0.0, 4.0], [10.0, 4.0]])
    )
    near_left, near_right = make_boundaries(
        centerline=np.array([[-2.0, 0.2], [0.0, 0.2], [10.0, 0.2]])
    )
    far_left = PerceivedLaneLine("far_left", far_left.points)
    far_right = PerceivedLaneLine("far_right", far_right.points)
    near_left = PerceivedLaneLine("near_left", near_left.points)
    near_right = PerceivedLaneLine("near_right", near_right.points)

    result = build_perception_lane_reference(
        (far_left, far_right, near_left, near_right)
    )

    assert result is not None
    assert np.allclose(result.reference_path[:, 1], 0.2, atol=1e-6)


def test_reverses_backward_lane_lines() -> None:
    lines = make_boundaries(
        centerline=np.array([[10.0, 0.0], [0.0, 0.0], [-2.0, 0.0]])
    )

    result = build_perception_lane_reference(lines)

    assert result is not None
    assert result.reference_path[-1, 0] > result.reference_path[0, 0]
    assert result.reference_path[0, 2] == pytest.approx(0.0)


def test_rejects_low_confidence_lane_lines() -> None:
    lines = make_boundaries(
        centerline=np.array([[-1.0, 0.0], [5.0, 0.0]]),
        confidence=0.2,
    )

    result = build_perception_lane_reference(
        lines,
        minimum_confidence=0.5,
    )

    assert result is None


def test_reference_path_has_arc_length_and_curvature() -> None:
    lines = make_boundaries(
        centerline=np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.2],
                [4.0, 0.8],
                [6.0, 1.8],
            ]
        )
    )

    result = build_perception_lane_reference(lines)

    assert result is not None
    path = result.reference_path
    assert path.shape[1] == 5
    assert path[0, 3] == pytest.approx(0.0)
    assert np.all(np.diff(path[:, 3]) > 0.0)
    assert np.all(np.isfinite(path[:, 4]))


def test_local_reference_converts_to_world() -> None:
    lines = make_boundaries(
        centerline=np.array([[0.0, 0.0], [2.0, 0.0]])
    )
    result = build_perception_lane_reference(lines)
    assert result is not None

    world = local_reference_path_to_world(
        result.reference_path,
        VehicleState(
            x=10.0,
            y=20.0,
            yaw=math.pi / 2.0,
            speed=0.0,
        ),
    )

    assert world[0, 0] == pytest.approx(10.0)
    assert world[0, 1] == pytest.approx(20.0)
    assert world[-1, 0] == pytest.approx(10.0)
    assert world[-1, 1] == pytest.approx(22.0)
    assert world[0, 2] == pytest.approx(math.pi / 2.0)
