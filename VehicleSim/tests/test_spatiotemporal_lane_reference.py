from __future__ import annotations

import math

import numpy as np
import pytest

from sim2d.perception import PerceivedLaneSegment
from sim2d.planning.spatiotemporal_planner import (
    build_perception_lane_reference,
    local_reference_path_to_world,
)
from sim2d.types import VehicleState


def make_lane(
    *,
    lane_id: str,
    centerline: np.ndarray,
    confidence: float = 1.0,
) -> PerceivedLaneSegment:
    centerline = np.asarray(centerline, dtype=np.float64)
    left = centerline + np.array([0.0, 1.75], dtype=np.float64)
    right = centerline - np.array([0.0, 1.75], dtype=np.float64)
    return PerceivedLaneSegment(
        map_lane_id=lane_id,
        centerline=centerline,
        left_boundary=left,
        right_boundary=right,
        confidence=confidence,
    )


def test_selects_lane_nearest_to_ego() -> None:
    near = make_lane(
        lane_id="near",
        centerline=np.array(
            [[-2.0, 0.2], [0.0, 0.2], [10.0, 0.2]],
            dtype=np.float64,
        ),
    )
    far = make_lane(
        lane_id="far",
        centerline=np.array(
            [[-2.0, 4.0], [0.0, 4.0], [10.0, 4.0]],
            dtype=np.float64,
        ),
    )

    result = build_perception_lane_reference((far, near))

    assert result is not None
    assert result.lane_id == "near"
    assert np.allclose(result.reference_path[:, 1], 0.2)


def test_reverses_backward_centerline() -> None:
    lane = make_lane(
        lane_id="reverse",
        centerline=np.array(
            [[10.0, 0.0], [0.0, 0.0], [-2.0, 0.0]],
            dtype=np.float64,
        ),
    )

    result = build_perception_lane_reference((lane,))

    assert result is not None
    assert result.reference_path[-1, 0] > result.reference_path[0, 0]
    assert result.reference_path[0, 2] == pytest.approx(0.0)


def test_rejects_low_confidence_lane() -> None:
    lane = make_lane(
        lane_id="uncertain",
        centerline=np.array(
            [[-1.0, 0.0], [5.0, 0.0]],
            dtype=np.float64,
        ),
        confidence=0.2,
    )

    result = build_perception_lane_reference(
        (lane,),
        minimum_confidence=0.5,
    )

    assert result is None


def test_reference_path_has_arc_length_and_curvature() -> None:
    lane = make_lane(
        lane_id="curve",
        centerline=np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.2],
                [4.0, 0.8],
                [6.0, 1.8],
            ],
            dtype=np.float64,
        ),
    )

    result = build_perception_lane_reference((lane,))

    assert result is not None
    path = result.reference_path
    assert path.shape[1] == 5
    assert path[0, 3] == pytest.approx(0.0)
    assert np.all(np.diff(path[:, 3]) > 0.0)
    assert np.all(np.isfinite(path[:, 4]))


def test_local_reference_converts_to_world() -> None:
    lane = make_lane(
        lane_id="straight",
        centerline=np.array(
            [[0.0, 0.0], [2.0, 0.0]],
            dtype=np.float64,
        ),
    )
    result = build_perception_lane_reference((lane,))
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
