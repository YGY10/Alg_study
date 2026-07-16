from __future__ import annotations

import numpy as np
import pytest

from sim2d.map.route import (
    NoRouteError,
    build_same_lane_route,
)
from sim2d.map.types import (
    Lane,
    LaneType,
    Polyline2D,
    RoadNetwork,
)


def make_lane(
    lane_id: str,
    points: np.ndarray,
    *,
    y_half_width: float = 1.75,
) -> Lane:
    centerline = np.asarray(
        points,
        dtype=np.float64,
    )

    left = centerline.copy()
    right = centerline.copy()

    left[:, 1] += y_half_width
    right[:, 1] -= y_half_width

    return Lane(
        lane_id=lane_id,
        lane_type=LaneType.DRIVING,
        centerline=Polyline2D(centerline),
        left_boundary=Polyline2D(left),
        right_boundary=Polyline2D(right),
    )


def make_network() -> RoadNetwork:
    lane_a = make_lane(
        "lane_a",
        np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 2.0],
                [15.0, 2.0],
            ],
            dtype=np.float64,
        ),
    )

    lane_b = make_lane(
        "lane_b",
        np.array(
            [
                [0.0, 10.0],
                [10.0, 10.0],
            ],
            dtype=np.float64,
        ),
    )

    return RoadNetwork(
        lanes=(
            lane_a,
            lane_b,
        ),
        source_type="test",
    )


def test_build_same_lane_route_forward() -> None:
    route = build_same_lane_route(
        make_network(),
        start_x=1.0,
        start_y=0.2,
        goal_x=14.0,
        goal_y=2.1,
        snap_distance=1.0,
    )

    assert route.lane_ids == ("lane_a",)
    assert route.reference_path.shape[1] == 5
    assert np.all(np.diff(route.reference_path[:, 3]) > 0.0)


def test_route_uses_exact_projection_endpoints() -> None:
    network = make_network()

    start = network.nearest_lane_point(
        x=2.5,
        y=0.4,
        max_distance=1.0,
    )
    goal = network.nearest_lane_point(
        x=12.0,
        y=2.2,
        max_distance=1.0,
    )

    assert start is not None
    assert goal is not None

    route = build_same_lane_route(
        network,
        start_x=2.5,
        start_y=0.4,
        goal_x=12.0,
        goal_y=2.2,
        snap_distance=1.0,
    )

    np.testing.assert_allclose(
        route.reference_path[0, 0:2],
        start.point,
    )
    np.testing.assert_allclose(
        route.reference_path[-1, 0:2],
        goal.point,
    )


def test_goal_behind_start_raises() -> None:
    with pytest.raises(
        NoRouteError,
        match="behind",
    ) as error_info:
        build_same_lane_route(
            make_network(),
            start_x=12.0,
            start_y=2.0,
            goal_x=2.0,
            goal_y=0.0,
            snap_distance=1.0,
        )

    assert error_info.value.reason == "goal_behind_start"


def test_different_lane_raises() -> None:
    with pytest.raises(NoRouteError) as error_info:
        build_same_lane_route(
            make_network(),
            start_x=1.0,
            start_y=0.0,
            goal_x=5.0,
            goal_y=10.0,
            snap_distance=1.0,
        )

    assert error_info.value.reason == "different_lane"


def test_start_projection_failure_raises() -> None:
    with pytest.raises(NoRouteError) as error_info:
        build_same_lane_route(
            make_network(),
            start_x=-100.0,
            start_y=-100.0,
            goal_x=5.0,
            goal_y=0.0,
            snap_distance=1.0,
        )

    assert error_info.value.reason == "start_projection_failed"


def test_goal_projection_failure_raises() -> None:
    with pytest.raises(NoRouteError) as error_info:
        build_same_lane_route(
            make_network(),
            start_x=1.0,
            start_y=0.0,
            goal_x=100.0,
            goal_y=100.0,
            snap_distance=1.0,
        )

    assert error_info.value.reason == "goal_projection_failed"


def test_same_segment_route() -> None:
    route = build_same_lane_route(
        make_network(),
        start_x=1.0,
        start_y=0.0,
        goal_x=4.0,
        goal_y=0.0,
        snap_distance=0.5,
    )

    assert route.reference_path.shape == (
        2,
        5,
    )

    np.testing.assert_allclose(
        route.reference_path[:, 0:2],
        np.array(
            [
                [1.0, 0.0],
                [4.0, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        route.reference_path[:, 4],
        0.0,
    )
