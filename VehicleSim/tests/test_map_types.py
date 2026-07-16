import math

import numpy as np
import pytest

from sim2d.map import (
    Lane,
    LaneProjection,
    LaneType,
    Polyline2D,
    RoadNetwork,
)


def make_polyline(
    y: float = 0.0,
) -> Polyline2D:
    return Polyline2D(
        points=np.array(
            [
                [0.0, y],
                [5.0, y],
                [10.0, y],
            ],
            dtype=np.float64,
        )
    )


def make_lane(
    lane_id: str = "lane_001",
    *,
    center_y: float = 0.0,
    lane_type: LaneType = LaneType.DRIVING,
    predecessor_ids: tuple[str, ...] = (),
    successor_ids: tuple[str, ...] = (),
) -> Lane:
    return Lane(
        lane_id=lane_id,
        lane_type=lane_type,
        centerline=make_polyline(y=center_y),
        left_boundary=make_polyline(y=center_y + 1.75),
        right_boundary=make_polyline(y=center_y - 1.75),
        speed_limit=13.89,
        predecessor_ids=predecessor_ids,
        successor_ids=successor_ids,
    )


def test_polyline_length() -> None:
    polyline = make_polyline()

    assert polyline.point_count == 3

    assert polyline.length == pytest.approx(10.0)


def test_polyline_copies_input_array() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    polyline = Polyline2D(points=points)

    points[0, 0] = 100.0

    assert polyline.points[
        0,
        0,
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "points",
    [
        np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                ]
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [
                    0.0,
                    0.0,
                ]
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    np.nan,
                    1.0,
                ],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_invalid_polyline_raises(
    points: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        Polyline2D(points=points)


def test_lane_validation() -> None:
    lane = make_lane()

    assert lane.lane_id == "lane_001"

    assert lane.lane_type is LaneType.DRIVING

    assert lane.speed_limit == pytest.approx(13.89)


def test_empty_lane_id_raises() -> None:
    with pytest.raises(ValueError):
        make_lane(lane_id="")


def test_invalid_speed_limit_raises() -> None:
    with pytest.raises(ValueError):
        Lane(
            lane_id="lane_001",
            lane_type=LaneType.DRIVING,
            centerline=make_polyline(),
            left_boundary=make_polyline(1.75),
            right_boundary=make_polyline(-1.75),
            speed_limit=0.0,
        )


def test_road_network_lane_lookup() -> None:
    lane = make_lane()

    road_network = RoadNetwork(
        lanes=(lane,),
        source_type="manual",
        source_name="unit_test",
    )

    assert road_network.lane_count == 1

    assert road_network.get_lane("lane_001") is lane

    assert road_network.find_lane("missing") is None


def test_duplicate_lane_ids_raise() -> None:
    lane_a = make_lane(lane_id="lane_001")

    lane_b = make_lane(lane_id="lane_001")

    with pytest.raises(ValueError):
        RoadNetwork(
            lanes=(
                lane_a,
                lane_b,
            ),
            source_type="manual",
        )


def test_unknown_successor_raises() -> None:
    lane = make_lane(
        lane_id="lane_001",
        successor_ids=("lane_missing",),
    )

    with pytest.raises(ValueError):
        RoadNetwork(
            lanes=(lane,),
            source_type="manual",
        )


def test_valid_lane_topology() -> None:
    lane_a = make_lane(
        lane_id="lane_a",
        successor_ids=("lane_b",),
    )

    lane_b = make_lane(
        lane_id="lane_b",
        predecessor_ids=("lane_a",),
    )

    road_network = RoadNetwork(
        lanes=(
            lane_a,
            lane_b,
        ),
        source_type="manual",
    )

    assert road_network.get_lane("lane_a").successor_ids == ("lane_b",)

    assert road_network.get_lane("lane_b").predecessor_ids == ("lane_a",)


def test_driving_lanes_filter() -> None:
    driving_lane = make_lane(lane_id="driving")

    sidewalk_lane = make_lane(
        lane_id="sidewalk",
        center_y=4.0,
        lane_type=LaneType.SIDEWALK,
    )

    road_network = RoadNetwork(
        lanes=(
            driving_lane,
            sidewalk_lane,
        ),
        source_type="manual",
    )

    assert road_network.driving_lanes() == (driving_lane,)


def test_nearest_lane_projection_on_straight_lane() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=2.0,
        y=3.0,
    )

    assert projection is not None

    assert isinstance(
        projection,
        LaneProjection,
    )

    assert projection.lane_id == "lane_001"

    np.testing.assert_allclose(
        projection.point,
        np.array(
            [
                2.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    )

    assert projection.distance == pytest.approx(3.0)

    assert projection.lateral_offset == pytest.approx(3.0)

    assert projection.yaw == pytest.approx(0.0)

    assert projection.segment_index == 0

    assert projection.segment_ratio == pytest.approx(0.4)

    assert projection.arc_length == pytest.approx(2.0)


def test_projection_right_side_has_negative_offset() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=7.0,
        y=-2.5,
    )

    assert projection is not None

    assert projection.distance == pytest.approx(2.5)

    assert projection.lateral_offset == pytest.approx(-2.5)

    assert projection.segment_index == 1

    assert projection.segment_ratio == pytest.approx(0.4)

    assert projection.arc_length == pytest.approx(7.0)


def test_projection_clamps_to_polyline_endpoint() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=14.0,
        y=3.0,
    )

    assert projection is not None

    np.testing.assert_allclose(
        projection.point,
        np.array(
            [
                10.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    )

    assert projection.distance == pytest.approx(5.0)

    assert projection.segment_index == 1

    assert projection.segment_ratio == pytest.approx(1.0)

    assert projection.arc_length == pytest.approx(10.0)


def test_nearest_lane_selects_closest_lane() -> None:
    lane_a = make_lane(
        lane_id="lane_a",
        center_y=0.0,
    )

    lane_b = make_lane(
        lane_id="lane_b",
        center_y=8.0,
    )

    road_network = RoadNetwork(
        lanes=(
            lane_a,
            lane_b,
        ),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=4.0,
        y=6.5,
    )

    assert projection is not None

    assert projection.lane_id == "lane_b"

    assert projection.distance == pytest.approx(1.5)


def test_projection_defaults_to_driving_lanes() -> None:
    driving_lane = make_lane(
        lane_id="driving",
        center_y=0.0,
        lane_type=LaneType.DRIVING,
    )

    sidewalk_lane = make_lane(
        lane_id="sidewalk",
        center_y=5.0,
        lane_type=LaneType.SIDEWALK,
    )

    road_network = RoadNetwork(
        lanes=(
            driving_lane,
            sidewalk_lane,
        ),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=3.0,
        y=5.1,
    )

    assert projection is not None

    assert projection.lane_id == "driving"


def test_projection_can_query_specific_lane_type() -> None:
    driving_lane = make_lane(
        lane_id="driving",
        center_y=0.0,
        lane_type=LaneType.DRIVING,
    )

    sidewalk_lane = make_lane(
        lane_id="sidewalk",
        center_y=5.0,
        lane_type=LaneType.SIDEWALK,
    )

    road_network = RoadNetwork(
        lanes=(
            driving_lane,
            sidewalk_lane,
        ),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=3.0,
        y=5.1,
        lane_types=(LaneType.SIDEWALK,),
    )

    assert projection is not None

    assert projection.lane_id == "sidewalk"

    assert projection.distance == pytest.approx(0.1)


def test_projection_lane_types_none_queries_all() -> None:
    sidewalk_lane = make_lane(
        lane_id="sidewalk",
        center_y=5.0,
        lane_type=LaneType.SIDEWALK,
    )

    road_network = RoadNetwork(
        lanes=(sidewalk_lane,),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=2.0,
        y=5.0,
        lane_types=None,
    )

    assert projection is not None

    assert projection.lane_id == "sidewalk"


def test_projection_respects_max_distance() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=2.0,
        y=4.0,
        max_distance=3.0,
    )

    assert projection is None


def test_projection_at_max_distance_is_allowed() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=2.0,
        y=3.0,
        max_distance=3.0,
    )

    assert projection is not None


def test_projection_on_vertical_segment() -> None:
    lane = Lane(
        lane_id="vertical",
        lane_type=LaneType.DRIVING,
        centerline=Polyline2D(
            points=np.array(
                [
                    [1.0, 1.0],
                    [1.0, 6.0],
                ],
                dtype=np.float64,
            )
        ),
        left_boundary=Polyline2D(
            points=np.array(
                [
                    [0.0, 1.0],
                    [0.0, 6.0],
                ],
                dtype=np.float64,
            )
        ),
        right_boundary=Polyline2D(
            points=np.array(
                [
                    [2.0, 1.0],
                    [2.0, 6.0],
                ],
                dtype=np.float64,
            )
        ),
    )

    road_network = RoadNetwork(
        lanes=(lane,),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=-1.0,
        y=4.0,
    )

    assert projection is not None

    np.testing.assert_allclose(
        projection.point,
        np.array(
            [
                1.0,
                4.0,
            ],
            dtype=np.float64,
        ),
    )

    assert projection.yaw == pytest.approx(math.pi / 2.0)

    # 行驶方向朝 +y，世界坐标 -x 位于车辆左侧。
    assert projection.lateral_offset == pytest.approx(2.0)

    assert projection.arc_length == pytest.approx(3.0)


def test_projection_empty_lane_type_filter_returns_none() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=0.0,
        y=0.0,
        lane_types=(),
    )

    assert projection is None


def test_projection_with_no_lanes_returns_none() -> None:
    road_network = RoadNetwork(
        lanes=(),
        source_type="manual",
    )

    projection = road_network.nearest_lane_point(
        x=0.0,
        y=0.0,
    )

    assert projection is None


@pytest.mark.parametrize(
    (
        "x",
        "y",
    ),
    [
        (
            np.nan,
            0.0,
        ),
        (
            0.0,
            np.inf,
        ),
    ],
)
def test_projection_invalid_query_raises(
    x: float,
    y: float,
) -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    with pytest.raises(ValueError):
        road_network.nearest_lane_point(
            x=x,
            y=y,
        )


def test_projection_negative_max_distance_raises() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    with pytest.raises(ValueError):
        road_network.nearest_lane_point(
            x=0.0,
            y=0.0,
            max_distance=-1.0,
        )


def test_projection_invalid_lane_type_raises() -> None:
    road_network = RoadNetwork(
        lanes=(make_lane(),),
        source_type="manual",
    )

    with pytest.raises(TypeError):
        road_network.nearest_lane_point(
            x=0.0,
            y=0.0,
            lane_types=("driving",),
        )
