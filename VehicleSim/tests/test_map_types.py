import numpy as np
import pytest

from sim2d.map import (
    Lane,
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
    predecessor_ids: tuple[str, ...] = (),
    successor_ids: tuple[str, ...] = (),
) -> Lane:
    return Lane(
        lane_id=lane_id,
        lane_type=LaneType.DRIVING,
        centerline=make_polyline(y=0.0),
        left_boundary=make_polyline(y=1.75),
        right_boundary=make_polyline(y=-1.75),
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

    assert polyline.points[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "points",
    [
        np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
        np.array(
            [[0.0, 1.0, 2.0]],
            dtype=np.float64,
        ),
        np.array(
            [[0.0, 0.0]],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.0],
                [np.nan, 1.0],
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

    sidewalk_lane = Lane(
        lane_id="sidewalk",
        lane_type=LaneType.SIDEWALK,
        centerline=make_polyline(4.0),
        left_boundary=make_polyline(5.0),
        right_boundary=make_polyline(3.0),
    )

    road_network = RoadNetwork(
        lanes=(
            driving_lane,
            sidewalk_lane,
        ),
        source_type="manual",
    )

    assert road_network.driving_lanes() == (driving_lane,)
