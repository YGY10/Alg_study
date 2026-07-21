from __future__ import annotations

import numpy as np

from sim2d.map.topology_repair import repair_road_network_topology
from sim2d.map.types import Lane, LaneType, Polyline2D, RoadNetwork


def _lane(lane_id: str, points: list[list[float]]) -> Lane:
    center = np.asarray(points, dtype=np.float64)
    left = center + np.array([0.0, 1.0])
    right = center - np.array([0.0, 1.0])
    return Lane(
        lane_id=lane_id,
        lane_type=LaneType.DRIVING,
        centerline=Polyline2D(center),
        left_boundary=Polyline2D(left),
        right_boundary=Polyline2D(right),
    )


def test_repair_connects_continuous_lane_chain() -> None:
    network = RoadNetwork(
        lanes=(
            _lane("lane_1", [[0.0, 0.0], [10.0, 0.0]]),
            _lane("lane_2", [[10.05, 0.02], [20.0, 0.0]]),
            _lane("lane_3", [[20.02, 0.01], [30.0, 2.0]]),
        ),
        source_type="test",
    )

    repaired = repair_road_network_topology(network)

    assert repaired.get_lane("lane_1").successor_ids == ("lane_2",)
    assert repaired.get_lane("lane_2").predecessor_ids == ("lane_1",)
    assert repaired.get_lane("lane_2").successor_ids == ("lane_3",)
    assert repaired.get_lane("lane_3").predecessor_ids == ("lane_2",)
    assert repaired.metadata["topology_edge_count"] == 2


def test_repair_rejects_opposite_heading_candidate() -> None:
    network = RoadNetwork(
        lanes=(
            _lane("source", [[0.0, 0.0], [10.0, 0.0]]),
            _lane("opposite", [[10.02, 0.0], [0.0, 0.0]]),
        ),
        source_type="test",
    )

    repaired = repair_road_network_topology(network)

    assert repaired.get_lane("source").successor_ids == ()
    assert repaired.get_lane("opposite").predecessor_ids == ()


def test_repair_preserves_explicit_topology() -> None:
    lane_1 = _lane("lane_1", [[0.0, 0.0], [10.0, 0.0]])
    lane_2 = _lane("lane_2", [[50.0, 0.0], [60.0, 0.0]])

    lane_1 = Lane(
        lane_id=lane_1.lane_id,
        lane_type=lane_1.lane_type,
        centerline=lane_1.centerline,
        left_boundary=lane_1.left_boundary,
        right_boundary=lane_1.right_boundary,
        successor_ids=("lane_2",),
    )
    lane_2 = Lane(
        lane_id=lane_2.lane_id,
        lane_type=lane_2.lane_type,
        centerline=lane_2.centerline,
        left_boundary=lane_2.left_boundary,
        right_boundary=lane_2.right_boundary,
        predecessor_ids=("lane_1",),
    )

    repaired = repair_road_network_topology(
        RoadNetwork(
            lanes=(lane_1, lane_2),
            source_type="test",
        )
    )

    assert repaired.get_lane("lane_1").successor_ids == ("lane_2",)
    assert repaired.get_lane("lane_2").predecessor_ids == ("lane_1",)
    assert repaired.metadata["explicit_topology_edge_count"] == 1
