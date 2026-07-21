import numpy as np

from sim2d.map import (
    Lane,
    LaneType,
    Polyline2D,
    RoadNetwork,
)
from sim2d.world import (
    RoadDeformationConfig,
    deform_road_network,
)


def _polyline(points):
    return Polyline2D(
        np.asarray(points, dtype=np.float64)
    )


def test_deformation_preserves_connected_lane_endpoint():
    lane_1 = Lane(
        lane_id="lane_1",
        lane_type=LaneType.DRIVING,
        centerline=_polyline(((0.0, 0.0), (10.0, 0.0))),
        left_boundary=_polyline(((0.0, 1.5), (10.0, 1.5))),
        right_boundary=_polyline(((0.0, -1.5), (10.0, -1.5))),
        successor_ids=("lane_2",),
    )

    lane_2 = Lane(
        lane_id="lane_2",
        lane_type=LaneType.DRIVING,
        centerline=_polyline(((10.0, 0.0), (20.0, 4.0))),
        left_boundary=_polyline(((10.0, 1.5), (19.4, 5.4))),
        right_boundary=_polyline(((10.0, -1.5), (20.6, 2.6))),
        predecessor_ids=("lane_1",),
    )

    road_network = RoadNetwork(
        lanes=(lane_1, lane_2),
        source_type="test",
    )

    world_lanes = deform_road_network(
        road_network,
        RoadDeformationConfig(
            offset_x=0.8,
            offset_y=-0.4,
            yaw_offset=0.05,
            longitudinal_scale=1.01,
            lateral_scale=0.98,
            local_longitudinal_amplitude=0.3,
            local_lateral_amplitude=0.8,
            local_wavelength=30.0,
        ),
    )

    world_lane_1, world_lane_2 = world_lanes

    np.testing.assert_allclose(
        world_lane_1.centerline[-1],
        world_lane_2.centerline[0],
        atol=1.0e-12,
    )

    assert world_lane_1.successor_ids == ("lane_2",)
    assert world_lane_2.predecessor_ids == ("lane_1",)


def test_deformation_varies_continuously_along_lane():
    lane = Lane(
        lane_id="lane",
        lane_type=LaneType.DRIVING,
        centerline=_polyline(((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))),
        left_boundary=_polyline(((0.0, 1.5), (10.0, 1.5), (20.0, 1.5))),
        right_boundary=_polyline(((0.0, -1.5), (10.0, -1.5), (20.0, -1.5))),
    )

    road_network = RoadNetwork(
        lanes=(lane,),
        source_type="test",
    )

    world_lane = deform_road_network(
        road_network,
        RoadDeformationConfig(
            local_lateral_amplitude=1.0,
            local_wavelength=20.0,
        ),
    )[0]

    offsets = world_lane.centerline - lane.centerline.points

    assert not np.allclose(offsets[0], offsets[1])
