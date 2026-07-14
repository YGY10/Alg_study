from pathlib import Path

import numpy as np
import pytest

from sim2d.map import (
    LaneType,
    OpenDriveLane,
    OpenDriveLaneSection,
    OpenDriveLaneSide,
    OpenDriveLaneWidth,
    OpenDriveLineGeometry,
    OpenDriveRoad,
    convert_road,
    load_opendrive_road_network,
    make_lane_uid,
    map_lane_type,
)


def make_width(
    value: float,
) -> OpenDriveLaneWidth:
    return OpenDriveLaneWidth(
        s_offset=0.0,
        a=value,
        b=0.0,
        c=0.0,
        d=0.0,
    )


def make_two_way_road() -> OpenDriveRoad:
    return OpenDriveRoad(
        road_id="1",
        name="two_way",
        length=20.0,
        geometries=(
            OpenDriveLineGeometry(
                s=0.0,
                x=0.0,
                y=0.0,
                heading=0.0,
                length=20.0,
            ),
        ),
        lane_sections=(
            OpenDriveLaneSection(
                s=0.0,
                lanes=(
                    OpenDriveLane(
                        lane_id=0,
                        side=OpenDriveLaneSide.CENTER,
                        lane_type="none",
                    ),
                    OpenDriveLane(
                        lane_id=1,
                        side=OpenDriveLaneSide.LEFT,
                        lane_type="driving",
                        widths=(make_width(3.5),),
                    ),
                    OpenDriveLane(
                        lane_id=-1,
                        side=OpenDriveLaneSide.RIGHT,
                        lane_type="driving",
                        widths=(make_width(3.5),),
                    ),
                ),
            ),
        ),
    )


def test_map_lane_type() -> None:
    assert map_lane_type("driving") is LaneType.DRIVING

    assert map_lane_type("shoulder") is LaneType.SHOULDER

    assert map_lane_type("biking") is LaneType.BIKE

    assert map_lane_type("something_new") is LaneType.UNKNOWN


def test_make_lane_uid() -> None:
    result = make_lane_uid(
        road_id="12",
        section_index=3,
        lane_id=-2,
    )

    assert result == ("road_12/section_3/lane_-2")


def test_convert_right_lane_direction() -> None:
    lanes = convert_road(
        road=make_two_way_road(),
        sample_step=1.0,
    )

    right_lane = next(lane for lane in lanes if lane.metadata["lane_id"] == -1)

    assert right_lane.lane_type is LaneType.DRIVING

    # 右侧车道沿 road s 正方向行驶。
    assert right_lane.centerline.start[0] == pytest.approx(0.0)

    assert right_lane.centerline.end[0] == pytest.approx(20.0)

    np.testing.assert_allclose(
        right_lane.centerline.points[:, 1],
        -1.75,
    )

    # 行驶方向为 +x 时，左边界更靠近参考线。
    np.testing.assert_allclose(
        right_lane.left_boundary.points[:, 1],
        0.0,
    )

    np.testing.assert_allclose(
        right_lane.right_boundary.points[:, 1],
        -3.5,
    )


def test_convert_left_lane_reverses_direction() -> None:
    lanes = convert_road(
        road=make_two_way_road(),
        sample_step=1.0,
    )

    left_lane = next(lane for lane in lanes if lane.metadata["lane_id"] == 1)

    # 左侧车道在 RHT 下逆 road s 行驶。
    assert left_lane.centerline.start[0] == pytest.approx(20.0)

    assert left_lane.centerline.end[0] == pytest.approx(0.0)

    np.testing.assert_allclose(
        left_lane.centerline.points[:, 1],
        1.75,
    )

    # 行驶方向为 -x 时，
    # 世界坐标更高的 y 方向是车辆右侧。
    np.testing.assert_allclose(
        left_lane.left_boundary.points[:, 1],
        3.5,
    )

    np.testing.assert_allclose(
        left_lane.right_boundary.points[:, 1],
        0.0,
    )


def test_convert_road_creates_unique_lane_ids() -> None:
    lanes = convert_road(
        road=make_two_way_road(),
        sample_step=1.0,
    )

    lane_ids = {lane.lane_id for lane in lanes}

    assert lane_ids == {
        "road_1/section_0/lane_1",
        "road_1/section_0/lane_-1",
    }


SIMPLE_XODR = """
<OpenDRIVE>
    <header
        revMajor="1"
        revMinor="7"
        name="converter_test"
        version="1.00"
        date=""
        north="0"
        south="0"
        east="0"
        west="0"
    />

    <road
        name="Main Road"
        length="10.0"
        id="1"
        junction="-1"
    >
        <planView>
            <geometry
                s="0.0"
                x="0.0"
                y="0.0"
                hdg="0.0"
                length="10.0"
            >
                <line/>
            </geometry>
        </planView>

        <lanes>
            <laneSection s="0.0">
                <center>
                    <lane
                        id="0"
                        type="none"
                        level="false"
                    />
                </center>

                <right>
                    <lane
                        id="-1"
                        type="driving"
                        level="false"
                    >
                        <width
                            sOffset="0.0"
                            a="3.5"
                            b="0.0"
                            c="0.0"
                            d="0.0"
                        />
                    </lane>
                </right>
            </laneSection>
        </lanes>
    </road>
</OpenDRIVE>
"""


def test_load_opendrive_road_network(
    tmp_path: Path,
) -> None:
    path = tmp_path / "simple.xodr"

    path.write_text(
        SIMPLE_XODR,
        encoding="utf-8",
    )

    road_network = load_opendrive_road_network(
        path,
        sample_step=1.0,
    )

    assert road_network.source_type == "opendrive"

    assert road_network.source_name == "simple.xodr"

    assert road_network.lane_count == 1

    lane = road_network.get_lane("road_1/section_0/lane_-1")

    assert lane.lane_type is LaneType.DRIVING

    assert lane.centerline.length == pytest.approx(10.0)

    assert road_network.metadata["road_count"] == 1

    assert road_network.metadata["traffic_rule"] == "RHT"


def test_invalid_sample_step_raises() -> None:
    with pytest.raises(ValueError):
        load_opendrive_road_network(
            "missing.xodr",
            sample_step=0.0,
        )
