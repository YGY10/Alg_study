import numpy as np
import pytest

from sim2d.map import (
    OpenDriveArcGeometry,
    OpenDriveLane,
    OpenDriveLaneSection,
    OpenDriveLaneSide,
    OpenDriveLaneWidth,
    OpenDriveLineGeometry,
    OpenDriveRoad,
)


def test_line_geometry_validation() -> None:
    geometry = OpenDriveLineGeometry(
        s=0.0,
        x=1.0,
        y=2.0,
        heading=0.5,
        length=10.0,
    )

    assert geometry.length == pytest.approx(10.0)


def test_invalid_line_length_raises() -> None:
    with pytest.raises(ValueError):
        OpenDriveLineGeometry(
            s=0.0,
            x=0.0,
            y=0.0,
            heading=0.0,
            length=0.0,
        )


def test_arc_curvature_must_be_nonzero() -> None:
    with pytest.raises(ValueError):
        OpenDriveArcGeometry(
            s=0.0,
            x=0.0,
            y=0.0,
            heading=0.0,
            length=10.0,
            curvature=0.0,
        )


def test_lane_width_evaluate_scalar() -> None:
    width = OpenDriveLaneWidth(
        s_offset=0.0,
        a=3.0,
        b=0.5,
        c=0.0,
        d=0.0,
    )

    assert width.evaluate(2.0) == pytest.approx(4.0)


def test_lane_width_evaluate_array() -> None:
    width = OpenDriveLaneWidth(
        s_offset=0.0,
        a=3.0,
        b=0.5,
        c=0.0,
        d=0.0,
    )

    result = width.evaluate(
        np.array(
            [
                0.0,
                1.0,
                2.0,
            ],
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                3.0,
                3.5,
                4.0,
            ],
            dtype=np.float64,
        ),
    )


def test_lane_width_switches_records() -> None:
    lane = OpenDriveLane(
        lane_id=-1,
        side=OpenDriveLaneSide.RIGHT,
        lane_type="driving",
        widths=(
            OpenDriveLaneWidth(
                s_offset=0.0,
                a=3.0,
                b=0.0,
                c=0.0,
                d=0.0,
            ),
            OpenDriveLaneWidth(
                s_offset=5.0,
                a=4.0,
                b=0.0,
                c=0.0,
                d=0.0,
            ),
        ),
    )

    result = lane.width_at(
        np.array(
            [
                0.0,
                4.9,
                5.0,
                8.0,
            ],
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                3.0,
                3.0,
                4.0,
                4.0,
            ],
            dtype=np.float64,
        ),
    )


@pytest.mark.parametrize(
    (
        "lane_id",
        "side",
    ),
    [
        (
            -1,
            OpenDriveLaneSide.LEFT,
        ),
        (
            1,
            OpenDriveLaneSide.RIGHT,
        ),
        (
            1,
            OpenDriveLaneSide.CENTER,
        ),
    ],
)
def test_invalid_lane_side_and_id_raise(
    lane_id: int,
    side: OpenDriveLaneSide,
) -> None:
    with pytest.raises(ValueError):
        OpenDriveLane(
            lane_id=lane_id,
            side=side,
            lane_type="driving",
        )


def test_lane_section_lookup() -> None:
    lane = OpenDriveLane(
        lane_id=-1,
        side=OpenDriveLaneSide.RIGHT,
        lane_type="driving",
    )

    section = OpenDriveLaneSection(
        s=0.0,
        lanes=(lane,),
    )

    assert section.get_lane(-1) is lane

    with pytest.raises(KeyError):
        section.get_lane(-2)


def test_valid_road() -> None:
    road = OpenDriveRoad(
        road_id="1",
        name="test_road",
        length=20.0,
        geometries=(
            OpenDriveLineGeometry(
                s=0.0,
                x=0.0,
                y=0.0,
                heading=0.0,
                length=10.0,
            ),
            OpenDriveArcGeometry(
                s=10.0,
                x=10.0,
                y=0.0,
                heading=0.0,
                length=10.0,
                curvature=0.1,
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
                        lane_id=-1,
                        side=OpenDriveLaneSide.RIGHT,
                        lane_type="driving",
                        widths=(
                            OpenDriveLaneWidth(
                                s_offset=0.0,
                                a=3.5,
                                b=0.0,
                                c=0.0,
                                d=0.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert road.road_id == "1"
    assert len(road.geometries) == 2
    assert len(road.lane_sections) == 1


def test_unsorted_geometry_raises() -> None:
    with pytest.raises(ValueError):
        OpenDriveRoad(
            road_id="1",
            length=20.0,
            geometries=(
                OpenDriveLineGeometry(
                    s=10.0,
                    x=10.0,
                    y=0.0,
                    heading=0.0,
                    length=10.0,
                ),
                OpenDriveLineGeometry(
                    s=0.0,
                    x=0.0,
                    y=0.0,
                    heading=0.0,
                    length=10.0,
                ),
            ),
            lane_sections=(),
        )


def test_geometry_beyond_road_length_raises() -> None:
    with pytest.raises(ValueError):
        OpenDriveRoad(
            road_id="1",
            length=10.0,
            geometries=(
                OpenDriveLineGeometry(
                    s=0.0,
                    x=0.0,
                    y=0.0,
                    heading=0.0,
                    length=11.0,
                ),
            ),
            lane_sections=(),
        )
