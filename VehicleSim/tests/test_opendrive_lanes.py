import numpy as np
import pytest

from sim2d.map import (
    OpenDriveLane,
    OpenDriveLaneSection,
    OpenDriveLaneSide,
    OpenDriveLaneWidth,
    OpenDriveLineGeometry,
    OpenDriveRoad,
    extract_reference_interval,
    interpolate_reference_line,
    sample_lane_section,
    sample_road_reference_line,
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


def make_straight_road(
    lanes: tuple[OpenDriveLane, ...],
    *,
    length: float = 20.0,
) -> OpenDriveRoad:
    return OpenDriveRoad(
        road_id="road_001",
        length=length,
        geometries=(
            OpenDriveLineGeometry(
                s=0.0,
                x=0.0,
                y=0.0,
                heading=0.0,
                length=length,
            ),
        ),
        lane_sections=(
            OpenDriveLaneSection(
                s=0.0,
                lanes=lanes,
            ),
        ),
    )


def test_interpolate_reference_line() -> None:
    road = make_straight_road(lanes=())

    reference = sample_road_reference_line(
        road=road,
        step=2.0,
    )

    result = interpolate_reference_line(
        reference_line=reference,
        query_s=np.array(
            [
                1.0,
                3.0,
                7.5,
            ],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        result[:, 0],
        np.array(
            [
                1.0,
                3.0,
                7.5,
            ],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        result[:, 1],
        result[:, 0],
    )

    np.testing.assert_allclose(
        result[:, 2],
        0.0,
    )


def test_extract_interval_adds_exact_endpoints() -> None:
    road = make_straight_road(lanes=())

    reference = sample_road_reference_line(
        road=road,
        step=2.0,
    )

    result = extract_reference_interval(
        reference_line=reference,
        s_start=1.0,
        s_end=7.0,
    )

    assert result[0, 0] == pytest.approx(1.0)

    assert result[-1, 0] == pytest.approx(7.0)

    np.testing.assert_allclose(
        result[:, 1],
        result[:, 0],
    )


def test_single_left_lane_geometry() -> None:
    road = make_straight_road(
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
                widths=(make_width(4.0),),
            ),
        )
    )

    reference = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    samples = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )

    assert len(samples) == 1

    lane = samples[0]

    assert lane.lane_id == 1

    np.testing.assert_allclose(
        lane.inner_boundary[:, 1],
        0.0,
    )

    np.testing.assert_allclose(
        lane.centerline[:, 1],
        2.0,
    )

    np.testing.assert_allclose(
        lane.outer_boundary[:, 1],
        4.0,
    )

    np.testing.assert_allclose(
        lane.width,
        4.0,
    )


def test_single_right_lane_geometry() -> None:
    road = make_straight_road(
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
                widths=(make_width(3.5),),
            ),
        )
    )

    reference = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    samples = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )

    assert len(samples) == 1

    lane = samples[0]

    assert lane.lane_id == -1

    np.testing.assert_allclose(
        lane.inner_boundary[:, 1],
        0.0,
    )

    np.testing.assert_allclose(
        lane.centerline[:, 1],
        -1.75,
    )

    np.testing.assert_allclose(
        lane.outer_boundary[:, 1],
        -3.5,
    )


def test_multiple_left_lanes_accumulate_width() -> None:
    road = make_straight_road(
        lanes=(
            OpenDriveLane(
                lane_id=1,
                side=OpenDriveLaneSide.LEFT,
                lane_type="driving",
                widths=(make_width(3.0),),
            ),
            OpenDriveLane(
                lane_id=2,
                side=OpenDriveLaneSide.LEFT,
                lane_type="driving",
                widths=(make_width(4.0),),
            ),
        )
    )

    reference = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    samples = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )

    lane_1 = next(lane for lane in samples if lane.lane_id == 1)

    lane_2 = next(lane for lane in samples if lane.lane_id == 2)

    np.testing.assert_allclose(
        lane_1.centerline[:, 1],
        1.5,
    )

    np.testing.assert_allclose(
        lane_1.outer_boundary[:, 1],
        3.0,
    )

    np.testing.assert_allclose(
        lane_2.inner_boundary[:, 1],
        3.0,
    )

    np.testing.assert_allclose(
        lane_2.centerline[:, 1],
        5.0,
    )

    np.testing.assert_allclose(
        lane_2.outer_boundary[:, 1],
        7.0,
    )


def test_multiple_right_lanes_accumulate_width() -> None:
    road = make_straight_road(
        lanes=(
            OpenDriveLane(
                lane_id=-1,
                side=OpenDriveLaneSide.RIGHT,
                lane_type="driving",
                widths=(make_width(3.0),),
            ),
            OpenDriveLane(
                lane_id=-2,
                side=OpenDriveLaneSide.RIGHT,
                lane_type="driving",
                widths=(make_width(4.0),),
            ),
        )
    )

    reference = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    samples = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )

    lane_1 = next(lane for lane in samples if lane.lane_id == -1)

    lane_2 = next(lane for lane in samples if lane.lane_id == -2)

    np.testing.assert_allclose(
        lane_1.centerline[:, 1],
        -1.5,
    )

    np.testing.assert_allclose(
        lane_1.outer_boundary[:, 1],
        -3.0,
    )

    np.testing.assert_allclose(
        lane_2.inner_boundary[:, 1],
        -3.0,
    )

    np.testing.assert_allclose(
        lane_2.centerline[:, 1],
        -5.0,
    )

    np.testing.assert_allclose(
        lane_2.outer_boundary[:, 1],
        -7.0,
    )


def test_variable_width_lane() -> None:
    lane = OpenDriveLane(
        lane_id=1,
        side=OpenDriveLaneSide.LEFT,
        lane_type="driving",
        widths=(
            OpenDriveLaneWidth(
                s_offset=0.0,
                a=2.0,
                b=0.1,
                c=0.0,
                d=0.0,
            ),
        ),
    )

    road = make_straight_road(
        lanes=(lane,),
        length=10.0,
    )

    reference = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    samples = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )

    sampled_lane = samples[0]

    assert sampled_lane.width[0] == pytest.approx(2.0)

    assert sampled_lane.width[-1] == pytest.approx(3.0)

    assert sampled_lane.centerline[0, 1] == pytest.approx(1.0)

    assert sampled_lane.centerline[-1, 1] == pytest.approx(1.5)


def test_lane_section_uses_its_own_interval() -> None:
    road = OpenDriveRoad(
        road_id="road_001",
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
                        lane_id=-1,
                        side=OpenDriveLaneSide.RIGHT,
                        lane_type="driving",
                        widths=(make_width(3.0),),
                    ),
                ),
            ),
            OpenDriveLaneSection(
                s=8.0,
                lanes=(
                    OpenDriveLane(
                        lane_id=-1,
                        side=OpenDriveLaneSide.RIGHT,
                        lane_type="driving",
                        widths=(make_width(4.0),),
                    ),
                ),
            ),
        ),
    )

    reference = sample_road_reference_line(
        road=road,
        step=3.0,
    )

    first = sample_lane_section(
        road=road,
        section_index=0,
        reference_line=reference,
    )[0]

    second = sample_lane_section(
        road=road,
        section_index=1,
        reference_line=reference,
    )[0]

    assert first.road_s[0] == pytest.approx(0.0)

    assert first.road_s[-1] == pytest.approx(8.0)

    assert second.road_s[0] == pytest.approx(8.0)

    assert second.road_s[-1] == pytest.approx(20.0)

    np.testing.assert_allclose(
        first.width,
        3.0,
    )

    np.testing.assert_allclose(
        second.width,
        4.0,
    )
