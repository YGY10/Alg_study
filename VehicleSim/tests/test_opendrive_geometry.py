import math

import numpy as np
import pytest

from sim2d.map import (
    OpenDriveArcGeometry,
    OpenDriveLineGeometry,
    OpenDriveRoad,
    sample_arc_geometry,
    sample_line_geometry,
    sample_local_distances,
    sample_road_reference_line,
)


def test_sample_local_distances_contains_endpoint() -> None:
    result = sample_local_distances(
        length=10.0,
        step=3.0,
    )

    assert result[0] == pytest.approx(0.0)

    assert result[-1] == pytest.approx(10.0)

    assert np.all(np.diff(result) > 0.0)


def test_sample_line_geometry_horizontal() -> None:
    geometry = OpenDriveLineGeometry(
        s=5.0,
        x=10.0,
        y=2.0,
        heading=0.0,
        length=4.0,
    )

    result = sample_line_geometry(
        geometry=geometry,
        step=1.0,
    )

    np.testing.assert_allclose(
        result[:, 0],
        np.array(
            [
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        result[:, 1],
        np.array(
            [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
            ],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        result[:, 2],
        2.0,
    )

    np.testing.assert_allclose(
        result[:, 3],
        0.0,
    )

    np.testing.assert_allclose(
        result[:, 4],
        0.0,
    )


def test_sample_line_geometry_with_heading() -> None:
    geometry = OpenDriveLineGeometry(
        s=0.0,
        x=1.0,
        y=2.0,
        heading=math.pi / 2.0,
        length=3.0,
    )

    result = sample_line_geometry(
        geometry=geometry,
        step=1.0,
    )

    np.testing.assert_allclose(
        result[-1, 1:3],
        np.array(
            [
                1.0,
                5.0,
            ]
        ),
        atol=1e-10,
    )


def test_sample_positive_arc_turns_left() -> None:
    geometry = OpenDriveArcGeometry(
        s=0.0,
        x=0.0,
        y=0.0,
        heading=0.0,
        length=math.pi * 5.0 / 2.0,
        curvature=0.2,
    )

    result = sample_arc_geometry(
        geometry=geometry,
        step=0.5,
    )

    np.testing.assert_allclose(
        result[-1, 1:3],
        np.array(
            [
                5.0,
                5.0,
            ],
            dtype=np.float64,
        ),
        atol=1e-8,
    )

    assert result[-1, 3] == pytest.approx(math.pi / 2.0)

    np.testing.assert_allclose(
        result[:, 4],
        0.2,
    )


def test_sample_negative_arc_turns_right() -> None:
    geometry = OpenDriveArcGeometry(
        s=0.0,
        x=0.0,
        y=0.0,
        heading=0.0,
        length=math.pi * 5.0 / 2.0,
        curvature=-0.2,
    )

    result = sample_arc_geometry(
        geometry=geometry,
        step=0.5,
    )

    np.testing.assert_allclose(
        result[-1, 1:3],
        np.array(
            [
                5.0,
                -5.0,
            ],
            dtype=np.float64,
        ),
        atol=1e-8,
    )

    assert result[-1, 3] == pytest.approx(-math.pi / 2.0)

    np.testing.assert_allclose(
        result[:, 4],
        -0.2,
    )


def test_sample_road_reference_line_removes_duplicate_joint() -> None:
    road = OpenDriveRoad(
        road_id="road_001",
        length=20.0,
        geometries=(
            OpenDriveLineGeometry(
                s=0.0,
                x=0.0,
                y=0.0,
                heading=0.0,
                length=10.0,
            ),
            OpenDriveLineGeometry(
                s=10.0,
                x=10.0,
                y=0.0,
                heading=0.0,
                length=10.0,
            ),
        ),
        lane_sections=(),
    )

    result = sample_road_reference_line(
        road=road,
        step=1.0,
    )

    assert result.shape == (
        21,
        5,
    )

    assert (
        np.count_nonzero(
            np.isclose(
                result[:, 0],
                10.0,
            )
        )
        == 1
    )

    assert result[-1, 0] == pytest.approx(20.0)

    assert result[-1, 1] == pytest.approx(20.0)


def test_sample_line_then_arc() -> None:
    arc_length = math.pi * 5.0 / 2.0

    road = OpenDriveRoad(
        road_id="road_001",
        length=10.0 + arc_length,
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
                length=arc_length,
                curvature=0.2,
            ),
        ),
        lane_sections=(),
    )

    result = sample_road_reference_line(
        road=road,
        step=0.5,
    )

    np.testing.assert_allclose(
        result[-1, 1:3],
        np.array(
            [
                15.0,
                5.0,
            ],
            dtype=np.float64,
        ),
        atol=1e-8,
    )

    assert result[-1, 3] == pytest.approx(math.pi / 2.0)


def test_discontinuous_geometry_position_raises() -> None:
    road = OpenDriveRoad(
        road_id="road_001",
        length=20.0,
        geometries=(
            OpenDriveLineGeometry(
                s=0.0,
                x=0.0,
                y=0.0,
                heading=0.0,
                length=10.0,
            ),
            OpenDriveLineGeometry(
                s=10.0,
                x=11.0,
                y=0.0,
                heading=0.0,
                length=10.0,
            ),
        ),
        lane_sections=(),
    )

    with pytest.raises(
        ValueError,
        match="positions are not continuous",
    ):
        sample_road_reference_line(
            road=road,
            step=1.0,
        )


def test_invalid_sample_step_raises() -> None:
    geometry = OpenDriveLineGeometry(
        s=0.0,
        x=0.0,
        y=0.0,
        heading=0.0,
        length=10.0,
    )

    with pytest.raises(ValueError):
        sample_line_geometry(
            geometry=geometry,
            step=0.0,
        )
