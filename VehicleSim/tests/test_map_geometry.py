import numpy as np
import pytest

from sim2d.map import (
    cumulative_arc_length,
    left_normals,
    offset_polyline,
    unit_tangents,
)


def test_cumulative_arc_length_straight_line() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    result = cumulative_arc_length(points)

    np.testing.assert_allclose(
        result,
        np.array(
            [
                0.0,
                3.0,
                7.0,
            ],
            dtype=np.float64,
        ),
    )


def test_unit_tangents_on_horizontal_line() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    tangents = unit_tangents(points)

    expected = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        tangents,
        expected,
    )


def test_left_normals_on_horizontal_line() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    normals = left_normals(points)

    expected = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        normals,
        expected,
    )


def test_offset_polyline_positive_moves_left() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = offset_polyline(
        points,
        2.0,
    )

    expected = np.array(
        [
            [0.0, 2.0],
            [5.0, 2.0],
            [10.0, 2.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_offset_polyline_negative_moves_right() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = offset_polyline(
        points,
        -1.5,
    )

    expected = np.array(
        [
            [0.0, -1.5],
            [5.0, -1.5],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_offset_polyline_accepts_per_point_offsets() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    offsets = np.array(
        [
            1.0,
            2.0,
            3.0,
        ],
        dtype=np.float64,
    )

    result = offset_polyline(
        points,
        offsets,
    )

    expected = np.array(
        [
            [0.0, 1.0],
            [5.0, 2.0],
            [10.0, 3.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_degenerate_points_raise() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        unit_tangents(points)


def test_invalid_offset_shape_raises() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        offset_polyline(
            points,
            np.array(
                [
                    1.0,
                    2.0,
                ],
                dtype=np.float64,
            ),
        )
