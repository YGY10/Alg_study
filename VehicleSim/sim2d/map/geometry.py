from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def validate_points(
    points: FloatArray,
) -> FloatArray:
    """
    校验并规范化二维点集。

    要求形状为 [N, 2]，N >= 2。
    """
    array = np.asarray(
        points,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError("points must be a 2D array, " f"got ndim={array.ndim}")

    if array.shape[1] != 2:
        raise ValueError("points must have shape [N, 2], " f"got {array.shape}")

    if array.shape[0] < 2:
        raise ValueError("points must contain at least two points")

    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")

    return array


def cumulative_arc_length(
    points: FloatArray,
) -> FloatArray:
    """
    计算折线各点的累计弧长。

    返回形状：
        [N]

    第一个元素固定为 0。
    """
    array = validate_points(points)

    segment_vectors = np.diff(
        array,
        axis=0,
    )

    segment_lengths = np.linalg.norm(
        segment_vectors,
        axis=1,
    )

    return np.concatenate(
        [
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(segment_lengths),
        ]
    )


def unit_tangents(
    points: FloatArray,
) -> FloatArray:
    """
    计算折线各采样点的单位切向量。

    内部点使用前后点形成的中心差分；
    首尾点使用单边差分。

    返回形状：
        [N, 2]
    """
    array = validate_points(points)

    tangents = np.empty_like(array)

    tangents[0] = array[1] - array[0]

    tangents[-1] = array[-1] - array[-2]

    if array.shape[0] > 2:
        tangents[1:-1] = array[2:] - array[:-2]

    norms = np.linalg.norm(
        tangents,
        axis=1,
    )

    if np.any(norms <= 1e-12):
        raise ValueError("Cannot compute tangent for duplicate " "or degenerate points")

    return tangents / norms[:, None]


def left_normals(
    points: FloatArray,
) -> FloatArray:
    """
    计算每个采样点的单位左法向量。

    对切向量：

        t = [tx, ty]

    左法向量定义为：

        n_left = [-ty, tx]

    该定义与 VehicleSim 坐标约定一致：
        +x 前方
        +y 左侧
        yaw 正方向逆时针
    """
    tangents = unit_tangents(points)

    normals = np.empty_like(tangents)

    normals[:, 0] = -tangents[:, 1]

    normals[:, 1] = tangents[:, 0]

    return normals


def offset_polyline(
    points: FloatArray,
    offsets: float | FloatArray,
) -> FloatArray:
    """
    沿折线左法向进行横向偏移。

    offsets 可以是：
        单个 float
        形状为 [N] 的数组

    符号约定：
        offset > 0：向左偏移
        offset < 0：向右偏移

    返回形状：
        [N, 2]
    """
    array = validate_points(points)

    normals = left_normals(array)

    if np.isscalar(offsets):
        offset_array = np.full(
            array.shape[0],
            float(offsets),
            dtype=np.float64,
        )
    else:
        offset_array = np.asarray(
            offsets,
            dtype=np.float64,
        )

        if offset_array.shape != (array.shape[0],):
            raise ValueError(
                "offsets must be a scalar or have "
                f"shape ({array.shape[0]},), "
                f"got {offset_array.shape}"
            )

        if not np.all(np.isfinite(offset_array)):
            raise ValueError("offsets must contain only finite values")

    return array + offset_array[:, None] * normals
