from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from sim2d.map.opendrive_types import (
    OpenDriveArcGeometry,
    OpenDriveGeometry,
    OpenDriveLineGeometry,
    OpenDriveRoad,
)

FloatArray = NDArray[np.float64]


def normalize_angle(
    angle: float | FloatArray,
) -> float | FloatArray:
    """
    将角度归一化到 [-pi, pi)。
    """
    value = np.asarray(
        angle,
        dtype=np.float64,
    )

    result = (value + math.pi) % (2.0 * math.pi) - math.pi

    if np.isscalar(angle):
        return float(result)

    return result.astype(
        np.float64,
        copy=False,
    )


def sample_local_distances(
    length: float,
    step: float,
) -> FloatArray:
    """
    为单个 geometry 生成局部弧长采样。

    返回值一定满足：

        result[0] == 0
        result[-1] == length

    即使 length 不能被 step 整除，
    也会显式包含 geometry 终点。
    """
    if not np.isfinite(length):
        raise ValueError("length must be finite")

    if not np.isfinite(step):
        raise ValueError("step must be finite")

    if length <= 0.0:
        raise ValueError("length must be positive")

    if step <= 0.0:
        raise ValueError("step must be positive")

    sample_count = max(
        int(math.ceil(length / step)),
        1,
    )

    return np.linspace(
        0.0,
        length,
        sample_count + 1,
        dtype=np.float64,
    )


def sample_line_geometry(
    geometry: OpenDriveLineGeometry,
    step: float,
) -> FloatArray:
    """
    采样 OpenDRIVE line geometry。

    输出形状：

        [N, 5]

    列定义：

        0: road absolute s
        1: x
        2: y
        3: heading
        4: curvature
    """
    local_s = sample_local_distances(
        length=geometry.length,
        step=step,
    )

    cos_heading = math.cos(geometry.heading)

    sin_heading = math.sin(geometry.heading)

    x = geometry.x + local_s * cos_heading

    y = geometry.y + local_s * sin_heading

    absolute_s = geometry.s + local_s

    heading = np.full_like(
        local_s,
        normalize_angle(geometry.heading),
    )

    curvature = np.zeros_like(local_s)

    return np.column_stack(
        [
            absolute_s,
            x,
            y,
            heading,
            curvature,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


def sample_arc_geometry(
    geometry: OpenDriveArcGeometry,
    step: float,
) -> FloatArray:
    """
    采样 OpenDRIVE arc geometry。

    圆弧参数方程：

        heading(ds) = heading0 + curvature * ds

        x(ds) = x0
              + (
                    sin(heading(ds))
                    - sin(heading0)
                ) / curvature

        y(ds) = y0
              - (
                    cos(heading(ds))
                    - cos(heading0)
                ) / curvature
    """
    local_s = sample_local_distances(
        length=geometry.length,
        step=step,
    )

    raw_heading = geometry.heading + geometry.curvature * local_s

    x = (
        geometry.x
        + (np.sin(raw_heading) - math.sin(geometry.heading)) / geometry.curvature
    )

    y = (
        geometry.y
        - (np.cos(raw_heading) - math.cos(geometry.heading)) / geometry.curvature
    )

    absolute_s = geometry.s + local_s

    heading = normalize_angle(raw_heading)

    curvature = np.full_like(
        local_s,
        geometry.curvature,
    )

    return np.column_stack(
        [
            absolute_s,
            x,
            y,
            heading,
            curvature,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


def sample_geometry(
    geometry: OpenDriveGeometry,
    step: float,
) -> FloatArray:
    """
    按具体类型采样一个 OpenDRIVE geometry。
    """
    if isinstance(
        geometry,
        OpenDriveLineGeometry,
    ):
        return sample_line_geometry(
            geometry=geometry,
            step=step,
        )

    if isinstance(
        geometry,
        OpenDriveArcGeometry,
    ):
        return sample_arc_geometry(
            geometry=geometry,
            step=step,
        )

    raise TypeError(
        "Unsupported OpenDRIVE geometry type: " f"{type(geometry).__name__}"
    )


def sample_road_reference_line(
    road: OpenDriveRoad,
    step: float = 0.5,
) -> FloatArray:
    """
    采样整条 OpenDRIVE road 的 reference line。

    相邻 geometry 的连接点会重复出现，因此拼接时会删除
    后一段 geometry 的第一个采样点。

    输出形状：

        [N, 5]

    列定义：

        s, x, y, heading, curvature
    """
    if not np.isfinite(step):
        raise ValueError("step must be finite")

    if step <= 0.0:
        raise ValueError("step must be positive")

    sampled_parts: list[FloatArray] = []

    previous_end: FloatArray | None = None

    for geometry_index, geometry in enumerate(road.geometries):
        sampled = sample_geometry(
            geometry=geometry,
            step=step,
        )

        if previous_end is not None:
            position_error = float(np.linalg.norm(sampled[0, 1:3] - previous_end[1:3]))

            s_error = abs(float(sampled[0, 0] - previous_end[0]))

            heading_error = abs(float(normalize_angle(sampled[0, 3] - previous_end[3])))

            if s_error > 1e-6:
                raise ValueError(
                    "OpenDRIVE geometry s values are "
                    "not continuous between geometry "
                    f"{geometry_index - 1} and "
                    f"{geometry_index}: "
                    f"s_error={s_error}"
                )

            if position_error > 1e-4:
                raise ValueError(
                    "OpenDRIVE geometry positions are "
                    "not continuous between geometry "
                    f"{geometry_index - 1} and "
                    f"{geometry_index}: "
                    f"position_error={position_error}"
                )

            if heading_error > 1e-4:
                raise ValueError(
                    "OpenDRIVE geometry headings are "
                    "not continuous between geometry "
                    f"{geometry_index - 1} and "
                    f"{geometry_index}: "
                    f"heading_error={heading_error}"
                )

            sampled = sampled[1:]

        if sampled.shape[0] > 0:
            sampled_parts.append(sampled)

        complete_geometry = sample_geometry(
            geometry=geometry,
            step=geometry.length,
        )

        previous_end = complete_geometry[-1]

    if not sampled_parts:
        raise ValueError("OpenDriveRoad produced no reference-line samples")

    result = np.vstack(sampled_parts)

    if not np.all(np.diff(result[:, 0]) > 0.0):
        raise ValueError(
            "Sampled road reference-line s values " "must be strictly increasing"
        )

    return result.astype(
        np.float64,
        copy=False,
    )
