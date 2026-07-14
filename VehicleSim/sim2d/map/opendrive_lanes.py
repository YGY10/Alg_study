from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim2d.map.opendrive_types import (
    OpenDriveLane,
    OpenDriveLaneSection,
    OpenDriveLaneSide,
    OpenDriveRoad,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OpenDriveLaneSample:
    """
    一条 OpenDRIVE lane 在某个 laneSection 内的离散几何。

    所有点均按照 road reference line 的 s 增大方向排列。

    road_s:
        road reference line 的绝对弧长，形状 [N]。

    centerline:
        车道中心线，形状 [N, 2]。

    inner_boundary:
        靠近道路参考线的边界，形状 [N, 2]。

    outer_boundary:
        远离道路参考线的边界，形状 [N, 2]。

    width:
        每个采样点处的车道宽度，形状 [N]。

    注意：
        inner_boundary 和 outer_boundary 暂时不是最终
        Lane.left_boundary / Lane.right_boundary。

        最终左右边界必须结合车道实际行驶方向判断。
    """

    road_id: str
    section_index: int

    section_s_start: float
    section_s_end: float

    lane_id: int
    side: OpenDriveLaneSide
    lane_type: str

    road_s: FloatArray

    centerline: FloatArray
    inner_boundary: FloatArray
    outer_boundary: FloatArray

    width: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "road_s",
            np.asarray(
                self.road_s,
                dtype=np.float64,
            ).copy(),
        )

        object.__setattr__(
            self,
            "centerline",
            np.asarray(
                self.centerline,
                dtype=np.float64,
            ).copy(),
        )

        object.__setattr__(
            self,
            "inner_boundary",
            np.asarray(
                self.inner_boundary,
                dtype=np.float64,
            ).copy(),
        )

        object.__setattr__(
            self,
            "outer_boundary",
            np.asarray(
                self.outer_boundary,
                dtype=np.float64,
            ).copy(),
        )

        object.__setattr__(
            self,
            "width",
            np.asarray(
                self.width,
                dtype=np.float64,
            ).copy(),
        )

        self.validate()

    def validate(self) -> None:
        if not self.road_id.strip():
            raise ValueError("OpenDriveLaneSample.road_id cannot be empty")

        if self.section_index < 0:
            raise ValueError(
                "OpenDriveLaneSample.section_index must " "be non-negative"
            )

        if not np.isfinite(self.section_s_start):
            raise ValueError("section_s_start must be finite")

        if not np.isfinite(self.section_s_end):
            raise ValueError("section_s_end must be finite")

        if self.section_s_end <= self.section_s_start:
            raise ValueError("section_s_end must be greater than " "section_s_start")

        if self.road_s.ndim != 1:
            raise ValueError("road_s must have shape [N]")

        point_count = self.road_s.shape[0]

        if point_count < 2:
            raise ValueError("OpenDriveLaneSample must contain " "at least two samples")

        for name, array in (
            (
                "centerline",
                self.centerline,
            ),
            (
                "inner_boundary",
                self.inner_boundary,
            ),
            (
                "outer_boundary",
                self.outer_boundary,
            ),
        ):
            if array.shape != (
                point_count,
                2,
            ):
                raise ValueError(
                    f"{name} must have shape "
                    f"({point_count}, 2), "
                    f"got {array.shape}"
                )

        if self.width.shape != (point_count,):
            raise ValueError(
                "width must have shape " f"({point_count},), " f"got {self.width.shape}"
            )

        arrays = (
            self.road_s,
            self.centerline,
            self.inner_boundary,
            self.outer_boundary,
            self.width,
        )

        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("OpenDriveLaneSample contains " "non-finite values")

        if not np.all(np.diff(self.road_s) > 0.0):
            raise ValueError("road_s must be strictly increasing")

        if np.any(self.width < 0.0):
            raise ValueError("lane width must be non-negative")


def interpolate_reference_line(
    reference_line: FloatArray,
    query_s: FloatArray,
) -> FloatArray:
    """
    在指定绝对 road s 位置插值参考线。

    reference_line 列定义：

        0: s
        1: x
        2: y
        3: heading
        4: curvature

    返回同样为：

        [s, x, y, heading, curvature]
    """
    reference = np.asarray(
        reference_line,
        dtype=np.float64,
    )

    query = np.asarray(
        query_s,
        dtype=np.float64,
    )

    if reference.ndim != 2:
        raise ValueError("reference_line must be a 2D array")

    if reference.shape[1] != 5:
        raise ValueError("reference_line must have shape [N, 5]")

    if reference.shape[0] < 2:
        raise ValueError("reference_line must contain at least " "two samples")

    if query.ndim != 1:
        raise ValueError("query_s must have shape [N]")

    if query.shape[0] < 1:
        raise ValueError("query_s cannot be empty")

    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_line contains non-finite values")

    if not np.all(np.isfinite(query)):
        raise ValueError("query_s contains non-finite values")

    reference_s = reference[:, 0]

    if not np.all(np.diff(reference_s) > 0.0):
        raise ValueError("reference_line s values must be " "strictly increasing")

    tolerance = 1e-9

    if query[0] < reference_s[0] - tolerance:
        raise ValueError("query_s starts before reference_line")

    if query[-1] > reference_s[-1] + tolerance:
        raise ValueError("query_s ends after reference_line")

    if not np.all(np.diff(query) >= 0.0):
        raise ValueError("query_s must be sorted")

    # heading 不能直接在 [-pi, pi) 上线性插值，
    # 否则跨越 pi/-pi 时会发生错误跳变。
    unwrapped_heading = np.unwrap(reference[:, 3])

    x = np.interp(
        query,
        reference_s,
        reference[:, 1],
    )

    y = np.interp(
        query,
        reference_s,
        reference[:, 2],
    )

    heading_unwrapped = np.interp(
        query,
        reference_s,
        unwrapped_heading,
    )

    heading = (heading_unwrapped + np.pi) % (2.0 * np.pi) - np.pi

    curvature = np.interp(
        query,
        reference_s,
        reference[:, 4],
    )

    return np.column_stack(
        [
            query,
            x,
            y,
            heading,
            curvature,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


def extract_reference_interval(
    reference_line: FloatArray,
    s_start: float,
    s_end: float,
) -> FloatArray:
    """
    提取参考线的闭区间 [s_start, s_end]。

    即使原始采样点中没有精确包含区间端点，
    返回结果也会通过插值显式加入两个端点。
    """
    if not np.isfinite(s_start):
        raise ValueError("s_start must be finite")

    if not np.isfinite(s_end):
        raise ValueError("s_end must be finite")

    if s_end <= s_start:
        raise ValueError("s_end must be greater than s_start")

    reference = np.asarray(
        reference_line,
        dtype=np.float64,
    )

    if reference.ndim != 2 or reference.shape[1] != 5:
        raise ValueError("reference_line must have shape [N, 5]")

    original_s = reference[:, 0]

    interior_s = original_s[(original_s > s_start) & (original_s < s_end)]

    query_s = np.concatenate(
        [
            np.array(
                [s_start],
                dtype=np.float64,
            ),
            interior_s,
            np.array(
                [s_end],
                dtype=np.float64,
            ),
        ]
    )

    # 某些浮点边界情况下可能产生重复值。
    query_s = np.unique(query_s)

    return interpolate_reference_line(
        reference_line=reference,
        query_s=query_s,
    )


def offset_reference_points(
    reference_samples: FloatArray,
    offsets: float | FloatArray,
) -> FloatArray:
    """
    使用参考线 heading 将采样点横向偏移。

    正偏移：
        朝参考线左侧。

    负偏移：
        朝参考线右侧。
    """
    samples = np.asarray(
        reference_samples,
        dtype=np.float64,
    )

    if samples.ndim != 2 or samples.shape[1] != 5:
        raise ValueError("reference_samples must have shape [N, 5]")

    point_count = samples.shape[0]

    if np.isscalar(offsets):
        offset_array = np.full(
            point_count,
            float(offsets),
            dtype=np.float64,
        )
    else:
        offset_array = np.asarray(
            offsets,
            dtype=np.float64,
        )

        if offset_array.shape != (point_count,):
            raise ValueError(
                "offsets must be a scalar or have "
                f"shape ({point_count},), "
                f"got {offset_array.shape}"
            )

    if not np.all(np.isfinite(offset_array)):
        raise ValueError("offsets must contain finite values")

    heading = samples[:, 3]

    left_normal = np.column_stack(
        [
            -np.sin(heading),
            np.cos(heading),
        ]
    )

    return samples[:, 1:3] + offset_array[:, None] * left_normal


def _evaluate_lane_width(
    lane: OpenDriveLane,
    section_local_s: FloatArray,
) -> FloatArray:
    """
    计算并校验一条车道的宽度。
    """
    width = np.asarray(
        lane.width_at(section_local_s),
        dtype=np.float64,
    )

    if width.shape != section_local_s.shape:
        raise ValueError("lane width result has an invalid shape")

    if not np.all(np.isfinite(width)):
        raise ValueError(f"Lane {lane.lane_id} width contains " "non-finite values")

    tolerance = 1e-9

    if np.any(width < -tolerance):
        minimum_width = float(np.min(width))

        raise ValueError(f"Lane {lane.lane_id} has negative width: " f"{minimum_width}")

    # 允许多项式浮点误差产生极小负值。
    return np.maximum(
        width,
        0.0,
    )


def _sample_side_lanes(
    *,
    road: OpenDriveRoad,
    section: OpenDriveLaneSection,
    section_index: int,
    section_s_end: float,
    reference_samples: FloatArray,
    side: OpenDriveLaneSide,
) -> tuple[OpenDriveLaneSample, ...]:
    """
    采样 laneSection 某一侧的全部车道。

    车道必须按照从参考线向外排列：

        LEFT:
            1, 2, 3, ...

        RIGHT:
            -1, -2, -3, ...
    """
    if side is OpenDriveLaneSide.LEFT:
        lanes = sorted(
            (lane for lane in section.lanes if lane.side is OpenDriveLaneSide.LEFT),
            key=lambda lane: lane.lane_id,
        )

        side_sign = 1.0

    elif side is OpenDriveLaneSide.RIGHT:
        lanes = sorted(
            (lane for lane in section.lanes if lane.side is OpenDriveLaneSide.RIGHT),
            key=lambda lane: abs(lane.lane_id),
        )

        side_sign = -1.0

    else:
        raise ValueError("_sample_side_lanes only supports " "LEFT or RIGHT")

    road_s = reference_samples[:, 0]

    section_local_s = road_s - section.s

    cumulative_width = np.zeros(
        road_s.shape[0],
        dtype=np.float64,
    )

    result: list[OpenDriveLaneSample] = []

    for lane in lanes:
        width = _evaluate_lane_width(
            lane=lane,
            section_local_s=section_local_s,
        )

        inner_distance = cumulative_width.copy()

        outer_distance = cumulative_width + width

        center_distance = 0.5 * (inner_distance + outer_distance)

        inner_offset = side_sign * inner_distance

        outer_offset = side_sign * outer_distance

        center_offset = side_sign * center_distance

        inner_boundary = offset_reference_points(
            reference_samples,
            inner_offset,
        )

        outer_boundary = offset_reference_points(
            reference_samples,
            outer_offset,
        )

        centerline = offset_reference_points(
            reference_samples,
            center_offset,
        )

        result.append(
            OpenDriveLaneSample(
                road_id=road.road_id,
                section_index=section_index,
                section_s_start=section.s,
                section_s_end=section_s_end,
                lane_id=lane.lane_id,
                side=lane.side,
                lane_type=lane.lane_type,
                road_s=road_s,
                centerline=centerline,
                inner_boundary=inner_boundary,
                outer_boundary=outer_boundary,
                width=width,
            )
        )

        cumulative_width = outer_distance

    return tuple(result)


def sample_lane_section(
    *,
    road: OpenDriveRoad,
    section_index: int,
    reference_line: FloatArray,
) -> tuple[OpenDriveLaneSample, ...]:
    """
    采样一个 laneSection 的左右车道。

    中心车道 lane_id=0 不生成几何，因为它通常只表示
    road reference line 或道路标线，而不是可行驶车道。
    """
    if section_index < 0:
        raise IndexError("section_index must be non-negative")

    if section_index >= len(road.lane_sections):
        raise IndexError(f"section_index out of range: " f"{section_index}")

    section = road.lane_sections[section_index]

    if section_index + 1 < len(road.lane_sections):
        section_s_end = road.lane_sections[section_index + 1].s
    else:
        section_s_end = road.length

    if section_s_end <= section.s:
        raise ValueError("laneSection has invalid s interval")

    reference_samples = extract_reference_interval(
        reference_line=reference_line,
        s_start=section.s,
        s_end=section_s_end,
    )

    left_lanes = _sample_side_lanes(
        road=road,
        section=section,
        section_index=section_index,
        section_s_end=section_s_end,
        reference_samples=reference_samples,
        side=OpenDriveLaneSide.LEFT,
    )

    right_lanes = _sample_side_lanes(
        road=road,
        section=section,
        section_index=section_index,
        section_s_end=section_s_end,
        reference_samples=reference_samples,
        side=OpenDriveLaneSide.RIGHT,
    )

    return left_lanes + right_lanes


def sample_road_lanes(
    *,
    road: OpenDriveRoad,
    reference_line: FloatArray,
) -> tuple[OpenDriveLaneSample, ...]:
    """
    采样一条 road 中的全部 laneSection。
    """
    result: list[OpenDriveLaneSample] = []

    for section_index in range(len(road.lane_sections)):
        result.extend(
            sample_lane_section(
                road=road,
                section_index=section_index,
                reference_line=reference_line,
            )
        )

    return tuple(result)
