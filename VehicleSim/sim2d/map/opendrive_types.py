from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OpenDriveGeometryType(str, Enum):
    """OpenDRIVE planView 几何类型。"""

    LINE = "line"
    ARC = "arc"


class OpenDriveLaneSide(str, Enum):
    """车道相对于道路参考线的位置。"""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass(frozen=True)
class OpenDriveLineGeometry:
    """
    OpenDRIVE 直线几何。

    s:
        该 geometry 在 road reference line 上的起始弧长。

    x、y:
        geometry 起点世界坐标，单位 m。

    heading:
        起点航向角，单位 rad。

    length:
        geometry 长度，单位 m。
    """

    s: float
    x: float
    y: float
    heading: float
    length: float

    geometry_type: OpenDriveGeometryType = OpenDriveGeometryType.LINE

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = np.array(
            [
                self.s,
                self.x,
                self.y,
                self.heading,
                self.length,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("OpenDriveLineGeometry contains " "non-finite values")

        if self.s < 0.0:
            raise ValueError("OpenDriveLineGeometry.s must be " "non-negative")

        if self.length <= 0.0:
            raise ValueError("OpenDriveLineGeometry.length must " "be positive")


@dataclass(frozen=True)
class OpenDriveArcGeometry:
    """
    OpenDRIVE 圆弧几何。

    curvature:
        有符号曲率，单位 1/m。

        curvature > 0:
            向左转，航向角随 s 增大。

        curvature < 0:
            向右转，航向角随 s 减小。
    """

    s: float
    x: float
    y: float
    heading: float
    length: float
    curvature: float

    geometry_type: OpenDriveGeometryType = OpenDriveGeometryType.ARC

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = np.array(
            [
                self.s,
                self.x,
                self.y,
                self.heading,
                self.length,
                self.curvature,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("OpenDriveArcGeometry contains " "non-finite values")

        if self.s < 0.0:
            raise ValueError("OpenDriveArcGeometry.s must be " "non-negative")

        if self.length <= 0.0:
            raise ValueError("OpenDriveArcGeometry.length must " "be positive")

        if abs(self.curvature) <= 1e-12:
            raise ValueError("OpenDriveArcGeometry.curvature must " "be non-zero")


OpenDriveGeometry = OpenDriveLineGeometry | OpenDriveArcGeometry


@dataclass(frozen=True)
class OpenDriveLaneWidth:
    """
    OpenDRIVE lane width 三次多项式。

    width(ds) = a + b*ds + c*ds² + d*ds³

    s_offset:
        相对于当前 laneSection 起点的偏移。

    ds:
        相对于该 width record 起点的距离：

            ds = section_s - s_offset
    """

    s_offset: float

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = np.array(
            [
                self.s_offset,
                self.a,
                self.b,
                self.c,
                self.d,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("OpenDriveLaneWidth contains " "non-finite values")

        if self.s_offset < 0.0:
            raise ValueError("OpenDriveLaneWidth.s_offset must be " "non-negative")

    def evaluate(
        self,
        ds: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        计算宽度。

        参数 ds 是相对于当前 width record 起点的距离，
        而不是相对于 laneSection 起点的距离。
        """
        value = np.asarray(
            ds,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                "Lane width evaluation distance must " "contain finite values"
            )

        if np.any(value < 0.0):
            raise ValueError("Lane width evaluation distance must " "be non-negative")

        result = (
            self.a
            + self.b * value
            + self.c * value * value
            + self.d * value * value * value
        )

        if np.isscalar(ds):
            return float(result)

        return result.astype(
            np.float64,
            copy=False,
        )


@dataclass(frozen=True)
class OpenDriveLane:
    """
    OpenDRIVE 原始车道描述。

    lane_id:
        OpenDRIVE 车道编号。

        左侧车道通常为正数；
        右侧车道通常为负数；
        中心车道通常为 0。

    lane_type:
        保留 OpenDRIVE XML 中的原始 type 字符串。

    level:
        对应 XML lane 的 level 属性。

    widths:
        按 s_offset 从小到大排列的宽度记录。
    """

    lane_id: int
    side: OpenDriveLaneSide
    lane_type: str

    widths: tuple[OpenDriveLaneWidth, ...] = ()

    predecessor_id: int | None = None
    successor_id: int | None = None

    level: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.widths,
            tuple,
        ):
            object.__setattr__(
                self,
                "widths",
                tuple(self.widths),
            )

        self.validate()

    def validate(self) -> None:
        if not self.lane_type.strip():
            raise ValueError("OpenDriveLane.lane_type cannot be empty")

        if self.side is OpenDriveLaneSide.LEFT and self.lane_id <= 0:
            raise ValueError("Left OpenDRIVE lane ID must be positive")

        if self.side is OpenDriveLaneSide.RIGHT and self.lane_id >= 0:
            raise ValueError("Right OpenDRIVE lane ID must be negative")

        if self.side is OpenDriveLaneSide.CENTER and self.lane_id != 0:
            raise ValueError("Center OpenDRIVE lane ID must be zero")

        width_offsets = [width.s_offset for width in self.widths]

        if width_offsets != sorted(width_offsets):
            raise ValueError("OpenDriveLane.widths must be sorted " "by s_offset")

        if len(width_offsets) != len(set(width_offsets)):
            raise ValueError(
                "OpenDriveLane.widths contains duplicate " "s_offset records"
            )

    def width_at(
        self,
        section_s: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        计算 laneSection 局部坐标 section_s 处的车道宽度。

        section_s 必须从当前 laneSection 起点开始计算。
        """
        values = np.asarray(
            section_s,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("section_s must contain finite values")

        if np.any(values < 0.0):
            raise ValueError("section_s must be non-negative")

        if not self.widths:
            result = np.zeros_like(
                values,
                dtype=np.float64,
            )

            if np.isscalar(section_s):
                return float(result)

            return result

        offsets = np.array(
            [width.s_offset for width in self.widths],
            dtype=np.float64,
        )

        flat_values = values.reshape(-1)

        record_indices = (
            np.searchsorted(
                offsets,
                flat_values,
                side="right",
            )
            - 1
        )

        record_indices = np.maximum(
            record_indices,
            0,
        )

        result_flat = np.empty_like(
            flat_values,
            dtype=np.float64,
        )

        for record_index, width in enumerate(self.widths):
            mask = record_indices == record_index

            if not np.any(mask):
                continue

            local_ds = flat_values[mask] - width.s_offset

            # 在第一个 width record 起点之前查询时，
            # 使用第一个 record 在 ds=0 的值。
            local_ds = np.maximum(
                local_ds,
                0.0,
            )

            result_flat[mask] = width.evaluate(local_ds)

        result = result_flat.reshape(values.shape)

        if np.isscalar(section_s):
            return float(result)

        return result


@dataclass(frozen=True)
class OpenDriveLaneSection:
    """
    OpenDRIVE laneSection。

    s:
        laneSection 在 road reference line 上的绝对起始弧长。

    lanes:
        包括左、中心、右车道。
    """

    s: float
    lanes: tuple[OpenDriveLane, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.lanes,
            tuple,
        ):
            object.__setattr__(
                self,
                "lanes",
                tuple(self.lanes),
            )

        self.validate()

    def validate(self) -> None:
        if not np.isfinite(self.s):
            raise ValueError("OpenDriveLaneSection.s must be finite")

        if self.s < 0.0:
            raise ValueError("OpenDriveLaneSection.s must be " "non-negative")

        lane_ids = [lane.lane_id for lane in self.lanes]

        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("OpenDriveLaneSection contains duplicate " "lane IDs")

    def get_lane(
        self,
        lane_id: int,
    ) -> OpenDriveLane:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane

        raise KeyError(f"Unknown OpenDRIVE lane ID: {lane_id}")


@dataclass(frozen=True)
class OpenDriveRoad:
    """
    一条 OpenDRIVE road 的最小内部表示。
    """

    road_id: str
    length: float

    geometries: tuple[OpenDriveGeometry, ...]
    lane_sections: tuple[OpenDriveLaneSection, ...]

    name: str | None = None
    junction_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.geometries,
            tuple,
        ):
            object.__setattr__(
                self,
                "geometries",
                tuple(self.geometries),
            )

        if not isinstance(
            self.lane_sections,
            tuple,
        ):
            object.__setattr__(
                self,
                "lane_sections",
                tuple(self.lane_sections),
            )

        self.validate()

    def validate(self) -> None:
        if not self.road_id.strip():
            raise ValueError("OpenDriveRoad.road_id cannot be empty")

        if not np.isfinite(self.length):
            raise ValueError("OpenDriveRoad.length must be finite")

        if self.length <= 0.0:
            raise ValueError("OpenDriveRoad.length must be positive")

        if not self.geometries:
            raise ValueError("OpenDriveRoad must contain at least " "one geometry")

        geometry_starts = [geometry.s for geometry in self.geometries]

        if geometry_starts != sorted(geometry_starts):
            raise ValueError("OpenDriveRoad.geometries must be sorted " "by s")

        if len(geometry_starts) != len(set(geometry_starts)):
            raise ValueError("OpenDriveRoad.geometries contains " "duplicate s values")

        lane_section_starts = [section.s for section in self.lane_sections]

        if lane_section_starts != sorted(lane_section_starts):
            raise ValueError("OpenDriveRoad.lane_sections must be " "sorted by s")

        if len(lane_section_starts) != len(set(lane_section_starts)):
            raise ValueError(
                "OpenDriveRoad.lane_sections contains " "duplicate s values"
            )

        geometry_end = max(geometry.s + geometry.length for geometry in self.geometries)

        if geometry_end > self.length + 1e-6:
            raise ValueError("OpenDriveRoad geometry extends beyond " "road length")

        for section in self.lane_sections:
            if section.s > self.length:
                raise ValueError("OpenDriveLaneSection starts beyond " "road length")
