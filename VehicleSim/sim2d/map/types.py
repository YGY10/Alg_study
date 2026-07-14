from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class LaneType(str, Enum):
    """统一地图层支持的车道类型。"""

    DRIVING = "driving"
    SHOULDER = "shoulder"
    PARKING = "parking"
    BIKE = "bike"
    SIDEWALK = "sidewalk"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Polyline2D:
    """
    二维折线。

    points 的形状必须为：

        [N, 2]

    每一行表示：

        [x, y]

    坐标单位统一为米。
    """

    points: FloatArray

    def __post_init__(self) -> None:
        array = np.asarray(
            self.points,
            dtype=np.float64,
        )

        self._validate_array(array)

        # 创建独立副本，防止外部数组修改内部数据。
        array = array.copy()

        # frozen dataclass 仍可在 __post_init__ 中
        # 使用 object.__setattr__ 完成规范化。
        object.__setattr__(
            self,
            "points",
            array,
        )

    @staticmethod
    def _validate_array(
        array: FloatArray,
    ) -> None:
        if array.ndim != 2:
            raise ValueError(
                "Polyline2D.points must be a 2D array, " f"got ndim={array.ndim}"
            )

        if array.shape[1] != 2:
            raise ValueError(
                "Polyline2D.points must have shape [N, 2], " f"got {array.shape}"
            )

        if array.shape[0] < 2:
            raise ValueError("Polyline2D must contain at least two points")

        if not np.all(np.isfinite(array)):
            raise ValueError("Polyline2D contains non-finite values")

    def validate(self) -> None:
        """
        重新校验当前折线。

        正常情况下构造时已经完成校验；
        该方法主要供 RoadNetwork 递归校验使用。
        """
        self._validate_array(
            np.asarray(
                self.points,
                dtype=np.float64,
            )
        )

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def start(self) -> FloatArray:
        """返回起点副本。"""
        return self.points[0].copy()

    @property
    def end(self) -> FloatArray:
        """返回终点副本。"""
        return self.points[-1].copy()

    @property
    def length(self) -> float:
        """返回折线累计长度，单位米。"""
        segment_vectors = np.diff(
            self.points,
            axis=0,
        )

        segment_lengths = np.linalg.norm(
            segment_vectors,
            axis=1,
        )

        return float(np.sum(segment_lengths))


@dataclass(frozen=True)
class Lane:
    """
    统一车道表示。

    lane_id:
        RoadNetwork 内唯一标识。

    centerline:
        沿车道行驶方向排列的中心线。

    left_boundary / right_boundary:
        以车道行驶方向为基准的左右边界。

    speed_limit:
        单位 m/s。None 表示未知。

    predecessor_ids / successor_ids:
        前驱和后继车道 ID。
    """

    lane_id: str
    lane_type: LaneType

    centerline: Polyline2D
    left_boundary: Polyline2D
    right_boundary: Polyline2D

    speed_limit: float | None = None

    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()

    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.predecessor_ids,
            tuple,
        ):
            object.__setattr__(
                self,
                "predecessor_ids",
                tuple(self.predecessor_ids),
            )

        if not isinstance(
            self.successor_ids,
            tuple,
        ):
            object.__setattr__(
                self,
                "successor_ids",
                tuple(self.successor_ids),
            )

        # 防止 frozen dataclass 中 metadata
        # 指向一个仍可由外部修改的普通 dict。
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

        self.validate()

    def validate(self) -> None:
        if not self.lane_id.strip():
            raise ValueError("Lane.lane_id cannot be empty")

        if not isinstance(
            self.lane_type,
            LaneType,
        ):
            raise TypeError(
                "Lane.lane_type must be a LaneType, "
                f"got {type(self.lane_type).__name__}"
            )

        self.centerline.validate()
        self.left_boundary.validate()
        self.right_boundary.validate()

        if self.speed_limit is not None and self.speed_limit <= 0.0:
            raise ValueError("Lane.speed_limit must be positive " "when provided")

        if self.lane_id in self.predecessor_ids:
            raise ValueError("Lane cannot list itself as a predecessor")

        if self.lane_id in self.successor_ids:
            raise ValueError("Lane cannot list itself as a successor")

        if len(set(self.predecessor_ids)) != len(self.predecessor_ids):
            raise ValueError("Lane.predecessor_ids contains duplicates")

        if len(set(self.successor_ids)) != len(self.successor_ids):
            raise ValueError("Lane.successor_ids contains duplicates")


@dataclass(frozen=True)
class RoadNetwork:
    """
    VehicleSim 内部统一道路网络。

    所有外部地图格式最终都必须转换为该类型：

        OpenDRIVE
        nuScenes
        nuPlan

    source_type 示例：

        "manual"
        "opendrive"
        "nuscenes"
        "nuplan"
    """

    lanes: tuple[Lane, ...]

    source_type: str
    source_name: str | None = None

    metadata: Mapping[str, object] = field(default_factory=dict)

    _lane_by_id: Mapping[str, Lane] = field(
        init=False,
        repr=False,
        compare=False,
    )

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

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

        lane_by_id = {lane.lane_id: lane for lane in self.lanes}

        object.__setattr__(
            self,
            "_lane_by_id",
            MappingProxyType(lane_by_id),
        )

        self.validate()

    def validate(self) -> None:
        if not self.source_type.strip():
            raise ValueError("RoadNetwork.source_type cannot be empty")

        lane_ids: list[str] = []

        for lane in self.lanes:
            lane.validate()
            lane_ids.append(lane.lane_id)

        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("RoadNetwork contains duplicate lane IDs")

        lane_id_set = set(lane_ids)

        for lane in self.lanes:
            for predecessor_id in lane.predecessor_ids:
                if predecessor_id not in lane_id_set:
                    raise ValueError(
                        "Lane "
                        f"{lane.lane_id!r} references unknown "
                        "predecessor "
                        f"{predecessor_id!r}"
                    )

            for successor_id in lane.successor_ids:
                if successor_id not in lane_id_set:
                    raise ValueError(
                        "Lane "
                        f"{lane.lane_id!r} references unknown "
                        "successor "
                        f"{successor_id!r}"
                    )

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    def get_lane(
        self,
        lane_id: str,
    ) -> Lane:
        """
        根据 lane_id 获取车道。

        找不到时抛出 KeyError。
        """
        try:
            return self._lane_by_id[lane_id]
        except KeyError as error:
            raise KeyError(f"Unknown lane_id: {lane_id!r}") from error

    def find_lane(
        self,
        lane_id: str,
    ) -> Lane | None:
        """
        根据 lane_id 查询车道。

        找不到时返回 None。
        """
        return self._lane_by_id.get(lane_id)

    def driving_lanes(
        self,
    ) -> tuple[Lane, ...]:
        """返回所有可行驶车道。"""
        return tuple(lane for lane in self.lanes if lane.lane_type is LaneType.DRIVING)
