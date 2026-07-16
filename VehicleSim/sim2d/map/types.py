from __future__ import annotations

import math
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
class LaneProjection:
    """
    世界坐标点在车道中心线上的最近投影结果。

    lane_id:
        投影所属车道。

    point:
        中心线上的投影坐标 [x, y]。

    yaw:
        投影所在折线段的行驶方向，单位 rad。

    distance:
        查询点到投影点的欧氏距离，单位 m。

    lateral_offset:
        查询点相对车道中心线的有符号横向偏移。

        正值：
            查询点位于车道行驶方向左侧。

        负值：
            查询点位于车道行驶方向右侧。

    segment_index:
        投影所在中心线线段的起始点下标。

        对应线段：

            points[segment_index]
            →
            points[segment_index + 1]

    segment_ratio:
        投影在线段内的比例，范围 [0, 1]。

    arc_length:
        从车道中心线起点到投影点的累计长度。
    """

    lane_id: str
    point: FloatArray

    yaw: float
    distance: float
    lateral_offset: float

    segment_index: int
    segment_ratio: float
    arc_length: float

    def __post_init__(self) -> None:
        point = np.asarray(
            self.point,
            dtype=np.float64,
        )

        if point.shape != (2,):
            raise ValueError(
                "LaneProjection.point must have shape (2,), " f"got {point.shape}"
            )

        if not np.all(np.isfinite(point)):
            raise ValueError("LaneProjection.point must contain " "finite values")

        object.__setattr__(
            self,
            "point",
            point.copy(),
        )

        self.validate()

    def validate(self) -> None:
        if not self.lane_id.strip():
            raise ValueError("LaneProjection.lane_id cannot be empty")

        scalar_values = np.array(
            [
                self.yaw,
                self.distance,
                self.lateral_offset,
                self.segment_ratio,
                self.arc_length,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("LaneProjection contains non-finite values")

        if self.distance < 0.0:
            raise ValueError("LaneProjection.distance must be non-negative")

        if self.segment_index < 0:
            raise ValueError("LaneProjection.segment_index must " "be non-negative")

        if not (0.0 <= self.segment_ratio <= 1.0):
            raise ValueError("LaneProjection.segment_ratio must be " "within [0, 1]")

        if self.arc_length < 0.0:
            raise ValueError("LaneProjection.arc_length must " "be non-negative")


@dataclass(frozen=True)
class RoadNetwork:
    """
    VehicleSim 内部统一道路网络。

    所有外部地图格式最终都必须转换为该类型：

        OpenDRIVE
        nuScenes
        nuPlan
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

    def nearest_lane_point(
        self,
        *,
        x: float,
        y: float,
        lane_types: tuple[LaneType, ...] | None = (LaneType.DRIVING,),
        max_distance: float | None = None,
    ) -> LaneProjection | None:
        """
        查询世界坐标点在车道中心线上的最近投影。

        参数：
            x、y:
                世界坐标，单位 m。

            lane_types:
                允许参与查询的车道类型。

                默认只查询 DRIVING 车道。

                传入 None 时查询全部车道类型。

            max_distance:
                最大允许投影距离，单位 m。

                查询结果超过该距离时返回 None。
                None 表示不限制距离。

        返回：
            最近的 LaneProjection。

            没有满足条件的车道时返回 None。

        当前实现会遍历全部车道和全部中心线段。
        后续地图规模增大后，可以在不改变接口的情况下
        使用空间索引加速。
        """
        query = np.array(
            [
                x,
                y,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(query)):
            raise ValueError("Projection query coordinates must " "be finite")

        if max_distance is not None:
            if not np.isfinite(max_distance):
                raise ValueError("max_distance must be finite")

            if max_distance < 0.0:
                raise ValueError("max_distance must be non-negative")

        allowed_types: frozenset[LaneType] | None

        if lane_types is None:
            allowed_types = None

        else:
            if not isinstance(
                lane_types,
                tuple,
            ):
                lane_types = tuple(lane_types)

            for lane_type in lane_types:
                if not isinstance(
                    lane_type,
                    LaneType,
                ):
                    raise TypeError("lane_types must contain " "LaneType values")

            allowed_types = frozenset(lane_types)

        best_projection: LaneProjection | None = None
        best_distance_squared = math.inf

        for lane in self.lanes:
            if allowed_types is not None and lane.lane_type not in allowed_types:
                continue

            projection = self._project_to_lane(
                query=query,
                lane=lane,
            )

            if projection is None:
                continue

            distance_squared = projection.distance * projection.distance

            if distance_squared < best_distance_squared:
                best_projection = projection
                best_distance_squared = distance_squared

        if best_projection is None:
            return None

        if max_distance is not None and best_projection.distance > max_distance:
            return None

        return best_projection

    @staticmethod
    def _project_to_lane(
        *,
        query: FloatArray,
        lane: Lane,
    ) -> LaneProjection | None:
        """
        将点投影到单条车道中心线。

        退化的零长度线段会被跳过。
        """
        points = lane.centerline.points

        segment_starts = points[:-1]
        segment_ends = points[1:]

        segment_vectors = segment_ends - segment_starts

        segment_length_squared = np.einsum(
            "ij,ij->i",
            segment_vectors,
            segment_vectors,
        )

        valid_segments = segment_length_squared > 1e-18

        if not np.any(valid_segments):
            return None

        query_vectors = query[None, :] - segment_starts

        segment_ratios = np.zeros(
            segment_length_squared.shape,
            dtype=np.float64,
        )

        segment_ratios[valid_segments] = (
            np.einsum(
                "ij,ij->i",
                query_vectors[valid_segments],
                segment_vectors[valid_segments],
            )
            / segment_length_squared[valid_segments]
        )

        segment_ratios = np.clip(
            segment_ratios,
            0.0,
            1.0,
        )

        projected_points = segment_starts + segment_ratios[:, None] * segment_vectors

        error_vectors = query[None, :] - projected_points

        distance_squared = np.einsum(
            "ij,ij->i",
            error_vectors,
            error_vectors,
        )

        distance_squared[~valid_segments] = math.inf

        segment_index = int(np.argmin(distance_squared))

        best_distance_squared = float(distance_squared[segment_index])

        if not np.isfinite(best_distance_squared):
            return None

        segment_vector = segment_vectors[segment_index]

        segment_length = math.sqrt(float(segment_length_squared[segment_index]))

        tangent = segment_vector / segment_length

        projected_point = projected_points[segment_index]

        offset_vector = query - projected_point

        # 二维叉乘 tangent × offset。
        # 正值表示查询点位于行驶方向左侧。
        lateral_offset = float(tangent[0]) * float(offset_vector[1]) - float(
            tangent[1]
        ) * float(offset_vector[0])

        segment_lengths = np.sqrt(
            np.maximum(
                segment_length_squared,
                0.0,
            )
        )

        arc_length_before_segment = float(np.sum(segment_lengths[:segment_index]))

        segment_ratio = float(segment_ratios[segment_index])

        arc_length = arc_length_before_segment + segment_ratio * segment_length

        yaw = math.atan2(
            float(segment_vector[1]),
            float(segment_vector[0]),
        )

        return LaneProjection(
            lane_id=lane.lane_id,
            point=projected_point,
            yaw=yaw,
            distance=math.sqrt(best_distance_squared),
            lateral_offset=lateral_offset,
            segment_index=segment_index,
            segment_ratio=segment_ratio,
            arc_length=arc_length,
        )
