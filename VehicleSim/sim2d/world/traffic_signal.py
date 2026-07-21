from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sim2d.map.types import TrafficSignal


class TrafficLightState(str, Enum):
    UNKNOWN = "unknown"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"


@dataclass
class WorldTrafficSignal:
    """
    真实世界中的交通灯实体。

    与地图层 TrafficSignal 的区别：

    Map:
        描述地图先验中“这里应该有一个灯”。

    World:
        描述真实世界中“当前实际存在的灯”，包括真实位姿
        和当前动态状态。

    map_signal_id:
        与地图层 TrafficSignal.signal_id 的关联 ID。
        可以为 None，表示真实世界中存在、但地图中没有对应记录。

    x/y/yaw:
        真实世界 ground-truth 位姿，可以与地图层存在偏差。

    state:
        当前真实灯色。
    """

    entity_id: str
    map_signal_id: str | None

    x: float
    y: float
    yaw: float

    state: TrafficLightState = TrafficLightState.UNKNOWN
    remaining_time: float | None = None

    @classmethod
    def from_map_signal(
        cls,
        signal: TrafficSignal,
        *,
        position_offset_x: float = 0.0,
        position_offset_y: float = 0.0,
        yaw_offset: float = 0.0,
        state: TrafficLightState = TrafficLightState.UNKNOWN,
    ) -> WorldTrafficSignal:
        """
        根据地图层交通灯创建真实世界交通灯。

        offset 参数用于模拟地图层与真实世界层之间的位姿误差。
        当前默认 offset 全为 0，因此初始化时两层完全重合。
        """
        signal_id = str(signal.signal_id)

        return cls(
            entity_id=f"world_signal_{signal_id}",
            map_signal_id=signal_id,
            x=float(signal.x) + float(position_offset_x),
            y=float(signal.y) + float(position_offset_y),
            yaw=float(signal.yaw) + float(yaw_offset),
            state=state,
            remaining_time=None,
        )
