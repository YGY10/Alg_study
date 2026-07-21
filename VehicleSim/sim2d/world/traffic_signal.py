from __future__ import annotations

import math
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
    真实世界中的动态交通灯实体。

    Map 层 TrafficSignal 只描述静态先验；本类型保存真实位姿、当前灯色
    以及完整周期参数。World.step() 每帧调用 advance() 推进状态机。
    """

    entity_id: str
    map_signal_id: str | None

    x: float
    y: float
    yaw: float

    state: TrafficLightState = TrafficLightState.UNKNOWN
    remaining_time: float | None = None

    red_duration: float = 12.0
    green_duration: float = 10.0
    yellow_duration: float = 3.0
    phase_offset: float = 0.0

    def __post_init__(self) -> None:
        durations = (
            self.red_duration,
            self.green_duration,
            self.yellow_duration,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("traffic-light durations must be finite and positive")
        if not math.isfinite(self.phase_offset):
            raise ValueError("phase_offset must be finite")

        # UNKNOWN 仅用于尚未配置周期的临时状态。配置了正常周期后，
        # 初始化为红灯并给出准确剩余时间。
        if self.state is TrafficLightState.UNKNOWN:
            self.state = TrafficLightState.RED

        if self.remaining_time is None:
            self.set_cycle_time(self.phase_offset)

    @property
    def cycle_duration(self) -> float:
        return self.red_duration + self.green_duration + self.yellow_duration

    def set_cycle_time(self, cycle_time: float) -> None:
        """按周期内绝对时间设置灯色和剩余时间。"""
        if not math.isfinite(cycle_time):
            raise ValueError("cycle_time must be finite")

        phase = cycle_time % self.cycle_duration

        if phase < self.red_duration:
            self.state = TrafficLightState.RED
            self.remaining_time = self.red_duration - phase
            return

        phase -= self.red_duration
        if phase < self.green_duration:
            self.state = TrafficLightState.GREEN
            self.remaining_time = self.green_duration - phase
            return

        phase -= self.green_duration
        self.state = TrafficLightState.YELLOW
        self.remaining_time = self.yellow_duration - phase

    def advance(self, dt: float) -> None:
        """推进一个仿真时间步，支持跨越多个灯色阶段。"""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        remaining = self.remaining_time
        if remaining is None:
            self.set_cycle_time(self.phase_offset)
            remaining = self.remaining_time

        assert remaining is not None
        remaining -= dt

        while remaining <= 1.0e-12:
            overshoot = -remaining
            if self.state is TrafficLightState.RED:
                self.state = TrafficLightState.GREEN
                remaining = self.green_duration - overshoot
            elif self.state is TrafficLightState.GREEN:
                self.state = TrafficLightState.YELLOW
                remaining = self.yellow_duration - overshoot
            else:
                self.state = TrafficLightState.RED
                remaining = self.red_duration - overshoot

        self.remaining_time = remaining

    @classmethod
    def from_map_signal(
        cls,
        signal: TrafficSignal,
        *,
        position_offset_x: float = 0.0,
        position_offset_y: float = 0.0,
        yaw_offset: float = 0.0,
        state: TrafficLightState = TrafficLightState.RED,
        phase_offset: float = 0.0,
    ) -> WorldTrafficSignal:
        signal_id = str(signal.signal_id)

        return cls(
            entity_id=f"world_signal_{signal_id}",
            map_signal_id=signal_id,
            x=float(signal.x) + float(position_offset_x),
            y=float(signal.y) + float(position_offset_y),
            yaw=float(signal.yaw) + float(yaw_offset),
            state=state,
            phase_offset=phase_offset,
        )
