from __future__ import annotations

import math
from dataclasses import dataclass

from sim2d.map.types import TrafficSignal
from sim2d.types import VehicleState
from sim2d.world.state import WorldState
from sim2d.world.traffic_signal import (
    TrafficLightState,
    WorldTrafficSignal,
)


@dataclass
class World:
    """
    仿真真实世界容器。

    World 负责提交同一个仿真时间步中的 ground-truth 状态更新：

        - 推进世界时间；
        - 更新自车真实状态；
        - 初始化和维护真实世界交通灯；
        - 后续更新交通灯周期、NPC、行人等世界实体。

    车辆动力学计算暂时仍由 DrivingEnv 中的 DynamicsBackend 完成。
    World.step() 只接收已经计算出的下一时刻自车状态。
    """

    state: WorldState

    def step(
        self,
        dt: float,
        *,
        ego_state: VehicleState | None = None,
    ) -> WorldState:
        """
        将真实世界推进一个仿真时间步。
        """
        if not isinstance(
            dt,
            (int, float),
        ):
            raise TypeError("dt must be a real number, " f"got {type(dt).__name__}")

        dt = float(dt)

        if not math.isfinite(dt):
            raise ValueError(f"dt must be finite, got {dt}")

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        if ego_state is not None:
            self.state.ego_state = ego_state

        self.state.time += dt

        return self.state

    def initialize_traffic_signals(
        self,
        map_signals: tuple[TrafficSignal, ...],
        *,
        position_offset_x: float = 0.0,
        position_offset_y: float = 0.0,
        yaw_offset: float = 0.0,
        initial_state: TrafficLightState = TrafficLightState.UNKNOWN,
    ) -> tuple[WorldTrafficSignal, ...]:
        """
        从地图层交通灯创建真实世界交通灯实体。

        当前对所有地图灯应用相同的位姿偏差，先建立最小可用链路。
        后续可以扩展为按 signal_id 配置不同误差、缺失灯和额外灯。

        再次调用该方法会整体替换当前 WorldState.traffic_signals。
        """
        world_signals = tuple(
            WorldTrafficSignal.from_map_signal(
                signal,
                position_offset_x=position_offset_x,
                position_offset_y=position_offset_y,
                yaw_offset=yaw_offset,
                state=initial_state,
            )
            for signal in map_signals
        )

        self.state.traffic_signals = world_signals

        return world_signals
