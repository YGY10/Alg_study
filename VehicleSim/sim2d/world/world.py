from __future__ import annotations

import math
from dataclasses import dataclass

from sim2d.map.topology_repair import (
    repair_road_network_topology,
)
from sim2d.map.types import RoadNetwork, TrafficSignal
from sim2d.types import VehicleState
from sim2d.world.road_deformation import (
    RoadDeformationConfig,
    deform_pose,
    deform_road_network,
)
from sim2d.world.state import WorldState
from sim2d.world.traffic_signal import (
    TrafficLightState,
    WorldTrafficSignal,
)


@dataclass
class World:
    """仿真真实世界容器。"""

    state: WorldState

    def step(
        self,
        dt: float,
        *,
        ego_state: VehicleState | None = None,
    ) -> WorldState:
        if not isinstance(dt, (int, float)):
            raise TypeError(
                "dt must be a real number, "
                f"got {type(dt).__name__}"
            )

        dt = float(dt)

        if not math.isfinite(dt):
            raise ValueError(
                f"dt must be finite, got {dt}"
            )

        if dt <= 0.0:
            raise ValueError(
                f"dt must be positive, got {dt}"
            )

        if ego_state is not None:
            self.state.ego_state = ego_state

        for signal in self.state.traffic_signals:
            signal.advance(dt)

        self.state.time += dt

        return self.state

    def initialize_map_geometry(
        self,
        road_network: RoadNetwork,
        deformation: RoadDeformationConfig,
        *,
        initial_signal_state: TrafficLightState = TrafficLightState.RED,
    ) -> None:
        """
        使用同一连续形变场初始化世界道路与世界交通灯。

        世界交通灯按索引分配相位偏移，使同一地图上的灯不会全部同步
        变色。当前是确定性测试周期，后续可替换为 controller 分组周期。
        """
        repaired_network = repair_road_network_topology(
            road_network
        )

        self.state.road_lanes = deform_road_network(
            repaired_network,
            deformation,
        )

        self.state.traffic_signals = tuple(
            self._deform_traffic_signal(
                signal,
                deformation,
                initial_signal_state,
                phase_offset=(index % 4) * 6.25,
            )
            for index, signal in enumerate(
                repaired_network.traffic_signals
            )
        )

    @staticmethod
    def _deform_traffic_signal(
        signal: TrafficSignal,
        deformation: RoadDeformationConfig,
        state: TrafficLightState,
        *,
        phase_offset: float = 0.0,
    ) -> WorldTrafficSignal:
        x, y, yaw = deform_pose(
            x=signal.x,
            y=signal.y,
            yaw=signal.yaw,
            config=deformation,
        )

        signal_id = str(signal.signal_id)

        return WorldTrafficSignal(
            entity_id=f"world_signal_{signal_id}",
            map_signal_id=signal_id,
            x=x,
            y=y,
            yaw=yaw,
            state=state,
            phase_offset=phase_offset,
        )

    def initialize_traffic_signals(
        self,
        map_signals: tuple[TrafficSignal, ...],
        *,
        position_offset_x: float = 0.0,
        position_offset_y: float = 0.0,
        yaw_offset: float = 0.0,
        initial_state: TrafficLightState = TrafficLightState.RED,
    ) -> tuple[WorldTrafficSignal, ...]:
        """保留旧接口，用统一刚体偏差初始化动态交通灯。"""
        world_signals = tuple(
            WorldTrafficSignal.from_map_signal(
                signal,
                position_offset_x=position_offset_x,
                position_offset_y=position_offset_y,
                yaw_offset=yaw_offset,
                state=initial_state,
                phase_offset=(index % 4) * 6.25,
            )
            for index, signal in enumerate(map_signals)
        )

        self.state.traffic_signals = world_signals

        return world_signals
