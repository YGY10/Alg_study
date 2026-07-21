from __future__ import annotations

from dataclasses import dataclass

from sim2d.types import (
    Obstacle,
    VehicleState,
)

from sim2d.world.traffic_signal import (
    WorldTrafficSignal,
)


@dataclass
class WorldState:
    """
    仿真世界某一时刻的真实状态。

    注意：

    这里保存的是 World Layer 数据。

    与 Map Layer 区别：

        map:
            OpenDRIVE 中的静态道路、
            静态交通灯位置。

        world:
            当前仿真时刻真实存在的状态。

    后续扩展：

        obstacles:
            动态障碍物、静态障碍物。

        traffic_signals:
            当前红绿灯状态。

        npc_vehicles:
            非自车车辆。

        pedestrians:
            行人。
    """

    # 仿真时间
    time: float

    # 自车真实状态
    ego_state: VehicleState

    # 世界中的障碍物
    obstacles: tuple[Obstacle, ...] = ()

    # 世界中的真实交通灯状态
    traffic_signals: tuple[WorldTrafficSignal, ...] = ()
