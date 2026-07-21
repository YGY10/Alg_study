from __future__ import annotations

import math
from typing import Protocol

from sim2d.core.environment import DrivingEnv
from sim2d.perception import (
    GroundTruthLocalPerception,
    PerceptionConfig,
    PlanningInput,
)
from sim2d.types import BoxObstacle, CircleObstacle, PlanResult

_INSTALLED = False


class PlannerProtocol(Protocol):
    """规划器只依赖稳定的输入/输出契约。"""

    def plan(self, planning_input: PlanningInput) -> PlanResult:
        ...

    def reset(self) -> None:
        ...


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_post_init = DrivingEnv.__post_init__
    original_reset = DrivingEnv.reset

    def post_init(self: DrivingEnv) -> None:
        original_post_init(self)
        self._perception_model = GroundTruthLocalPerception()
        self._perception_map_network = None

    def reset(self: DrivingEnv, *args, **kwargs):
        self._perception_model.reset()
        return original_reset(self, *args, **kwargs)

    DrivingEnv.__post_init__ = post_init
    DrivingEnv.reset = reset
    DrivingEnv.set_perception_config = _set_perception_config
    DrivingEnv.set_perception_map_network = _set_perception_map_network
    DrivingEnv.get_perception_snapshot = _get_perception_snapshot
    DrivingEnv.get_planning_input = _get_planning_input
    DrivingEnv.get_observation = _get_planning_input

    _INSTALLED = True


def _set_perception_config(
    self: DrivingEnv,
    config: PerceptionConfig,
) -> None:
    config.validate()
    self._perception_model = GroundTruthLocalPerception(config)


def _set_perception_map_network(
    self: DrivingEnv,
    road_network,
) -> None:
    """保存地图先验引用；感知本身仍只读取 World。"""
    self._perception_map_network = road_network


def _get_perception_snapshot(self: DrivingEnv):
    self._require_initialized()
    return self._perception_model.observe(
        self.world.state,
        frame=self.frame,
    )


def _vehicle_to_world(snapshot, x: float, y: float) -> tuple[float, float]:
    """将车辆坐标系点转换回全局坐标，仅供旧规划器兼容字段使用。"""
    ego = snapshot.ego
    c = math.cos(ego.yaw)
    s = math.sin(ego.yaw)
    return (
        ego.x + c * float(x) - s * float(y),
        ego.y + s * float(x) + c * float(y),
    )


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _perceived_obstacles(snapshot) -> tuple[object, ...]:
    """生成旧规划器使用的全局坐标障碍物兼容视图。

    新规划器应直接使用 snapshot.objects，其中所有目标均为车辆坐标系。
    """
    obstacles: list[object] = []
    for item in snapshot.objects:
        world_x, world_y = _vehicle_to_world(snapshot, item.x, item.y)
        if item.object_type == "circle":
            obstacles.append(
                CircleObstacle(
                    obstacle_id=item.object_id,
                    x=world_x,
                    y=world_y,
                    radius=0.5 * max(item.length, item.width),
                )
            )
        else:
            obstacles.append(
                BoxObstacle(
                    obstacle_id=item.object_id,
                    x=world_x,
                    y=world_y,
                    yaw=_normalize_angle(snapshot.ego.yaw + item.yaw),
                    length=item.length,
                    width=item.width,
                )
            )
    return tuple(obstacles)


def _get_planning_input(self: DrivingEnv) -> PlanningInput:
    """返回规划器输入。

    perception 内的目标、交通灯和车道几何均为车辆坐标系。为保持现有
    BezierPlanner 可运行，顶层 obstacles 仍提供转换后的全局坐标兼容视图。
    """
    self._require_initialized()
    assert self._goal is not None

    perception = _get_perception_snapshot(self)
    previous = self._copy_optional_array(self._planned_trajectory)

    return PlanningInput(
        time=perception.publish_time,
        frame=self.frame,
        ego=perception.ego,
        obstacles=_perceived_obstacles(perception),
        goal=self._goal,
        perception=perception,
        map_network=self._perception_map_network,
        previous_trajectory=previous,
    )
