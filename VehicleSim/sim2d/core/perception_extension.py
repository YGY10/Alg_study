from __future__ import annotations

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


def _perceived_obstacles(snapshot) -> tuple[object, ...]:
    obstacles: list[object] = []
    for item in snapshot.objects:
        if item.object_type == "circle":
            obstacles.append(
                CircleObstacle(
                    obstacle_id=item.object_id,
                    x=item.x,
                    y=item.y,
                    radius=0.5 * max(item.length, item.width),
                )
            )
        else:
            obstacles.append(
                BoxObstacle(
                    obstacle_id=item.object_id,
                    x=item.x,
                    y=item.y,
                    yaw=item.yaw,
                    length=item.length,
                    width=item.width,
                )
            )
    return tuple(obstacles)


def _get_planning_input(self: DrivingEnv) -> PlanningInput:
    """
    返回规划器输入。

    旧规划器继续读取 time/frame/ego/obstacles/goal；新规划器可进一步读取
    perception、map_network 和 previous_trajectory。
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
