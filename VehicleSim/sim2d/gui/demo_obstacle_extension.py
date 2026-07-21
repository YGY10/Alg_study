from __future__ import annotations

import math

from sim2d.core.environment import DrivingEnv
from sim2d.types import BoxObstacle, CircleObstacle

_INSTALLED = False


def install() -> None:
    """将 GUI 示例障碍物从固定世界坐标改为相对自车坐标。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_reset = DrivingEnv.reset

    def reset(self: DrivingEnv, *args, **kwargs):
        initial_state = kwargs.get("initial_state")
        obstacles = kwargs.get("obstacles")

        if initial_state is not None and obstacles is not None:
            kwargs["obstacles"] = _transform_demo_obstacles(
                initial_state,
                tuple(obstacles),
            )

        return original_reset(self, *args, **kwargs)

    DrivingEnv.reset = reset
    _INSTALLED = True


def _transform_demo_obstacles(initial_state, obstacles):
    """
    main_window.py 中的两个示例障碍物原本写成固定世界坐标
    (12, 0) 和 (20, -2.4)。把它们解释为车辆坐标：

        x: 前向距离
        y: 左向距离

    再根据当前自车初始位姿转换到世界坐标。这样无论用户从地图
    哪个位置起步，示例目标都会位于自车附近，感知图中的相对位置
    也具有稳定语义。
    """
    c = math.cos(initial_state.yaw)
    s = math.sin(initial_state.yaw)

    transformed = []
    for obstacle in obstacles:
        is_demo_circle = (
            isinstance(obstacle, CircleObstacle)
            and obstacle.obstacle_id == "circle_001"
        )
        is_demo_box = (
            isinstance(obstacle, BoxObstacle)
            and obstacle.obstacle_id == "box_001"
        )

        if not (is_demo_circle or is_demo_box):
            transformed.append(obstacle)
            continue

        local_x = float(obstacle.x)
        local_y = float(obstacle.y)
        world_x = initial_state.x + c * local_x - s * local_y
        world_y = initial_state.y + s * local_x + c * local_y

        if is_demo_circle:
            transformed.append(
                CircleObstacle(
                    obstacle_id=obstacle.obstacle_id,
                    x=world_x,
                    y=world_y,
                    radius=obstacle.radius,
                )
            )
        else:
            transformed.append(
                BoxObstacle(
                    obstacle_id=obstacle.obstacle_id,
                    x=world_x,
                    y=world_y,
                    yaw=initial_state.yaw + obstacle.yaw,
                    length=obstacle.length,
                    width=obstacle.width,
                )
            )

    return tuple(transformed)


__all__ = ["install"]
