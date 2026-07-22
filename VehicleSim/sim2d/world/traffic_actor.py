from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from sim2d.types import BoxObstacle, CircleObstacle, Obstacle


class TrafficActorType(str, Enum):
    """规划与感知使用的交通参与者语义类型。"""

    PEDESTRIAN = "pedestrian"
    SMALL_CAR = "small_car"
    LARGE_VEHICLE = "large_vehicle"


@dataclass(frozen=True)
class PedestrianObstacle(CircleObstacle):
    """行人：圆形碰撞包络，但对外语义始终是 pedestrian。"""

    yaw: float = 0.0
    speed: float = 0.0
    semantic_type: str = TrafficActorType.PEDESTRIAN.value

    def validate(self) -> None:
        super().validate()
        _validate_motion(self.yaw, self.speed)


@dataclass(frozen=True)
class SmallCarObstacle(BoxObstacle):
    """小车：有方向矩形碰撞包络。"""

    speed: float = 0.0
    semantic_type: str = TrafficActorType.SMALL_CAR.value

    def validate(self) -> None:
        super().validate()
        _validate_motion(self.yaw, self.speed)


@dataclass(frozen=True)
class LargeVehicleObstacle(BoxObstacle):
    """大车：有方向矩形碰撞包络。"""

    speed: float = 0.0
    semantic_type: str = TrafficActorType.LARGE_VEHICLE.value

    def validate(self) -> None:
        super().validate()
        _validate_motion(self.yaw, self.speed)


TrafficActor = PedestrianObstacle | SmallCarObstacle | LargeVehicleObstacle


def create_traffic_actor(
    actor_type: TrafficActorType | str,
    *,
    actor_id: str,
    x: float,
    y: float,
    yaw: float = 0.0,
    speed: float = 0.0,
) -> TrafficActor:
    """代码场景与 GUI 共用的语义交通参与者工厂。

    x、y、yaw 均为世界坐标；speed 沿 actor 自身 yaw 方向运动。
    """
    normalized = TrafficActorType(str(actor_type))

    if normalized is TrafficActorType.PEDESTRIAN:
        actor: TrafficActor = PedestrianObstacle(
            obstacle_id=actor_id,
            x=float(x),
            y=float(y),
            radius=0.35,
            yaw=float(yaw),
            speed=float(speed),
        )
    elif normalized is TrafficActorType.SMALL_CAR:
        actor = SmallCarObstacle(
            obstacle_id=actor_id,
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            length=4.5,
            width=1.8,
            speed=float(speed),
        )
    else:
        actor = LargeVehicleObstacle(
            obstacle_id=actor_id,
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            length=10.0,
            width=2.5,
            speed=float(speed),
        )

    actor.validate()
    return actor


def advance_obstacle(obstacle: Obstacle, dt: float) -> Obstacle:
    """按恒速、恒航向推进支持运动属性的交通参与者。"""
    speed = float(getattr(obstacle, "speed", 0.0))
    yaw = float(getattr(obstacle, "yaw", 0.0))
    if abs(speed) <= 1e-12:
        return obstacle
    return replace(
        obstacle,
        x=float(obstacle.x) + speed * math.cos(yaw) * float(dt),
        y=float(obstacle.y) + speed * math.sin(yaw) * float(dt),
    )


def semantic_type_of(obstacle: Obstacle) -> str:
    return str(getattr(obstacle, "semantic_type", "unknown"))


def _validate_motion(yaw: float, speed: float) -> None:
    if not math.isfinite(float(yaw)):
        raise ValueError("traffic actor yaw must be finite")
    if not math.isfinite(float(speed)) or float(speed) < 0.0:
        raise ValueError("traffic actor speed must be finite and non-negative")


__all__ = [
    "LargeVehicleObstacle",
    "PedestrianObstacle",
    "SmallCarObstacle",
    "TrafficActor",
    "TrafficActorType",
    "advance_obstacle",
    "create_traffic_actor",
    "semantic_type_of",
]
