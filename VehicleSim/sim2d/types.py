from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ObstacleShape(str, Enum):
    """仿真器第一阶段支持的障碍物形状。"""

    CIRCLE = "circle"
    BOX = "box"


@dataclass(frozen=True)
class VehicleState:
    """
    车辆状态。

    坐标约定：
        x、y：世界坐标，单位 m
        yaw：航向角，单位 rad
        speed：纵向速度，单位 m/s

    yaw = 0 表示车头朝 +x。
    yaw 正方向为逆时针。
    """

    x: float
    y: float
    yaw: float
    speed: float

    def as_array(self) -> FloatArray:
        """转换为规划器常用的 NumPy 状态向量。"""
        return np.array(
            [
                self.x,
                self.y,
                self.yaw,
                self.speed,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        value: FloatArray,
    ) -> VehicleState:
        """从长度为 4 的数组创建车辆状态。"""
        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if array.shape != (4,):
            raise ValueError(
                "VehicleState array must have " f"shape (4,), got {array.shape}"
            )

        return cls(
            x=float(array[0]),
            y=float(array[1]),
            yaw=float(array[2]),
            speed=float(array[3]),
        )


@dataclass(frozen=True)
class VehicleControl:
    """
    车辆控制量。

    acceleration：
        纵向加速度，单位 m/s²。

    steering：
        前轮转角，单位 rad。
    """

    acceleration: float
    steering: float

    def as_array(self) -> FloatArray:
        return np.array(
            [
                self.acceleration,
                self.steering,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        value: FloatArray,
    ) -> VehicleControl:
        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if array.shape != (2,):
            raise ValueError(
                "VehicleControl array must have " f"shape (2,), got {array.shape}"
            )

        return cls(
            acceleration=float(array[0]),
            steering=float(array[1]),
        )


@dataclass(frozen=True)
class VehicleConfig:
    """车辆几何尺寸和控制限制。"""

    length: float = 4.6
    width: float = 1.85
    wheel_base: float = 2.7

    acceleration_min: float = -3.0
    acceleration_max: float = 2.0

    steering_min: float = -0.45
    steering_max: float = 0.45

    speed_min: float = 0.0
    speed_max: float = 15.0

    def validate(self) -> None:
        if self.length <= 0.0:
            raise ValueError("Vehicle length must be positive")

        if self.width <= 0.0:
            raise ValueError("Vehicle width must be positive")

        if self.wheel_base <= 0.0:
            raise ValueError("Wheel base must be positive")

        if self.acceleration_min >= self.acceleration_max:
            raise ValueError("acceleration_min must be less than " "acceleration_max")

        if self.steering_min >= self.steering_max:
            raise ValueError("steering_min must be less than " "steering_max")

        if self.speed_min >= self.speed_max:
            raise ValueError("speed_min must be less than speed_max")


@dataclass(frozen=True)
class CircleObstacle:
    """圆形静态障碍物。"""

    obstacle_id: str
    x: float
    y: float
    radius: float

    shape: ObstacleShape = field(
        default=ObstacleShape.CIRCLE,
        init=False,
    )

    def validate(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id cannot be empty")

        if self.radius <= 0.0:
            raise ValueError("Circle obstacle radius must be positive")


@dataclass(frozen=True)
class BoxObstacle:
    """
    有方向矩形障碍物。

    yaw 的定义与车辆航向角一致。
    """

    obstacle_id: str
    x: float
    y: float
    yaw: float
    length: float
    width: float

    shape: ObstacleShape = field(
        default=ObstacleShape.BOX,
        init=False,
    )

    def validate(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id cannot be empty")

        if self.length <= 0.0:
            raise ValueError("Box obstacle length must be positive")

        if self.width <= 0.0:
            raise ValueError("Box obstacle width must be positive")


Obstacle = CircleObstacle | BoxObstacle


@dataclass(frozen=True)
class GoalState:
    """规划终点及允许误差。"""

    state: VehicleState

    position_tolerance: float = 0.5
    yaw_tolerance: float = 0.15
    speed_tolerance: float = 0.5

    def validate(self) -> None:
        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive")

        if self.yaw_tolerance <= 0.0:
            raise ValueError("yaw_tolerance must be positive")

        if self.speed_tolerance <= 0.0:
            raise ValueError("speed_tolerance must be positive")


@dataclass(frozen=True)
class Observation:
    """
    提供给规划器或 Torch 模型的观测。

    当前阶段包含：
        仿真时间
        当前帧号
        自车状态
        障碍物
        目标状态

    road_network 后面接入统一地图接口时再加入。
    """

    time: float
    frame: int
    ego: VehicleState
    obstacles: tuple[Obstacle, ...]
    goal: GoalState


@dataclass(frozen=True)
class PlanResult:
    """
    规划器统一返回结构。
    """

    action: VehicleControl
    trajectory: FloatArray | None = None
    controls: FloatArray | None = None
    reference_path: FloatArray | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Environment.step() 的返回结果。"""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrafficSignalSnapshot:
    """
    提供给 GUI 的只读真实世界交通灯快照。

    这里不直接暴露可变的 WorldTrafficSignal，避免 GUI 持有并修改
    WorldState 中的 ground-truth 实体。
    """

    entity_id: str
    map_signal_id: str | None

    x: float
    y: float
    yaw: float

    state: str
    remaining_time: float | None = None


@dataclass(frozen=True)
class SimulationSnapshot:
    """
    专门提供给 GUI 的只读仿真快照。

    GUI 只读取 SimulationSnapshot，
    不直接访问 DrivingEnv 内部状态。
    """

    frame: int
    time: float

    ego: VehicleState
    obstacles: tuple[Obstacle, ...]

    planned_trajectory: FloatArray | None
    reference_path: FloatArray | None
    history_trajectory: FloatArray

    collision: bool
    min_clearance: float | None

    world_traffic_signals: tuple[TrafficSignalSnapshot, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)
