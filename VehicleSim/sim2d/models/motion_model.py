from __future__ import annotations

from abc import ABC, abstractmethod

from sim2d.types import (
    VehicleControl,
    VehicleState,
)


class MotionModel(ABC):
    """
    供规划器闭环 rollout 使用的无状态车辆运动模型。

    与 DynamicsBackend 的区别：

    - DynamicsBackend 持有真实仿真状态；
    - MotionModel 从调用方提供的任意假设状态进行一步预测。
    """

    @abstractmethod
    def predict_step(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        """
        从给定假设状态预测一个时间步。
        """
        raise NotImplementedError
