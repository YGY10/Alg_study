from __future__ import annotations

from abc import ABC, abstractmethod

from sim2d.types import VehicleControl, VehicleState


class DynamicsBackend(ABC):
    """车辆动力学后端统一接口。"""

    @abstractmethod
    def reset(self, initial_state: VehicleState) -> VehicleState:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> VehicleState:
        raise NotImplementedError
