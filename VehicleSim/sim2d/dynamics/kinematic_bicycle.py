from __future__ import annotations

from dataclasses import replace

from sim2d.core.dynamics import KinematicBicycleModel
from sim2d.dynamics.base import DynamicsBackend
from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


class KinematicBicycleBackend(DynamicsBackend):
    """基于现有 KinematicBicycleModel 的后端封装。"""

    def __init__(self, vehicle_config: VehicleConfig) -> None:
        vehicle_config.validate()
        self._model = KinematicBicycleModel(
            config=vehicle_config,
        )
        self._state: VehicleState | None = None

    def reset(
        self,
        initial_state: VehicleState,
    ) -> VehicleState:
        self._state = replace(initial_state)
        return self._state

    def step(
        self,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        if self._state is None:
            raise RuntimeError(
                "Dynamics backend has not been reset. " "Call reset() before step()."
            )

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._state = self._model.step(
            state=self._state,
            control=control,
            dt=dt,
        )
        return self._state

    def get_state(self) -> VehicleState:
        if self._state is None:
            raise RuntimeError(
                "Dynamics backend has not been reset. "
                "Call reset() before get_state()."
            )

        return self._state
