from __future__ import annotations

from sim2d.core.dynamics import (
    KinematicBicycleModel,
)
from sim2d.models.motion_model import (
    MotionModel,
)
from sim2d.types import (
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


class KinematicBicyclePredictionModel(MotionModel):
    """
    基于现有 KinematicBicycleModel 的无状态预测适配器。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
    ) -> None:
        vehicle_config.validate()

        self._model = KinematicBicycleModel(
            config=vehicle_config,
        )

    def predict_step(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        return self._model.step(
            state=state,
            control=control,
            dt=dt,
        )
