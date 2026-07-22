from __future__ import annotations

import numpy as np

from sim2d.models.kinematic_bicycle import (
    KinematicBicyclePredictionModel,
)
from sim2d.types import VehicleConfig, VehicleControl, VehicleState

from .types import SpatiotemporalTrajectory


class TrajectoryRollout:
    def __init__(
        self,
        vehicle_config: VehicleConfig,
        dt: float,
    ) -> None:
        vehicle_config.validate()

        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        self.dt = dt
        self.vehicle_config = vehicle_config
        self.motion_model = KinematicBicyclePredictionModel(
            vehicle_config=vehicle_config,
        )

    def rollout(
        self,
        initial_state: VehicleState,
        controls: np.ndarray,
    ) -> SpatiotemporalTrajectory:
        controls = np.asarray(controls, dtype=np.float64)

        if controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError(
                "controls must have shape [N, 2], "
                f"got {controls.shape}"
            )

        if not np.all(np.isfinite(controls)):
            raise ValueError("controls contain non-finite values")

        states = [initial_state.as_array()]
        current_state = initial_state
        clipped_controls = np.empty_like(controls)

        for index, row in enumerate(controls):
            acceleration = float(
                np.clip(
                    row[0],
                    self.vehicle_config.acceleration_min,
                    self.vehicle_config.acceleration_max,
                )
            )

            steering = float(
                np.clip(
                    row[1],
                    self.vehicle_config.steering_min,
                    self.vehicle_config.steering_max,
                )
            )

            control = VehicleControl(
                acceleration=acceleration,
                steering=steering,
            )

            clipped_controls[index] = (
                acceleration,
                steering,
            )

            current_state = self.motion_model.predict_step(
                current_state,
                control,
                self.dt,
            )

            states.append(current_state.as_array())

        times = np.arange(
            len(states),
            dtype=np.float64,
        ) * self.dt

        return SpatiotemporalTrajectory(
            times=times,
            states=np.asarray(states, dtype=np.float64),
            controls=clipped_controls,
        )
