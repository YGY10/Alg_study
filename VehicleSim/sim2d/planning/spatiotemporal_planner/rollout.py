from __future__ import annotations

import numpy as np

from sim2d.models.kinematic_bicycle import (
    KinematicBicyclePredictionModel,
)
from sim2d.types import VehicleConfig, VehicleControl, VehicleState

from .types import SpatiotemporalTrajectory


class TrajectoryRollout:
    """在规划时刻冻结的自车坐标系中执行车辆动力学 rollout。"""

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

    @staticmethod
    def local_initial_state(ego_state: VehicleState) -> VehicleState:
        """从当前自车状态创建冻结局部坐标系的初始状态。"""
        return VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=float(ego_state.speed),
        )

    def rollout_from_ego(
        self,
        ego_state: VehicleState,
        controls: np.ndarray,
    ) -> SpatiotemporalTrajectory:
        """忽略当前自车世界位姿，只保留速度并从局部原点 rollout。"""
        return self.rollout(
            initial_state=self.local_initial_state(ego_state),
            controls=controls,
        )

    def rollout(
        self,
        initial_state: VehicleState,
        controls: np.ndarray,
    ) -> SpatiotemporalTrajectory:
        """从给定局部状态生成冻结自车坐标系中的未来轨迹。

        ``initial_state`` 和输出的全部状态必须被解释为同一个固定局部
        坐标系中的量。常规规划入口应优先调用 ``rollout_from_ego``，确保
        每个规划周期都从 (0, 0, 0, current_speed) 开始。
        """
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
