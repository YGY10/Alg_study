from __future__ import annotations

import numpy as np

from sim2d.perception import PerceivedObject

from .types import (
    ObjectPredictionSet,
    PredictedObjectTrajectory,
)


class ConstantVelocityPredictor:
    """在规划时刻冻结的自车坐标系中进行恒速、恒航向预测。"""

    def predict(
        self,
        objects: tuple[PerceivedObject, ...],
        times: np.ndarray,
    ) -> ObjectPredictionSet:
        times = np.asarray(times, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(
                "times must have shape [N], "
                f"got {times.shape}"
            )
        if times.size == 0:
            raise ValueError("times must not be empty")
        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")
        if abs(float(times[0])) > 1e-12:
            raise ValueError("times must start at zero")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing")

        trajectories = tuple(
            self._predict_object(obj, times)
            for obj in objects
        )

        return ObjectPredictionSet(
            times=times,
            trajectories=trajectories,
        )

    @staticmethod
    def _predict_object(
        obj: PerceivedObject,
        times: np.ndarray,
    ) -> PredictedObjectTrajectory:
        cosine = float(np.cos(obj.yaw))
        sine = float(np.sin(obj.yaw))

        positions = np.column_stack(
            (
                obj.x + obj.speed * cosine * times,
                obj.y + obj.speed * sine * times,
            )
        )
        yaws = np.full(
            times.shape,
            obj.yaw,
            dtype=np.float64,
        )

        return PredictedObjectTrajectory(
            object_id=obj.object_id,
            object_type=obj.object_type,
            semantic_type=obj.semantic_type,
            times=times,
            positions=positions,
            yaws=yaws,
            speed=float(obj.speed),
            length=float(obj.length),
            width=float(obj.width),
            confidence=float(obj.confidence),
        )


__all__ = ["ConstantVelocityPredictor"]
