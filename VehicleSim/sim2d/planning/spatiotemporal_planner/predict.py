from __future__ import annotations

import numpy as np

from sim2d.perception import PerceivedObject

from .types import (
    ObjectPredictionSet,
    PredictedObjectTrajectory,
)


class ConstantVelocityPredictor:
    def predict(
        self,
        objects: tuple[PerceivedObject, ...],
        times: np.ndarray,
    ) -> ObjectPredictionSet:
        times = np.asarray(times, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(f"times must have shape [N], got {times.shape}")

        if times.size == 0:
            raise ValueError("times must not be empty")

        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")

        if abs(float(times[0])) > 1e-12:
            raise ValueError("times must start at zero")

        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing")

        trajectories = tuple(self._predict_object(obj, times) for obj in objects)

        return ObjectPredictionSet(
            times=times,
            trajectories=trajectories,
        )

    def _predict_object(
        self,
        obj: PerceivedObject,
        times: np.ndarray,
    ) -> PredictedObjectTrajectory:
        cosine = np.cos(obj.yaw)
        sine = np.sin(obj.yaw)

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
            speed=obj.speed,
            length=obj.length,
            width=obj.width,
            confidence=obj.confidence,
        )
