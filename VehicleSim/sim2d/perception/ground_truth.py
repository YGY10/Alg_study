from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import numpy as np

from sim2d.perception.types import (
    PerceivedLaneSegment,
    PerceivedObject,
    PerceivedTrafficSignal,
    PerceptionConfig,
    PerceptionSnapshot,
)
from sim2d.types import BoxObstacle, CircleObstacle, VehicleState
from sim2d.world.state import WorldState


class GroundTruthLocalPerception:
    """
    第一版局部真值感知。

    数据来自 World，但只发布自车周围有限范围内的信息。配置中已经预留
    噪声、漏检、刷新周期和延迟，规划器无需因后续误差模型而修改接口。
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._history: deque[PerceptionSnapshot] = deque()
        self._latest_measurement: PerceptionSnapshot | None = None

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.random_seed)
        self._history.clear()
        self._latest_measurement = None

    def observe(self, world_state: WorldState, *, frame: int) -> PerceptionSnapshot:
        now = float(world_state.time)
        latest = self._latest_measurement
        if latest is None or now - latest.measurement_time + 1e-12 >= self.config.update_period:
            latest = self._measure(world_state, frame=frame)
            self._latest_measurement = latest
            self._history.append(latest)

        cutoff = now - self.config.latency
        published = self._history[0]
        for candidate in self._history:
            if candidate.measurement_time <= cutoff + 1e-12:
                published = candidate
            else:
                break

        while len(self._history) > 2 and self._history[1].measurement_time <= cutoff:
            self._history.popleft()

        return replace(published, publish_time=now)

    def _measure(self, world_state: WorldState, *, frame: int) -> PerceptionSnapshot:
        ego = self._noisy_ego(world_state.ego_state)
        objects = tuple(
            item
            for obstacle in world_state.obstacles
            if self._inside_view(ego, float(obstacle.x), float(obstacle.y))
            if self._rng.random() >= self.config.object_dropout_probability
            for item in (self._perceive_obstacle(obstacle),)
        )
        signals = tuple(
            self._perceive_signal(signal)
            for signal in world_state.traffic_signals
            if self._inside_view(ego, signal.x, signal.y)
            if self._rng.random() >= self.config.signal_dropout_probability
        )
        road_segments = tuple(
            segment
            for lane in world_state.road_lanes
            if self._rng.random() >= self.config.lane_dropout_probability
            for segment in (self._clip_lane(lane, ego),)
            if segment is not None
        )

        return PerceptionSnapshot(
            measurement_time=float(world_state.time),
            publish_time=float(world_state.time),
            frame=int(frame),
            ego=ego,
            objects=objects,
            traffic_signals=signals,
            road_segments=road_segments,
            debug={
                "forward_range": self.config.forward_range,
                "rear_range": self.config.rear_range,
                "lateral_range": self.config.lateral_range,
                "update_period": self.config.update_period,
                "latency": self.config.latency,
            },
        )

    def _inside_view(self, ego: VehicleState, x: float, y: float) -> bool:
        dx = float(x) - ego.x
        dy = float(y) - ego.y
        c = math.cos(ego.yaw)
        s = math.sin(ego.yaw)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        if longitudinal > self.config.forward_range or longitudinal < -self.config.rear_range:
            return False
        if abs(lateral) > self.config.lateral_range:
            return False
        if self.config.field_of_view < 2.0 * math.pi - 1e-9:
            angle = abs(math.atan2(lateral, longitudinal))
            if angle > 0.5 * self.config.field_of_view:
                return False
        return True

    def _noisy_ego(self, ego: VehicleState) -> VehicleState:
        return VehicleState(
            x=ego.x + float(self._rng.normal(0.0, self.config.position_noise_std)),
            y=ego.y + float(self._rng.normal(0.0, self.config.position_noise_std)),
            yaw=ego.yaw + float(self._rng.normal(0.0, self.config.yaw_noise_std)),
            speed=max(0.0, ego.speed + float(self._rng.normal(0.0, self.config.speed_noise_std))),
        )

    def _perceive_obstacle(self, obstacle) -> PerceivedObject:
        x = float(obstacle.x) + float(self._rng.normal(0.0, self.config.position_noise_std))
        y = float(obstacle.y) + float(self._rng.normal(0.0, self.config.position_noise_std))
        if isinstance(obstacle, CircleObstacle):
            return PerceivedObject(
                object_id=obstacle.obstacle_id,
                object_type="circle",
                x=x,
                y=y,
                yaw=0.0,
                speed=0.0,
                length=2.0 * obstacle.radius,
                width=2.0 * obstacle.radius,
            )
        if isinstance(obstacle, BoxObstacle):
            return PerceivedObject(
                object_id=obstacle.obstacle_id,
                object_type="box",
                x=x,
                y=y,
                yaw=obstacle.yaw + float(self._rng.normal(0.0, self.config.yaw_noise_std)),
                speed=0.0,
                length=obstacle.length,
                width=obstacle.width,
            )
        raise TypeError(f"Unsupported obstacle type: {type(obstacle).__name__}")

    def _perceive_signal(self, signal) -> PerceivedTrafficSignal:
        return PerceivedTrafficSignal(
            entity_id=signal.entity_id,
            map_signal_id=signal.map_signal_id,
            x=signal.x + float(self._rng.normal(0.0, self.config.position_noise_std)),
            y=signal.y + float(self._rng.normal(0.0, self.config.position_noise_std)),
            yaw=signal.yaw + float(self._rng.normal(0.0, self.config.yaw_noise_std)),
            state=signal.state.value,
            remaining_time=signal.remaining_time,
        )

    def _clip_lane(self, lane, ego: VehicleState) -> PerceivedLaneSegment | None:
        centerline = np.asarray(lane.centerline, dtype=np.float64)
        visible = np.asarray(
            [self._inside_view(ego, point[0], point[1]) for point in centerline],
            dtype=bool,
        )
        indices = np.flatnonzero(visible)
        if indices.size == 0:
            return None
        start = max(0, int(indices[0]) - 1)
        end = min(centerline.shape[0], int(indices[-1]) + 2)
        if end - start < 2:
            return None

        def slice_points(points) -> np.ndarray:
            array = np.asarray(points, dtype=np.float64)[start:end].copy()
            if self.config.position_noise_std > 0.0:
                array += self._rng.normal(0.0, self.config.position_noise_std, array.shape)
            return array

        return PerceivedLaneSegment(
            map_lane_id=lane.map_lane_id,
            centerline=slice_points(lane.centerline),
            left_boundary=slice_points(lane.left_boundary),
            right_boundary=slice_points(lane.right_boundary),
            predecessor_ids=lane.predecessor_ids,
            successor_ids=lane.successor_ids,
        )
