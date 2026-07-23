from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import numpy as np

from sim2d.perception.ideal_lane_sensor import perceive_lane_corridors
from sim2d.perception.types import (
    PerceivedObject,
    PerceivedTrafficSignal,
    PerceptionConfig,
    PerceptionSnapshot,
)
from sim2d.types import BoxObstacle, CircleObstacle, VehicleState
from sim2d.world.state import WorldState


class GroundTruthLocalPerception:
    """局部理想感知。

    数据来自 World，但只通过有限视野的局部几何扫描发布给 PNC。
    目标、交通灯和车道线统一位于车辆坐标系：+x 向前，+y 向左。
    感知层只发布车道线点列，不发布 lane、走廊、左右配对或拓扑。
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
        if (
            latest is None
            or now - latest.measurement_time + 1e-12 >= self.config.update_period
        ):
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

        while (
            len(self._history) > 2
            and self._history[1].measurement_time <= cutoff
        ):
            self._history.popleft()

        return replace(published, publish_time=now)

    def _measure(
        self,
        world_state: WorldState,
        *,
        frame: int,
    ) -> PerceptionSnapshot:
        ego = self._noisy_ego(world_state.ego_state)

        objects = tuple(
            item
            for obstacle in world_state.obstacles
            if self._inside_view(ego, float(obstacle.x), float(obstacle.y))
            if self._rng.random() >= self.config.object_dropout_probability
            for item in (self._perceive_obstacle(obstacle, ego),)
        )

        signals = tuple(
            self._perceive_signal(signal, ego)
            for signal in world_state.traffic_signals
            if self._inside_view(ego, signal.x, signal.y)
            if self._rng.random() >= self.config.signal_dropout_probability
        )

        # 兼容性能调试层的旧函数名；返回值已经是 PerceivedLaneLine 集合。
        lane_lines = perceive_lane_corridors(
            world_state.road_lanes,
            ego,
            self.config,
            self._rng,
        )

        return PerceptionSnapshot(
            measurement_time=float(world_state.time),
            publish_time=float(world_state.time),
            frame=int(frame),
            ego=ego,
            objects=objects,
            traffic_signals=signals,
            road_segments=(),
            lane_lines=lane_lines,
            source="ideal_lane_line_scan",
            coordinate_frame="vehicle",
            debug={
                "coordinate_frame": "vehicle",
                "x_axis": "forward",
                "y_axis": "left",
                "forward_range": self.config.forward_range,
                "rear_range": self.config.rear_range,
                "lateral_range": self.config.lateral_range,
                "update_period": self.config.update_period,
                "latency": self.config.latency,
                "lane_sensor": "transverse_scan_plus_fan_rays",
                "lane_transverse_spacing": self.config.lane_transverse_spacing,
                "lane_ray_count": self.config.lane_ray_count,
                "lane_ray_half_angle": self.config.lane_ray_half_angle,
                "lane_line_count": len(lane_lines),
                "lane_corridor_count": 0,
                "lane_topology_published": False,
                "lane_pairing_published": False,
            },
        )

    def _inside_view(self, ego: VehicleState, x: float, y: float) -> bool:
        longitudinal, lateral = self._world_to_vehicle(ego, x, y)
        if (
            longitudinal > self.config.forward_range
            or longitudinal < -self.config.rear_range
        ):
            return False
        if abs(lateral) > self.config.lateral_range:
            return False
        if self.config.field_of_view < 2.0 * math.pi - 1e-9:
            angle = abs(math.atan2(lateral, longitudinal))
            if angle > 0.5 * self.config.field_of_view:
                return False
        return True

    @staticmethod
    def _world_to_vehicle(
        ego: VehicleState,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        dx = float(x) - ego.x
        dy = float(y) - ego.y
        cosine = math.cos(ego.yaw)
        sine = math.sin(ego.yaw)
        return (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _noisy_ego(self, ego: VehicleState) -> VehicleState:
        return VehicleState(
            x=ego.x + float(self._rng.normal(0.0, self.config.position_noise_std)),
            y=ego.y + float(self._rng.normal(0.0, self.config.position_noise_std)),
            yaw=self._normalize_angle(
                ego.yaw + float(self._rng.normal(0.0, self.config.yaw_noise_std))
            ),
            speed=max(
                0.0,
                ego.speed
                + float(self._rng.normal(0.0, self.config.speed_noise_std)),
            ),
        )

    def _noisy_local_position(
        self,
        ego: VehicleState,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        local_x, local_y = self._world_to_vehicle(ego, x, y)
        if self.config.position_noise_std > 0.0:
            local_x += float(
                self._rng.normal(0.0, self.config.position_noise_std)
            )
            local_y += float(
                self._rng.normal(0.0, self.config.position_noise_std)
            )
        return local_x, local_y

    def _perceive_obstacle(
        self,
        obstacle,
        ego: VehicleState,
    ) -> PerceivedObject:
        local_x, local_y = self._noisy_local_position(
            ego,
            obstacle.x,
            obstacle.y,
        )

        if isinstance(obstacle, CircleObstacle):
            return PerceivedObject(
                object_id=obstacle.obstacle_id,
                object_type="circle",
                x=local_x,
                y=local_y,
                yaw=0.0,
                speed=0.0,
                length=2.0 * obstacle.radius,
                width=2.0 * obstacle.radius,
            )

        if isinstance(obstacle, BoxObstacle):
            relative_yaw = self._normalize_angle(
                obstacle.yaw
                - ego.yaw
                + float(self._rng.normal(0.0, self.config.yaw_noise_std))
            )
            return PerceivedObject(
                object_id=obstacle.obstacle_id,
                object_type="box",
                x=local_x,
                y=local_y,
                yaw=relative_yaw,
                speed=0.0,
                length=obstacle.length,
                width=obstacle.width,
            )

        raise TypeError(
            f"Unsupported obstacle type: {type(obstacle).__name__}"
        )

    def _perceive_signal(
        self,
        signal,
        ego: VehicleState,
    ) -> PerceivedTrafficSignal:
        local_x, local_y = self._noisy_local_position(
            ego,
            signal.x,
            signal.y,
        )
        relative_yaw = self._normalize_angle(
            signal.yaw
            - ego.yaw
            + float(self._rng.normal(0.0, self.config.yaw_noise_std))
        )
        return PerceivedTrafficSignal(
            entity_id=signal.entity_id,
            map_signal_id=signal.map_signal_id,
            x=local_x,
            y=local_y,
            yaw=relative_yaw,
            state=signal.state.value,
            remaining_time=signal.remaining_time,
        )
