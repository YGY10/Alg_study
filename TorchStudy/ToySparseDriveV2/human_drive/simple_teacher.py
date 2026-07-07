from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from auto_policy import AutoDriveObservation, AutoDriveScene, AutoPolicyConfig
from drive_sim import DriveAction, DriveState, obstacle_center_at


@dataclass
class SimplePlan:
    xy: np.ndarray
    speed: np.ndarray
    cost: float
    collision: bool
    behavior: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def make_bezier_points(
    start_xy: np.ndarray,
    start_yaw: float,
    target_xy: np.ndarray,
    target_dir: np.ndarray,
    num_points: int,
) -> np.ndarray:
    chord = target_xy - start_xy
    distance = float(np.linalg.norm(chord))
    start_dir = np.array([math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float32)
    handle = max(3.0, min(12.0, 0.35 * distance))
    p0 = start_xy.astype(np.float32)
    p1 = p0 + start_dir * handle
    p3 = target_xy.astype(np.float32)
    p2 = p3 - target_dir.astype(np.float32) * handle
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    ).astype(np.float32)


def trajectory_collision_cost(
    xy: np.ndarray,
    times: np.ndarray,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
) -> tuple[bool, float]:
    ego_half = np.array([config.ego_length, config.ego_width], dtype=np.float32) * 0.5
    min_clearance = 1.0e6
    collision = False
    for point, time_s in zip(xy, times):
        for obstacle in scene.obstacles:
            center = obstacle_center_at(obstacle, float(time_s))
            obstacle_half = np.asarray(obstacle["size_xy"], dtype=np.float32) * 0.5
            delta = np.abs(point - center) - (ego_half + obstacle_half)
            outside = np.maximum(delta, 0.0)
            clearance = float(
                np.linalg.norm(outside) + min(max(delta[0], delta[1]), 0.0)
            )
            min_clearance = min(min_clearance, clearance)
            if delta[0] <= float(config.safety_margin) and delta[1] <= float(
                config.safety_margin
            ):
                collision = True
    return collision, min_clearance


def score_candidate(
    xy: np.ndarray,
    speed: np.ndarray,
    times: np.ndarray,
    state: DriveState,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    lateral_offset: float,
    target_speed: float,
) -> SimplePlan:
    collision, min_clearance = trajectory_collision_cost(xy, times, scene, config)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float32)
    start_xy = np.array([state.x, state.y], dtype=np.float32)
    start_goal_distance = float(np.linalg.norm(goal_xy - start_xy))
    end_goal_distance = float(np.linalg.norm(goal_xy - xy[-1]))
    progress = start_goal_distance - end_goal_distance
    speed_change = abs(float(target_speed) - float(state.speed))
    path_delta = np.diff(xy, axis=0)
    smoothness = float(np.linalg.norm(np.diff(path_delta, axis=0), axis=1).mean())
    clearance_cost = 0.0
    if min_clearance < 2.0:
        clearance_cost = (2.0 - min_clearance) ** 2 * 8.0
    cost = (
        -8.0 * progress
        + 0.35 * abs(float(lateral_offset))
        + 0.8 * speed_change
        + 2.0 * smoothness
        + clearance_cost
    )
    if collision:
        cost += 1.0e6
    if target_speed < 0.5 and start_goal_distance > 5.0:
        # Stopping is a fallback, not the default driving style.
        cost += 15.0
    behavior = (
        "stop"
        if target_speed < 0.5
        else "pass" if abs(lateral_offset) > 1.0 else "direct"
    )
    return SimplePlan(
        xy=xy,
        speed=speed.astype(np.float32),
        cost=float(cost),
        collision=bool(collision),
        behavior=behavior,
    )


def build_candidate_plan(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    lateral_offset: float,
    target_speed: float,
) -> SimplePlan:
    state = observation.state
    start_xy = np.array([state.x, state.y], dtype=np.float32)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float32)
    goal_dir = unit_vector(
        goal_xy - start_xy,
        np.array([math.cos(state.yaw), math.sin(state.yaw)], dtype=np.float32),
    )
    normal = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)
    times = np.arange(
        config.plan_dt, config.horizon + 1.0e-6, config.plan_dt, dtype=np.float32
    )
    avg_speed = max(0.2, 0.5 * (float(state.speed) + float(target_speed)))
    distance_to_goal = float(np.linalg.norm(goal_xy - start_xy))
    progress = min(distance_to_goal, max(3.0, avg_speed * float(config.horizon)))
    target_xy = start_xy + goal_dir * progress + normal * float(lateral_offset)
    if distance_to_goal <= progress + 3.0:
        target_xy = goal_xy + normal * float(lateral_offset) * 0.2
    xy = make_bezier_points(
        start_xy=start_xy,
        start_yaw=float(state.yaw),
        target_xy=target_xy,
        target_dir=goal_dir,
        num_points=len(times),
    )
    speed = np.linspace(
        float(state.speed), float(target_speed), len(times), dtype=np.float32
    )
    return score_candidate(
        xy=xy,
        speed=speed,
        times=times,
        state=state,
        scene=scene,
        config=config,
        lateral_offset=lateral_offset,
        target_speed=target_speed,
    )


def track_plan_to_action(
    plan: SimplePlan,
    observation: AutoDriveObservation,
    config: AutoPolicyConfig,
) -> DriveAction:
    state = observation.state
    lookahead_index = min(3, len(plan.xy) - 1)
    target = plan.xy[lookahead_index]
    dx = float(target[0] - state.x)
    dy = float(target[1] - state.y)
    target_yaw = math.atan2(dy, dx)
    heading_error = math.atan2(
        math.sin(target_yaw - state.yaw), math.cos(target_yaw - state.yaw)
    )
    steer = clamp(
        math.atan2(
            2.0 * config.wheelbase * math.sin(heading_error),
            max(math.hypot(dx, dy), 1.0),
        ),
        -math.radians(config.max_steer_deg),
        math.radians(config.max_steer_deg),
    )
    desired_speed = float(plan.speed[min(1, len(plan.speed) - 1)])
    speed_error = desired_speed - float(state.speed)
    accel = clamp(
        float(config.speed_kp) * speed_error,
        -float(config.brake),
        float(config.accel),
    )
    return DriveAction(key=plan.behavior, accel=float(accel), steer=float(steer))


def plan_simple_teacher_action(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
) -> DriveAction:
    state = observation.state
    cruise_speed = min(float(config.cruise_speed), float(config.max_speed), 6.0)
    speed_options = [cruise_speed, min(cruise_speed, 4.0), min(cruise_speed, 2.0), 0.0]
    lateral_offsets = [0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0]

    plans: list[SimplePlan] = []
    for lateral_offset in lateral_offsets:
        for target_speed in speed_options:
            plans.append(
                build_candidate_plan(
                    observation=observation,
                    scene=scene,
                    config=config,
                    lateral_offset=lateral_offset,
                    target_speed=target_speed,
                )
            )

    safe_plans = [plan for plan in plans if not plan.collision]
    if safe_plans:
        best = min(safe_plans, key=lambda item: item.cost)
    else:
        stop_plans = [plan for plan in plans if plan.behavior == "stop"]
        best = min(stop_plans or plans, key=lambda item: item.cost)

    if bool(getattr(config, "debug_policy", False)):
        print(
            "[simple_teacher] selected "
            f"behavior={best.behavior} cost={best.cost:.2f} collision={best.collision}"
        )
    return track_plan_to_action(best, observation, config)
