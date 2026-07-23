from __future__ import annotations

import math

import numpy as np

from sim2d.perception import PlanningInput
from sim2d.types import GoalState, VehicleState

from .types import LocalPlanningContext, SpatiotemporalTrajectory


def normalize_angle(angle: float | np.ndarray):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_state_to_local(
    state: VehicleState,
    origin: VehicleState,
) -> VehicleState:
    dx = float(state.x) - float(origin.x)
    dy = float(state.y) - float(origin.y)
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    return VehicleState(
        x=cosine * dx + sine * dy,
        y=-sine * dx + cosine * dy,
        yaw=float(normalize_angle(state.yaw - origin.yaw)),
        speed=float(state.speed),
    )


def local_state_to_world(
    state: VehicleState,
    origin: VehicleState,
) -> VehicleState:
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    return VehicleState(
        x=float(origin.x) + cosine * state.x - sine * state.y,
        y=float(origin.y) + sine * state.x + cosine * state.y,
        yaw=float(normalize_angle(state.yaw + origin.yaw)),
        speed=float(state.speed),
    )


def world_goal_to_local(
    goal: GoalState,
    origin: VehicleState,
) -> GoalState:
    return GoalState(
        state=world_state_to_local(goal.state, origin),
        position_tolerance=goal.position_tolerance,
        yaw_tolerance=goal.yaw_tolerance,
        speed_tolerance=goal.speed_tolerance,
    )


def world_reference_path_to_local(
    reference_path: np.ndarray | None,
    origin: VehicleState,
) -> np.ndarray | None:
    if reference_path is None:
        return None

    path = np.asarray(reference_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 3:
        raise ValueError(
            "reference_path must have shape [N, M], M >= 3, "
            f"got {path.shape}"
        )
    if not np.all(np.isfinite(path)):
        raise ValueError("reference_path contains non-finite values")

    local = path.copy()
    dx = path[:, 0] - float(origin.x)
    dy = path[:, 1] - float(origin.y)
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    local[:, 0] = cosine * dx + sine * dy
    local[:, 1] = -sine * dx + cosine * dy
    local[:, 2] = normalize_angle(path[:, 2] - float(origin.yaw))
    return local


def local_trajectory_to_world(
    trajectory: SpatiotemporalTrajectory,
    origin: VehicleState,
) -> SpatiotemporalTrajectory:
    states = trajectory.states.copy()
    local_x = trajectory.states[:, 0]
    local_y = trajectory.states[:, 1]
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    states[:, 0] = float(origin.x) + cosine * local_x - sine * local_y
    states[:, 1] = float(origin.y) + sine * local_x + cosine * local_y
    states[:, 2] = normalize_angle(
        trajectory.states[:, 2] + float(origin.yaw)
    )
    return SpatiotemporalTrajectory(
        times=trajectory.times,
        states=states,
        controls=trajectory.controls,
    )


def build_local_planning_context(
    planning_input: PlanningInput,
    reference_path: np.ndarray | None,
) -> LocalPlanningContext:
    """构造冻结自车坐标 PNC 上下文。

    感知车道线、目标和交通灯已经位于 vehicle frame，直接传给 PNC；
    PNC 内部的 map 小模块随后根据 lane_lines 构造候选 reference line。
    """
    perception = planning_input.perception
    if perception.coordinate_frame != "vehicle":
        raise ValueError(
            "spatiotemporal planner requires vehicle-frame perception, "
            f"got {perception.coordinate_frame!r}"
        )

    world_origin = planning_input.ego
    local_ego = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=float(world_origin.speed),
    )

    return LocalPlanningContext(
        world_origin=world_origin,
        ego=local_ego,
        goal=world_goal_to_local(planning_input.goal, world_origin),
        reference_path=world_reference_path_to_local(
            reference_path,
            world_origin,
        ),
        objects=perception.objects,
        lane_lines=perception.lane_lines,
        traffic_signals=perception.traffic_signals,
    )


__all__ = [
    "build_local_planning_context",
    "local_state_to_world",
    "local_trajectory_to_world",
    "normalize_angle",
    "world_goal_to_local",
    "world_reference_path_to_local",
    "world_state_to_local",
]
