from __future__ import annotations

import numpy as np

from sim2d.planning.spatiotemporal_planner import (
    ObjectPredictionSet,
    SpatiotemporalPlannerConfig,
    SpatiotemporalTrajectory,
)
from sim2d.planning.spatiotemporal_planner.cost import SpatiotemporalCost
from sim2d.types import VehicleConfig


def _trajectory(y: float, steering: float = 0.0, speed: float = 8.0):
    times = np.arange(6, dtype=np.float64) * 0.1
    x = np.linspace(0.0, 5.0, 6)
    states = np.column_stack(
        (
            x,
            np.full_like(x, y),
            np.zeros_like(x),
            np.full_like(x, speed),
        )
    )
    controls = np.zeros((5, 2), dtype=np.float64)
    controls[:, 1] = steering
    return SpatiotemporalTrajectory(times=times, states=states, controls=controls)


def _reference_and_boundaries():
    x = np.linspace(0.0, 10.0, 21)
    yaw = np.zeros_like(x)
    arc = x.copy()
    curvature = np.zeros_like(x)
    reference = np.column_stack((x, np.zeros_like(x), yaw, arc, curvature))
    left = np.column_stack((x, np.full_like(x, 2.0)))
    right = np.column_stack((x, np.full_like(x, -2.0)))
    return reference, left, right


def _evaluate(trajectory):
    config = SpatiotemporalPlannerConfig(
        target_speed=8.0,
        weight_reference=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_acceleration=0.0,
        weight_steering=0.0,
        weight_acceleration_rate=0.0,
        weight_steering_rate=0.0,
        weight_collision=0.0,
        weight_terminal=0.0,
        weight_corridor=1.0,
        weight_frenet_progress=1.0,
        weight_lateral_acceleration=1.0,
    )
    cost = SpatiotemporalCost(config=config, vehicle_config=VehicleConfig())
    reference, left, right = _reference_and_boundaries()
    predictions = ObjectPredictionSet(times=trajectory.times, trajectories=())
    return cost.evaluate(
        trajectory=trajectory,
        predictions=predictions,
        reference_path=reference,
        left_boundary=left,
        right_boundary=right,
    )


def test_corridor_cost_penalizes_vehicle_body_outside_lane() -> None:
    _, inside_terms = _evaluate(_trajectory(y=0.0))
    _, outside_terms = _evaluate(_trajectory(y=1.6))
    assert inside_terms["corridor"] == 0.0
    assert outside_terms["corridor"] > 0.0


def test_lateral_acceleration_cost_penalizes_fast_large_steering() -> None:
    _, straight_terms = _evaluate(_trajectory(y=0.0, steering=0.0, speed=10.0))
    _, turning_terms = _evaluate(_trajectory(y=0.0, steering=0.35, speed=10.0))
    assert straight_terms["lateral_acceleration"] == 0.0
    assert turning_terms["lateral_acceleration"] > 0.0


def test_frenet_progress_penalizes_backward_projection() -> None:
    forward = _trajectory(y=0.0)
    backward_states = forward.states.copy()
    backward_states[:, 0] = backward_states[::-1, 0]
    backward = SpatiotemporalTrajectory(
        times=forward.times,
        states=backward_states,
        controls=forward.controls,
    )
    _, forward_terms = _evaluate(forward)
    _, backward_terms = _evaluate(backward)
    assert forward_terms["frenet_progress"] == 0.0
    assert backward_terms["frenet_progress"] > 0.0
