from __future__ import annotations

import numpy as np

from sim2d.planning.spatiotemporal_planner import (
    SpatiotemporalPlanner,
    SpatiotemporalPlannerConfig,
)
from sim2d.types import VehicleConfig, VehicleState


def _left_turn_reference(radius: float = 8.0, point_count: int = 101):
    theta = np.linspace(-0.5 * np.pi, 0.0, point_count)
    x = radius * np.cos(theta)
    y = radius + radius * np.sin(theta)
    yaw = theta + 0.5 * np.pi
    arc = radius * (theta + 0.5 * np.pi)
    curvature = np.full_like(theta, 1.0 / radius)
    reference = np.column_stack((x, y, yaw, arc, curvature))
    normal = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    left = reference[:, :2] + 2.0 * normal
    right = reference[:, :2] - 2.0 * normal
    return reference, left, right


def test_reference_feedforward_brakes_and_turns_before_optimizer() -> None:
    config = SpatiotemporalPlannerConfig(
        horizon_steps=20,
        target_speed=10.0,
        max_iterations=1,
    )
    planner = SpatiotemporalPlanner(
        vehicle_config=VehicleConfig(),
        config=config,
    )
    reference, _, _ = _left_turn_reference()

    controls = planner._make_reference_feedforward_controls(
        ego_state=VehicleState(0.0, 0.0, 0.0, 8.0),
        reference_path=reference,
    )

    assert controls.shape == (20, 2)
    assert controls[0, 0] < 0.0
    assert np.max(controls[:, 1]) > 0.05


def test_feedforward_initialization_reduces_corridor_violation() -> None:
    config = SpatiotemporalPlannerConfig(
        horizon_steps=20,
        target_speed=10.0,
        max_iterations=1,
    )
    planner = SpatiotemporalPlanner(
        vehicle_config=VehicleConfig(),
        config=config,
    )
    ego = VehicleState(0.0, 0.0, 0.0, 8.0)
    reference, left, right = _left_turn_reference()

    feedforward = planner._make_reference_feedforward_controls(
        ego_state=ego,
        reference_path=reference,
    )
    straight = np.zeros_like(feedforward)
    straight[:, 0] = planner.vehicle_config.acceleration_max

    feedforward_trajectory = planner.optimizer.rollout.rollout_from_ego(
        ego_state=ego,
        controls=feedforward,
    )
    straight_trajectory = planner.optimizer.rollout.rollout_from_ego(
        ego_state=ego,
        controls=straight,
    )
    feedforward_diagnostics = planner.optimizer._trajectory_diagnostics(
        trajectory=feedforward_trajectory,
        reference_path=reference,
        left_boundary=left,
        right_boundary=right,
    )
    straight_diagnostics = planner.optimizer._trajectory_diagnostics(
        trajectory=straight_trajectory,
        reference_path=reference,
        left_boundary=left,
        right_boundary=right,
    )

    assert float(feedforward_diagnostics["max_footprint_violation"]) < float(
        straight_diagnostics["max_footprint_violation"]
    )
