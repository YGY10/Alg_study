from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from drive_sim import DriveAction, DriveState, obstacle_center_at, step_ego

# =============================================================================
# Config / public data containers
# =============================================================================


@dataclass
class AutoPolicyConfig:
    """ToySparseDriveV2 auto policy config.

    The public entry function remains:

        plan_auto_action(observation, scene, config) -> DriveAction

    Internally this file implements an EPSILON-style architecture for the toy
    environment:

        behavior branching
            -> spatio-temporal semantic corridor, SSC
            -> Bezier trajectory QP
            -> trajectory tracking to one DriveAction

    This is an architecture-level implementation adapted to the current toy
    scene representation. It does not require ROS, HD maps, or OOQP.
    """

    name: str = "epsilon_style_ssc_bezier_qp_v1"

    # Vehicle and low-level limits.
    wheelbase: float = 2.8
    accel: float = 1.6
    brake: float = 3.0
    max_steer_deg: float = 18.0
    max_speed: float = 12.0

    # Speed control.
    cruise_speed: float = 6.5
    speed_kp: float = 0.9

    # Legacy target-point parameters used only for emergency fallback.
    lookahead_base: float = 7.0
    goal_bias: float = 0.62
    avoid_lateral_offset: float = 5.0
    corridor_half_width: float = 2.5
    obstacle_lookahead: float = 28.0

    # Planning horizon.
    horizon: float = 4.0
    plan_dt: float = 0.2
    bezier_control_points: int = 6

    # Toy SSC road/corridor assumptions.
    # Positive l means left of route tangent.
    ssc_lateral_bound: float = 6.0
    lane_keep_lateral_bound: float = 2.2
    pass_lateral_offset: float = 4.0
    nudge_lateral_offset: float = 2.0

    # Ego and safety geometry.
    ego_length: float = 4.8
    ego_width: float = 2.0
    safety_margin: float = 0.35
    desired_clearance: float = 2.0
    static_s_buffer: float = 2.0
    static_l_buffer: float = 0.6
    dynamic_time_buffer: float = 0.8

    # Bezier QP objective weights.
    qp_s_ref_weight: float = 1.0
    qp_l_ref_weight: float = 16.0
    qp_s_smooth_weight: float = 3.0
    qp_l_smooth_weight: float = 24.0
    qp_s_jerk_weight: float = 0.5
    qp_l_jerk_weight: float = 4.0
    qp_progress_weight: float = 2.0
    qp_terminal_l_weight: float = 18.0
    qp_terminal_s_weight: float = 1.5

    # Final trajectory scoring weights.
    collision_cost: float = 1.0e6
    clearance_weight: float = 160.0
    route_weight: float = 4.0
    goal_weight: float = 2.0
    progress_weight: float = 10.0
    heading_weight: float = 1.0
    speed_weight: float = 0.3
    smooth_weight: float = 1.0
    behavior_bias_weight: float = 1.0
    stop_without_reason_cost: float = 40.0

    # Controller tracking.
    tracking_lookahead_time: float = 0.7
    tracking_min_lookahead_index: int = 2

    # Solver behavior.
    use_qp: bool = True
    qp_solver: str = "OSQP"
    qp_verbose: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "AutoPolicyConfig":
        """Build config from argparse-like args.

        All fields use getattr(..., default), so existing callers do not need
        new argparse flags immediately.
        """
        defaults = cls()
        values: dict[str, Any] = {}
        for item in fields(cls):
            default_value = getattr(defaults, item.name)
            raw_value = getattr(args, item.name, default_value)
            if isinstance(default_value, bool):
                values[item.name] = bool(raw_value)
            elif isinstance(default_value, int) and not isinstance(default_value, bool):
                values[item.name] = int(raw_value)
            elif isinstance(default_value, float):
                values[item.name] = float(raw_value)
            else:
                values[item.name] = raw_value
        return cls(**values)

    def to_metadata(self) -> dict[str, float | str | bool]:
        data = asdict(self)
        data["type"] = data.pop("name")
        return data


@dataclass
class AutoDriveScene:
    route_path: np.ndarray
    goal_xy: np.ndarray
    obstacles: list[dict[str, Any]]

    @property
    def route_xy(self) -> np.ndarray:
        return self.route_path[:, :2].astype(np.float32)

    @property
    def route_s(self) -> np.ndarray:
        return build_route_progress(self.route_xy)


@dataclass
class AutoDriveObservation:
    state: DriveState
    time_s: float
    history: list[list[float]]


# =============================================================================
# Internal planner data structures
# =============================================================================


@dataclass
class FrenetProjection:
    s: float
    l: float
    route_yaw: float
    index: int
    distance: float


@dataclass
class BehaviorCandidate:
    name: str
    target_l: float
    target_speed: float
    side: str  # "center", "left", "right"
    mode: str  # "keep", "pass", "follow", "stop", "yield"
    bias: float = 0.0


@dataclass
class SSCCorridor:
    behavior: BehaviorCandidate
    times: np.ndarray
    s_ref: np.ndarray
    l_ref: np.ndarray
    s_min: np.ndarray
    s_max: np.ndarray
    l_min: np.ndarray
    l_max: np.ndarray
    feasible: bool
    reason: str = ""


@dataclass
class PlannedTrajectory:
    behavior: BehaviorCandidate
    source: str
    times: np.ndarray
    s: np.ndarray
    l: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    cost: float
    valid: bool
    collision: bool
    min_clearance: float
    debug: dict[str, Any]


# =============================================================================
# Public policy entry
# =============================================================================


def plan_auto_action(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
) -> DriveAction:
    """Plan one action using EPSILON-style SSC + Bezier QP.

    External interface is unchanged. The function internally computes a short
    horizon trajectory and returns only the first executable control.
    """
    state = observation.state
    route_xy = scene.route_xy
    route_s = scene.route_s

    if len(route_xy) < 2:
        return emergency_stop_action(state, config)

    ego_proj = project_to_route(
        xy=np.array([state.x, state.y], dtype=np.float32),
        route_xy=route_xy,
        route_s=route_s,
    )

    behaviors = generate_behavior_candidates(
        observation=observation,
        scene=scene,
        config=config,
        ego_proj=ego_proj,
    )

    best: PlannedTrajectory | None = None
    for behavior in behaviors:
        corridor = build_ssc_corridor(
            observation=observation,
            scene=scene,
            config=config,
            ego_proj=ego_proj,
            behavior=behavior,
        )
        if not corridor.feasible:
            continue

        plan = optimize_bezier_trajectory(
            observation=observation,
            scene=scene,
            config=config,
            ego_proj=ego_proj,
            corridor=corridor,
        )

        if not plan.valid:
            continue

        plan = evaluate_planned_trajectory(
            plan=plan,
            observation=observation,
            scene=scene,
            config=config,
            ego_proj=ego_proj,
        )

        if best is None or plan.cost < best.cost:
            best = plan

    if best is None:
        # Defensive fallback: if all SSC/QP candidates fail, use the old
        # target-point policy instead of returning an arbitrary zero command.
        return legacy_pure_pursuit_action(observation, scene, config)

    return track_trajectory_to_action(
        trajectory=best,
        observation=observation,
        config=config,
    )


# =============================================================================
# Behavior branching
# =============================================================================


def generate_behavior_candidates(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
) -> list[BehaviorCandidate]:
    """Guided behavior branching.

    This is the EPSILON-style behavior layer adapted to the toy scene:
    instead of solving a full POMDP over rich semantic inputs, it creates a
    compact set of closed-loop semantic intents that the SSC/QP layer can test.
    """
    state = observation.state
    front = find_front_obstacle(
        observation=observation,
        scene=scene,
        config=config,
        ego_proj=ego_proj,
    )

    cruise = min(float(config.cruise_speed), float(config.max_speed))
    near_goal_speed = target_speed_near_goal(state, scene.goal_xy, cruise)
    base_speed = min(cruise, near_goal_speed)

    behaviors: list[BehaviorCandidate] = [
        BehaviorCandidate(
            name="KEEP_ROUTE",
            target_l=0.0,
            target_speed=base_speed,
            side="center",
            mode="keep",
            bias=0.0,
        ),
        BehaviorCandidate(
            name="NUDGE_LEFT",
            target_l=clamp(
                config.nudge_lateral_offset,
                -config.ssc_lateral_bound,
                config.ssc_lateral_bound,
            ),
            target_speed=min(base_speed, 5.0),
            side="left",
            mode="pass",
            bias=0.8,
        ),
        BehaviorCandidate(
            name="NUDGE_RIGHT",
            target_l=clamp(
                -config.nudge_lateral_offset,
                -config.ssc_lateral_bound,
                config.ssc_lateral_bound,
            ),
            target_speed=min(base_speed, 5.0),
            side="right",
            mode="pass",
            bias=0.9,
        ),
        BehaviorCandidate(
            name="PASS_LEFT",
            target_l=clamp(
                config.pass_lateral_offset,
                -config.ssc_lateral_bound,
                config.ssc_lateral_bound,
            ),
            target_speed=min(base_speed, 5.5),
            side="left",
            mode="pass",
            bias=1.8,
        ),
        BehaviorCandidate(
            name="PASS_RIGHT",
            target_l=clamp(
                -config.pass_lateral_offset,
                -config.ssc_lateral_bound,
                config.ssc_lateral_bound,
            ),
            target_speed=min(base_speed, 5.5),
            side="right",
            mode="pass",
            bias=2.0,
        ),
    ]

    if front is not None:
        front_gap = float(front["gap"])
        front_speed = float(front["speed_along_route"])
        follow_speed = clamp(min(base_speed, front_speed + 0.5), 0.0, base_speed)
        if front_gap < 18.0:
            follow_speed = min(follow_speed, 3.5)
        if front_gap < 10.0:
            follow_speed = min(follow_speed, 1.5)

        behaviors.append(
            BehaviorCandidate(
                name="FOLLOW_FRONT",
                target_l=0.0,
                target_speed=follow_speed,
                side="center",
                mode="follow",
                bias=-0.5,
            )
        )

        behaviors.append(
            BehaviorCandidate(
                name="YIELD_BEHIND",
                target_l=0.0,
                target_speed=min(follow_speed, 1.5),
                side="center",
                mode="yield",
                bias=0.2,
            )
        )

        if front_gap < 7.0:
            stop_bias = -1.0
        else:
            stop_bias = 2.0
        behaviors.append(
            BehaviorCandidate(
                name="STOP",
                target_l=0.0,
                target_speed=0.0,
                side="center",
                mode="stop",
                bias=stop_bias,
            )
        )

    return behaviors


def find_front_obstacle(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
) -> dict[str, float] | None:
    nearest: dict[str, float] | None = None
    for obstacle in scene.obstacles:
        center = obstacle_center_at(obstacle, observation.time_s)
        obs_proj = project_to_route(center, scene.route_xy, scene.route_s)
        ds = obs_proj.s - ego_proj.s
        if ds <= -2.0 or ds > config.obstacle_lookahead:
            continue

        size_xy = np.asarray(obstacle["size_xy"], dtype=np.float32)
        lateral_gate = (
            0.5 * config.ego_width
            + 0.5 * float(size_xy[1])
            + config.safety_margin
            + 0.6
        )
        if abs(obs_proj.l - ego_proj.l) > lateral_gate:
            continue

        speed_along = estimate_obstacle_route_speed(
            obstacle=obstacle,
            time_s=observation.time_s,
            route_xy=scene.route_xy,
            route_s=scene.route_s,
        )
        gap = max(ds - 0.5 * config.ego_length - 0.5 * float(size_xy[0]), 0.0)
        item = {
            "s": float(obs_proj.s),
            "l": float(obs_proj.l),
            "gap": float(gap),
            "speed_along_route": float(speed_along),
        }
        if nearest is None or item["gap"] < nearest["gap"]:
            nearest = item
    return nearest


def estimate_obstacle_route_speed(
    obstacle: dict[str, Any],
    time_s: float,
    route_xy: np.ndarray,
    route_s: np.ndarray,
) -> float:
    p0 = obstacle_center_at(obstacle, time_s)
    p1 = obstacle_center_at(obstacle, time_s + 0.5)
    s0 = project_to_route(p0, route_xy, route_s).s
    s1 = project_to_route(p1, route_xy, route_s).s
    return max(0.0, (s1 - s0) / 0.5)


# =============================================================================
# SSC construction
# =============================================================================


def build_ssc_corridor(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
    behavior: BehaviorCandidate,
) -> SSCCorridor:
    """Build a toy spatio-temporal semantic corridor in Frenet space.

    The corridor is expressed as box constraints at future time samples:

        s_min(t_i) <= s(t_i) <= s_max(t_i)
        l_min(t_i) <= l(t_i) <= l_max(t_i)

    Dynamic obstacles are converted to time-indexed forbidden regions in
    (s, l, t). Depending on the behavior, the corridor either stays behind
    them, passes on the left, or passes on the right.
    """
    dt = max(float(config.plan_dt), 1.0e-3)
    n = max(3, int(math.ceil(config.horizon / dt)) + 1)
    times = np.linspace(0.0, config.horizon, n, dtype=np.float32)

    route_s_end = float(scene.route_s[-1])
    speed = max(0.0, min(float(behavior.target_speed), float(config.max_speed)))

    s_ref = ego_proj.s + speed * times
    s_ref = np.minimum(s_ref, route_s_end)

    # Smooth lateral reference from current l to behavior target l.
    tau = times / max(config.horizon, 1.0e-6)
    smooth = 3.0 * tau * tau - 2.0 * tau * tau * tau
    l_ref = ego_proj.l + (behavior.target_l - ego_proj.l) * smooth

    lat_bound = float(config.ssc_lateral_bound)
    if behavior.mode in {"keep", "follow", "yield", "stop"}:
        # Keep/follow behavior should not wander across the full road width.
        local_bound = min(lat_bound, float(config.lane_keep_lateral_bound))
    else:
        local_bound = lat_bound

    l_min = np.full(n, -local_bound, dtype=np.float32)
    l_max = np.full(n, local_bound, dtype=np.float32)

    # Start exactly at current lateral position.
    l_min[0] = min(l_min[0], ego_proj.l)
    l_max[0] = max(l_max[0], ego_proj.l)

    s_min = np.full(n, max(0.0, ego_proj.s - 0.5), dtype=np.float32)
    s_max = np.full(n, route_s_end, dtype=np.float32)
    s_min[0] = ego_proj.s
    s_max[0] = ego_proj.s

    # Apply obstacle semantics.
    for obstacle in scene.obstacles:
        size_xy = np.asarray(obstacle["size_xy"], dtype=np.float32)
        obs_half_s = (
            0.5 * float(size_xy[0])
            + 0.5 * config.ego_length
            + config.safety_margin
            + config.static_s_buffer
        )
        obs_half_l = (
            0.5 * float(size_xy[1])
            + 0.5 * config.ego_width
            + config.safety_margin
            + config.static_l_buffer
        )

        for i, t_rel in enumerate(times):
            if i == 0:
                continue

            t_abs = float(observation.time_s + t_rel)
            center = obstacle_center_at(obstacle, t_abs)
            obs_proj = project_to_route(center, scene.route_xy, scene.route_s)

            # Only obstacles near the planning horizon matter.
            if obs_proj.s < ego_proj.s - 4.0:
                continue
            if obs_proj.s > ego_proj.s + max(
                config.obstacle_lookahead, speed * config.horizon + 12.0
            ):
                continue

            # Time-expanded longitudinal overlap with nominal reference.
            longitudinal_overlap = abs(float(s_ref[i]) - obs_proj.s) <= (
                obs_half_s + max(2.0, speed * config.dynamic_time_buffer)
            )

            if behavior.side == "left" and behavior.mode == "pass":
                if longitudinal_overlap:
                    # Pass to the left of the obstacle.
                    new_min = obs_proj.l + obs_half_l
                    l_min[i] = max(l_min[i], new_min)
            elif behavior.side == "right" and behavior.mode == "pass":
                if longitudinal_overlap:
                    # Pass to the right of the obstacle.
                    new_max = obs_proj.l - obs_half_l
                    l_max[i] = min(l_max[i], new_max)
            else:
                # Keep/follow/yield/stop: stay behind obstacles blocking the
                # current route-centered corridor.
                lateral_overlap = abs(obs_proj.l) <= obs_half_l
                if lateral_overlap and obs_proj.s > ego_proj.s:
                    stop_s = max(ego_proj.s, obs_proj.s - obs_half_s)
                    # The later the time, the stronger the upper bound. This
                    # creates a valid ST corridor behind the obstacle.
                    if float(s_ref[i]) >= stop_s - 1.0:
                        s_max[i] = min(s_max[i], stop_s)
                        s_ref[i] = min(s_ref[i], stop_s)

    # Stop behavior explicitly chooses a short stopping corridor.
    if behavior.mode == "stop":
        stop_distance = stopping_distance(
            max(float(observation.state.speed), 0.0), config
        )
        stop_s = min(route_s_end, ego_proj.s + stop_distance)
        for i in range(1, n):
            s_max[i] = min(s_max[i], stop_s + 0.4)
            s_ref[i] = min(s_ref[i], stop_s)

    # Repair minor numerical issues and check feasibility.
    eps = 1.0e-3
    for i in range(n):
        if l_min[i] > l_max[i] - eps:
            return SSCCorridor(
                behavior=behavior,
                times=times,
                s_ref=s_ref,
                l_ref=l_ref,
                s_min=s_min,
                s_max=s_max,
                l_min=l_min,
                l_max=l_max,
                feasible=False,
                reason=f"l corridor infeasible at i={i}",
            )
        if s_min[i] > s_max[i] + eps:
            return SSCCorridor(
                behavior=behavior,
                times=times,
                s_ref=s_ref,
                l_ref=l_ref,
                s_min=s_min,
                s_max=s_max,
                l_min=l_min,
                l_max=l_max,
                feasible=False,
                reason=f"s corridor infeasible at i={i}",
            )
        s_ref[i] = clamp(float(s_ref[i]), float(s_min[i]), float(s_max[i]))
        l_ref[i] = clamp(float(l_ref[i]), float(l_min[i]), float(l_max[i]))

    return SSCCorridor(
        behavior=behavior,
        times=times,
        s_ref=s_ref.astype(np.float32),
        l_ref=l_ref.astype(np.float32),
        s_min=s_min.astype(np.float32),
        s_max=s_max.astype(np.float32),
        l_min=l_min.astype(np.float32),
        l_max=l_max.astype(np.float32),
        feasible=True,
        reason="ok",
    )


def stopping_distance(speed: float, config: AutoPolicyConfig) -> float:
    brake = max(float(config.brake), 0.1)
    return max(1.0, speed * speed / (2.0 * brake) + 1.5)


# =============================================================================
# Bezier QP optimization
# =============================================================================


def optimize_bezier_trajectory(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
    corridor: SSCCorridor,
) -> PlannedTrajectory:
    if config.use_qp:
        try:
            plan = optimize_bezier_trajectory_qp(
                observation=observation,
                scene=scene,
                config=config,
                ego_proj=ego_proj,
                corridor=corridor,
            )
            if plan.valid:
                return plan
        except Exception as exc:  # pragma: no cover - robust runtime fallback
            # Keep the policy usable if cvxpy/solver is unavailable.
            pass

    return optimize_bezier_trajectory_sampling(
        observation=observation,
        scene=scene,
        config=config,
        ego_proj=ego_proj,
        corridor=corridor,
    )


def optimize_bezier_trajectory_qp(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
    corridor: SSCCorridor,
) -> PlannedTrajectory:
    import cvxpy as cp

    n_eval = len(corridor.times)
    n_ctrl = max(4, int(config.bezier_control_points))
    degree = n_ctrl - 1

    tau = corridor.times / max(float(corridor.times[-1]), 1.0e-6)
    basis = bezier_basis_matrix(n_ctrl, tau)
    d1 = finite_difference_matrix(n_eval, order=1, dt=float(config.plan_dt))
    d2 = finite_difference_matrix(n_eval, order=2, dt=float(config.plan_dt))
    d3 = finite_difference_matrix(n_eval, order=3, dt=float(config.plan_dt))

    ctrl_s = cp.Variable(n_ctrl)
    ctrl_l = cp.Variable(n_ctrl)

    s_eval = basis @ ctrl_s
    l_eval = basis @ ctrl_l

    constraints = [
        ctrl_s[0] == float(ego_proj.s),
        ctrl_l[0] == float(ego_proj.l),
        s_eval >= corridor.s_min,
        s_eval <= corridor.s_max,
        l_eval >= corridor.l_min,
        l_eval <= corridor.l_max,
    ]

    # Initial derivative from Bezier endpoint property:
    # p'(0) = degree * (P1 - P0) / T
    horizon = max(float(config.horizon), 1.0e-6)
    start_speed_s = max(
        0.0,
        float(observation.state.speed)
        * math.cos(wrap_angle(float(observation.state.yaw) - ego_proj.route_yaw)),
    )
    constraints.append(ctrl_s[1] - ctrl_s[0] >= 0.0)
    constraints.append(
        ctrl_s[1] - ctrl_s[0] <= max(config.max_speed * horizon / degree, 0.1)
    )
    # Do not force exact l-dot; keep it soft through objective.

    if n_eval >= 2:
        s_dot = d1 @ s_eval
        constraints.append(s_dot >= -0.05)
        constraints.append(s_dot <= float(config.max_speed) * 1.15)

    obj = 0.0
    obj += config.qp_s_ref_weight * cp.sum_squares(s_eval - corridor.s_ref)
    obj += config.qp_l_ref_weight * cp.sum_squares(l_eval - corridor.l_ref)

    if d2.shape[0] > 0:
        obj += config.qp_s_smooth_weight * cp.sum_squares(d2 @ s_eval)
        obj += config.qp_l_smooth_weight * cp.sum_squares(d2 @ l_eval)
    if d3.shape[0] > 0:
        obj += config.qp_s_jerk_weight * cp.sum_squares(d3 @ s_eval)
        obj += config.qp_l_jerk_weight * cp.sum_squares(d3 @ l_eval)

    obj += config.qp_terminal_l_weight * cp.square(
        l_eval[-1] - corridor.behavior.target_l
    )
    obj += config.qp_terminal_s_weight * cp.square(s_eval[-1] - corridor.s_ref[-1])

    # Linear progress reward. This remains convex because minimizing a negative
    # affine term is allowed.
    obj += -config.qp_progress_weight * s_eval[-1]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    solver_name = str(config.qp_solver).upper()

    try:
        if solver_name == "OSQP":
            problem.solve(solver=cp.OSQP, verbose=config.qp_verbose, warm_start=True)
        else:
            problem.solve(verbose=config.qp_verbose, warm_start=True)
    except Exception:
        problem.solve(verbose=config.qp_verbose, warm_start=True)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        return invalid_plan(corridor, "qp_infeasible")

    s = np.asarray(s_eval.value, dtype=np.float32)
    l = np.asarray(l_eval.value, dtype=np.float32)

    # Numerical projection back into corridor.
    s = np.minimum(np.maximum(s, corridor.s_min), corridor.s_max)
    l = np.minimum(np.maximum(l, corridor.l_min), corridor.l_max)

    return build_planned_trajectory_from_frenet(
        behavior=corridor.behavior,
        source="bezier_qp",
        times=corridor.times,
        s=s,
        l=l,
        scene=scene,
        base_cost=float(problem.value) if problem.value is not None else 0.0,
        valid=True,
        debug={"qp_status": problem.status},
    )


def optimize_bezier_trajectory_sampling(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
    corridor: SSCCorridor,
) -> PlannedTrajectory:
    """Fallback if cvxpy/OSQP is unavailable.

    This still uses Bezier-shaped trajectories, but chooses the best sampled
    endpoint instead of solving a QP.
    """
    best: PlannedTrajectory | None = None
    times = corridor.times
    tau = times / max(float(times[-1]), 1.0e-6)
    smooth = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    l_targets = [
        corridor.behavior.target_l,
        0.7 * corridor.behavior.target_l,
        1.15 * corridor.behavior.target_l,
        0.0,
    ]
    speed_targets = [
        corridor.behavior.target_speed,
        max(0.0, corridor.behavior.target_speed - 1.5),
        min(config.max_speed, corridor.behavior.target_speed + 1.0),
    ]

    for l_target in l_targets:
        l_candidate = ego_proj.l + (l_target - ego_proj.l) * smooth
        l_candidate = np.minimum(
            np.maximum(l_candidate, corridor.l_min), corridor.l_max
        )

        for speed in speed_targets:
            s_candidate = ego_proj.s + max(0.0, speed) * times
            s_candidate = np.minimum(
                np.maximum(s_candidate, corridor.s_min), corridor.s_max
            )
            # Enforce nondecreasing s after clipping.
            s_candidate = np.maximum.accumulate(s_candidate)

            plan = build_planned_trajectory_from_frenet(
                behavior=corridor.behavior,
                source="bezier_sampling_fallback",
                times=times,
                s=s_candidate.astype(np.float32),
                l=l_candidate.astype(np.float32),
                scene=scene,
                base_cost=0.0,
                valid=True,
                debug={"fallback": True},
            )
            plan = evaluate_planned_trajectory(
                plan=plan,
                observation=observation,
                scene=scene,
                config=config,
                ego_proj=ego_proj,
            )
            if best is None or plan.cost < best.cost:
                best = plan

    return best if best is not None else invalid_plan(corridor, "sampling_failed")


def build_planned_trajectory_from_frenet(
    behavior: BehaviorCandidate,
    source: str,
    times: np.ndarray,
    s: np.ndarray,
    l: np.ndarray,
    scene: AutoDriveScene,
    base_cost: float,
    valid: bool,
    debug: dict[str, Any] | None = None,
) -> PlannedTrajectory:
    xy, route_yaw = frenet_to_world(s, l, scene.route_xy, scene.route_s)

    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)

    yaw = estimate_yaw_from_xy(x, y, fallback_yaw=route_yaw)
    v = estimate_speed_from_xy(x, y, times)

    return PlannedTrajectory(
        behavior=behavior,
        source=source,
        times=times.astype(np.float32),
        s=s.astype(np.float32),
        l=l.astype(np.float32),
        x=x,
        y=y,
        yaw=yaw.astype(np.float32),
        v=v.astype(np.float32),
        cost=float(base_cost),
        valid=valid,
        collision=False,
        min_clearance=1.0e6,
        debug=debug or {},
    )


def invalid_plan(corridor: SSCCorridor, reason: str) -> PlannedTrajectory:
    zeros = np.zeros_like(corridor.times, dtype=np.float32)
    return PlannedTrajectory(
        behavior=corridor.behavior,
        source="invalid",
        times=corridor.times,
        s=zeros,
        l=zeros,
        x=zeros,
        y=zeros,
        yaw=zeros,
        v=zeros,
        cost=1.0e12,
        valid=False,
        collision=True,
        min_clearance=0.0,
        debug={"reason": reason},
    )


def bezier_basis_matrix(n_ctrl: int, tau: np.ndarray) -> np.ndarray:
    degree = n_ctrl - 1
    basis = np.zeros((len(tau), n_ctrl), dtype=np.float64)
    for i, u in enumerate(tau):
        u = float(clamp(float(u), 0.0, 1.0))
        for k in range(n_ctrl):
            basis[i, k] = math.comb(degree, k) * (u**k) * ((1.0 - u) ** (degree - k))
    return basis


def finite_difference_matrix(n: int, order: int, dt: float) -> np.ndarray:
    if order <= 0:
        return np.eye(n, dtype=np.float64)
    if n <= order:
        return np.zeros((0, n), dtype=np.float64)

    if order == 1:
        mat = np.zeros((n - 1, n), dtype=np.float64)
        for i in range(n - 1):
            mat[i, i] = -1.0 / dt
            mat[i, i + 1] = 1.0 / dt
        return mat

    prev = finite_difference_matrix(n, order - 1, dt)
    if prev.shape[0] <= 1:
        return np.zeros((0, n), dtype=np.float64)
    diff = finite_difference_matrix(prev.shape[0], 1, dt)
    return diff @ prev


# =============================================================================
# Trajectory evaluation and tracking
# =============================================================================


def evaluate_planned_trajectory(
    plan: PlannedTrajectory,
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
    ego_proj: FrenetProjection,
) -> PlannedTrajectory:
    if not plan.valid:
        return plan

    collision = False
    min_clearance = 1.0e6
    heading_error_sum = 0.0

    for i in range(len(plan.times)):
        state_i = DriveState(
            x=float(plan.x[i]),
            y=float(plan.y[i]),
            yaw=float(plan.yaw[i]),
            speed=float(plan.v[i]),
        )
        t_abs = float(observation.time_s + plan.times[i])
        step_collision, step_clearance = collision_and_clearance_at_state(
            state=state_i,
            time_s=t_abs,
            obstacles=scene.obstacles,
            config=config,
        )
        collision = collision or step_collision
        min_clearance = min(min_clearance, step_clearance)

        proj_i = project_to_route(
            np.array([state_i.x, state_i.y], dtype=np.float32),
            scene.route_xy,
            scene.route_s,
        )
        heading_error_sum += abs(wrap_angle(float(state_i.yaw) - proj_i.route_yaw))

    final_xy = np.array([plan.x[-1], plan.y[-1]], dtype=np.float32)
    final_goal_distance = float(np.linalg.norm(final_xy - scene.goal_xy))
    progress = max(0.0, float(plan.s[-1]) - ego_proj.s)
    mean_abs_l = float(np.mean(np.abs(plan.l)))
    mean_heading_error = heading_error_sum / max(1, len(plan.times))

    clearance_violation = max(0.0, config.desired_clearance - min_clearance)

    # Smoothness from discrete trajectory derivatives.
    smooth_cost = trajectory_smoothness_cost(plan)

    speed_target = float(plan.behavior.target_speed)
    mean_speed_error = float(np.mean(np.abs(plan.v - speed_target)))

    cost = 0.0
    if collision:
        cost += config.collision_cost
    cost += config.clearance_weight * clearance_violation * clearance_violation
    cost += config.route_weight * mean_abs_l
    cost += config.goal_weight * final_goal_distance
    cost -= config.progress_weight * progress
    cost += config.heading_weight * mean_heading_error
    cost += config.speed_weight * mean_speed_error
    cost += config.smooth_weight * smooth_cost
    cost += config.behavior_bias_weight * float(plan.behavior.bias)

    obstacle_pressure = obstacle_limited_speed(
        state=observation.state,
        obstacles=scene.obstacles,
        time_s=observation.time_s,
        corridor_half_width=config.corridor_half_width,
        lookahead_m=config.obstacle_lookahead,
    )
    if obstacle_pressure is None and float(plan.v[-1]) < 0.5:
        cost += config.stop_without_reason_cost

    # Prefer center behavior when no obstacle forces passing.
    if obstacle_pressure is None and plan.behavior.mode == "pass":
        cost += 8.0

    plan.cost = float(cost)
    plan.collision = bool(collision)
    plan.min_clearance = float(min_clearance)
    plan.debug.update(
        {
            "final_goal_distance": final_goal_distance,
            "progress": progress,
            "mean_abs_l": mean_abs_l,
            "mean_heading_error": mean_heading_error,
            "smooth_cost": smooth_cost,
        }
    )
    return plan


def trajectory_smoothness_cost(plan: PlannedTrajectory) -> float:
    if len(plan.times) < 4:
        return 0.0
    dt = float(np.mean(np.diff(plan.times)))
    if dt <= 1.0e-6:
        return 0.0

    dx = np.gradient(plan.x, dt)
    dy = np.gradient(plan.y, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    accel_cost = float(np.mean(ddx * ddx + ddy * ddy))

    dyaw = np.diff(np.unwrap(plan.yaw))
    yaw_rate = dyaw / dt
    yaw_rate_cost = float(np.mean(yaw_rate * yaw_rate)) if len(yaw_rate) else 0.0

    return accel_cost + 0.2 * yaw_rate_cost


def track_trajectory_to_action(
    trajectory: PlannedTrajectory,
    observation: AutoDriveObservation,
    config: AutoPolicyConfig,
) -> DriveAction:
    state = observation.state

    if len(trajectory.times) < 2:
        return emergency_stop_action(state, config)

    idx = int(round(config.tracking_lookahead_time / max(config.plan_dt, 1.0e-3)))
    idx = max(int(config.tracking_min_lookahead_index), idx)
    idx = min(idx, len(trajectory.times) - 1)

    target_xy = np.array([trajectory.x[idx], trajectory.y[idx]], dtype=np.float32)
    target_local = local_xy(state, target_xy)

    alpha = math.atan2(float(target_local[1]), max(float(target_local[0]), 1.0e-3))
    target_distance = max(float(np.linalg.norm(target_local)), 1.0)

    steer = math.atan2(
        2.0 * config.wheelbase * math.sin(alpha),
        target_distance,
    )
    steer = clamp(
        steer,
        -math.radians(config.max_steer_deg),
        math.radians(config.max_steer_deg),
    )

    target_speed = float(trajectory.v[min(idx, len(trajectory.v) - 1)])
    # Add a small terminal safety: if the selected plan is nearly stopped,
    # do not let tracking accelerate into it.
    if trajectory.behavior.mode == "stop":
        target_speed = min(target_speed, 0.5)

    accel = config.speed_kp * (target_speed - float(state.speed))
    accel = clamp(accel, -config.brake, config.accel)
    if target_speed < 0.1 and float(state.speed) < 0.2:
        accel = 0.0

    return DriveAction(key="auto", accel=float(accel), steer=float(steer))


# =============================================================================
# Frenet / geometry utilities
# =============================================================================


def build_route_progress(route_xy: np.ndarray) -> np.ndarray:
    if len(route_xy) <= 1:
        return np.zeros((len(route_xy),), dtype=np.float32)
    segment = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment)]).astype(np.float32)


def project_to_route(
    xy: np.ndarray,
    route_xy: np.ndarray,
    route_s: np.ndarray,
) -> FrenetProjection:
    xy = np.asarray(xy, dtype=np.float32)

    if len(route_xy) == 0:
        return FrenetProjection(s=0.0, l=0.0, route_yaw=0.0, index=0, distance=0.0)
    if len(route_xy) == 1:
        diff = xy - route_xy[0]
        return FrenetProjection(
            s=0.0,
            l=float(diff[1]),
            route_yaw=0.0,
            index=0,
            distance=float(np.linalg.norm(diff)),
        )

    best_dist = 1.0e12
    best_s = 0.0
    best_l = 0.0
    best_yaw = 0.0
    best_index = 0

    for i in range(len(route_xy) - 1):
        p0 = route_xy[i]
        p1 = route_xy[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1.0e-6:
            continue

        tangent = seg / seg_len
        rel = xy - p0
        ratio = clamp(float(np.dot(rel, tangent)) / seg_len, 0.0, 1.0)
        proj = p0 + ratio * seg
        diff = xy - proj
        dist = float(np.linalg.norm(diff))

        if dist < best_dist:
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            best_dist = dist
            best_s = float(route_s[i] + ratio * seg_len)
            best_l = float(np.dot(diff, normal))
            best_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
            best_index = i

    return FrenetProjection(
        s=best_s,
        l=best_l,
        route_yaw=best_yaw,
        index=best_index,
        distance=best_dist,
    )


def frenet_to_world(
    s_values: np.ndarray,
    l_values: np.ndarray,
    route_xy: np.ndarray,
    route_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((len(s_values), 2), dtype=np.float32)
    route_yaw = np.zeros((len(s_values),), dtype=np.float32)

    if len(route_xy) == 0:
        return xy, route_yaw
    if len(route_xy) == 1:
        xy[:, :] = route_xy[0]
        xy[:, 1] += l_values
        return xy, route_yaw

    for i, (s, l) in enumerate(zip(s_values, l_values)):
        s = float(clamp(float(s), float(route_s[0]), float(route_s[-1])))

        idx = int(np.searchsorted(route_s, s, side="right") - 1)
        idx = min(max(idx, 0), len(route_xy) - 2)

        s0 = float(route_s[idx])
        s1 = float(route_s[idx + 1])
        p0 = route_xy[idx]
        p1 = route_xy[idx + 1]
        seg = p1 - p0
        seg_len = max(float(np.linalg.norm(seg)), 1.0e-6)
        ratio = clamp((s - s0) / max(s1 - s0, 1.0e-6), 0.0, 1.0)

        tangent = seg / seg_len
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        base = p0 + ratio * seg

        xy[i] = base + normal * float(l)
        route_yaw[i] = math.atan2(float(tangent[1]), float(tangent[0]))

    return xy, route_yaw


def estimate_yaw_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    fallback_yaw: np.ndarray,
) -> np.ndarray:
    n = len(x)
    yaw = np.zeros((n,), dtype=np.float32)
    if n <= 1:
        return np.asarray(fallback_yaw, dtype=np.float32)

    for i in range(n):
        if i == 0:
            dx = float(x[1] - x[0])
            dy = float(y[1] - y[0])
        elif i == n - 1:
            dx = float(x[-1] - x[-2])
            dy = float(y[-1] - y[-2])
        else:
            dx = float(x[i + 1] - x[i - 1])
            dy = float(y[i + 1] - y[i - 1])

        if abs(dx) + abs(dy) < 1.0e-6:
            yaw[i] = float(fallback_yaw[i])
        else:
            yaw[i] = math.atan2(dy, dx)
    return yaw


def estimate_speed_from_xy(
    x: np.ndarray, y: np.ndarray, times: np.ndarray
) -> np.ndarray:
    n = len(x)
    speed = np.zeros((n,), dtype=np.float32)
    if n <= 1:
        return speed

    for i in range(n):
        if i == 0:
            ds = math.hypot(float(x[1] - x[0]), float(y[1] - y[0]))
            dt = max(float(times[1] - times[0]), 1.0e-6)
        else:
            ds = math.hypot(float(x[i] - x[i - 1]), float(y[i] - y[i - 1]))
            dt = max(float(times[i] - times[i - 1]), 1.0e-6)
        speed[i] = ds / dt
    return speed


def collision_and_clearance_at_state(
    state: DriveState,
    time_s: float,
    obstacles: list[dict[str, Any]],
    config: AutoPolicyConfig,
) -> tuple[bool, float]:
    """Approximate ego oriented box vs obstacle box.

    Obstacle orientation is not part of the current toy obstacle dictionary, so
    obstacles are treated as boxes aligned to the ego frame for conservative
    checking.
    """
    min_clearance = 1.0e6
    collision = False

    ego_half_x = 0.5 * float(config.ego_length)
    ego_half_y = 0.5 * float(config.ego_width)

    for obstacle in obstacles:
        center = obstacle_center_at(obstacle, time_s)
        rel = local_xy(state, center)
        size_xy = np.asarray(obstacle["size_xy"], dtype=np.float32)

        obs_half_x = 0.5 * float(size_xy[0])
        obs_half_y = 0.5 * float(size_xy[1])

        safe_x = ego_half_x + obs_half_x + float(config.safety_margin)
        safe_y = ego_half_y + obs_half_y + float(config.safety_margin)

        dx = abs(float(rel[0])) - safe_x
        dy = abs(float(rel[1])) - safe_y

        if dx <= 0.0 and dy <= 0.0:
            collision = True
            min_clearance = 0.0
            continue

        outside_x = max(dx, 0.0)
        outside_y = max(dy, 0.0)
        clearance = math.hypot(outside_x, outside_y)
        min_clearance = min(min_clearance, clearance)

    return bool(collision), float(min_clearance)


def local_xy(state: DriveState, xy: np.ndarray) -> np.ndarray:
    dx = float(xy[0]) - float(state.x)
    dy = float(xy[1]) - float(state.y)
    c = math.cos(float(state.yaw))
    s = math.sin(float(state.yaw))
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)


# =============================================================================
# Fallback legacy policy
# =============================================================================


def legacy_pure_pursuit_action(
    observation: AutoDriveObservation,
    scene: AutoDriveScene,
    config: AutoPolicyConfig,
) -> DriveAction:
    """Emergency fallback used only when SSC/QP returns no valid plan."""
    state = observation.state
    route_target, route_index = pick_lookahead_point(
        route_xy=scene.route_xy,
        route_s=scene.route_s,
        state=state,
        lookahead_m=clamp(config.lookahead_base + 0.45 * state.speed, 5.0, 12.0),
    )

    goal_distance = float(
        np.linalg.norm(np.array([state.x, state.y], dtype=np.float32) - scene.goal_xy)
    )
    blended = (
        config.goal_bias * scene.goal_xy + (1.0 - config.goal_bias) * route_target
    ).astype(np.float32)

    target_local = local_xy(state, blended)
    alpha = math.atan2(float(target_local[1]), max(float(target_local[0]), 1.0e-3))
    target_distance = max(float(np.linalg.norm(target_local)), 1.0)

    steer = math.atan2(2.0 * config.wheelbase * math.sin(alpha), target_distance)
    steer = clamp(
        steer, -math.radians(config.max_steer_deg), math.radians(config.max_steer_deg)
    )

    turn_amount = estimate_route_turn(scene.route_xy, route_index)
    target_speed = config.cruise_speed
    if turn_amount > 0.35:
        target_speed = min(target_speed, 4.5)
    if abs(steer) > math.radians(10.0):
        target_speed = min(target_speed, 4.0)
    if goal_distance < 14.0:
        target_speed = min(target_speed, max(1.5, 0.45 * goal_distance))

    limited_speed = obstacle_limited_speed(
        state=state,
        obstacles=scene.obstacles,
        time_s=observation.time_s,
        corridor_half_width=config.corridor_half_width,
        lookahead_m=config.obstacle_lookahead,
    )
    if limited_speed is not None:
        target_speed = min(target_speed, limited_speed)

    accel = config.speed_kp * (target_speed - state.speed)
    accel = clamp(accel, -config.brake, config.accel)
    if target_speed < 0.1 and state.speed < 0.2:
        accel = 0.0

    return DriveAction(key="auto", accel=float(accel), steer=float(steer))


def pick_lookahead_point(
    route_xy: np.ndarray,
    route_s: np.ndarray,
    state: DriveState,
    lookahead_m: float,
) -> tuple[np.ndarray, int]:
    ego_xy = np.array([state.x, state.y], dtype=np.float32)
    distances = np.linalg.norm(route_xy - ego_xy[None], axis=1)
    nearest_index = int(np.argmin(distances))
    target_s = float(route_s[nearest_index]) + float(lookahead_m)
    target_index = int(np.searchsorted(route_s, target_s, side="left"))
    target_index = min(max(target_index, nearest_index), len(route_xy) - 1)
    return route_xy[target_index], target_index


def estimate_route_turn(
    route_xy: np.ndarray, start_index: int, window: int = 8
) -> float:
    if len(route_xy) < 3:
        return 0.0
    i0 = max(0, start_index - 1)
    i1 = min(len(route_xy) - 1, start_index + window)
    if i1 <= i0 + 1:
        return 0.0
    before = route_xy[i0 + 1] - route_xy[i0]
    after = route_xy[i1] - route_xy[i1 - 1]
    before_yaw = math.atan2(float(before[1]), float(before[0]))
    after_yaw = math.atan2(float(after[1]), float(after[0]))
    return abs(wrap_angle(after_yaw - before_yaw))


def obstacle_limited_speed(
    state: DriveState,
    obstacles: list[dict[str, Any]],
    time_s: float,
    corridor_half_width: float,
    lookahead_m: float,
) -> float | None:
    nearest = None
    for obstacle in obstacles:
        center = obstacle_center_at(obstacle, time_s)
        rel = local_xy(state, center)
        size_xy = np.asarray(obstacle["size_xy"], dtype=np.float32)
        front_margin = 0.5 * float(size_xy[0]) + 2.5
        side_margin = corridor_half_width + 0.5 * float(size_xy[1])
        if rel[0] <= -1.0 or rel[0] > lookahead_m:
            continue
        if abs(float(rel[1])) > side_margin:
            continue
        gap = max(float(rel[0]) - front_margin, 0.0)
        nearest = gap if nearest is None else min(nearest, gap)
    if nearest is None:
        return None
    if nearest < 4.0:
        return 0.0
    if nearest < 10.0:
        return 1.5
    if nearest < 18.0:
        return 3.5
    return 5.0


def emergency_stop_action(state: DriveState, config: AutoPolicyConfig) -> DriveAction:
    accel = clamp(config.speed_kp * (0.0 - float(state.speed)), -config.brake, 0.0)
    if float(state.speed) < 0.2:
        accel = 0.0
    return DriveAction(key="auto", accel=float(accel), steer=0.0)


# =============================================================================
# Misc utilities
# =============================================================================


def target_speed_near_goal(
    state: DriveState, goal_xy: np.ndarray, cruise: float
) -> float:
    ego_xy = np.array([state.x, state.y], dtype=np.float32)
    goal_distance = float(np.linalg.norm(goal_xy - ego_xy))
    if goal_distance < 14.0:
        return min(cruise, max(1.5, 0.45 * goal_distance))
    return cruise


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
