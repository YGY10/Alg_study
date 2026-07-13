"""
Augmented-Lagrangian Constrained iLQR (AL-CILQR) demo.

Compared with a penalty-only iLQR implementation, this version does two
important things:

1. Control bounds are solved inside the backward pass as a box-constrained QP.
2. State inequality constraints are handled by an augmented-Lagrangian outer
   loop with Lagrange multipliers and constraint-residual convergence checks.

State:
    x = [px, py, yaw, v]
Control:
    u = [acceleration, steering]

Run:
    python CILQR_true.py

Optional:
    python CILQR_true.py --animate
    python CILQR_true.py --save cilqr_result.png
    python CILQR_true.py --single-start
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    # Horizon
    N: int = 80
    dt: float = 0.1

    # Vehicle
    wheel_base: float = 2.7

    # State: [px, py, yaw, v]
    start: np.ndarray | None = None
    goal: np.ndarray | None = None

    # Hard control bounds: [acceleration, steering]
    a_min: float = -3.0
    a_max: float = 2.0
    delta_min: float = -0.45
    delta_max: float = 0.45

    # State inequality constraints
    y_min: float = -4.0
    y_max: float = 4.0
    v_ref: float = 3.0
    v_min: float = 0.0
    v_max: float = 6.0

    # Circular obstacle: (x, y, safe_radius)
    obstacle: Tuple[float, float, float] = (12.0, 1.0, 2.40)

    # Original objective weights. Constraints are NOT hidden in these weights.
    w_y: float = 0.08
    w_yaw: float = 0.05
    w_v: float = 2.0
    w_acc: float = 0.03
    w_delta: float = 0.20

    terminal_Q: np.ndarray | None = None

    # Inner constrained-iLQR settings
    max_inner_iter: int = 50
    initial_reg: float = 1e-4
    reg_min: float = 1e-7
    reg_max: float = 1e10
    inner_cost_tol: float = 1e-6

    # Augmented-Lagrangian outer-loop settings
    max_outer_iter: int = 14
    rho_init: float = 10.0
    rho_scale: float = 10.0
    rho_max: float = 1e9
    violation_reduction_ratio: float = 0.5
    constraint_tol: float = 1e-5

    # Numerical tolerances
    box_qp_tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.start is None:
            self.start = np.array([0.0, 0.0, 0.0, 2.0], dtype=float)
        if self.goal is None:
            self.goal = np.array([24.0, 0.0, 0.0, 3.0], dtype=float)
        if self.terminal_Q is None:
            self.terminal_Q = np.diag([150.0, 150.0, 60.0, 50.0])


@dataclass
class SolverHistory:
    inner_merit: List[float] = field(default_factory=list)
    outer_objective: List[float] = field(default_factory=list)
    outer_violation: List[float] = field(default_factory=list)
    outer_rho: List[float] = field(default_factory=list)
    active_control_count: List[int] = field(default_factory=list)
    selected_seed: str = ""


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def control_lower(p: Params) -> np.ndarray:
    return np.array([p.a_min, p.delta_min], dtype=float)


def control_upper(p: Params) -> np.ndarray:
    return np.array([p.a_max, p.delta_max], dtype=float)


def project_control(u: np.ndarray, p: Params) -> np.ndarray:
    return np.minimum(
        np.maximum(np.asarray(u, dtype=float), control_lower(p)), control_upper(p)
    )


def bicycle_dynamics(x: np.ndarray, u: np.ndarray, p: Params) -> np.ndarray:
    """Discrete kinematic bicycle model."""
    px, py, yaw, v = x
    a, delta = u

    next_x = np.zeros(4, dtype=float)
    next_x[0] = px + v * np.cos(yaw) * p.dt
    next_x[1] = py + v * np.sin(yaw) * p.dt
    next_x[2] = wrap_angle(yaw + v / p.wheel_base * np.tan(delta) * p.dt)
    next_x[3] = v + a * p.dt
    return next_x


def dynamics_jacobians(
    x: np.ndarray, u: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order dynamics linearization around (x, u)."""
    _, _, yaw, v = x
    _, delta = u

    A = np.eye(4, dtype=float)
    B = np.zeros((4, 2), dtype=float)

    A[0, 2] = -v * np.sin(yaw) * p.dt
    A[0, 3] = np.cos(yaw) * p.dt
    A[1, 2] = v * np.cos(yaw) * p.dt
    A[1, 3] = np.sin(yaw) * p.dt
    A[2, 3] = np.tan(delta) / p.wheel_base * p.dt

    B[2, 1] = v / p.wheel_base / (np.cos(delta) ** 2) * p.dt
    B[3, 0] = p.dt
    return A, B


# -----------------------------------------------------------------------------
# Original objective: tracking + control effort only.
# State constraints are represented separately as g(x) <= 0.
# -----------------------------------------------------------------------------


def stage_objective(x: np.ndarray, u: np.ndarray, p: Params) -> float:
    _, py, yaw, v = x
    a, delta = u
    return float(
        p.w_y * py**2
        + p.w_yaw * wrap_angle(yaw) ** 2
        + p.w_v * (v - p.v_ref) ** 2
        + p.w_acc * a**2
        + p.w_delta * delta**2
    )


def stage_objective_derivatives(
    x: np.ndarray, u: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic derivatives of the unconstrained stage objective."""
    _, py, yaw, v = x
    a, delta = u

    l_x = np.array(
        [
            0.0,
            2.0 * p.w_y * py,
            2.0 * p.w_yaw * wrap_angle(yaw),
            2.0 * p.w_v * (v - p.v_ref),
        ],
        dtype=float,
    )
    l_u = np.array([2.0 * p.w_acc * a, 2.0 * p.w_delta * delta], dtype=float)

    l_xx = np.diag([0.0, 2.0 * p.w_y, 2.0 * p.w_yaw, 2.0 * p.w_v])
    l_uu = np.diag([2.0 * p.w_acc, 2.0 * p.w_delta])
    l_ux = np.zeros((2, 4), dtype=float)
    return l_x, l_u, l_xx, l_uu, l_ux


def terminal_objective(x: np.ndarray, p: Params) -> float:
    e = np.asarray(x - p.goal, dtype=float)
    e[2] = wrap_angle(e[2])
    return float(e.T @ p.terminal_Q @ e)


def terminal_objective_derivatives(
    x: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray]:
    e = np.asarray(x - p.goal, dtype=float)
    e[2] = wrap_angle(e[2])
    return 2.0 * p.terminal_Q @ e, 2.0 * p.terminal_Q.copy()


# -----------------------------------------------------------------------------
# Explicit state constraints g_i(x) <= 0.
# g0: obstacle safe-radius violation
# g1/g2: upper/lower road-boundary violation
# g3/g4: lower/upper speed violation
# -----------------------------------------------------------------------------


def state_constraints_with_derivatives(
    x: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        g      : [nc], each constraint is g_i(x) <= 0
        grad_g : [nc, nx]
        hess_g : [nc, nx, nx]
    """
    px, py, _, v = x
    ox, oy, safe_radius = p.obstacle

    q = np.array([px - ox, py - oy], dtype=float)
    dist = max(float(np.linalg.norm(q)), 1e-9)

    g = np.array(
        [
            safe_radius - dist,
            py - p.y_max,
            p.y_min - py,
            p.v_min - v,
            v - p.v_max,
        ],
        dtype=float,
    )

    grad_g = np.zeros((5, 4), dtype=float)
    hess_g = np.zeros((5, 4, 4), dtype=float)

    # g_obstacle = r - ||q||
    grad_g[0, 0:2] = -q / dist
    hess_dist = np.eye(2) / dist - np.outer(q, q) / (dist**3)
    hess_g[0, 0:2, 0:2] = -hess_dist

    grad_g[1, 1] = 1.0
    grad_g[2, 1] = -1.0
    grad_g[3, 3] = -1.0
    grad_g[4, 3] = 1.0
    return g, grad_g, hess_g


def augmented_constraint_terms(
    x: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    p: Params,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Powell-Hestenes-Rockafellar augmented Lagrangian for g(x) <= 0:

        phi_i = (max(0, lambda_i + rho*g_i)^2 - lambda_i^2) / (2*rho)

    The multiplier update is:

        lambda_i <- max(0, lambda_i + rho*g_i)
    """
    g, grad_g, hess_g = state_constraints_with_derivatives(x, p)

    value = 0.0
    grad = np.zeros(4, dtype=float)
    hess = np.zeros((4, 4), dtype=float)

    for i in range(len(g)):
        shifted = multipliers[i] + rho * g[i]
        value -= multipliers[i] ** 2 / (2.0 * rho)

        if shifted <= 0.0:
            continue

        value += shifted**2 / (2.0 * rho)
        grad += shifted * grad_g[i]
        hess += rho * np.outer(grad_g[i], grad_g[i]) + shifted * hess_g[i]

    return float(value), grad, 0.5 * (hess + hess.T)


def augmented_stage_derivatives(
    x: np.ndarray,
    u: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_x, l_u, l_xx, l_uu, l_ux = stage_objective_derivatives(x, u, p)
    _, c_x, c_xx = augmented_constraint_terms(x, multipliers, rho, p)
    return l_x + c_x, l_u, l_xx + c_xx, l_uu, l_ux


def augmented_terminal_derivatives(
    x: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray]:
    V_x, V_xx = terminal_objective_derivatives(x, p)
    _, c_x, c_xx = augmented_constraint_terms(x, multipliers, rho, p)
    return V_x + c_x, V_xx + c_xx


def rollout(U: np.ndarray, p: Params) -> np.ndarray:
    X = np.zeros((p.N + 1, 4), dtype=float)
    X[0] = p.start.copy()
    for t in range(p.N):
        X[t + 1] = bicycle_dynamics(X[t], project_control(U[t], p), p)
    return X


def original_total_cost(X: np.ndarray, U: np.ndarray, p: Params) -> float:
    value = sum(stage_objective(X[t], U[t], p) for t in range(p.N))
    return float(value + terminal_objective(X[-1], p))


def augmented_total_cost(
    X: np.ndarray,
    U: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    p: Params,
) -> float:
    value = 0.0
    for t in range(p.N):
        c, _, _ = augmented_constraint_terms(X[t], multipliers[t], rho, p)
        value += stage_objective(X[t], U[t], p) + c

    c_terminal, _, _ = augmented_constraint_terms(X[-1], multipliers[-1], rho, p)
    value += terminal_objective(X[-1], p) + c_terminal
    return float(value)


def trajectory_constraint_values(X: np.ndarray, p: Params) -> np.ndarray:
    values = np.zeros((p.N + 1, 5), dtype=float)
    for t in range(p.N + 1):
        values[t] = state_constraints_with_derivatives(X[t], p)[0]
    return values


def max_constraint_violation(X: np.ndarray, p: Params) -> float:
    g = trajectory_constraint_values(X, p)
    return float(np.max(np.maximum(g, 0.0)))


# -----------------------------------------------------------------------------
# Exact small box-QP solver used in the backward pass.
# -----------------------------------------------------------------------------


def solve_box_qp(
    H: np.ndarray,
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the strictly convex box QP:

        min_z  0.5*z.T*H*z + g.T*z
        s.t.   lower <= z <= upper

    The control dimension is two in this demo, so enumerating the 3^nu active
    sets is exact, simple, and dependency-free.

    Returns:
        z         : optimal feedforward control increment
        free_mask : dimensions not clamped at a lower/upper bound
    """
    n = len(g)
    best_z: np.ndarray | None = None
    best_status: Tuple[int, ...] | None = None
    best_obj = np.inf

    # status: -1 = lower bound, 0 = free, +1 = upper bound
    for status in product((-1, 0, 1), repeat=n):
        z = np.zeros(n, dtype=float)
        active = [i for i, s in enumerate(status) if s != 0]
        free = [i for i, s in enumerate(status) if s == 0]

        for i in active:
            z[i] = lower[i] if status[i] < 0 else upper[i]

        if free:
            H_ff = H[np.ix_(free, free)]
            rhs = -g[free]
            if active:
                rhs -= H[np.ix_(free, active)] @ z[active]
            try:
                z[free] = np.linalg.solve(H_ff, rhs)
            except np.linalg.LinAlgError:
                continue

        if np.any(z < lower - tol) or np.any(z > upper + tol):
            continue

        z = np.minimum(np.maximum(z, lower), upper)
        obj = 0.5 * float(z.T @ H @ z) + float(g.T @ z)
        if obj < best_obj:
            best_obj = obj
            best_z = z.copy()
            best_status = status

    if best_z is None or best_status is None:
        raise np.linalg.LinAlgError("box QP has no numerically valid active set")

    free_mask = np.array([s == 0 for s in best_status], dtype=bool)
    return best_z, free_mask


def backward_pass(
    X: np.ndarray,
    U: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    reg: float,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Constrained iLQR backward pass with a box-QP at every time step."""
    nx = 4
    nu = 2
    k_seq = np.zeros((p.N, nu), dtype=float)
    K_seq = np.zeros((p.N, nu, nx), dtype=float)

    V_x, V_xx = augmented_terminal_derivatives(X[-1], multipliers[-1], rho, p)
    active_count = 0

    u_min = control_lower(p)
    u_max = control_upper(p)

    for t in reversed(range(p.N)):
        A, B = dynamics_jacobians(X[t], U[t], p)
        l_x, l_u, l_xx, l_uu, l_ux = augmented_stage_derivatives(
            X[t], U[t], multipliers[t], rho, p
        )

        Q_x = l_x + A.T @ V_x
        Q_u = l_u + B.T @ V_x
        Q_xx = l_xx + A.T @ V_xx @ A
        Q_uu = l_uu + B.T @ V_xx @ B
        Q_ux = l_ux + B.T @ V_xx @ A

        Q_xx = 0.5 * (Q_xx + Q_xx.T)
        Q_uu = 0.5 * (Q_uu + Q_uu.T)
        Q_uu_reg = Q_uu + reg * np.eye(nu)

        # Bounds on the control increment k, not on an already-computed control.
        du_lower = u_min - U[t]
        du_upper = u_max - U[t]
        k, free_mask = solve_box_qp(Q_uu_reg, Q_u, du_lower, du_upper, p.box_qp_tol)

        K = np.zeros((nu, nx), dtype=float)
        free = np.flatnonzero(free_mask)
        if len(free) > 0:
            H_ff = Q_uu_reg[np.ix_(free, free)]
            K[free] = -np.linalg.solve(H_ff, Q_ux[free])

        active_count += int(nu - len(free))
        k_seq[t] = k
        K_seq[t] = K

        # Substitute du = k + K*dx into the local Q function.
        V_x = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
        V_xx = Q_xx + K.T @ Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K
        V_xx = 0.5 * (V_xx + V_xx.T)

    return k_seq, K_seq, active_count


def constrained_ilqr_inner(
    U_init: np.ndarray,
    multipliers: np.ndarray,
    rho: float,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    """One inner constrained-iLQR solve for fixed AL multipliers and rho."""
    U = np.asarray(U_init, dtype=float).copy()
    for t in range(p.N):
        U[t] = project_control(U[t], p)

    X = rollout(U, p)
    merit = augmented_total_cost(X, U, multipliers, rho, p)
    merit_history = [merit]
    reg = p.initial_reg
    last_active_count = 0

    alphas = (1.0, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01)

    for _ in range(p.max_inner_iter):
        try:
            k_seq, K_seq, active_count = backward_pass(X, U, multipliers, rho, reg, p)
        except np.linalg.LinAlgError:
            reg *= 10.0
            if reg > p.reg_max:
                break
            merit_history.append(merit)
            continue

        accepted = False
        for alpha in alphas:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = p.start.copy()

            for t in range(p.N):
                dx = X_new[t] - X[t]
                dx[2] = wrap_angle(dx[2])

                # Active control dimensions have K rows equal to zero. Projection
                # remains as a numerical safeguard for the nonlinear rollout.
                u_trial = U[t] + alpha * k_seq[t] + K_seq[t] @ dx
                U_new[t] = project_control(u_trial, p)
                X_new[t + 1] = bicycle_dynamics(X_new[t], U_new[t], p)

            merit_new = augmented_total_cost(X_new, U_new, multipliers, rho, p)
            if merit_new < merit - 1e-12:
                X, U, merit = X_new, U_new, merit_new
                reg = max(reg / 2.0, p.reg_min)
                accepted = True
                last_active_count = active_count
                break

        if not accepted:
            reg *= 10.0

        merit_history.append(merit)

        if reg > p.reg_max:
            break
        if len(merit_history) >= 6:
            recent = merit_history[-6:]
            if max(recent) - min(recent) < p.inner_cost_tol:
                break

    return X, U, merit_history, last_active_count


def al_cilqr_optimize(
    U_init: np.ndarray,
    p: Params,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, SolverHistory]:
    """
    Augmented-Lagrangian constrained iLQR.

    The inner loop solves a control-limited iLQR problem. The outer loop updates
    inequality-constraint multipliers and the penalty parameter until primal
    constraint residuals are small.
    """
    multipliers = np.zeros((p.N + 1, 5), dtype=float)
    rho = p.rho_init
    previous_violation = np.inf
    U = np.asarray(U_init, dtype=float).copy()
    X = rollout(U, p)
    history = SolverHistory()

    for outer in range(p.max_outer_iter):
        X, U, inner_history, active_count = constrained_ilqr_inner(
            U, multipliers, rho, p
        )
        history.inner_merit.extend(inner_history)

        g = trajectory_constraint_values(X, p)
        violation = float(np.max(np.maximum(g, 0.0)))
        objective = original_total_cost(X, U, p)

        history.outer_objective.append(objective)
        history.outer_violation.append(violation)
        history.outer_rho.append(rho)
        history.active_control_count.append(active_count)

        if verbose:
            print(
                f"outer={outer:02d} objective={objective:12.6f} "
                f"max_violation={violation:10.6e} rho={rho:9.2e} "
                f"active_controls={active_count}"
            )

        if violation <= p.constraint_tol:
            break

        # KKT multiplier projection for g(x) <= 0.
        multipliers = np.maximum(0.0, multipliers + rho * g)

        # Increase rho only when primal feasibility is not improving enough.
        if violation > p.violation_reduction_ratio * previous_violation:
            rho = min(rho * p.rho_scale, p.rho_max)
        previous_violation = violation

    return X, U, history


def make_initial_controls(p: Params, direction: float = 0.0) -> np.ndarray:
    """
    Initial control sequence.

    direction = 0: straight seed
    direction = +1/-1: smooth left/right S-curve seed

    Multiple seeds are useful because obstacle avoidance has different topology
    classes, and a local optimizer should not be expected to invent both sides
    from one perfectly symmetric trajectory.
    """
    U = np.zeros((p.N, 2), dtype=float)
    total_time = p.N * p.dt
    U[:, 0] = (p.v_ref - p.start[3]) / total_time

    if direction != 0.0:
        t = np.arange(p.N, dtype=float) * p.dt
        first_turn = np.exp(-0.5 * ((t - 2.2) / 0.75) ** 2)
        return_turn = np.exp(-0.5 * ((t - 4.6) / 0.95) ** 2)
        settle_turn = np.exp(-0.5 * ((t - 6.4) / 0.75) ** 2)
        U[:, 1] = direction * (
            0.24 * first_turn - 0.28 * return_turn + 0.10 * settle_turn
        )

    for i in range(p.N):
        U[i] = project_control(U[i], p)
    return U


def optimize_with_multistart(
    p: Params,
    single_start: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SolverHistory]:
    candidates = (
        [("straight", 0.0)]
        if single_start
        else [
            ("left", 1.0),
            ("right", -1.0),
        ]
    )

    best = None
    for name, direction in candidates:
        print(f"\n=== seed: {name} ===")
        U0 = make_initial_controls(p, direction)
        X0 = rollout(U0, p)
        X, U, history = al_cilqr_optimize(U0, p, verbose=True)

        violation = max_constraint_violation(X, p)
        objective = original_total_cost(X, U, p)
        feasible = violation <= p.constraint_tol
        rank = (not feasible, objective if feasible else violation, violation)

        if best is None or rank < best[0]:
            history.selected_seed = name
            best = (rank, X0, X, U, history)

    assert best is not None
    _, X0_best, X_best, U_best, history_best = best
    return X0_best, X_best, U_best, history_best


def compute_clearance(X: np.ndarray, p: Params) -> float:
    ox, oy, safe_radius = p.obstacle
    dists = np.hypot(X[:, 0] - ox, X[:, 1] - oy)
    return float(np.min(dists - safe_radius))


def plot_result(
    X0: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    history: SolverHistory,
    p: Params,
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax_traj = fig.add_subplot(2, 2, (1, 3))
    ax_conv = fig.add_subplot(2, 2, 2)
    ax_ctrl = fig.add_subplot(2, 2, 4)

    ax_traj.axhline(p.y_min, linestyle="--", linewidth=1)
    ax_traj.axhline(p.y_max, linestyle="--", linewidth=1)
    ax_traj.fill_between(
        [p.start[0] - 2.0, p.goal[0] + 2.0], p.y_min, p.y_max, alpha=0.08
    )

    ox, oy, safe_radius = p.obstacle
    ax_traj.add_patch(plt.Circle((ox, oy), safe_radius, fill=False, linewidth=2))
    ax_traj.add_patch(plt.Circle((ox, oy), 1.2, fill=True, alpha=0.25))

    ax_traj.plot(
        X0[:, 0], X0[:, 1], "--", label=f"initial seed ({history.selected_seed})"
    )
    ax_traj.plot(X[:, 0], X[:, 1], linewidth=2.5, label="AL-CILQR optimized")
    ax_traj.scatter([p.start[0]], [p.start[1]], s=80, marker="o", label="start")
    ax_traj.scatter([p.goal[0]], [p.goal[1]], s=120, marker="*", label="goal")
    ax_traj.set_title("Augmented-Lagrangian constrained iLQR")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend(loc="best")

    outer = np.arange(len(history.outer_objective))
    ax_conv.plot(outer, history.outer_objective, marker="o", label="original objective")
    ax_conv.set_title("Outer-loop convergence")
    ax_conv.set_xlabel("AL outer iteration")
    ax_conv.set_ylabel("objective")
    ax_conv.grid(True)

    ax_violation = ax_conv.twinx()
    safe_violation = np.maximum(np.asarray(history.outer_violation), 1e-12)
    ax_violation.semilogy(
        outer, safe_violation, marker="x", linestyle="--", label="max violation"
    )
    ax_violation.axhline(p.constraint_tol, linestyle=":", linewidth=1)
    ax_violation.set_ylabel("max constraint violation")

    lines1, labels1 = ax_conv.get_legend_handles_labels()
    lines2, labels2 = ax_violation.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labels1 + labels2, loc="best")

    time = np.arange(p.N) * p.dt
    ax_ctrl.plot(time, U[:, 0], label="acceleration a")
    ax_ctrl.plot(time, U[:, 1], label="steering delta")
    ax_ctrl.axhline(p.a_min, linestyle="--", linewidth=1)
    ax_ctrl.axhline(p.a_max, linestyle="--", linewidth=1)
    ax_ctrl.axhline(p.delta_min, linestyle=":", linewidth=1)
    ax_ctrl.axhline(p.delta_max, linestyle=":", linewidth=1)
    ax_ctrl.set_title("Controls: bounds solved in backward-pass QP")
    ax_ctrl.set_xlabel("time [s]")
    ax_ctrl.grid(True)
    ax_ctrl.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved figure to: {save_path}")
    plt.show()


def animate_result(X: np.ndarray, p: Params) -> None:
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(p.y_min, linestyle="--", linewidth=1)
    ax.axhline(p.y_max, linestyle="--", linewidth=1)

    ox, oy, safe_radius = p.obstacle
    ax.add_patch(plt.Circle((ox, oy), safe_radius, fill=False, linewidth=2))
    ax.add_patch(plt.Circle((ox, oy), 1.2, fill=True, alpha=0.25))
    ax.plot(X[:, 0], X[:, 1], "--", alpha=0.5, label="optimized trajectory")
    (point,) = ax.plot([], [], marker="o", markersize=8, label="ego")

    ax.scatter([p.start[0]], [p.start[1]], s=80, marker="o", label="start")
    ax.scatter([p.goal[0]], [p.goal[1]], s=120, marker="*", label="goal")
    ax.set_xlim(p.start[0] - 2.0, p.goal[0] + 2.0)
    ax.set_ylim(p.y_min - 1.0, p.y_max + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("AL-CILQR trajectory animation")

    def update(i: int):
        point.set_data([X[i, 0]], [X[i, 1]])
        ax.set_xlabel(f"frame={i}, x={X[i, 0]:.2f}, y={X[i, 1]:.2f}, v={X[i, 3]:.2f}")
        return (point,)

    FuncAnimation(fig, update, frames=len(X), interval=60, blit=True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument(
        "--single-start",
        action="store_true",
        help="use only the straight initial trajectory; multistart is safer for symmetric obstacles",
    )
    args = parser.parse_args()

    p = Params()
    X0, X, U, history = optimize_with_multistart(p, args.single_start)

    print("\n=== selected solution ===")
    print("seed:", history.selected_seed)
    print("original objective:", original_total_cost(X, U, p))
    print("final state:", X[-1])
    print("goal state :", p.goal)
    print("minimum obstacle clearance:", compute_clearance(X, p))
    print("maximum state-constraint violation:", max_constraint_violation(X, p))
    print("max |steer|:", float(np.max(np.abs(U[:, 1]))))
    print("max |acc|:", float(np.max(np.abs(U[:, 0]))))

    plot_result(X0, X, U, history, p, args.save)
    if args.animate:
        animate_result(X, p)


if __name__ == "__main__":
    main()
