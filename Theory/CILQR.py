"""
Educational CILQR demo:
- Bicycle model
- Obstacle avoidance
- Road boundary constraints
- Speed/control constraints
- Matplotlib visualization

Run:
    python CILQR.py

Optional:
    python CILQR.py --animate
    python CILQR.py --save cilqr_result.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    # Horizon
    N: int = 80
    dt: float = 0.1

    # Vehicle
    wheel_base: float = 2.7

    # State: [px, py, yaw, v]
    start: np.ndarray = None
    goal: np.ndarray = None

    # Control: [acceleration, steering]
    a_min: float = -3.0
    a_max: float = 2.0
    delta_min: float = -0.45
    delta_max: float = 0.45

    # Road boundary
    y_min: float = -4.0
    y_max: float = 4.0

    # Speed
    v_ref: float = 3.0
    v_min: float = 0.0
    v_max: float = 6.0

    # Obstacle: (x, y, safe_radius)
    obstacle: Tuple[float, float, float] = (12.0, -0.0, 2.40)

    # Cost weights
    w_y: float = 0.08
    w_yaw: float = 0.05
    w_v: float = 2.0
    w_acc: float = 0.03
    w_delta: float = 0.20
    w_obstacle: float = 1000.0
    w_boundary: float = 1000.0
    w_v_limit: float = 50.0

    # Terminal cost
    terminal_Q: np.ndarray = None

    # Optimization
    max_iter: int = 45
    initial_reg: float = 1e-4  # 正则化系数，防止矩阵求逆数值爆增
    finite_diff_eps: float = 1e-4

    def __post_init__(self):
        if self.start is None:
            self.start = np.array([0.0, 0.0, 0.0, 2.0], dtype=float)

        if self.goal is None:
            self.goal = np.array([24.0, 0.0, 0.0, 3.0], dtype=float)

        if self.terminal_Q is None:
            self.terminal_Q = np.diag([150.0, 150.0, 60.0, 50.0])


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def softplus(z: float, beta: float = 5.0) -> float:
    """
    Smooth approximation of max(0, z).
    Larger beta -> closer to ReLU.
    """
    x = beta * z
    return float(np.log1p(np.exp(-abs(x))) / beta + max(x, 0.0) / beta)


def clip_control(u: np.ndarray, p: Params) -> np.ndarray:
    u = np.array(u, dtype=float).copy()
    u[0] = np.clip(u[0], p.a_min, p.a_max)
    u[1] = np.clip(u[1], p.delta_min, p.delta_max)
    return u


def bicycle_dynamics(x: np.ndarray, u: np.ndarray, p: Params) -> np.ndarray:
    """
    Discrete bicycle model.

    State:
        x = [px, py, yaw, v]

    Control:
        u = [a, delta]
    """
    px, py, yaw, v = x
    a, delta = u

    dt = p.dt
    L = p.wheel_base

    next_x = np.zeros(4)
    next_x[0] = px + v * np.cos(yaw) * dt
    next_x[1] = py + v * np.sin(yaw) * dt
    next_x[2] = wrap_angle(yaw + v / L * np.tan(delta) * dt)
    next_x[3] = v + a * dt

    return next_x


def dynamics_jacobians(
    x: np.ndarray, u: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize dynamics:
        x_{t+1} ~= A x_t + B u_t
    around current trajectory point.
    """
    _, _, yaw, v = x
    _, delta = u

    dt = p.dt
    L = p.wheel_base

    A = np.eye(4)
    B = np.zeros((4, 2))

    A[0, 2] = -v * np.sin(yaw) * dt
    A[0, 3] = np.cos(yaw) * dt

    A[1, 2] = v * np.cos(yaw) * dt
    A[1, 3] = np.sin(yaw) * dt

    A[2, 3] = np.tan(delta) / L * dt

    B[2, 1] = v / L * (1.0 / np.cos(delta) ** 2) * dt
    B[3, 0] = dt

    return A, B


def stage_cost(x: np.ndarray, u: np.ndarray, p: Params) -> float:
    """
    One-step cost.

    This is where we put soft constraints:
    - obstacle avoidance
    - road boundary
    - speed limit
    """
    px, py, yaw, v = x
    a, delta = u
    cost = 0.0
    cost += p.w_y * py**2
    cost += p.w_yaw * wrap_angle(yaw) ** 2

    # Desired speed
    cost += p.w_v * (v - p.v_ref) ** 2

    # Control effort
    cost += p.w_acc * a**2
    cost += p.w_delta * delta**2

    # Obstacle soft constraint
    ox, oy, safe_radius = p.obstacle
    dist = np.hypot(px - ox, py - oy) + 1e-6
    obstacle_violation = softplus(safe_radius - dist, beta=5.0)
    cost += p.w_obstacle * obstacle_violation**2
    # Road Boundary soft constraint
    upper_violation = softplus(py - p.y_max, beta=5.0)
    lower_violation = softplus(p.y_min - py, beta=5.0)
    cost += p.w_boundary * (upper_violation**2 + lower_violation**2)

    # Speed limit soft constraint
    v_low_violation = softplus(p.v_min - v, beta=5.0)
    v_high_violation = softplus(v - p.v_max, beta=5.0)
    cost += p.w_v_limit * (v_low_violation**2 + v_high_violation**2)

    return float(cost)


def terminal_cost(x: np.ndarray, p: Params) -> float:
    e = x - p.goal
    e[2] = wrap_angle(e[2])
    return float(e.T @ p.terminal_Q @ e)


def total_cost(X: np.ndarray, U: np.ndarray, p: Params) -> float:
    cost = 0.0
    for t in range(p.N):
        cost += stage_cost(X[t], U[t], p)
    cost += terminal_cost(X[-1], p)
    return float(cost)


def rollout(U: np.ndarray, p: Params) -> np.ndarray:
    X = np.zeros((p.N + 1, 4))
    X[0] = p.start.copy()

    for t in range(p.N):
        u = clip_control(U[t], p)
        X[t + 1] = bicycle_dynamics(X[t], u, p)
    return X


def stage_cost_derivatives(
    x: np.ndarray, u: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerical derivatives of stage cost.

    Returns:
        l_x   : [nx]
        l_u   : [nu]
        l_xx  : [nx, nx]
        l_uu  : [nu, nu]
        l_ux  : [nu, nx]

    For learning, numerical derivatives are easier to read.
    For industrial use, analytic/autodiff derivatives are preferred.
    """
    nx = 4
    nu = 2
    eps = p.finite_diff_eps

    l0 = stage_cost(x, u, p)

    l_x = np.zeros(nx)
    l_u = np.zeros(nu)
    l_xx = np.zeros((nx, nx))
    l_uu = np.zeros((nu, nu))
    l_ux = np.zeros((nu, nx))

    # State gradient and diagonal Hessian.
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        fp = stage_cost(x + dx, u, p)
        fm = stage_cost(x - dx, u, p)

        l_x[i] = (fp - fm) / (2.0 * eps)
        l_xx[i, i] = (fp - 2.0 * l0 + fm) / (eps**2)

    # Control gradient and diagonal Hessian.
    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        fp = stage_cost(x, u + du, p)
        fm = stage_cost(x, u - du, p)

        l_u[i] = (fp - fm) / (2.0 * eps)
        l_uu[i, i] = (fp - 2.0 * l0 + fm) / (eps**2)

    # State Hessian off-diagonal.
    for i in range(nx):
        for j in range(i + 1, nx):
            dxi = np.zeros(nx)
            dxj = np.zeros(nx)
            dxi[i] = eps
            dxj[j] = eps

            fpp = stage_cost(x + dxi + dxj, u, p)
            fpm = stage_cost(x + dxi - dxj, u, p)
            fmp = stage_cost(x - dxi + dxj, u, p)
            fmm = stage_cost(x - dxi - dxj, u, p)

            value = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            l_xx[i, j] = value
            l_xx[j, i] = value

    # Control Hessian off-diagonal.
    for i in range(nu):
        for j in range(i + 1, nu):
            dui = np.zeros(nu)
            duj = np.zeros(nu)
            dui[i] = eps
            duj[j] = eps

            fpp = stage_cost(x, u + dui + duj, p)
            fpm = stage_cost(x, u + dui - duj, p)
            fmp = stage_cost(x, u - dui + duj, p)
            fmm = stage_cost(x, u - dui - duj, p)

            value = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            l_uu[i, j] = value
            l_uu[j, i] = value

    # Mixed Hessian: d^2 l / du dx
    for i in range(nu):
        for j in range(nx):
            du = np.zeros(nu)
            dx = np.zeros(nx)
            du[i] = eps
            dx[j] = eps

            fpp = stage_cost(x + dx, u + du, p)
            fpm = stage_cost(x - dx, u + du, p)
            fmp = stage_cost(x + dx, u - du, p)
            fmm = stage_cost(x - dx, u - du, p)

            l_ux[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)

    return l_x, l_u, l_xx, l_uu, l_ux


def terminal_cost_derivatives(
    x: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray]:
    e = x - p.goal
    e[2] = wrap_angle(e[2])

    V_x = 2.0 * p.terminal_Q @ e
    V_xx = 2.0 * p.terminal_Q

    return V_x, V_xx


def cilqr_optimize(
    U_init: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    iLQR / CILQR-style optimization.

    Backward pass:
        local LQR approximation

    Forward pass:
        line search + control clipping

    Constraints:
        obstacle, boundary, speed limits are soft penalties.
        acceleration and steering are clipped as hard control limits.
    """
    U = U_init.copy()
    X = rollout(U, p)
    J = total_cost(X, U, p)

    history = [J]
    reg = p.initial_reg

    alphas = [1.0, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]

    for iteration in range(p.max_iter):
        # Terminal value function.
        # 终点代价对终点状态的导数，V：从某个状态出发，未来还会产生多少总代价
        # 终点只有当前代价
        V_x, V_xx = terminal_cost_derivatives(X[-1], p)
        # k_seq 用来保存每个时间步的前馈控制修正， k_t = [Δa, Δdelta]
        k_seq = np.zeros_like(U)
        # K_seq 保存每个时间步的反馈矩阵。
        K_seq = np.zeros((p.N, 2, 4))

        backward_ok = True

        # Backward pass.
        for t in reversed(range(p.N)):
            # 从最后一个时刻往前遍历
            # 取当前时刻状态和控制量
            x = X[t]
            u = U[t]

            A, B = dynamics_jacobians(x, u, p)
            # 在当前状态和控制附近，对stage cost做二次近似
            """
            l_x: 状态稍微变化，当前这一步 cost 怎么变

            l_u: 控制稍微变化，当前这一步 cost 怎么变

            l_xx: cost 关于状态的曲率

            l_uu: cost 关于控制的曲率

            l_ux: 控制和状态之间的耦合曲率
            """
            l_x, l_u, l_xx, l_uu, l_ux = stage_cost_derivatives(x, u, p)

            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = l_ux + B.T @ V_xx @ A

            # Regularization improves numerical stability.
            Q_uu = 0.5 * (Q_uu + Q_uu.T)
            Q_uu_reg = Q_uu + reg * np.eye(2)

            try:
                k = -np.linalg.solve(Q_uu_reg, Q_u)
                K = -np.linalg.solve(Q_uu_reg, Q_ux)
            except np.linalg.LinAlgError:
                backward_ok = False
                break

            k_seq[t] = k
            K_seq[t] = K

            # Update value function.
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)

        if not backward_ok:
            reg *= 10.0
            history.append(J)
            print(f"iter={iteration:02d} backward failed, increase reg={reg:.2e}")
            continue

        # Forward line search.
        accepted = False

        for alpha in alphas:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = p.start.copy()

            for t in range(p.N):
                dx = X_new[t] - X[t]
                dx[2] = wrap_angle(dx[2])

                u_new = U[t] + alpha * k_seq[t] + K_seq[t] @ dx
                u_new = clip_control(u_new, p)

                U_new[t] = u_new
                X_new[t + 1] = bicycle_dynamics(X_new[t], u_new, p)

            J_new = total_cost(X_new, U_new, p)

            if J_new < J:
                X = X_new
                U = U_new
                J = J_new
                accepted = True
                reg = max(reg / 2.0, 1e-6)
                break

        if not accepted:
            reg *= 10.0

        history.append(J)

        print(
            f"iter={iteration:02d} "
            f"cost={J:.3f} "
            f"reg={reg:.2e} "
            f"accepted={accepted}"
        )

        if len(history) > 6:
            recent = history[-6:]
            improvement = max(recent) - min(recent)
            if improvement < 1e-3:
                print("Converged: recent cost improvement is very small.")
                break

    return X, U, history


def make_initial_controls(p: Params) -> np.ndarray:
    """
    Initial control sequence.

    Start from almost straight driving.
    The obstacle is slightly off-center, so the gradient can push the car around it.
    """
    U = np.zeros((p.N, 2))

    # Small constant acceleration toward v_ref.
    total_time = p.N * p.dt
    U[:, 0] = (p.v_ref - p.start[3]) / total_time

    # Steering initially zero.
    U[:, 1] = 0.0

    return U


def compute_clearance(X: np.ndarray, p: Params) -> float:
    ox, oy, safe_radius = p.obstacle
    dists = np.hypot(X[:, 0] - ox, X[:, 1] - oy)
    return float(np.min(dists - safe_radius))


def plot_result(
    X0: np.ndarray,
    U0: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    history: List[float],
    p: Params,
    save_path: str | None,
):
    fig = plt.figure(figsize=(12, 8))

    ax_traj = fig.add_subplot(2, 2, (1, 3))
    ax_cost = fig.add_subplot(2, 2, 2)
    ax_ctrl = fig.add_subplot(2, 2, 4)

    # Road boundary.
    ax_traj.axhline(p.y_min, linestyle="--", linewidth=1)
    ax_traj.axhline(p.y_max, linestyle="--", linewidth=1)
    ax_traj.fill_between(
        [p.start[0] - 2.0, p.goal[0] + 2.0],
        p.y_min,
        p.y_max,
        alpha=0.08,
    )

    # Obstacle.
    ox, oy, safe_radius = p.obstacle
    safe_circle = plt.Circle((ox, oy), safe_radius, fill=False, linewidth=2)
    physical_circle = plt.Circle((ox, oy), 1.2, fill=True, alpha=0.25)
    ax_traj.add_patch(safe_circle)
    ax_traj.add_patch(physical_circle)

    # Trajectories.
    ax_traj.plot(X0[:, 0], X0[:, 1], "--", label="initial rollout")
    ax_traj.plot(X[:, 0], X[:, 1], linewidth=2.5, label="CILQR optimized")

    # Start / goal.
    ax_traj.scatter([p.start[0]], [p.start[1]], s=80, marker="o", label="start")
    ax_traj.scatter([p.goal[0]], [p.goal[1]], s=120, marker="*", label="goal")

    ax_traj.set_title("Trajectory optimization with obstacle and road constraints")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend(loc="best")

    # Cost.
    ax_cost.plot(history, marker="o")
    ax_cost.set_title("Cost convergence")
    ax_cost.set_xlabel("iteration")
    ax_cost.set_ylabel("total cost")
    ax_cost.grid(True)

    # Controls.
    t = np.arange(p.N) * p.dt
    ax_ctrl.plot(t, U[:, 0], label="acceleration a")
    ax_ctrl.plot(t, U[:, 1], label="steering delta")
    ax_ctrl.axhline(p.a_min, linestyle="--", linewidth=1)
    ax_ctrl.axhline(p.a_max, linestyle="--", linewidth=1)
    ax_ctrl.axhline(p.delta_min, linestyle=":", linewidth=1)
    ax_ctrl.axhline(p.delta_max, linestyle=":", linewidth=1)
    ax_ctrl.set_title("Optimized controls")
    ax_ctrl.set_xlabel("time [s]")
    ax_ctrl.grid(True)
    ax_ctrl.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved figure to: {save_path}")

    plt.show()


def animate_result(X: np.ndarray, p: Params):
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
    ax.set_title("CILQR trajectory animation")

    def update(i):
        point.set_data([X[i, 0]], [X[i, 1]])
        ax.set_xlabel(f"frame={i}, x={X[i, 0]:.2f}, y={X[i, 1]:.2f}, v={X[i, 3]:.2f}")
        return (point,)

    ani = FuncAnimation(fig, update, frames=len(X), interval=60, blit=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animate", action="store_true", help="show animation after static plots"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="save static figure to path, e.g. cilqr_result.png",
    )
    args = parser.parse_args()

    p = Params()

    U0 = make_initial_controls(p)
    X0 = rollout(U0, p)

    print("Initial cost:", total_cost(X0, U0, p))
    print("Initial obstacle clearance:", compute_clearance(X0, p))

    X, U, history = cilqr_optimize(U0, p)

    print()
    print("Final cost:", history[-1])
    print("Final state:", X[-1])
    print("Goal state :", p.goal)
    print("Final obstacle clearance:", compute_clearance(X, p))
    print("Max |steer|:", np.max(np.abs(U[:, 1])))
    print("Max |acc|:", np.max(np.abs(U[:, 0])))

    plot_result(X0, U0, X, U, history, p, args.save)

    if args.animate:
        animate_result(X, p)


if __name__ == "__main__":
    main()
