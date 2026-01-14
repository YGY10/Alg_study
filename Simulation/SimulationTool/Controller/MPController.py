import math
import numpy as np
import cvxpy as cp
from SimulationTool.Common.common import normalize_angle


class MPCController:
    def __init__(
        self,
        wheelbase=2.7,
        v_ref=2.0,
        dt=0.05,
        horizon=25,
        Q=np.diag([200.0, 1.0]),
        R=0.01,
        Rd=1.0,
    ):
        self.L, self.v, self.dt, self.N = wheelbase, v_ref, dt, horizon
        self.Q, self.R, self.Rd = Q, R, Rd
        self.max_steer = math.radians(35)
        self.last_delta = 0.0
        self.last_idx = 0

    def compute_control(self, ego, traj, dt):
        if not traj:
            return 0.0, 0.0, 0.0

        # 1. 单调最近点搜索
        min_dist, idx = float("inf"), self.last_idx
        search_end = min(self.last_idx + 30, len(traj))
        for i in range(self.last_idx, search_end):
            d = (traj[i].x - ego.x) ** 2 + (traj[i].y - ego.y) ** 2
            if d < min_dist:
                min_dist, idx = d, i
        self.last_idx = idx
        ref = traj[idx]

        # 2. Frenet 误差（垂足投影）
        # 计算横向误差 e_y: 向量 (ego-ref) 在 ref_yaw 法线方向上的投影
        # dx = ego.x - ref.x
        # dy = ego.y - ref.y
        # 这里的 e_y 符号需与运动学模型匹配：左正右负
        # e_y = -math.sin(ref.yaw) * dx + math.cos(ref.yaw) * dy
        dx, dy = ego.x - ref.x, ego.y - ref.y
        nx, ny = -math.sin(ref.yaw), math.cos(ref.yaw)
        e_y = dx * nx + dy * ny
        e_yaw = normalize_angle(ego.yaw - ref.yaw)
        x0 = np.array([e_y, e_yaw])

        # 3. MPC 模型
        v_curr = max(abs(ego.v), 0.5)
        A = np.array([[1.0, v_curr * self.dt], [0.0, 1.0]])
        B = np.array([[0.0], [v_curr / self.L * self.dt]])

        x = cp.Variable((2, self.N + 1))
        u = cp.Variable((1, self.N))
        cost, constr = 0, [x[:, 0] == x0]

        for k in range(self.N):
            # === 曲率前馈扰动（关键修复）===
            future_idx = min(idx + k, len(traj) - 1)
            if 0 < future_idx < len(traj) - 1:
                yaw_prev = traj[future_idx - 1].yaw
                yaw_next = traj[future_idx + 1].yaw
                ds = math.hypot(
                    traj[future_idx + 1].x - traj[future_idx - 1].x,
                    traj[future_idx + 1].y - traj[future_idx - 1].y,
                )
                kappa = normalize_angle(yaw_next - yaw_prev) / max(ds, 1e-3)
            else:
                kappa = 0.0

            d = np.array([0.0, -v_curr * kappa * self.dt])

            # 代价
            cost += cp.quad_form(x[:, k], self.Q)
            cost += self.R * cp.square(u[:, k])

            prev_u = self.last_delta if k == 0 else u[:, k - 1]
            cost += self.Rd * cp.square(u[:, k] - prev_u)

            constr += [
                x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + d,
                cp.abs(u[:, k]) <= self.max_steer,
            ]

        prob = cp.Problem(
            cp.Minimize(cost + cp.quad_form(x[:, self.N], self.Q)), constr
        )
        prob.solve(solver=cp.OSQP, warm_start=True)

        delta = float(u.value[0, 0]) if u.value is not None else self.last_delta

        self.last_delta = delta
        accel = 1.0 * (traj[idx].v - ego.v)
        print(
            f"[Control] 输出 a={accel:.2f},e_y: {e_y:.4f} | e_yaw: {math.degrees(e_yaw):.2f}° | Delta: {math.degrees(delta):.2f}°"
        )
        return accel, delta, e_y
