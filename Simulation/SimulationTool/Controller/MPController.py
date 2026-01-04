import math
import numpy as np
import cvxpy as cp
from SimulationTool.Common.common import normalize_angle


class MPCController:
    """
    MPC 横向控制器
    状态量: [e_y, e_yaw] (横向距离误差, 偏航角误差)
    控制量: delta (前轮转向角)
    """

    def __init__(
        self,
        wheelbase=2.7,
        v_ref=2.0,
        dt=0.05,
        horizon=15,
        Q=np.diag([10.0, 5.0]),  # 状态权重
        R=0.1,  # 控制量大小权重
        Rd=10.0,  # 控制量变化率权重 (防止转向抖动)
        max_steer=math.radians(35),
    ):
        self.L = wheelbase
        self.v = v_ref
        self.dt = dt
        self.N = horizon
        self.Q = Q
        self.R = R
        self.Rd = Rd
        self.max_steer = max_steer

        self.last_idx = 0
        self.last_delta = 0.0  # 用于计算变化率

    def find_nearest_forward_index(self, ego, traj):
        # 增加局部搜索窗口，防止轨迹重叠或环路时的搜索错误
        search_range = 50
        start = self.last_idx
        end = min(start + search_range, len(traj))

        min_dist = float("inf")
        best_idx = start

        for i in range(start, end):
            dx = traj[i].x - ego.x
            dy = traj[i].y - ego.y
            d = dx**2 + dy**2
            if d < min_dist:
                min_dist = d
                best_idx = i

        self.last_idx = best_idx
        return best_idx

    def compute_curvature(self, traj, idx):
        if idx <= 0 or idx >= len(traj) - 1:
            return 0.0
        # 使用中心差分法计算曲率 kappa = d_yaw / ds
        yaw_prev = traj[idx - 1].yaw
        yaw_next = traj[idx + 1].yaw

        # 处理角度突变
        dyaw = normalize_angle(yaw_next - yaw_prev)

        ds = math.hypot(
            traj[idx + 1].x - traj[idx - 1].x,
            traj[idx + 1].y - traj[idx - 1].y,
        )
        if ds < 1e-3:
            return 0.0
        return dyaw / ds

    def compute_control(self, ego, traj, dt):
        if traj is None or len(traj) == 0:
            return 0.0, 0.0, 0.0

        # 1. 寻找参考点并计算 Frenet 误差
        idx = self.find_nearest_forward_index(ego, traj)
        ref = traj[idx]

        # 计算横向误差 e_y: 向量 (ego-ref) 在 ref_yaw 法线方向上的投影
        dx = ego.x - ref.x
        dy = ego.y - ref.y
        # 这里的 e_y 符号需与运动学模型匹配：左正右负
        e_y = -math.sin(ref.yaw) * dx + math.cos(ref.yaw) * dy
        e_yaw = normalize_angle(ego.yaw - ref.yaw)

        x0 = np.array([e_y, e_yaw])

        # 2. 离散化系统矩阵 (线性化运动学模型)
        # e_y_dot = v * sin(e_yaw) ≈ v * e_yaw
        # e_yaw_dot = v/L * tan(delta) - v * kappa ≈ v/L * delta - v * kappa
        v = max(self.v, 0.5)  # 防止静止时矩阵奇异
        A = np.array([[1.0, v * self.dt], [0.0, 1.0]])
        B = np.array([[0.0], [v / self.L * self.dt]])

        # 3. 建立 CVXPY 优化问题
        x = cp.Variable((2, self.N + 1))
        u = cp.Variable((1, self.N))

        cost = 0
        constr = [x[:, 0] == x0]

        for k in range(self.N):
            # 获取预测时域内每个点的曲率（关键修正：前馈项随路径变化）
            future_idx = min(idx + k, len(traj) - 1)
            kappa_ref = self.compute_curvature(traj, future_idx)
            disturb = np.array([0.0, -v * kappa_ref * self.dt])

            # 状态代价
            cost += cp.quad_form(x[:, k], self.Q)
            # 控制代价
            cost += self.R * cp.square(u[:, k])

            # 变化率代价 (平滑转向)
            if k == 0:
                cost += self.Rd * cp.square(u[:, k] - self.last_delta)
            else:
                cost += self.Rd * cp.square(u[:, k] - u[:, k - 1])

            # 动力学约束与执行器限制
            constr += [
                x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + disturb,
                cp.abs(u[:, k]) <= self.max_steer,
            ]

        # 终端代价
        cost += cp.quad_form(x[:, self.N], self.Q)

        # 4. 求解
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if u.value is None:
            delta = self.last_delta
        else:
            delta = float(u.value[0, 0])

        self.last_delta = delta

        # 5. 简单的纵向加速度控制 (P控制)
        target_v = traj[idx].v if hasattr(traj[idx], "v") else self.v
        accel = 1.0 * (target_v - ego.v)

        return accel, delta, e_y
