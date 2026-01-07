import math
import numpy as np
from SimulationTool.Common.common import TrajPoint
from typing import List


# ===============================
# 多项式工具
# ===============================
def quintic_poly(p0, v0, a0, p1, v1, a1, T):
    A = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [T**5, T**4, T**3, T**2, T, 1],
            [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
            [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
        ]
    )
    b = np.array([p0, v0, a0, p1, v1, a1])
    return np.linalg.solve(A, b)


def quartic_poly(p0, v0, a0, v1, a1, T):
    A = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 2, 0, 0],
            [4 * T**3, 3 * T**2, 2 * T, 1, 0],
            [12 * T**2, 6 * T, 2, 0, 0],
        ]
    )
    b = np.array([p0, v0, a0, v1, a1])
    return np.linalg.solve(A, b)


# ===============================
# ST Frenet Planner（固定10秒规划）
# ===============================
class STFrenetPlanner:
    def __init__(self):
        self.l_samples = [-3.5, 0.0, 3.5]
        self.T_samples = [10.0]
        self.dt = 0.2  # 轨迹采样 dt
        self.plan_dt = 0.05  # ★ 实际规划周期

        self.last_traj = List[TrajPoint]
        self.last_plan_time = None

        # ★ 保存上周期的多项式模型
        self.last_lon_coef = None
        self.last_lat_coef = None
        self.last_T_lat = None

    # -------------------------------------------------
    def plan(self, ego, ref_path, obstacles):
        if not ref_path:
            self._reset_warm_start()
            return []

        # ===== 1. 当前规划时刻 =====
        if self.last_plan_time is None:
            t_now = 0.0
        else:
            t_now = self.last_plan_time + self.plan_dt

        # ===== 2. 规划起点（多项式 warm-start）=====
        use_last = (
            self.last_lon_coef is not None
            and self.last_lat_coef is not None
            and self.last_traj
            and self.is_traj_consistent(ego, self.last_traj)
        )

        if use_last:
            ret = self.eval_traj_by_poly(t_now)
            if ret is None:
                use_last = False

        if use_last:
            s0, l0, v0 = ret
        else:
            s0 = self.find_nearest_s(ego, ref_path)
            l0 = self.compute_lateral_error(ego, ref_path, s0)
            v0 = ego.v
            t_now = 0.0

        best_traj = None
        best_cost = float("inf")

        best_lon_coef = None
        best_lat_coef = None
        best_T_lat = None

        # ===== 3. ST 采样 =====
        for l_target in self.l_samples:
            for T in self.T_samples:
                ref_end = self.find_ref_by_s(ref_path, s0 + v0 * T)
                v_target = ref_end.v

                traj, lon_coef, lat_coef, T_lat = self.generate_traj(
                    s0, v0, l0, l_target, v_target, T, ref_path
                )

                if self.check_collision(traj, obstacles):
                    continue

                cost = self.compute_cost(traj, l_target)
                if cost < best_cost:
                    best_cost = cost
                    best_traj = traj
                    best_lon_coef = lon_coef
                    best_lat_coef = lat_coef
                    best_T_lat = T_lat

        # ===== 4. 打印（不动）=====
        if best_traj and len(best_traj) >= 3:
            print("[ST-Frenet] First three points:")
            print(
                f"  P0: x={best_traj[0].x:.3f}, y={best_traj[0].y:.3f}, "
                f"v={best_traj[0].v:.2f}, t={best_traj[0].t:.2f}"
            )
            print(
                f"  P1: x={best_traj[1].x:.3f}, y={best_traj[1].y:.3f}, "
                f"v={best_traj[1].v:.2f}, t={best_traj[1].t:.2f}"
            )
            print(
                f"  P1: x={best_traj[2].x:.3f}, y={best_traj[2].y:.3f}, "
                f"v={best_traj[2].v:.2f}, t={best_traj[2].t:.2f}"
            )
            print(f"  Ego: x={ego.x:.3f}, y={ego.y:.3f}")

        # ===== 5. 更新 warm-start 状态 =====
        if best_traj:
            self.last_plan_time = t_now
            self.last_traj = best_traj
            self.last_lon_coef = best_lon_coef
            self.last_lat_coef = best_lat_coef
            self.last_T_lat = best_T_lat
        else:
            self._reset_warm_start()

        return best_traj if best_traj else []

    # -------------------------------------------------
    def eval_traj_by_poly(self, t_now):
        """
        用上周期多项式模型，在时间 t_now 直接求 Frenet 状态
        """
        if self.last_lon_coef is None or self.last_lat_coef is None:
            return None

        t_lat = min(t_now, self.last_T_lat)

        s = np.polyval(self.last_lon_coef, t_now)
        l = np.polyval(self.last_lat_coef, t_lat)

        lon_coef_d = np.polyder(self.last_lon_coef)
        v = max(0.0, np.polyval(lon_coef_d, t_now))

        return s, l, v

    # -------------------------------------------------
    def generate_traj(self, s0, v0, l0, l_target, v_target, T, ref_path):
        T_lat = min(2.0, T)

        lat_coef = quintic_poly(l0, 0.0, 0.0, l_target, 0.0, 0.0, T_lat)
        lon_coef = quartic_poly(s0, v0, 0.0, v_target, 0.0, T)
        lon_coef_d = np.polyder(lon_coef)

        traj = []
        t = 0.0

        while t <= T:
            l = np.polyval(lat_coef, t) if t <= T_lat else l_target
            s = np.polyval(lon_coef, t)
            v = max(0.0, np.polyval(lon_coef_d, t))

            ref = self.find_ref_by_s(ref_path, s)
            nx, ny = -math.sin(ref.yaw), math.cos(ref.yaw)

            traj.append(
                TrajPoint(
                    x=ref.x + l * nx,
                    y=ref.y + l * ny,
                    yaw=ref.yaw,
                    v=v,
                    t=t,
                    s=s,
                    l=l,
                )
            )
            t += self.dt

        return traj, lon_coef, lat_coef, T_lat

    # -------------------------------------------------
    def is_traj_consistent(self, ego, traj):
        p = min(traj, key=lambda q: (q.x - ego.x) ** 2 + (q.y - ego.y) ** 2)
        dist = math.hypot(ego.x - p.x, ego.y - p.y)
        return dist < 1.5

    # -------------------------------------------------
    def _reset_warm_start(self):
        self.last_traj = None
        self.last_plan_time = None
        self.last_lon_coef = None
        self.last_lat_coef = None
        self.last_T_lat = None

    # -------------------------------------------------
    def check_collision(self, traj, obstacles):
        for p in traj:
            for obs in obstacles:
                if abs(p.x - obs.x) < 1.5 and abs(p.y - obs.y) < 1.0:
                    return True
        return False

    # -------------------------------------------------
    def compute_cost(self, traj, l_target):
        return 10.0 * abs(l_target) + 0.1 * len(traj)

    # -------------------------------------------------
    def find_nearest_s(self, ego, ref_path):
        return min(
            ref_path,
            key=lambda p: (p.x - ego.x) ** 2 + (p.y - ego.y) ** 2,
        ).s

    # -------------------------------------------------
    def compute_lateral_error(self, ego, ref_path, s):
        ref = self.find_ref_by_s(ref_path, s)
        dx, dy = ego.x - ref.x, ego.y - ref.y
        return -math.sin(ref.yaw) * dx + math.cos(ref.yaw) * dy

    # -------------------------------------------------
    def find_ref_by_s(self, ref_path, s):
        if s <= ref_path[0].s:
            return ref_path[0]
        if s >= ref_path[-1].s:
            return ref_path[-1]

        for i in range(len(ref_path) - 1):
            p0 = ref_path[i]
            p1 = ref_path[i + 1]
            if p0.s <= s <= p1.s:
                ratio = (s - p0.s) / max(p1.s - p0.s, 1e-6)
                return TrajPoint(
                    x=p0.x + ratio * (p1.x - p0.x),
                    y=p0.y + ratio * (p1.y - p0.y),
                    yaw=p0.yaw + ratio * (p1.yaw - p0.yaw),
                    v=p0.v + ratio * (p1.v - p0.v),
                    s=s,
                    t=0.0,
                )

        return ref_path[-1]
