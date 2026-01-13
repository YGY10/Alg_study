import math
import numpy as np
import csv
import os
from SimulationTool.Common.common import TrajPoint


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
# ST Frenet Planner
# ===============================
class STFrenetPlanner:
    def __init__(self):
        self.l_samples = [-3.5, 0.0, 3.5]
        self.T_samples = [10.0]
        self.dt = 0.2
        self.plan_dt = 0.1

        self.last_traj = None
        self.last_plan_time = None
        self.last_lon_coef = None
        self.last_lat_coef = None
        self.last_T_lat = None

        self.plan_id = 0
        self.log_path = "st_frenet_traj_log.csv"
        self._init_log_file()
        self.log_cropped_traj = True

    def _init_log_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                pass

    def _log_traj_block(self, traj, ego, sim_time, use_warm_start, t_now):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", f"{sim_time:.3f}"])
            writer.writerow(
                [
                    "plan_id",
                    "use_warm_start",
                    "t_now",
                    "point_idx",
                    "t",
                    "s",
                    "l",
                    "x",
                    "y",
                    "yaw",
                    "v",
                    "ego_x",
                    "ego_y",
                ]
            )
            for idx, p in enumerate(traj):
                writer.writerow(
                    [
                        self.plan_id,
                        int(use_warm_start),
                        t_now,
                        idx,
                        p.t,
                        p.s,
                        p.l,
                        p.x,
                        p.y,
                        p.yaw,
                        p.v,
                        ego.x,
                        ego.y,
                    ]
                )
            writer.writerow([])
        self.plan_id += 1

    # =================================================
    # 主入口
    # =================================================
    def plan(self, ego, ref_path, obstacles, sim_time):
        if not ref_path:
            self._reset_warm_start()
            return []

        t_now, planning_ego, use_warm_start = self._select_planning_start(ego, ref_path)
        print(f"  t_now={t_now:.1f}, start x={planning_ego.x}, y = {planning_ego.y}")

        best_traj_full, best_lon, best_lat, best_T_lat = self._search_best_trajectory(
            planning_ego, ref_path, obstacles, use_warm_start
        )

        if not best_traj_full:
            self._reset_warm_start()
            return []

        best_traj = self._post_process_and_log(
            best_traj_full, ego, sim_time, use_warm_start, t_now
        )

        if use_warm_start:
            # 如果执行落后太多，就冻结时间轴，避免越滚越超前
            dist_freeze = 1.0
            dist_now = math.hypot(ego.x - planning_ego.x, ego.y - planning_ego.y)
            if dist_now < dist_freeze:
                self.last_plan_time = t_now
            else:
                # 冻结：保持不变
                self.last_plan_time = self.last_plan_time
        else:
            self.last_plan_time = 0.0

        self.last_traj = best_traj_full
        self.last_lon_coef = best_lon
        self.last_lat_coef = best_lat
        self.last_T_lat = best_T_lat

        return best_traj

    # =================================================
    # ① 选择规划起点
    # =================================================
    def _select_planning_start(self, ego, ref_path):
        if self.last_plan_time is None:
            print("没上周期信息")
            return 0.0, ego, False

        # 1) 规划内部时间轴，按dt推进，因为规划每周期都是相对时间
        t_now = self.plan_dt

        if not self._can_use_warm_start(ego):
            print("上周期信息不可信")
            return 0.0, ego, False

        # 2) planner 预测的期望状态
        expected = self.eval_expected_state(t_now, ref_path)
        if expected is None:
            print("没算出来期望状态")
            return 0.0, ego, False

        # 3) 计算 ego 在参考线上的 Frenet（用于锚定到真实车）
        s_ego = self.find_nearest_s(ego, ref_path)
        l_ego = self.compute_lateral_error(ego, ref_path, s_ego)

        # 4) 用距离决定融合程度：dist 越大，越偏向 ego
        dist = math.hypot(ego.x - expected.x, ego.y - expected.y)
        dist_max = 1.0  # 你可以先用 1.0m；需要更“粘 ego”就调小
        alpha = 1.0 - dist / max(dist_max, 1e-6)
        alpha = max(0.0, min(1.0, alpha))

        # 5) 融合 Frenet 起点（关键：s/l/v 都要融合）
        s0 = alpha * expected.s + (1.0 - alpha) * s_ego
        l0 = alpha * expected.l + (1.0 - alpha) * l_ego
        v0 = ego.v  # v 用真实车更稳（也可融合，但先别复杂化）

        # 6) 把融合后的 (s0, l0) 映射回世界坐标，生成 planning_ego
        ref = self.find_ref_by_s(ref_path, s0)
        nx, ny = -math.sin(ref.yaw), math.cos(ref.yaw)

        planning_ego = TrajPoint(
            x=ref.x + l0 * nx,
            y=ref.y + l0 * ny,
            yaw=ref.yaw,
            v=v0,
            s=s0,
            l=l0,
            t=0.0,
        )

        # warm-start 仍然算 True（因为我们依然使用了上一周期的信息），
        # 但 planning_ego 已经被锚定到 ego，不会“飞”
        return t_now, planning_ego, True

    def _can_use_warm_start(self, ego):
        return (
            self.last_lon_coef is not None
            and self.last_lat_coef is not None
            and self.last_traj is not None
        )

    # =================================================
    # ② 轨迹搜索（★关键修复在这里）
    # =================================================
    def _search_best_trajectory(self, ego, ref_path, obstacles, use_warm_start):
        if use_warm_start:
            s0 = ego.s
            l0 = ego.l
            v0 = ego.v
        else:
            s0 = self.find_nearest_s(ego, ref_path)
            l0 = self.compute_lateral_error(ego, ref_path, s0)
            v0 = ego.v

        best_cost = float("inf")
        best = (None, None, None, None)

        for l_target in self.l_samples:
            for T in self.T_samples:
                ref_end = self.find_ref_by_s(ref_path, s0 + v0 * T)
                v_target = ref_end.v

                traj, lon, lat, T_lat = self.generate_traj(
                    s0, v0, l0, l_target, v_target, T, ref_path
                )

                if self.check_collision(traj, obstacles):
                    continue

                cost = self.compute_cost(traj, l_target)
                if cost < best_cost:
                    best_cost = cost
                    best = (traj, lon, lat, T_lat)

        return best

    # =================================================
    # ③ 输出、裁剪、日志（未改）
    # =================================================
    def _post_process_and_log(self, traj_full, ego, sim_time, use_warm_start, t_now):
        if traj_full and len(traj_full) >= 3:
            print("[ST-Frenet] First three points:")
            for i in range(3):
                p = traj_full[i]
                print(f"  P{i}: x={p.x:.3f}, y={p.y:.3f}, " f"v={p.v:.2f}, t={p.t:.2f}")
            print(f"  Ego: x={ego.x:.3f}, y={ego.y:.3f}")

        traj = self.crop_and_retime_traj(traj_full, self.plan_dt)
        if not traj:
            traj = traj_full

        if self.log_cropped_traj:
            self._log_traj_block(traj, ego, sim_time, use_warm_start, t_now)
        else:
            self._log_traj_block(traj_full, ego, sim_time, use_warm_start, t_now)

        return traj

    # =================================================
    # 工具函数（保持不变）
    # =================================================
    def eval_expected_state(self, t, ref_path):
        ret = self.eval_traj_by_poly(t)
        if ret is None:
            print("[WarmStart] Failed to evaluate polynomial state at t")
            return None

        s, l, v = ret
        print(f"[WarmStart][FrenetState] t={t:.2f}, s={s:.2f}, l={l:.2f}, v={v:.2f}")

        ref = self.find_ref_by_s(ref_path, s)
        print(
            f"[WarmStart][RefPoint] "
            f"s={s:.2f} -> x_ref={ref.x:.2f}, y_ref={ref.y:.2f}, yaw_ref={ref.yaw:.2f}"
        )

        nx, ny = -math.sin(ref.yaw), math.cos(ref.yaw)

        tmp = TrajPoint(
            x=ref.x + l * nx,
            y=ref.y + l * ny,
            yaw=ref.yaw,
            v=v,
            s=s,
            l=l,
            t=0.0,
        )

        print(
            f"[WarmStart][ExpectedEgo] "
            f"x={tmp.x:.2f}, y={tmp.y:.2f}, yaw={tmp.yaw:.2f}, v={tmp.v:.2f}"
        )

        return tmp

    def eval_traj_by_poly(self, t):
        t_lat = min(t, self.last_T_lat)
        s = np.polyval(self.last_lon_coef, t)
        l = np.polyval(self.last_lat_coef, t_lat)
        v = max(0.0, np.polyval(np.polyder(self.last_lon_coef), t))
        return s, l, v

    def generate_traj(self, s0, v0, l0, l_target, v_target, T, ref_path):
        T_lat = min(2.0, T)
        lat = quintic_poly(l0, 0.0, 0.0, l_target, 0.0, 0.0, T_lat)
        lon = quartic_poly(s0, v0, 0.0, v_target, 0.0, T)
        lon_d = np.polyder(lon)

        traj = []
        t = 0.0
        while t <= T + 1e-9:
            l = np.polyval(lat, t) if t <= T_lat else l_target
            s = np.polyval(lon, t)
            v = max(0.0, np.polyval(lon_d, t))
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
        return traj, lon, lat, T_lat

    def crop_and_retime_traj(self, traj, t_shift):
        out = []
        for p in traj:
            if p.t >= t_shift:
                out.append(
                    TrajPoint(
                        x=p.x,
                        y=p.y,
                        yaw=p.yaw,
                        v=p.v,
                        t=p.t - t_shift,
                        s=p.s,
                        l=p.l,
                    )
                )
        return out

    def is_traj_consistent(self, ego, traj):
        p = min(traj, key=lambda q: (q.x - ego.x) ** 2 + (q.y - ego.y) ** 2)
        return math.hypot(ego.x - p.x, ego.y - p.y) < 100

    def _reset_warm_start(self):
        self.last_traj = None
        self.last_plan_time = None
        self.last_lon_coef = None
        self.last_lat_coef = None
        self.last_T_lat = None

    def check_collision(self, traj, obstacles):
        for p in traj:
            for o in obstacles:
                if abs(p.x - o.x) < 1.5 and abs(p.y - o.y) < 1.0:
                    return True
        return False

    def compute_cost(self, traj, l_target):
        return 10.0 * abs(l_target) + 0.1 * len(traj)

    def find_nearest_s(self, ego, ref_path):
        return min(ref_path, key=lambda p: (p.x - ego.x) ** 2 + (p.y - ego.y) ** 2).s

    def compute_lateral_error(self, ego, ref_path, s):
        ref = self.find_ref_by_s(ref_path, s)
        dx, dy = ego.x - ref.x, ego.y - ref.y
        return -math.sin(ref.yaw) * dx + math.cos(ref.yaw) * dy

    def find_ref_by_s(self, ref_path, s):
        if s <= ref_path[0].s:
            return ref_path[0]
        if s >= ref_path[-1].s:
            return ref_path[-1]
        for i in range(len(ref_path) - 1):
            p0, p1 = ref_path[i], ref_path[i + 1]
            if p0.s <= s <= p1.s:
                r = (s - p0.s) / max(p1.s - p0.s, 1e-6)
                return TrajPoint(
                    x=p0.x + r * (p1.x - p0.x),
                    y=p0.y + r * (p1.y - p0.y),
                    yaw=p0.yaw + r * (p1.yaw - p0.yaw),
                    v=p0.v + r * (p1.v - p0.v),
                    s=s,
                    t=0.0,
                )
        return ref_path[-1]
