import math
import numpy as np
import csv
import os
from SimulationTool.Common.common import TrajPoint
from SimulationTool.SimModel.vehicle_model import VehicleKModel
from typing import List
from SimulationTool.Planner.NNPlanner import NNPlanner
from SimulationTool.Planner.trajectory_features import extract_nn_features


# ===============================
# ST Frenet Planner
# ===============================
class STFrenetPlanner:
    def __init__(self, ego_length, ego_width, nn_planner: NNPlanner):
        self.l_samples = [-3.5, 0.0, 3.5]
        self.T_samples = [10.0]
        self.dt = 0.2
        self.plan_dt = 0.1
        self.max_stitch_time = 1
        self.last_traj = None
        self.last_plan_time = None
        self.last_lon_coef = None
        self.last_lat_coef = None
        self.last_T_lat = None
        self.warm_start_mode = "COLD"  # COLD / POLY / STITCH
        self.plan_id = 0
        self.log_path = "st_frenet_traj_log.csv"
        self._init_log_file()
        self.log_cropped_traj = True
        self.ego_length = ego_length
        self.ego_width = ego_width
        self.nn_planner = nn_planner
        self.last_choice = None
        self.last_l_target = None
        self.nn_dataset_path = "nn_dataset.csv"
        self._init_nn_dataset()

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
    def plan(self, ego: VehicleKModel, ref_path, obstacles, sim_time):
        if not ref_path:
            self._reset_warm_start()
            return []
        if self.last_lon_coef is not None and self.warm_start_mode != "STITCH":
            self.warm_start_mode = "POLY"

        # ===== ① 选择规划起点 =====
        t_now, planning_ego, use_warm_start = self._select_planning_start(ego, ref_path)
        print(
            f"[Plan] 规划起点 x={planning_ego.x:.2f}, y = {planning_ego.y:.2f}, WarmStart={int(use_warm_start)}"
        )
        print(
            f"[Plan] 自车实际状态 x={ego.x}, y={ego.y}, v={ego.v:.2f}, yaw={math.degrees(ego.yaw):.2f}"
        )

        # ===== ② 轨迹拼接 =====
        stitched_traj = None
        if use_warm_start:
            stitched_traj = self._try_stitch_trajectory(planning_ego, obstacles)
        if stitched_traj:
            print("[Stitch] Use previous trajectory segment")
            print(
                f"[Stitch] points={len(stitched_traj)}, "
                f"t_end={stitched_traj[-1].t:.2f}"
            )

            print("[Stitch] 拼接的点:")
            for i in range(len(stitched_traj)):
                p = stitched_traj[i]
                print(
                    f"  S{i}: x={p.x:.3f}, y={p.y:.3f}, "
                    f"v={p.v:.2f}, t={p.t:.2f}, s={p.s:.2f}, l={p.l:.2f}, dl={p.dl:.2f}, ddl={p.ddl:.2f}"
                )
        else:
            print("[Stitch] Disabled (collision or deviation)")

        # ===== ③ 需要补规划的起点 =====
        if stitched_traj:
            self.warm_start_mode = "STITCH"
            self.last_lon_coef = None
            self.last_lat_coef = None
            self.last_T_lat = None
            replan_start = stitched_traj[-1]
        else:
            replan_start = planning_ego
        print(f"[Plan] 补齐位置 x={replan_start.x}, y={replan_start.y}")
        # ===== ④ 规划补齐段 =====
        best, frenet_ego = self._search_best_trajectory(
            replan_start, ref_path, obstacles, use_warm_start
        )
        if self.last_choice is not None:
            self.last_l_target = self.last_choice[0]
        else:
            self.last_l_target = None
        # 记录样本
        # ======================================================
        if self.last_choice is not None:
            feats = extract_nn_features(
                frenet_ego,
                ref_path,
                obstacles,
            )
            label_l, label_T = self.last_choice
            self._log_nn_sample(feats, label_l, label_T)
        # ======================================================

        best_traj_full, best_lon, best_lat, best_T_lat = best
        if not best_traj_full:
            self._reset_warm_start()
            return []

        if stitched_traj:
            # 重定时补规划段
            dt0 = stitched_traj[-1].t
            best_traj_full = best_traj_full[1:]
            for p in best_traj_full:
                p.t += dt0
            best_traj_full = stitched_traj + best_traj_full

        best_traj = self._post_process_and_log(
            best_traj_full, ego, sim_time, use_warm_start, t_now
        )

        self.last_traj = best_traj_full
        self.last_plan_time = t_now

        if self.warm_start_mode != "STITCH":
            self.last_lon_coef = best_lon
            self.last_lat_coef = best_lat
            self.last_T_lat = best_T_lat

        return best_traj

    # 轨迹拼接
    def _try_stitch_trajectory(self, planning_ego, obstacles):
        if self.last_traj is None or len(self.last_traj) < 2:
            print("[Stitch] 上周期规划轨迹不合理")
            return None

        dist = self.distance_to_last_traj_curve(planning_ego)
        print(f"[Stitch] 本周期规划起点到上周期轨迹的距离 d={dist:.2f}")
        if dist > 2:
            print("[Stitch] 本周期规划起点离上周期规划轨迹太远")
            return None

        t0 = self._find_time_on_last_traj(planning_ego)
        if t0 is None:
            return None

        stitched = self._resample_and_retime_from_last_traj(t0)
        if not stitched:
            return None

        if self.check_collision(stitched, obstacles):
            return None

        return stitched

    def distance_to_last_traj_curve(self, planning_ego):
        min_dist = float("inf")
        for i in range(len(self.last_traj) - 1):
            p0 = self.last_traj[i]
            p1 = self.last_traj[i + 1]
            d = self.point_to_segment_distance(
                planning_ego.x, planning_ego.y, p0.x, p0.y, p1.x, p1.y
            )
            min_dist = min(min_dist, d)
        return min_dist

    @staticmethod
    def point_to_segment_distance(px, py, x0, y0, x1, y1):
        vx, vy = x1 - x0, y1 - y0
        wx, wy = px - x0, py - y0

        vv = vx * vx + vy * vy
        if vv < 1e-9:
            return math.hypot(px - x0, py - y0)

        t = (wx * vx + wy * vy) / vv
        t = max(0.0, min(1.0, t))

        proj_x = x0 + t * vx
        proj_y = y0 + t * vy

        return math.hypot(px - proj_x, py - proj_y)

    # =================================================
    # 工具函数（保持不变）
    # =================================================
    def _log_nn_sample(self, features, label_l, label_T):
        assert len(features) == self.nn_planner.net[0].in_features
        row = list(features.astype(float)) + [float(label_l), float(label_T)]
        with open(self.nn_dataset_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _init_nn_dataset(self):
        if not os.path.exists(self.nn_dataset_path):
            with open(self.nn_dataset_path, "w", newline="") as f:
                writer = csv.writer(f)
                # 不写 header 也可以，训练时自己约定顺序
                # writer.writerow(["f0", "f1", ..., "label_l", "label_T"])
                pass

    def _interp_trajpoint_by_time(self, traj, t_query):
        # traj: list[TrajPoint] with increasing p.t
        if t_query <= traj[0].t:
            p = traj[0]
            return TrajPoint(
                x=p.x,
                y=p.y,
                yaw=p.yaw,
                v=p.v,
                t=t_query,
                s=p.s,
                l=p.l,
                dl=p.dl,
                ddl=p.ddl,
            )
        if t_query >= traj[-1].t:
            p = traj[-1]
            return TrajPoint(
                x=p.x,
                y=p.y,
                yaw=p.yaw,
                v=p.v,
                t=t_query,
                s=p.s,
                l=p.l,
                dl=p.dl,
                ddl=p.ddl,
            )

        # find segment
        for i in range(len(traj) - 1):
            p0, p1 = traj[i], traj[i + 1]
            if p0.t <= t_query <= p1.t:
                den = max(p1.t - p0.t, 1e-9)
                r = (t_query - p0.t) / den

                yaw = p0.yaw + r * (p1.yaw - p0.yaw)

                def lerp(a, b):
                    return a + r * (b - a)

                return TrajPoint(
                    x=lerp(p0.x, p1.x),
                    y=lerp(p0.y, p1.y),
                    yaw=yaw,
                    v=lerp(p0.v, p1.v),
                    t=t_query,
                    s=lerp(p0.s, p1.s),
                    ds=lerp(getattr(p0, "ds", p0.v), getattr(p1, "ds", p1.v)),
                    l=lerp(p0.l, p1.l),
                    dl=lerp(getattr(p0, "dl", 0.0), getattr(p1, "dl", 0.0)),
                    ddl=lerp(getattr(p0, "ddl", 0.0), getattr(p1, "ddl", 0.0)),
                )

        # fallback
        p = traj[-1]
        return TrajPoint(x=p.x, y=p.y, yaw=p.yaw, v=p.v, t=t_query, s=p.s, l=p.l)

    def _resample_and_retime_from_last_traj(self, t0):
        traj = self.last_traj
        if not traj:
            return None

        # we want stitched traj points at: t0 + k*dt, but output t = k*dt
        out = []
        k = 0
        while True:
            tq = t0 + k * self.dt
            if tq > traj[-1].t + 1e-9:
                break
            if tq - t0 > self.max_stitch_time:
                break
            p = self._interp_trajpoint_by_time(traj, tq)
            # retime to start at 0
            out.append(
                TrajPoint(
                    x=p.x,
                    y=p.y,
                    yaw=p.yaw,
                    v=p.v,
                    t=k * self.dt,
                    s=p.s,
                    ds=p.ds,
                    l=p.l,
                    dl=p.dl,
                    ddl=p.ddl,
                )
            )
            k += 1

        return out

    def _find_time_on_last_traj(self, planning_ego):
        traj = self.last_traj
        if not traj or len(traj) < 2:
            return None

        # find nearest point index by XY
        idx = min(
            range(len(traj)),
            key=lambda i: (traj[i].x - planning_ego.x) ** 2
            + (traj[i].y - planning_ego.y) ** 2,
        )

        # try project onto segment [idx, idx+1] or [idx-1, idx]
        cand = []

        def project(i0, i1):
            p0, p1 = traj[i0], traj[i1]
            vx, vy = p1.x - p0.x, p1.y - p0.y
            wx, wy = planning_ego.x - p0.x, planning_ego.y - p0.y
            vv = vx * vx + vy * vy
            if vv < 1e-9:
                return None
            r = (wx * vx + wy * vy) / vv
            r = max(0.0, min(1.0, r))
            t = p0.t + r * (p1.t - p0.t)
            x = p0.x + r * vx
            y = p0.y + r * vy
            d2 = (x - planning_ego.x) ** 2 + (y - planning_ego.y) ** 2
            return (d2, t)

        if idx < len(traj) - 1:
            ret = project(idx, idx + 1)
            if ret:
                cand.append(ret)
        if idx > 0:
            ret = project(idx - 1, idx)
            if ret:
                cand.append(ret)

        if not cand:
            return traj[idx].t

        cand.sort(key=lambda x: x[0])
        return cand[0][1]

    def _can_use_warm_start(self, ego):
        return self.last_traj is not None

    def _post_process_and_log(self, traj_full, ego, sim_time, use_warm_start, t_now):
        if traj_full and len(traj_full) >= 5:
            print("[ST-Frenet] 规划结果:")
            for i in range(min(15, len(traj_full))):
                p = traj_full[i]
                print(
                    f"  P{i}: x={p.x:.3f}, y={p.y:.3f}, "
                    f"v={p.v:.2f}, t={p.t:.2f}, s={p.s:.2f}, l={p.l:.2f}, dl={p.dl:.2f}, ddl={p.ddl:.2f}"
                )
            print(f"  Ego: x={ego.x:.3f}, y={ego.y:.3f}")

        traj = self.crop_and_retime_traj(traj_full, self.plan_dt)
        if not traj:
            traj = traj_full

        if self.log_cropped_traj:
            self._log_traj_block(traj, ego, sim_time, use_warm_start, t_now)
        else:
            self._log_traj_block(traj_full, ego, sim_time, use_warm_start, t_now)

        return traj

    def _select_planning_start(self, ego, ref_path):
        if self.warm_start_mode == "COLD":
            print("[WarmStart] Cold start")
            return 0.0, ego, False

        t_now = self.plan_dt

        if self.warm_start_mode == "STITCH":
            t0 = t_now
            planning_ego = self._interp_trajpoint_by_time(self.last_traj, t0)
            print("[WarmStart] 拼接")
            return t_now, planning_ego, True

        if self.warm_start_mode == "POLY":
            expected = self.eval_expected_state(t_now, ref_path)
            if expected is None:
                self.warm_start_mode = "COLD"
                return 0.0, ego, False
            return t_now, expected, True

        # 兜底（理论上不会走到）
        return 0.0, ego, False

    def _search_best_trajectory(
        self, ego: TrajPoint, ref_path, obstacles, use_warm_start
    ):
        self.last_choice = None
        # 搜索起点
        if use_warm_start:
            s0 = ego.s
            l0 = ego.l
            dl0 = ego.dl
            ddl0 = ego.ddl
            v0 = ego.v
        else:
            s0 = self.find_nearest_s(ego, ref_path)
            l0 = self.compute_lateral_error(ego, ref_path, s0)
            v0 = ego.v
            dl0 = 0
            ddl0 = 0

        frenet_ego = TrajPoint(
            s=s0,
            l=l0,
            dl=dl0,
            ddl=ddl0,
            v=v0,
            x=ego.x,  # 可选，仅用于 debug
            y=ego.y,
            yaw=ego.yaw,
            t=0.0,
        )

        best_cost = float("inf")
        best = (None, None, None, None)

        # 候选轨迹，NNPlanner优先
        candidates = []
        nn_candidates = self._nn_proposal(frenet_ego, ref_path, obstacles)
        candidates += nn_candidates
        # 原本规则采样（兜底）
        for l_target in self.l_samples:
            for T in self.T_samples:
                candidates.append((l_target, T))
        uniq = []
        seen = set()
        for l_target, T in candidates:
            key = (round(float(l_target), 3), round(float(T), 3))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((float(l_target), float(T)))
        candidates = uniq
        print("[Planner] candidates:", candidates)
        picked_from_nn = False
        for l_target, T in candidates:
            ref_end = self.find_ref_by_s(ref_path, s0 + v0 * T)
            v_target = ref_end.v

            traj, lon, lat, T_lat = self.generate_traj(
                s0, v0, l0, dl0, ddl0, l_target, v_target, T, ref_path
            )
            print(f"[SearchBest] l_target={l_target:.1f}, T={T:.2f}")
            if self.check_collision(traj, obstacles):
                print("[SearchBest] Danger continue")
                continue

            cost = self.compute_cost(traj, l_target)
            if cost < best_cost:
                best_cost = cost
                best = (traj, lon, lat, T_lat)
                self.last_choice = (l_target, T)
                picked_from_nn = (l_target, T) in nn_candidates

        print("[Planner] pick from NN:", picked_from_nn)
        return best, frenet_ego

    def _nn_proposal(
        self, ego: TrajPoint, ref_path: List[TrajPoint], obstacles: List[VehicleKModel]
    ):
        if self.nn_planner is None:
            return []

        feats = extract_nn_features(ego, ref_path, obstacles)
        l_nn, T_nn = self.nn_planner.propose(feats)
        # 基本裁剪，防止网络输出爆炸
        l_nn = float(np.clip(l_nn, -4.0, 4.0))
        T_nn = float(np.clip(T_nn, 2.0, 10.0))

        # 让 NN 提议排在前面 + 给一点扰动增强鲁棒性
        print(f"[NNPlanner] proposal l={l_nn:.2f}, T={T_nn:.2f}")
        return [
            (l_nn, T_nn),
            (l_nn + 0.5, T_nn),
            (l_nn - 0.5, T_nn),
        ]

    def _interp_traj_at_time(self, traj, t):
        for i in range(len(traj) - 1):
            p0, p1 = traj[i], traj[i + 1]
            if p0.t <= t <= p1.t:
                r = (t - p0.t) / max(p1.t - p0.t, 1e-6)
                return TrajPoint(
                    x=p0.x + r * (p1.x - p0.x),
                    y=p0.y + r * (p1.y - p0.y),
                    yaw=p0.yaw + r * (p1.yaw - p0.yaw),
                    v=p0.v + r * (p1.v - p0.v),
                    t=t,
                    s=p0.s + r * (p1.s - p0.s),
                    l=p0.l + r * (p1.l - p0.l),
                )
        return None

    def eval_expected_state(self, t, ref_path):
        ret = self.eval_traj_by_poly(t)
        if ret is None:
            print("[WarmStart] Failed to evaluate polynomial state at t")
            return None

        s, l, dl, ddl, v = ret
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
            dl=dl,
            ddl=ddl,
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
        lat_d = np.polyder(self.last_lat_coef, 1)
        lat_dd = np.polyder(self.last_lat_coef, 2)
        dl = np.polyval(lat_d, t_lat)
        ddl = np.polyval(lat_dd, t_lat)
        return s, l, dl, ddl, v

    def generate_traj(self, s0, v0, l0, dl0, ddl0, l_target, v_target, T, ref_path):
        T_lat = min(2.0, T)
        lat = quintic_poly(l0, dl0, ddl0, l_target, 0.0, 0.0, T_lat)
        lon = quartic_poly(s0, v0, 0.0, v_target, 0.0, T)
        lon_d = np.polyder(lon)

        traj = []
        t = 0.0
        while t <= T + 1e-9:
            l = np.polyval(lat, min(t, T_lat))
            lat_d = np.polyder(lat, 1)
            lat_dd = np.polyder(lat, 2)
            dl = np.polyval(lat_d, min(t, T_lat))
            ddl = np.polyval(lat_dd, min(t, T_lat))
            ddl = np.clip(ddl, -3.0, 3.0)  # m/s^2，先保守
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
                    ds=v,
                    l=l,
                    dl=dl,
                    ddl=ddl,
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
                        ds=p.ds,
                        l=p.l,
                        dl=p.dl,
                        ddl=p.ddl,
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
        self.last_l_target = None

    def check_collision(self, traj, obstacles):
        ego_half_l = self.ego_length / 2.0
        ego_half_w = self.ego_width / 2.0

        for p in traj:
            for o in obstacles:
                # 只检查前方障碍（非常重要）
                if o.x < p.x - self.ego_length:
                    continue

                dx = abs(p.x - o.x)
                dy = abs(p.y - o.y)

                obs_half_l = o.length / 2.0
                obs_half_w = o.width / 2.0

                if dx < obs_half_l + ego_half_l and dy < obs_half_w + ego_half_w:
                    print(
                        f"[Collision] 冲突轨迹点x={p.x:.2f} y={p.y:.2f}  障碍物x={o.x:.2f} y={o.y:.2f}"
                    )
                    return True
        return False

    def compute_cost(self, traj, l_target):
        l_end = traj[-1].l
        l_mean = np.mean([abs(p.l) for p in traj])

        # 偏离参考线惩罚
        cost_l = 10.0 * abs(l_target)
        cost_end = 20 * abs(l_end)
        cost_mean = 5 * l_mean
        # 决策一致性惩罚
        cost_consis = 0
        if self.last_l_target is not None:
            if abs(self.last_l_target) > 0.3 and abs(l_target) > 0.3:
                # 防止在0附近
                if self.last_l_target * l_target < 0:
                    cost_consis = 15
        print(
            f"[ComputeCost] cost_l={cost_l:.2f}, cost_end={cost_end:.2f}, cost_mean={cost_mean:.2f}, cost_consit={cost_consis:.2f}"
        )
        return cost_l + cost_end + cost_mean + cost_consis

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
