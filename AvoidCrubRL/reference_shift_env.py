import numpy as np
import matplotlib.pyplot as plt

from geometry_utils import (
    compute_s,
    compute_yaw_from_xy,
    compute_normals_from_yaw,
    apply_local_offset,
    resample_polyline_by_s,
    point_to_segment_projection,
    point_to_polyline_projection,
    compute_min_distance_to_boundary,
)

from scene_utils import (
    build_end_decay_window,
    build_turn_path,
    build_boundary_from_points,
)

from control_point_utils import build_l_from_basis_coeffs


class ReferenceShiftEnv:
    def __init__(
        self,
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=12,
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
        seed=0,
    ):
        self.num_ref_points = num_ref_points
        self.num_boundary_points = num_boundary_points
        self.num_ctrl = num_ctrl
        self.safe_dist = safe_dist
        self.hard_collision_dist = hard_collision_dist
        self.alpha_limit = alpha_limit
        self.rng = np.random.default_rng(seed)

        self.fig1 = plt.figure(1, figsize=(8, 8))
        self.fig2 = plt.figure(2, figsize=(10, 4))
        plt.ion()  # 开启交互模式

        self.ref_x = None
        self.ref_y = None
        self.s_ref = None
        self.yaw_ref = None
        self.boundary_x = None
        self.boundary_y = None
        self.ref_to_boundary_dist = None
        self.direction_sign = 0.0

    def _sample_boundary_points(self):
        """
        第一版先只做“参考线右侧边界”
        """
        base_x = self.rng.uniform(1.0, 2.2)

        p0 = (base_x + self.rng.uniform(-0.2, 0.2), 6.0)
        p1 = (base_x + self.rng.uniform(-0.2, 0.2), 24.0)
        p2 = (base_x + self.rng.uniform(-0.1, 0.6), 25.5)
        p3 = (base_x + self.rng.uniform(-0.3, 0.3), 26.3)
        p4 = (base_x + self.rng.uniform(-0.8, 0.2), 27.0)
        p5 = (base_x + self.rng.uniform(-0.6, 0.4), 27.2)

        return [p0, p1, p2, p3, p4, p5]

    def _build_state(self):
        ref_feat = np.stack([self.ref_x, self.ref_y], axis=1)
        bd_feat = np.stack([self.boundary_x, self.boundary_y], axis=1)
        dist_feat = self.ref_to_boundary_dist[:, None]

        state = np.concatenate(
            [ref_feat.flatten(), bd_feat.flatten(), dist_feat.flatten()], axis=0
        )
        return state.astype(np.float32)

    def reset(self):
        ref_x, ref_y = build_turn_path(r=10.0, straight_len=20.0, n_line=120, n_arc=120)
        self.ref_x, self.ref_y, self.s_ref = resample_polyline_by_s(
            ref_x, ref_y, num=self.num_ref_points
        )
        self.yaw_ref = compute_yaw_from_xy(self.ref_x, self.ref_y)

        boundary_points = self._sample_boundary_points()
        boundary_x, boundary_y = build_boundary_from_points(
            boundary_points, num_per_seg=40
        )
        self.boundary_x, self.boundary_y, _ = resample_polyline_by_s(
            boundary_x, boundary_y, num=self.num_boundary_points
        )

        self.ref_to_boundary_dist = compute_min_distance_to_boundary(
            self.ref_x, self.ref_y, self.boundary_x, self.boundary_y
        )

        mean_bd_x = float(np.mean(self.boundary_x))
        mean_ref_x = float(np.mean(self.ref_x))
        self.direction_sign = 1.0 if mean_bd_x > mean_ref_x else -1.0

        return self._build_state()

    def _compute_reward(self, l_offset, safe_x, safe_y, ctrl_values):
        d_after = compute_min_distance_to_boundary(
            safe_x, safe_y, self.boundary_x, self.boundary_y
        )
        d_min = float(np.min(d_after))

        # 1) 硬约束风格
        hard_violation = max(self.hard_collision_dist - d_min, 0.0)
        if hard_violation > 0.0:
            r_hard = -5000.0 - 2000.0 * (hard_violation**2)
        else:
            r_hard = 0.0

        # 2) 软约束：整段危险区
        danger_mask = self.ref_to_boundary_dist < 2.0
        if np.any(danger_mask):
            violation = np.maximum(self.safe_dist - d_after, 0.0)
            r_safe_area = -800.0 * float(np.mean((violation[danger_mask]) ** 2))

            clearance = np.maximum(d_after - self.safe_dist, 0.0)
            r_safe_bonus = 8.0 * float(np.mean(np.minimum(clearance[danger_mask], 0.5)))
        else:
            r_safe_area = 0.0
            r_safe_bonus = 0.0

        # 3) 非危险区尽量别乱偏
        # 3) 非危险区域尽量别乱偏
        non_danger_mask = self.ref_to_boundary_dist > 1.8
        transition_mask = (self.ref_to_boundary_dist > 1.2) & (
            self.ref_to_boundary_dist <= 1.8
        )

        if np.any(non_danger_mask):
            # 明显安全的地方，不该有太大偏移
            r_offset_far = -6.0 * float(np.mean(l_offset[non_danger_mask] ** 2))
        else:
            r_offset_far = 0.0

        if np.any(transition_mask):
            # 过渡区轻一点惩罚，避免策略突然开始乱偏
            r_offset_mid = -2.0 * float(np.mean(l_offset[transition_mask] ** 2))
        else:
            r_offset_mid = 0.0

        r_offset = r_offset_far + r_offset_mid

        # 4) 平滑性
        dl = np.diff(l_offset)
        ddl = np.diff(l_offset, n=2)
        dddl = np.diff(l_offset, n=3)

        r_smooth = -1.5 * float(np.mean(dl**2)) if len(dl) > 0 else 0.0
        r_curv = -0.6 * float(np.mean(ddl**2)) if len(ddl) > 0 else 0.0
        r_jerk = -0.8 * float(np.mean(dddl**2)) if len(dddl) > 0 else 0.0

        # 4.5) 控制点层面的平滑约束（非常关键）
        ctrl_dl = np.diff(ctrl_values)
        ctrl_ddl = np.diff(ctrl_values, n=2)

        r_ctrl_smooth = -4.0 * float(np.mean(ctrl_dl**2)) if len(ctrl_dl) > 0 else 0.0
        r_ctrl_curv = -8.0 * float(np.mean(ctrl_ddl**2)) if len(ctrl_ddl) > 0 else 0.0

        # 5) 方向引导
        if np.any(danger_mask):
            local_mean_l = float(np.mean(l_offset[danger_mask]))
        else:
            local_mean_l = float(np.mean(l_offset))
        r_direction = 2.0 * self.direction_sign * local_mean_l

        # 6) 头尾贴回原参考线
        start_penalty = -20.0 * float(l_offset[0] ** 2)
        end_penalty = -20.0 * float(l_offset[-1] ** 2)

        reward = (
            r_hard
            + r_safe_area
            + r_safe_bonus
            + r_offset
            + r_smooth
            + r_curv
            + r_ctrl_smooth
            + r_ctrl_curv
            + r_direction
            + start_penalty
            + end_penalty
            + r_jerk
        )

        reward_terms = {
            "d_min": d_min,
            "r_hard": r_hard,
            "r_safe_area": r_safe_area,
            "r_safe_bonus": r_safe_bonus,
            "r_offset_far": r_offset_far,
            "r_offset_mid": r_offset_mid,
            "r_offset": r_offset,
            "r_smooth": r_smooth,
            "r_curv": r_curv,
            "r_direction": r_direction,
            "start_penalty": start_penalty,
            "end_penalty": end_penalty,
            "r_ctrl_smooth": r_ctrl_smooth,
            "r_ctrl_curv": r_ctrl_curv,
        }
        return reward, d_after, reward_terms

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        assert action.shape[0] == self.num_ctrl, "action 维度不对"

        # ctrl_values = np.clip(action, -self.alpha_limit, self.alpha_limit)

        # # 控制点 -> l(s)
        # l_offset, ctrl_s = build_l_from_control_points(self.s_ref, ctrl_values)
        basis_coeffs = np.clip(action, -self.alpha_limit, self.alpha_limit)

        # basis coeffs -> l(s)
        l_offset, basis_debug = build_l_from_basis_coeffs(self.s_ref, basis_coeffs)

        # 头尾压回 0
        window = build_end_decay_window(self.s_ref)
        l_offset = l_offset * window

        safe_x, safe_y = apply_local_offset(
            self.ref_x, self.ref_y, self.yaw_ref, l_offset
        )

        reward, d_after, terms = self._compute_reward(
            l_offset, safe_x, safe_y, basis_coeffs
        )

        done = True
        info = {
            "basis_coeffs": basis_coeffs,
            "basis_debug": basis_debug,
            "l_offset": l_offset,
            "safe_x": safe_x,
            "safe_y": safe_y,
            "d_after": d_after,
            "d_min": terms["d_min"],
            "reward_terms": terms,
        }

        next_state = self._build_state()
        return next_state, reward, done, info

    def render(self, info, title_prefix=""):
        safe_x = info["safe_x"]
        safe_y = info["safe_y"]
        l_offset = info["l_offset"]
        d_min = info["d_min"]
        basis_debug = info["basis_debug"]

        # =========================
        # Figure 1：轨迹图
        # =========================
        plt.figure(self.fig1.number)
        plt.cla()  # 清当前轴（不是整张图）

        plt.plot(
            self.boundary_x,
            self.boundary_y,
            color="red",
            linewidth=2.5,
            label="boundary",
        )
        plt.plot(
            self.ref_x, self.ref_y, color="black", linewidth=2.5, label="reference"
        )
        plt.plot(safe_x, safe_y, color="green", linewidth=2.5, label="shifted")

        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.legend()
        plt.title(f"{title_prefix} d_min={d_min:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")

        # =========================
        # Figure 2：l(s)
        # =========================
        plt.figure(self.fig2.number)
        plt.cla()

        plt.plot(self.s_ref, basis_debug["l_raw"], linewidth=1.5, label="l_raw")
        plt.plot(self.s_ref, basis_debug["l_total"], linewidth=2.5, label="l_smooth")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.5)

        plt.grid(True)
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("l")
        plt.title("Per-point Offset")

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
        plt.pause(1.0)
