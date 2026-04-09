import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. 基础几何
# =========================================================
def build_end_decay_window(s):
    u = (s - s[0]) / max(s[-1] - s[0], 1e-6)
    return np.sin(np.pi * u) ** 2


def build_turn_path(r=10.0, straight_len=20.0, n_line=120, n_arc=120):
    y1 = np.linspace(0.0, straight_len, n_line)
    x1 = np.zeros_like(y1)

    theta = np.linspace(np.pi, np.pi / 2.0, n_arc)
    x2 = r + r * np.cos(theta)
    y2 = straight_len + r * np.sin(theta)

    x3 = np.linspace(r, r + straight_len, n_line)
    y3 = np.full_like(x3, straight_len + r)

    x = np.concatenate([x1, x2[1:], x3[1:]])
    y = np.concatenate([y1, y2[1:], y3[1:]])
    return x, y


def build_boundary_from_points(points, num_per_seg=40):
    pts = np.asarray(points, dtype=float)
    assert len(pts) >= 2, "至少需要两个点"

    xs = []
    ys = []

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]

        if i < len(pts) - 2:
            t = np.linspace(0.0, 1.0, num_per_seg, endpoint=False)
        else:
            t = np.linspace(0.0, 1.0, num_per_seg, endpoint=True)

        x_seg = p0[0] * (1.0 - t) + p1[0] * t
        y_seg = p0[1] * (1.0 - t) + p1[1] * t

        xs.append(x_seg)
        ys.append(y_seg)

    bx = np.concatenate(xs)
    by = np.concatenate(ys)
    return bx, by


def compute_s(x, y):
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def compute_yaw_from_xy(x, y):
    n = len(x)
    yaw = np.zeros(n)

    for i in range(n):
        if i == 0:
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
        elif i == n - 1:
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
        else:
            dx = 0.5 * (x[i + 1] - x[i - 1])
            dy = 0.5 * (y[i + 1] - y[i - 1])

        yaw[i] = np.arctan2(dy, dx)

    return yaw


def compute_normals_from_yaw(yaw):
    nx = -np.sin(yaw)
    ny = np.cos(yaw)
    return nx, ny


def apply_local_offset(x, y, yaw, offset_l):
    nx, ny = compute_normals_from_yaw(yaw)
    x_new = x + nx * offset_l
    y_new = y + ny * offset_l
    return x_new, y_new


def resample_polyline_by_s(x, y, num=200):
    s = compute_s(x, y)
    s_new = np.linspace(s[0], s[-1], num)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new, s_new


# =========================================================
# 2. 点到折线距离
# =========================================================
def point_to_segment_projection(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return ax, ay, np.hypot(px - ax, py - ay), 0.0

    t = (apx * abx + apy * aby) / ab2
    t = np.clip(t, 0.0, 1.0)

    qx = ax + t * abx
    qy = ay + t * aby
    dist = np.hypot(px - qx, py - qy)
    return qx, qy, dist, t


def point_to_polyline_projection(px, py, line_x, line_y):
    best_dist = np.inf
    best_qx = None
    best_qy = None

    for i in range(len(line_x) - 1):
        qx, qy, dist, _ = point_to_segment_projection(
            px, py, line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]
        )
        if dist < best_dist:
            best_dist = dist
            best_qx = qx
            best_qy = qy

    return best_qx, best_qy, best_dist


def compute_min_distance_to_boundary(path_x, path_y, boundary_x, boundary_y):
    dists = np.zeros(len(path_x))
    for i in range(len(path_x)):
        _, _, dist = point_to_polyline_projection(
            path_x[i], path_y[i], boundary_x, boundary_y
        )
        dists[i] = dist
    return dists


# =========================================================
# 3. Spline-control-point 动作表达
# =========================================================
def smooth_signal(y, num_iter=3):
    """
    一个轻量平滑器，避免控制点插值后出现折感
    """
    y = y.copy()
    for _ in range(num_iter):
        y_new = y.copy()
        y_new[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
        y = y_new
    return y


def build_l_from_control_points(s_ref, ctrl_values):
    """
    用控制点生成整条 l(s)
    改成固定“前密后疏”控制点分布
    """
    num_ctrl = len(ctrl_values)

    if num_ctrl == 8:
        ctrl_s_ratio = np.array([0.0, 0.06, 0.12, 0.22, 0.36, 0.55, 0.78, 1.0])

    elif num_ctrl == 18:
        ctrl_s_ratio = np.array(
            [
                0.00,
                0.03,
                0.06,
                0.09,
                0.12,
                0.16,
                0.20,
                0.25,
                0.30,
                0.36,
                0.43,
                0.50,
                0.58,
                0.67,
                0.76,
                0.85,
                0.93,
                1.00,
            ]
        )

    else:
        # 其它情况先退回均匀分布
        ctrl_s_ratio = np.linspace(0.0, 1.0, num_ctrl)

    ctrl_s = s_ref[0] + ctrl_s_ratio * (s_ref[-1] - s_ref[0])

    l = np.interp(s_ref, ctrl_s, ctrl_values)

    # 平滑几次，让它更像 spline
    l = smooth_signal(l, num_iter=6)
    return l, ctrl_s


# =========================================================
# 4. 最小 RL 环境
# =========================================================
class ReferenceShiftEnv:
    def __init__(
        self,
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=18,
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
        danger_weight = np.exp(-0.5 * (self.ref_to_boundary_dist / 1.5) ** 2)
        stay_weight = 1.0 - danger_weight
        r_offset = -6.0 * float(np.mean(stay_weight * (l_offset**2)))

        # 4) 平滑性
        dl = np.diff(l_offset)
        ddl = np.diff(l_offset, n=2)

        r_smooth = -1.5 * float(np.mean(dl**2)) if len(dl) > 0 else 0.0
        r_curv = -0.6 * float(np.mean(ddl**2)) if len(ddl) > 0 else 0.0

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
        )

        reward_terms = {
            "d_min": d_min,
            "r_hard": r_hard,
            "r_safe_area": r_safe_area,
            "r_safe_bonus": r_safe_bonus,
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

        ctrl_values = np.clip(action, -self.alpha_limit, self.alpha_limit)

        # 控制点 -> l(s)
        l_offset, ctrl_s = build_l_from_control_points(self.s_ref, ctrl_values)

        # 头尾压回 0
        window = build_end_decay_window(self.s_ref)
        l_offset = l_offset * window

        safe_x, safe_y = apply_local_offset(
            self.ref_x, self.ref_y, self.yaw_ref, l_offset
        )

        reward, d_after, terms = self._compute_reward(
            l_offset, safe_x, safe_y, ctrl_values
        )

        done = True
        info = {
            "ctrl_values": ctrl_values,
            "ctrl_s": ctrl_s,
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
        ctrl_s = info["ctrl_s"]
        ctrl_values = info["ctrl_values"]

        plt.figure(figsize=(8, 8))
        plt.plot(
            self.boundary_x,
            self.boundary_y,
            color="red",
            linewidth=2.5,
            label="boundary",
        )
        plt.plot(
            self.ref_x,
            self.ref_y,
            color="black",
            linewidth=2.5,
            label="original reference",
        )
        plt.plot(
            safe_x,
            safe_y,
            color="green",
            linewidth=2.5,
            label="shifted reference",
        )

        for s_c, l_c in zip(ctrl_s, ctrl_values):
            idx = np.argmin(np.abs(self.s_ref - s_c))
            px = self.ref_x[idx]
            py = self.ref_y[idx]
            yaw = self.yaw_ref[idx]
            nx = -np.sin(yaw)
            ny = np.cos(yaw)
            cx = px + nx * l_c
            cy = py + ny * l_c
            plt.scatter(cx, cy, color="blue", s=18)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.legend()
        plt.title(f"{title_prefix} d_min={d_min:.3f}m")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(self.s_ref, l_offset, linewidth=2, label="l(s)")
        plt.scatter(ctrl_s, ctrl_values, color="blue", s=25, label="control points")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.5)
        plt.grid(True)
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("l")
        plt.title("Offset Curve by Spline Control Points")
        plt.show()


# =========================================================
# 5. 简单测试
# =========================================================
def main():
    env = ReferenceShiftEnv(
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=18,
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
        seed=42,
    )

    state = env.reset()
    print("state shape =", state.shape)

    num_samples = 100

    best_reward = -1e18
    worst_reward = 1e18
    best_info = None
    worst_info = None
    best_action = None
    worst_action = None

    for _ in range(num_samples):
        action = np.random.uniform(-2.0, 2.0, size=env.num_ctrl)
        _, reward, _, info = env.step(action)

        if reward > best_reward:
            best_reward = reward
            best_info = info
            best_action = action

        if reward < worst_reward:
            worst_reward = reward
            worst_info = info
            worst_action = action

    print("\n===== BEST SAMPLE =====")
    print("reward =", best_reward)
    print("d_min =", best_info["d_min"])
    print("ctrl_values =", best_action)
    print("reward_terms =", best_info["reward_terms"])

    print("\n===== WORST SAMPLE =====")
    print("reward =", worst_reward)
    print("d_min =", worst_info["d_min"])
    print("ctrl_values =", worst_action)
    print("reward_terms =", worst_info["reward_terms"])

    print("\n>>> Visualizing BEST")
    env.render(best_info, title_prefix="BEST")

    print("\n>>> Visualizing WORST")
    env.render(worst_info, title_prefix="WORST")


if __name__ == "__main__":
    main()
