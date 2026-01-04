import math
from SimulationTool.Common.common import normalize_angle


class StanleyController:
    """
    Lookahead Stanley Controller with D damping term
    """

    def __init__(
        self,
        k=0.8,  # 横向误差增益
        kv=1.0,  # 纵向速度 P
        kd=0.3,  # 航向误差 D 项
        lookahead_distance=5.0,  # 默认前视距离（m）
        min_speed=1.0,  # Stanley 分母最小速度
        max_steer=math.radians(35),
    ):
        self.k = k
        self.kv = kv
        self.kd = kd
        self.lookahead_distance = lookahead_distance
        self.min_speed = min_speed
        self.max_steer = max_steer

        # 内部状态（用于 D 项）
        self.last_heading_error = 0.0

    def find_nearest_forward_index(self, ego, traj):
        min_dist = float("inf")
        idx = 0

        # 车头方向向量
        hx = math.cos(ego.yaw)
        hy = math.sin(ego.yaw)

        for i, p in enumerate(traj):
            dx = p.x - ego.x
            dy = p.y - ego.y

            # 判断是否在车前（点积）
            forward = dx * hx + dy * hy
            if forward < 0:
                continue  # 在车后，直接跳过

            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                idx = i

        return idx

    def find_lookahead_index(self, ego, traj):
        """
        从最近点开始，沿轨迹累计距离，找到 lookahead 距离对应的点
        """
        nearest_idx = self.find_nearest_forward_index(ego, traj)

        dist = 0.0
        for i in range(nearest_idx, len(traj) - 1):
            dx = traj[i + 1].x - traj[i].x
            dy = traj[i + 1].y - traj[i].y
            step = math.hypot(dx, dy)
            dist += step

            if dist >= self.lookahead_distance:
                return i + 1

        return len(traj) - 1

    def compute_control(self, ego, traj, dt):
        if traj is None or len(traj) == 0:
            return 0.0, 0.0, 0.0

        # ===============================
        # 1. 前视轨迹点
        # ===============================
        idx = self.find_lookahead_index(ego, traj)
        ref = traj[idx]

        # ===============================
        # 2. 航向误差
        # ===============================
        heading_error = normalize_angle(ref.yaw - ego.yaw)

        # ===============================
        # 3. 横向误差（带符号）
        # ===============================
        dx = ref.x - ego.x
        dy = ref.y - ego.y

        path_yaw = ref.yaw
        nx = -math.sin(path_yaw)
        ny = math.cos(path_yaw)
        cross_track_error = dx * nx + dy * ny

        # ===============================
        # 4. Lookahead Stanley 控制律（含 D 项）
        # ===============================
        v = max(abs(ego.v), self.min_speed)

        cte_term = math.atan2(self.k * cross_track_error, v)

        d_heading = (heading_error - self.last_heading_error) / max(dt, 1e-3)

        delta = heading_error + cte_term + self.kd * d_heading
        delta = normalize_angle(delta)
        delta = max(-self.max_steer, min(self.max_steer, delta))

        self.last_heading_error = heading_error

        # ===============================
        # 5. 纵向控制
        # ===============================
        a = self.kv * (ref.v - ego.v)

        return a, delta, cross_track_error
