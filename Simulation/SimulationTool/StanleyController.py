import math


class StanleyController:
    """
    Stanley Controller with D damping term
    """

    def __init__(
        self,
        k=0.8,  # 横向误差增益
        kv=1.0,  # 纵向速度 P
        kd=0.3,  # 航向误差 D 项
        min_speed=1.0,  # Stanley 分母最小速度
        max_steer=math.radians(35),
    ):
        self.k = k
        self.kv = kv
        self.kd = kd
        self.min_speed = min_speed
        self.max_steer = max_steer

        # 内部状态（用于 D 项）
        self.last_heading_error = 0.0

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def find_nearest_index(self, ego, traj):
        min_dist = float("inf")
        idx = 0
        for i, p in enumerate(traj):
            dx = p.x - ego.x
            dy = p.y - ego.y
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                idx = i
        return idx

    def compute_control(self, ego, traj, dt):
        if traj is None or len(traj) == 0:
            return 0.0, 0.0

        # ===============================
        # 1. 最近轨迹点
        # ===============================
        idx = self.find_nearest_index(ego, traj)
        ref = traj[idx]

        # ===============================
        # 2. 航向误差
        # ===============================
        heading_error = self.normalize_angle(ref.yaw - ego.yaw)

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
        # 4. Stanley 控制律（含 D 项）
        # ===============================
        v = max(abs(ego.v), self.min_speed)

        cte_term = math.atan2(self.k * cross_track_error, v)

        d_heading = (heading_error - self.last_heading_error) / max(dt, 1e-3)

        delta = heading_error + cte_term + self.kd * d_heading
        delta = self.normalize_angle(delta)
        delta = max(-self.max_steer, min(self.max_steer, delta))

        self.last_heading_error = heading_error

        # ===============================
        # 5. 纵向控制
        # ===============================
        a = self.kv * (ref.v - ego.v)

        return a, delta, cross_track_error
