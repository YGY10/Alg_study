import math


class StanleyController:
    """
    Stanley Trajectory Controller
    - 横向误差 + 航向误差
    - 纯输出控制量，不更新车辆
    """

    def __init__(
        self,
        k=1.0,  # Stanley 横向误差增益
        kv=1.0,  # 速度 P 控制
        lookahead_dist=1.0,  # 最近点搜索前视
        max_steer=math.radians(35),
    ):
        self.k = k
        self.kv = kv
        self.lookahead_dist = lookahead_dist
        self.max_steer = max_steer

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def find_target_index(self, ego, traj):
        """
        找到距离自车最近的轨迹点
        """
        min_dist = float("inf")
        target_idx = 0

        for i, p in enumerate(traj):
            dx = p.x - ego.x
            dy = p.y - ego.y
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                target_idx = i

        return target_idx

    def compute_control(self, ego, traj):
        if traj is None or len(traj) == 0:
            return 0.0, 0.0

        # ===============================
        # 1. 找最近轨迹点
        # ===============================
        idx = self.find_target_index(ego, traj)
        target = traj[idx]

        # ===============================
        # 2. 航向误差
        # ===============================
        heading_error = self.normalize_angle(target.yaw - ego.yaw)

        # ===============================
        # 3. 横向误差（符号很关键）
        # ===============================
        dx = target.x - ego.x
        dy = target.y - ego.y

        # 轨迹切向方向
        path_yaw = target.yaw
        perp_vec_x = math.cos(path_yaw + math.pi / 2)
        perp_vec_y = math.sin(path_yaw + math.pi / 2)

        cross_track_error = dx * perp_vec_x + dy * perp_vec_y

        # ===============================
        # 4. Stanley 控制律
        # ===============================
        v = max(ego.v, 0.1)  # 防止除零
        cte_term = math.atan2(self.k * cross_track_error, v)

        delta = heading_error + cte_term
        delta = self.normalize_angle(delta)
        delta = max(-self.max_steer, min(self.max_steer, delta))

        # ===============================
        # 5. 纵向控制（速度）
        # ===============================
        v_error = target.v - ego.v
        a = self.kv * v_error

        return a, delta
