# control/stanley_controller.py
import math


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class StanleyController:
    def __init__(
        self,
        k=1.0,
        ks=1.0,
        wheelbase=2.8,
        max_steer_rad=0.61,
        lookahead_base=3.0,
        lookahead_gain=0.5,
        low_speed_steer_scale=0.6,
        low_speed_threshold=1.0,
        low_speed_max_steer_rad=0.30,
    ):
        """
        k: 横向误差增益
        ks: 低速软化项，防止 v 很小时 cte 项过大
        wheelbase: 轴距（当前版本未直接使用，先保留）
        max_steer_rad: 正常情况下最大前轮转角近似值（约35度）
        lookahead_base: 最小前视距离
        lookahead_gain: 随速度增加的前视距离增益
        low_speed_steer_scale: 低速时额外缩放转向
        low_speed_threshold: 低速阈值
        low_speed_max_steer_rad: 低速时额外硬限幅的最大转角（约17度）
        """
        self.k = k
        self.ks = ks
        self.wheelbase = wheelbase
        self.max_steer_rad = max_steer_rad

        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain

        self.low_speed_steer_scale = low_speed_steer_scale
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_max_steer_rad = low_speed_max_steer_rad

    def calc_lookahead_distance(self, ego_speed):
        return self.lookahead_base + self.lookahead_gain * ego_speed

    def find_nearest_index(self, ego_x, ego_y, ref_path):
        min_dist_sq = float("inf")
        min_idx = 0

        for i, pt in enumerate(ref_path):
            dx = pt["x"] - ego_x
            dy = pt["y"] - ego_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_idx = i

        return min_idx

    def find_target_index(self, ego_x, ego_y, ego_speed, ref_path):
        """
        先找最近点，再按 s 向前找 lookahead 目标点
        """
        if not ref_path:
            return 0

        nearest_idx = self.find_nearest_index(ego_x, ego_y, ref_path)
        lookahead_dist = self.calc_lookahead_distance(ego_speed)
        target_s = ref_path[nearest_idx]["s"] + lookahead_dist

        target_idx = nearest_idx
        for i in range(nearest_idx, len(ref_path)):
            if ref_path[i]["s"] >= target_s:
                target_idx = i
                break
            target_idx = i

        return target_idx

    def compute_control(self, ego_x, ego_y, ego_yaw, ego_speed, ref_path):
        if not ref_path:
            return 0.0, 0, 0.0, 0.0

        target_idx = self.find_target_index(ego_x, ego_y, ego_speed, ref_path)
        target = ref_path[target_idx]

        ref_x = target["x"]
        ref_y = target["y"]
        ref_yaw = target["yaw"]

        dx = ego_x - ref_x
        dy = ego_y - ref_y

        # 参考路径左法向量
        nx = -math.sin(ref_yaw)
        ny = math.cos(ref_yaw)

        # signed cross-track error
        cte = dx * nx + dy * ny

        heading_error = normalize_angle(ref_yaw - ego_yaw)

        # 注意这里负号已经按你之前的实验修正好了
        cte_term = math.atan2(-self.k * cte, ego_speed + self.ks)

        steer_rad = heading_error + cte_term

        # 低速阶段更柔和，避免原地大角度拧方向
        if ego_speed < self.low_speed_threshold:
            steer_rad *= self.low_speed_steer_scale
            steer_rad = max(
                -self.low_speed_max_steer_rad,
                min(self.low_speed_max_steer_rad, steer_rad),
            )

        # 正常总体限幅
        steer_rad = max(-self.max_steer_rad, min(self.max_steer_rad, steer_rad))

        return steer_rad, target_idx, cte, heading_error

    def steer_rad_to_carla(self, steer_rad):
        """
        将理想前轮转角近似映射为 CARLA steer in [-1, 1]
        """
        steer = steer_rad / self.max_steer_rad
        steer = max(-1.0, min(1.0, steer))
        return steer
