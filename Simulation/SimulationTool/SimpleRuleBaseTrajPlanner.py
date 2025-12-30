from dataclasses import dataclass
import math


@dataclass
class TrajPoint:
    x: float
    y: float
    yaw: float
    v: float
    t: float


class RuleTrajectoryPlanner:
    def __init__(self, horizon=30, ds=1.0, base_speed=2.0):
        self.horizon = horizon  # 轨迹点数
        self.ds = ds  # 空间步长
        self.base_speed = base_speed

    def plan(self, ego, obstacles):
        traj = []

        # ===== 简单避障逻辑 =====
        lateral_offset = 0.0
        target_speed = self.base_speed

        for obs in obstacles:
            dx = obs.x - ego.x
            dy = obs.y - ego.y

            # 只关心前方障碍
            if 0 < dx < 10.0 and abs(dy) < 2.0:
                lateral_offset = 3.0 if dy <= 0 else -3.0
                target_speed = 1.0
                break

        t = 0.0
        x = ego.x
        y = ego.y

        for i in range(self.horizon):
            x += self.ds * math.cos(ego.yaw)
            y += self.ds * math.sin(ego.yaw) + lateral_offset * 0.02

            yaw = ego.yaw
            v = target_speed
            traj.append(TrajPoint(x, y, yaw, v, t))
            t += self.ds / max(v, 0.1)

        return traj
