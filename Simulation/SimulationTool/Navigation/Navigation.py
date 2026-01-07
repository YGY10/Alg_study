from SimulationTool.Common.common import TrajPoint, Vec2D
import math
from typing import List


class NavigationPathGenerator:
    # 导航路径生成器
    def __init__(self, ds=0.5):
        # 默认采样分辨率
        self.ds = ds

    def generate_straight_traj(
        self, start: Vec2D, end: Vec2D, v_ref
    ) -> List[TrajPoint]:
        traj: List[TrajPoint] = []
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return traj
        yaw = math.atan2(dy, dx)

        s = 0.0
        t = 0.0
        dist = 0.0
        while dist <= length:
            ratio = dist / length
            x = start.x + ratio * dx
            y = start.y + ratio * dy

            traj.append(
                TrajPoint(
                    x=x,
                    y=y,
                    yaw=yaw,
                    v=v_ref,
                    s=s,
                    t=t,
                )
            )

            dist += self.ds
            s += self.ds
            t += self.ds / max(v_ref, 0.1)

        return traj
