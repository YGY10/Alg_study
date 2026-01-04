import math
from SimulationTool.Common.common import TrajPoint


class RuleTrajectoryPlanner:
    """
    基于全局参考线的 Rule-Based 局部轨迹规划器
    """

    def __init__(self, horizon=30, ds=1.0, base_speed=2.0):
        self.horizon = horizon
        self.ds = ds
        self.base_speed = base_speed
        self.last_ref_idx = 0  # 保证 s 单调

    # -------------------------------------------------
    def find_nearest_forward_index(self, ego, path):
        """
        在全局参考线中找最近的前向点（不回头）
        """
        min_dist = float("inf")
        best_idx = self.last_ref_idx

        for i in range(self.last_ref_idx, len(path)):
            dx = path[i].x - ego.x
            dy = path[i].y - ego.y
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                best_idx = i

        self.last_ref_idx = best_idx
        return best_idx

    # -------------------------------------------------
    def compute_lateral_offset(self, ego, obstacles):
        """
        非常简单的 rule-based 避障：给一个横向偏移量
        """
        offset = 0.0
        target_speed = self.base_speed

        for obs in obstacles:
            dx = obs.x - ego.x
            dy = obs.y - ego.y

            # 只关心前方近距离障碍
            if 0 < dx < 10.0 and abs(dy) < 2.0:
                offset = 2.5 if dy <= 0 else -2.5
                target_speed = 1.0
                break

        return offset, target_speed

    # -------------------------------------------------
    def plan(self, ego, nav_path, obstacles):
        """
        核心规划函数
        """
        traj = []

        if nav_path is None or len(nav_path) == 0:
            return traj

        # ===== 1. 找到全局参考线起点 =====
        ref_idx = self.find_nearest_forward_index(ego, nav_path)

        # ===== 2. 计算横向偏移策略 =====
        lateral_offset, target_speed = self.compute_lateral_offset(ego, obstacles)

        t = 0.0

        # ===== 3. 沿参考线生成局部轨迹 =====
        for i in range(self.horizon):
            idx = min(ref_idx + i, len(nav_path) - 1)
            ref = nav_path[idx]

            # 法向方向
            nx = -math.sin(ref.yaw)
            ny = math.cos(ref.yaw)

            # 横向偏移（平滑比例）
            ratio = min(i / 10.0, 1.0)
            x = ref.x + ratio * lateral_offset * nx
            y = ref.y + ratio * lateral_offset * ny

            traj.append(
                TrajPoint(
                    x=x,
                    y=y,
                    yaw=ref.yaw,
                    v=target_speed,
                    t=t,
                )
            )

            t += self.ds / max(target_speed, 0.1)

        return traj
