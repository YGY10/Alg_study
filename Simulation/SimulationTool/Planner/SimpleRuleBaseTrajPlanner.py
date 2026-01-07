import math
from SimulationTool.Common.common import TrajPoint


class RuleTrajectoryPlanner:
    def __init__(self, horizon=40, ds=0.2, base_speed=2.0):
        self.horizon = horizon
        self.ds = ds
        self.base_speed = base_speed

        self.last_traj = None
        self.last_ref_idx = 0

        # ⭐ 新增：避障决策记忆
        # +1 = 左绕，-1 = 右绕，None = 尚未决定 / 不在避障
        self.avoid_side = None

    # -------------------------------------------------
    def find_nearest_forward_index(self, state, path):
        min_dist = float("inf")
        best_idx = self.last_ref_idx

        search_range = 100
        end = min(self.last_ref_idx + search_range, len(path))

        for i in range(self.last_ref_idx, end):
            dx = path[i].x - state.x
            dy = path[i].y - state.y
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                best_idx = i

        self.last_ref_idx = best_idx
        return best_idx

    # -------------------------------------------------
    def compute_lateral_offset(self, ego, obstacles):
        """
        ⭐ 改进版 rule-based 避障（带决策记忆）
        """
        target_offset = 0.0
        target_speed = self.base_speed

        obstacle_detected = False

        for obs in obstacles:
            dx = obs.x - ego.x
            dy = obs.y - ego.y

            if 0 < dx < 12.0 and abs(dy) < 2.0:
                obstacle_detected = True

                # ===== 第一次遇到障碍：做一次决策 =====
                if self.avoid_side is None:
                    self.avoid_side = 1 if dy <= 0 else -1

                target_offset = 2.8 * self.avoid_side
                target_speed = 1.2
                break

        # ===== 障碍物清除：允许恢复直行 =====
        if not obstacle_detected:
            self.avoid_side = None

        return target_offset, target_speed

    # -------------------------------------------------
    def find_stitching_point(self, ego):
        if self.last_traj is None or len(self.last_traj) < 2:
            return ego, "Initial"

        min_dist = float("inf")
        best_idx = 0

        for i, p in enumerate(self.last_traj):
            dx = p.x - ego.x
            dy = p.y - ego.y
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                best_idx = i

        return self.last_traj[best_idx], "Stitched"

    # -------------------------------------------------
    def plan(self, ego, nav_path, obstacles):
        if not nav_path:
            return []

        # ===== 1. 轨迹拼接 =====
        start_state, stitching_status = self.find_stitching_point(ego)

        # ===== 2. 参考线索引 =====
        ref_idx = self.find_nearest_forward_index(start_state, nav_path)

        # ===== 3. 避障决策（已稳定）=====
        target_offset, target_speed = self.compute_lateral_offset(ego, obstacles)

        # ===== 4. 当前 offset（连续性）=====
        ref_start = nav_path[ref_idx]
        dx_s = start_state.x - ref_start.x
        dy_s = start_state.y - ref_start.y
        current_offset = (
            -math.sin(ref_start.yaw) * dx_s + math.cos(ref_start.yaw) * dy_s
        )

        # ===== 5. 生成轨迹 =====
        new_traj = []
        t = 0.0

        for i in range(self.horizon):
            idx = min(ref_idx + i, len(nav_path) - 1)
            ref_pt = nav_path[idx]

            weight = min(i / 5.0, 1.0)
            planned_offset = current_offset * (1.0 - weight) + target_offset * weight

            nx, ny = -math.sin(ref_pt.yaw), math.cos(ref_pt.yaw)
            x = ref_pt.x + planned_offset * nx
            y = ref_pt.y + planned_offset * ny

            if i > 0:
                dx = x - new_traj[-1].x
                dy = y - new_traj[-1].y
                yaw = math.atan2(dy, dx)
            else:
                yaw = ref_pt.yaw

            new_traj.append(
                TrajPoint(
                    x=x,
                    y=y,
                    yaw=yaw,
                    v=target_speed,
                    t=t,
                )
            )

            t += self.ds / max(target_speed, 0.1)

        print(
            f"[Planner] Status: {stitching_status} | "
            f"Start_y: {new_traj[0].y:.3f} | Offset: {target_offset}"
        )

        self.last_traj = new_traj
        return new_traj
