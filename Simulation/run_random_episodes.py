import random
from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.Common.common import Vec2D
from SimulationTool.Navigation.Navigation import NavigationPathGenerator
from simulation_main import run_episode


def sample_episode():
    # ===== ego =====
    ego_v = random.uniform(6.0, 12.0)
    ego = VehicleKModel(x=0.0, y=0.0, yaw=0.0, v=ego_v)

    # ===== reference line =====
    nav_gen = NavigationPathGenerator(2)
    navi_traj = nav_gen.generate_straight_traj(
        Vec2D(0.0, 0.0),
        Vec2D(200.0, 0.0),
        ego_v,
    )

    # ===== obstacles (方案A：最小间距拒绝采样) =====
    obstacles = []
    max_tries = 200

    # 与 planner 的 AABB 碰撞模型保持一致的安全间距
    margin_x = 8.0  # m
    margin_y = 2.0  # m

    for k in range(2):
        placed = False
        for _ in range(max_tries):
            x = random.uniform(25.0, 65.0)
            y = random.uniform(-2.5, 0.5)
            cand = VehicleKModel(x=x, y=y, yaw=0.0, v=0.0)

            too_close = False
            for o in obstacles:
                dx = abs(cand.x - o.x)
                dy = abs(cand.y - o.y)

                half_l = cand.length / 2.0 + o.length / 2.0 + margin_x
                half_w = cand.width / 2.0 + o.width / 2.0 + margin_y

                # AABB 过近判定
                if dx < half_l and dy < half_w:
                    too_close = True
                    break

            if not too_close:
                obstacles.append(cand)
                placed = True
                break

        if not placed:
            # 实在放不下就少放一个，宁缺勿滥
            print("[Warn] Failed to place obstacle with safe distance")
            break

    return ego, obstacles, navi_traj


if __name__ == "__main__":
    for episode in range(1):
        print("\n" + "#" * 20)
        print(f"### EPISODE {episode}")
        print("#" * 20)

        ego, obstacles, navi_traj = sample_episode()
        # =================================
        print("[Episode Config]")
        print(f"  Ego:")
        print(f"    v = {ego.v:.2f} m/s")

        print(f"  Obstacles ({len(obstacles)}):")
        for i, obs in enumerate(obstacles):
            print(f"    [{i}] x={obs.x:.2f}, y={obs.y:.2f}, v={obs.v:.2f}")

        p0 = navi_traj[0]
        p1 = navi_traj[-1]
        print(
            f"  Reference Line:"
            f" start=({p0.x:.1f}, {p0.y:.1f}),"
            f" end=({p1.x:.1f}, {p1.y:.1f})"
        )
        print("-" * 20)
        # =================================
        run_episode(
            ego,
            obstacles,
            navi_traj,
            sim_time=10.0,
            dt=0.1,
        )
