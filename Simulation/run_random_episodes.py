import random
from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.Common.common import Vec2D
from SimulationTool.Navigation.Navigation import NavigationPathGenerator
from simulation_main import run_episode

EPI_OFFSET = 200


def sample_episode():
    # ===== ego =====
    ego_v = random.uniform(10.0, 12.0)
    ego = VehicleKModel(x=0.0, y=0.0, yaw=0.0, v=ego_v)

    # ===== reference line =====
    nav_gen = NavigationPathGenerator(2)
    navi_traj = nav_gen.generate_straight_traj(
        Vec2D(0.0, 0.0),
        Vec2D(200.0, 0.0),
        ego_v,
    )

    # ===== lane config =====
    lane_centers = [-3.5, 0.0, 3.5]
    lane_half_width = 0.8

    # ===== obstacle count =====
    r = random.random()
    if r < 0.33:
        num_obs = 0
    elif r < 0.66:
        num_obs = 1

    else:
        num_obs = 2

    print(f"[SamEpi] r={r:.2f}  num_obs={num_obs}")

    # ===== x range (分层) =====
    r_x = random.random()
    if r_x < 0.33:
        x_range = (75.0, 105.0)  # 很远
    elif r_x < 0.66:
        x_range = (40.0, 70.0)
    else:
        x_range = (20.0, 40.0)
    print(f"[SamEpi] r_x={r_x:.2f}  x_range={x_range}")
    # ===== choose blocked lanes =====
    blocked_lanes = []
    if num_obs == 1:
        blocked_lanes = random.sample(lane_centers, 1)
    elif num_obs == 2:
        blocked_lanes = random.sample(lane_centers, 2)

    print(f"[SamEpi] blocked_lanes={blocked_lanes}, x_range={x_range}")

    # ===== generate obstacles =====
    obstacles = []
    max_tries = 200

    margin_x = 6.0
    margin_y = 2.0

    for lane_y in blocked_lanes:
        placed = False
        for _ in range(max_tries):
            x = random.uniform(*x_range)
            y = random.uniform(lane_y - lane_half_width, lane_y + lane_half_width)
            cand = VehicleKModel(x=x, y=y, yaw=0.0, v=0.0)

            too_close = False
            for o in obstacles:
                dx = abs(cand.x - o.x)
                dy = abs(cand.y - o.y)

                half_l = cand.length / 2.0 + o.length / 2.0 + margin_x
                half_w = cand.width / 2.0 + o.width / 2.0 + margin_y

                if dx < half_l and dy < half_w:
                    too_close = True
                    break

            if not too_close:
                obstacles.append(cand)
                placed = True
                break

        if not placed:
            print("[Warn] Failed to place obstacle in lane", lane_y)

    return ego, obstacles, navi_traj


if __name__ == "__main__":
    for episode in range(1):
        print("\n" + "#" * 20)
        print(f"### EPISODE {episode}")
        print("#" * 20)

        ego, obstacles, navi_traj = sample_episode()

        print("[Episode Config]")
        print(f"  Ego v = {ego.v:.2f} m/s")
        print(f"  Obstacles ({len(obstacles)}):")
        for i, obs in enumerate(obstacles):
            print(f"    [{i}] x={obs.x:.2f}, y={obs.y:.2f}, v={obs.v:.2f}")
        epi_uni = episode + EPI_OFFSET
        run_episode(
            epi_uni,
            ego,
            obstacles,
            navi_traj,
            sim_time=10.0,
            dt=0.1,
        )
