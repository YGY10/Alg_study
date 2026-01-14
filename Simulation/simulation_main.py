import pygame
import math

from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.VisualizeTool.visualize import VehicleVisualizer
from SimulationTool.Controller.MPController import MPCController
from SimulationTool.Common.common import TrajPoint, Vec2D
from SimulationTool.Planner.SpatioTemporalFrenetPlanner import STFrenetPlanner
from SimulationTool.VisualizeTool.RTScope import RealtimeScope2D
from SimulationTool.Navigation.Navigation import NavigationPathGenerator


# ===============================
# Scope 初始化
# ===============================
scope_xy = RealtimeScope2D("Trajectory Tracking", "x", "y")
scope_xy.add_channel("reference", color="green", linestyle="--")
scope_xy.add_channel("ego", color="blue")

# scope_ey = RealtimeScope2D("Lateral Error", "t", "e_y")
# scope_ey.add_channel("e_y", color="red")

# scope_vt = RealtimeScope2D("Speed Trakcking", "t", "v")
# scope_vt.add_channel("v", color="blue")


# ===============================
# 生成导航参考线（全局）
# ===============================
navi_pathgenerator = NavigationPathGenerator(2)
navi_trajectory = navi_pathgenerator.generate_straight_traj(
    Vec2D(0.0, 0.0), Vec2D(200, 0.0), 10.0
)

ego = VehicleKModel(x=0.0, y=0.0, yaw=0.0, v=10.0)

# 障碍物
obs1 = VehicleKModel(x=15.0, y=0.0, yaw=0.0, v=0.0)
obs2 = VehicleKModel(x=25.0, y=3.5, yaw=0.0, v=0.0)
obstacles = [obs1]

# Planner + Controller
# planner = RuleTrajectoryPlanner(horizon=30, ds=1.0, base_speed=2.0)
planner = STFrenetPlanner()
controller = MPCController()

vis = VehicleVisualizer()

# ===============================
# 仿真参数
# ===============================
dt = 0.1
sim_time = 20
steps = int(sim_time / dt)
t = 0.0


# ===============================
# 主仿真循环
# ===============================
for step in range(steps):
    print("\n" + "=" * 80)
    print(f"[STEP {step}] START  t = {t:.2f}s")
    # ===== 1. Planner：从导航线生成局部轨迹 =====
    local_traj = planner.plan(ego, navi_trajectory, obstacles, step * dt)

    # ===== 2. Controller：跟踪局部轨迹 =====
    a, delta, e_y = controller.compute_control(ego, local_traj, dt)

    ego.update(a, delta)

    # ===== 3. Scope =====
    # scope_ey.update("e_y", t, e_y)
    # scope_vt.update("v", t, ego.v)
    scope_xy.update("ego", ego.x, ego.y)

    # 画 planner 参考轨迹（最近点）
    ref_idx = min(
        range(len(local_traj)),
        key=lambda i: (local_traj[i].x - ego.x) ** 2 + (local_traj[i].y - ego.y) ** 2,
    )
    ref = local_traj[ref_idx]
    scope_xy.update("reference", ref.x, ref.y)

    # ===== 4. 可视化 =====
    vis.run_step(ego, local_traj, obstacles)
    vis.clock.tick(1.0 / dt)

    t += dt
    print("=" * 80 + "\n")


print("Simulation finished. Close windows to exit.")

# ===============================
# 保持窗口
# ===============================
scope_xy.hold()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
