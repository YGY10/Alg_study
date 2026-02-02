import pygame
import math

from SimulationTool.SimModel.vehicle_model import VehicleKModel
from SimulationTool.VisualizeTool.visualize import VehicleVisualizer
from SimulationTool.Controller.MPController import MPCController
from SimulationTool.Common.common import TrajPoint, Vec2D
from SimulationTool.Planner.SpatioTemporalFrenetPlanner import STFrenetPlanner
from SimulationTool.NNPlanner.Model.NNPlanner import NNPlanner
from SimulationTool.VisualizeTool.RTScope import RealtimeScope2D
from SimulationTool.Navigation.Navigation import NavigationPathGenerator


def run_episode(episode, ego, obstacles, navi_trajectory, sim_time=20.0, dt=0.1):
    # scope_xy = RealtimeScope2D("Trajectory Tracking", "x", "y")
    # scope_xy.add_channel("reference", color="green", linestyle="--")
    # scope_xy.add_channel("ego", color="blue")

    nn_planner = NNPlanner(in_dim=10)
    nn_planner.load(
        "/home/yihang/Documents/Alg_study/Simulation/SimulationTool/nnplanner_two_head.pth"
    )
    planner = STFrenetPlanner(
        ego_length=ego.length, ego_width=ego.width, nn_planner=nn_planner
    )
    controller = MPCController()
    vis = VehicleVisualizer()

    steps = int(sim_time / dt)
    t = 0.0

    for step in range(steps):
        print("\n" + "=" * 80)
        print(f"[STEP {step}] START  t = {t:.2f}s")

        local_traj = planner.plan(ego, navi_trajectory, obstacles, episode, step * dt)
        a, delta, e_y = controller.compute_control(ego, local_traj, dt)
        ego.update(a, delta)

        # scope_xy.update("ego", ego.x, ego.y)

        ref_idx = min(
            range(len(local_traj)),
            key=lambda i: (local_traj[i].x - ego.x) ** 2
            + (local_traj[i].y - ego.y) ** 2,
        )
        ref = local_traj[ref_idx]
        # scope_xy.update("reference", ref.x, ref.y)

        vis.run_step(ego, local_traj, obstacles)
        vis.clock.tick(1.0 / dt)

        t += dt
        print("=" * 80 + "\n")

    pygame.quit()

    # scope_xy.hold()
