import math
import numpy as np
from typing import List
from SimulationTool.Common.common import TrajPoint
from SimulationTool.SimModel.vehicle_model import VehicleKModel


class NMPCPlanner:
    # 初始化
    def __init__(self, horizon_T=3.0, dt=0.2, wheelbase=2.7):
        self.dt = dt
        self.N = int(horizon_T / dt)
        self.wheelbase = wheelbase
        # cost weight
        self.w_ey = 5.0
        self.w_epsi = 2.0
        self.w_v = 1.0
        self.w_a = 0.1
        self.w_delta = 0.1
        # control limits
        self.a_min, self.a_max = -3.0, 2.0
        # 一些成员变量
        self.last_plan_traj = []
        self.warm_start_mode = "COLD"  # COLD: 默认初始化/自车已经偏离上周期轨迹太多 STICH

    # 规划主程序
    def plan(
        self,
        ego: VehicleKModel,
        ref_path: List[TrajPoint],
        obstacles: List[VehicleKModel],
        epi,
        sim_time,
    ) -> List[TrajPoint]:
        if not ref_path:
            return []

        # 1. 规划起点选取
        # 2. NMPC求解（时空联合，非恒定速度）
