import math
from SimulationTool.Common.common import normalize_angle


class VehicleKModel:
    # 车辆运动学模型
    # 状态： x, y, yaw(rad), v(m/s)
    # 控制量：a(m/s^2), delta(rad)前轮转角
    # 参数：wheelbase轴距， dt离散仿真时间步长
    def __init__(
        self,
        x=0.0,
        y=0.0,
        yaw=0.0,
        v=0.0,
        wheelbase=2.7,
        dt=0.05,
        length=4.5,
        width=1.8,
    ):
        # 初始化
        self.x = x
        self.y = y
        self.yaw = normalize_angle(yaw)
        self.v = v

        # 系统参数
        self.wheelbase = wheelbase
        self.dt = dt

        # 外型参数
        self.length = length
        self.width = width

        # 边界限制
        self.max_steer = math.radians(35)  # 最大转角
        self.max_speed = 35.0  # 上限速度
        self.min_speed = -5.0  # 下限速度

    def update(self, a, delta):
        delta = max(-self.max_steer, min(self.max_steer, delta))
        # 更新车辆状态
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.wheelbase * math.tan(delta) * self.dt
        # 归一化到-pi, pi中
        self.yaw = normalize_angle(self.yaw)
        self.v += a * self.dt
        self.v = max(self.min_speed, min(self.max_speed, self.v))

    def state(self):
        # 返回当前车辆状态（元组）
        return self.x, self.y, self.yaw, self.v

    def set_state(self, x, y, yaw, v=0.0):
        # 手动设置车辆状态
        self.x = x
        self.y = y
        self.yaw = normalize_angle(yaw)
        self.v = v

    def corners(self):
        # 返回车辆四个角点，[前右，前左，后左，后右]
        L = self.length
        W = self.width
        yaw = self.yaw

        # 车辆坐标系下四角点
        local_corners = [
            (+L / 2, -W / 2),  # 前右
            (+L / 2, +W / 2),  # 前左
            (-L / 2, +W / 2),  # 后左
            (-L / 2, -W / 2),  # 后右
        ]

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        world_corners = []
        for cx, cy in local_corners:
            wx = self.x + cx * cos_y - cy * sin_y
            wy = self.y + cx * sin_y + cy * cos_y
            world_corners.append((wx, wy))

        return world_corners
