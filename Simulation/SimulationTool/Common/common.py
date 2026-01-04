from dataclasses import dataclass
import math


@dataclass
class TrajPoint:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    t: float = 0.0
    s: float = 0.0


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def generate_sine_traj_from_origin(
    length=40.0,  # 轨迹长度（x 方向）
    A=2.0,  # 振幅（m）
    wavelength=20.0,  # 波长（m）
    v_ref=2.0,  # 参考速度
    ds=0.5,  # 采样间隔（x 方向）
):
    """
    从 (0, 0) 开始的正弦轨迹（Frenet 版）
        y = A * sin(2π / λ * x)

    返回的 TrajPoint 中包含：
        x, y, yaw, v, t, s
    """
    traj = []

    x = 0.0
    t = 0.0
    s = 0.0

    prev_x = None
    prev_y = None

    omega = 2.0 * math.pi / wavelength

    while x <= length:
        y = A * math.sin(omega * x)

        # dy/dx
        dy_dx = A * omega * math.cos(omega * x)

        # 航向角 = 切线方向
        yaw = math.atan2(dy_dx, 1.0)

        # ===== 累计弧长 s =====
        if prev_x is not None:
            dx = x - prev_x
            dy = y - prev_y
            ds_arc = math.hypot(dx, dy)  # 真实弧长
            s += ds_arc

        traj.append(
            TrajPoint(
                x=x,
                y=y,
                yaw=yaw,
                v=v_ref,
                t=t,
                s=s,
            )
        )

        prev_x = x
        prev_y = y

        x += ds
        t += ds / v_ref

    return traj
