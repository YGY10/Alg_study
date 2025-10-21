import numpy as np
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def generate_bezier_path(resolution, p_start, p_end, ctrl_ratio=0.01):
    """
    生成三阶Bezier路径
    p_start: dict {x, y, yaw}
    p_end: dict {x, y, yaw}
    """
    if p_end["x"] - 0.1 <= p_start["x"]:
        raise ValueError("end.x 必须大于 start.x，否则无法生成路径")

    # yaw方向平移一个点
    def pt_trans_yaw(p, yaw, dist):
        return {
            "x": p["x"] + dist * np.cos(yaw),
            "y": p["y"] + dist * np.sin(yaw)
        }

    delta_s = distance(p_start["x"], p_start["y"], p_end["x"], p_end["y"])
    c1 = pt_trans_yaw(p_start, p_start["yaw"], delta_s * ctrl_ratio)
    c2 = pt_trans_yaw(p_end, p_end["yaw"], -delta_s * ctrl_ratio)

    # Bezier 控制点
    P0 = np.array([p_start["x"], p_start["y"]])
    P1 = np.array([c1["x"], c1["y"]])
    P2 = np.array([c2["x"], c2["y"]])
    P3 = np.array([p_end["x"], p_end["y"]])

    total_num = int(delta_s / resolution) + 1
    path = []

    for i in range(1, total_num + 1):
        t = i / total_num
        # 三阶贝塞尔公式
        point = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
        path.append(point)

    return np.array(path), [P0, P1, P2, P3]

# 测试参数
p_start = {"x": 0.0, "y": 0.0, "yaw": np.deg2rad(0)}     # 起点 (0,0) 朝向 0度
p_end   = {"x": 50.0, "y": 20.0, "yaw": np.deg2rad(20)}  # 终点 (50,20) 朝向 20度
resolution = 1.0

# 生成曲线
path, control_pts = generate_bezier_path(resolution, p_start, p_end, ctrl_ratio=0.01)

# 画图
plt.figure(figsize=(8,5))
plt.plot(path[:,0], path[:,1], 'b-', label="Bezier Path")
plt.plot([p[0] for p in control_pts], [p[1] for p in control_pts], 'ro--', label="Control Points")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bezier Path Example")
plt.grid(True)
plt.axis("equal")
plt.show()
