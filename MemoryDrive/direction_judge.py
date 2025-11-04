import math
import matplotlib.pyplot as plt


def direction_judge(path, eps=1e-6):
    """
    判断曲线是左弯、右弯还是直线
    path: [(x0, y0), (x1, y1), ...]
    eps: 判定阈值
    """
    if len(path) < 3:
        return "unknown"

    z_sum = 0.0
    for i in range(len(path) - 2):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        x3, y3 = path[i + 2]
        # 向量叉积 z 分量
        v1x, v1y = x2 - x1, y2 - y1
        v2x, v2y = x3 - x2, y3 - y2
        cross_z = v1x * v2y - v1y * v2x
        z_sum += cross_z

    avg_z = z_sum / (len(path) - 2)

    if avg_z > eps:
        return "左弯"
    elif avg_z < -eps:
        return "右弯"
    else:
        return "直线"


mem_path = [
    (0, 0), (1, 0.5), (2, 0.7), (3, 0.7), (4, 0.6)
]

# 判断结果
result = direction_judge(mem_path)
print("曲线方向判断结果：", result)

# 绘制曲线
xs, ys = zip(*mem_path)
plt.plot(xs, ys, marker='o', label=f"Path ({result})")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("Curve Direction Detection")
plt.show()
