import numpy as np
import matplotlib.pyplot as plt

def affine_transformation(x, y, Matrix):
    # 构造齐次坐标
    pt = np.array([x, y, 1.0])
    # 应用变换
    # Matrix = { cos t, -sin t, t x;
    #            sin t,  cos t, t y; 
    #                0,      0,    1;}
    pt_trans = Matrix @ pt
    return pt_trans[0], pt_trans[1]

# 旋转 + 平移矩阵
theta = np.deg2rad(90)
cos_t, sin_t = np.cos(theta), np.sin(theta)

M = np.array([
    [cos_t, -sin_t, 0.0],
    [sin_t,  cos_t, 1.0],
    [0,      0,     1.0]
])

# 原始点
x, y = 1, 0
x_t, y_t = affine_transformation(x, y, M)

print("原始点:", (x, y))
print("变换后:", (x_t, y_t))

# 可视化
plt.figure()
plt.scatter(x, y, c='blue', label="original")
plt.scatter(x_t, y_t, c='red', label="transformed")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()
