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
theta2 = np.deg2rad(-90)
cos_t, sin_t = np.cos(theta), np.sin(theta)
cos_t2, sin_t2 = np.cos(theta2), np.sin(theta2)

M = np.array([
    [cos_t, -sin_t, 0.0],
    [sin_t,  cos_t, 0.0],
    [0,      0,     1.0]
])

M2 = np.array([
    [cos_t2, -sin_t2, 0.0],
    [sin_t2,  cos_t2, 0.0],
    [0,      0,     1.0]
])

# 原始点
x, y = 1, 0
x_t, y_t = affine_transformation(x, y, M)
x_t2, y_t2 = affine_transformation(x_t, y_t, M2)

print("原始点:", (x, y))
print("变换后:", (x_t, y_t))
print("变换后:", (x_t2, y_t2))

# 可视化
plt.figure()
plt.scatter(x, y, c='blue', label="original")
plt.scatter(x_t, y_t, c='red', label="transformed")
plt.scatter(x_t2, y_t2, c='green', label="transformed2")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()
