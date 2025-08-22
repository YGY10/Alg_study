import numpy as np
import matplotlib.pyplot as plt


# 定义卡尔曼滤波器类（使用修正后的版本）
class KalmanFilter:
    def __init__(self, A, B, x0, q, r, H, dt=0.1, p0=1.0):
        self.dt = dt
        self.A = A
        self.B = B
        self.x = x0  # 初始状态 [位置, 速度]
        self.Q = q * np.eye(x0.shape[0])
        self.H = H
        self.R = r * np.eye(self.H.shape[0])
        self.P = p0 * np.eye(x0.shape[0])
    
    def predict(self, u=None):
        self.x = np.dot(self.A, self.x)
        if u is not None:
            self.x += np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x.copy()
    
    def update(self, measure):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))
        self.x += np.dot(K, measure - np.dot(self.H, self.x))
        self.P = np.dot((np.eye(self.x.shape[0]) - np.dot(K, self.H)), self.P)
        return self.x.copy()

# 生成测试数据
def generate_data(dt=0.1, T=10, a=0.5, x0_true=0, v0_true=0):
    """
    生成真实轨迹和带噪声的观测值
    dt: 时间间隔
    T: 总时间
    a: 加速度（控制输入）
    """
    t = np.arange(0, T, dt)  # 时间序列
    n = len(t)
    
    # 真实状态：位置 = x0 + v0*t + 0.5*a*t²；速度 = v0 + a*t
    x_true = x0_true + v0_true * t + 0.5 * a * t**2
    v_true = v0_true + a * t
    
    # 生成带噪声的观测值（仅观测位置，添加高斯噪声）
    obs_noise = np.random.normal(0, np.sqrt(1.0), n)  # 观测噪声（方差1.0）
    z = x_true + obs_noise  # 观测位置
    
    return t, x_true, v_true, z, a

# 主测试函数
def test_kalman_filter():
    dt = 0.1  # 时间间隔
    # 1. 定义系统矩阵
    A = np.array([[1, dt],
                  [0, 1]])  # 状态转移矩阵
    B = np.array([[0.5 * dt**2],
                  [dt]])  # 控制矩阵（输入为加速度）
    H = np.array([[1, 0]])  # 观测矩阵（仅观测位置）
    
    # 2. 初始化滤波器参数
    x0 = np.array([[0], [0]])  # 初始状态估计（假设初始位置和速度为0）
    q = 0.05  # 过程噪声系数
    r = 1.0  # 观测噪声系数
    kf = KalmanFilter(A, B, x0, q, r, H, dt)
    
    # 3. 生成真实数据和观测值
    t, x_true, v_true, z, a = generate_data(dt=dt)
    n = len(t)
    
    # 4. 运行卡尔曼滤波
    x_est = []  # 估计位置
    v_est = []  # 估计速度
    for i in range(n):
        # 预测（控制输入为加速度a）
        kf.predict(u=np.array([[a]]))
        # 更新（使用当前观测值）
        x = kf.update(np.array([[z[i]]]))
        x_est.append(x[0, 0])
        v_est.append(x[1, 0])
    
    # 5. 绘图对比
    plt.figure(figsize=(12, 6))
    
    # 位置对比
    plt.subplot(2, 1, 1)
    plt.plot(t, x_true, label='True position', color='black')
    plt.plot(t, z, label='Measurement(with noise)', color='gray', alpha=0.5)
    plt.plot(t, x_est, label='Kalman estimate', color='red')
    plt.xlabel('Time(s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    
    # 速度对比
    plt.subplot(2, 1, 2)
    plt.plot(t, v_true, label='True velocity', color='black')
    plt.plot(t, v_est, label='Kalman estimate', color='blue')
    plt.xlabel('Time(s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行测试
if __name__ == "__main__":
    test_kalman_filter()
    