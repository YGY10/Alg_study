import numpy as np
import matplotlib.pyplot as plt

# =========================
# 场景参数
# =========================
dt = 0.05
T = 6.0
steps = int(T / dt)

d_safe = 2.0

# 前车（匀速）
xo0 = 20.0
vo = 5.0

# 自车初始状态
x0 = 0.0
v0 = 15.0  # 明显比前车快

# HOCBF 参数
k1 = 2.0
k0 = 1.0

# 物理与舒适约束
a_phys_min = -8.0  # 物理最大减速度（刹车上限）
a_comfort_min = -2.0  # 舒适减速度（乘坐体验上限）


# 工具函数：单步更新前车位置
def lead_pos(step):
    t = step * dt
    return xo0 + vo * t


# =========================
# 1. Naive：同时硬要安全 + 舒适（无 PA/RA）
# =========================
def simulate_naive():
    x, v = x0, v0
    xs, vs, hs, as_ = [], [], [], []

    for k in range(steps):
        xo = lead_pos(k)
        h = xo - x - d_safe
        dh = vo - v

        # HOCBF 上界：a <= a_safe_max
        a_safe_max = k1 * dh + k0 * h

        # 想同时满足：
        #   a >= a_comfort_min  （舒适）
        #   a >= a_phys_min     （物理）
        #   a <= a_safe_max     （安全）
        a_low = max(a_phys_min, a_comfort_min)
        a_high = a_safe_max

        if a_low <= a_high:
            # 约束可行，选最强刹车（区间下界）
            a = a_low
        else:
            # 约束冲突：区间为空
            # Naive 策略：硬上舒适刹车（安全会被破坏）
            a = a_comfort_min

        # 状态更新
        v += a * dt
        v = max(v, 0.0)
        x += v * dt

        xs.append(x)
        vs.append(v)
        hs.append(h)
        as_.append(a)

    return xs, vs, hs, as_


# =========================
# 2. PA-HOCBF：安全优先
# =========================
def simulate_pa_hocbf():
    x, v = x0, v0
    xs, vs, hs, as_ = [], [], [], []

    for k in range(steps):
        xo = lead_pos(k)
        h = xo - x - d_safe
        dh = vo - v

        a_safe_max = k1 * dh + k0 * h

        # 先尝试同时满足安全 + 舒适
        a_low = max(a_phys_min, a_comfort_min)
        a_high = a_safe_max

        if a_low <= a_high:
            # 有解：兼顾安全与舒适
            a = a_low
        else:
            # 无解：PA 策略 → 丢掉舒适约束，只保证安全 + 物理上限
            # 只要求：a >= a_phys_min, a <= a_safe_max
            a_phys_low = a_phys_min
            if a_phys_low <= a_safe_max:
                a = a_phys_low
            else:
                # 连物理极限也无法满足 HOCBF（h 已经太小、v 太大）
                # 这里就用物理极限，安全不可避免会被破坏
                a = a_phys_min

        v += a * dt
        v = max(v, 0.0)
        x += v * dt

        xs.append(x)
        vs.append(v)
        hs.append(h)
        as_.append(a)

    return xs, vs, hs, as_


# =========================
# 3. RA-HOCBF：舒适优先，安全可放松（带松弛）
# =========================
def simulate_ra_hocbf():
    x, v = x0, v0
    xs, vs, hs, as_, slacks = [], [], [], [], []

    for k in range(steps):
        xo = lead_pos(k)
        h = xo - x - d_safe
        dh = vo - v

        # RA：始终按舒适刹车
        a = a_comfort_min
        a = max(a, a_phys_min)  # 不能超过物理上限

        # HOCBF 约束加松弛：
        # -a + k1*dh + k0*h >= -delta, delta >= 0
        lhs = -a + k1 * dh + k0 * h
        delta = max(0.0, -lhs)  # 违反多少就需要多少 slack

        v += a * dt
        v = max(v, 0.0)
        x += v * dt

        xs.append(x)
        vs.append(v)
        hs.append(h)
        as_.append(a)
        slacks.append(delta)

    return xs, vs, hs, as_, slacks


# =========================
# 运行仿真
# =========================
x_naive, v_naive, h_naive, a_naive = simulate_naive()
x_pa, v_pa, h_pa, a_pa = simulate_pa_hocbf()
x_ra, v_ra, h_ra, a_ra, slack_ra = simulate_ra_hocbf()

time = np.arange(steps) * dt

# =========================
# 画图
# =========================
plt.figure(figsize=(12, 10))

# 1) 距离裕度 h
plt.subplot(4, 1, 1)
plt.plot(time, h_naive, label="Naive (no PA/RA)")
plt.plot(time, h_pa, label="PA-HOCBF (safety-priority)")
plt.plot(time, h_ra, label="RA-HOCBF (safety relaxed)")
plt.axhline(0, color="k", linestyle="--")
plt.ylabel("h (distance margin)")
plt.legend()
plt.grid()

# 2) 速度
plt.subplot(4, 1, 2)
plt.plot(time, v_naive, label="Naive")
plt.plot(time, v_pa, label="PA-HOCBF")
plt.plot(time, v_ra, label="RA-HOCBF")
plt.ylabel("Velocity")
plt.legend()
plt.grid()

# 3) 加速度
plt.subplot(4, 1, 3)
plt.plot(time, a_naive, label="Naive")
plt.plot(time, a_pa, label="PA-HOCBF")
plt.plot(time, a_ra, label="RA-HOCBF")
plt.axhline(a_comfort_min, color="gray", linestyle=":", label="Comfort limit")
plt.axhline(a_phys_min, color="k", linestyle="--", label="Physical limit")
plt.ylabel("Acceleration")
plt.legend()
plt.grid()

# 4) RA-HOCBF 松弛大小（安全违反程度）
plt.subplot(4, 1, 4)
plt.plot(time, slack_ra, label="Slack (RA-HOCBF)")
plt.ylabel("Slack δ")
plt.xlabel("Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
