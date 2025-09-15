import matplotlib.pyplot as plt
import numpy as np

class DS_DotProduct:  # 你的点积版
    def __init__(self):
        self.ds_vec = []

    def process(self, bba_vec):
        K = 0.0
        if not self.ds_vec:
            self.ds_vec = bba_vec.copy()
            return
        
        for i in range(len(bba_vec)):
            K += bba_vec[i] * self.ds_vec[i]

        for i in range(len(bba_vec)):
            self.ds_vec[i] = bba_vec[i] * self.ds_vec[i] / (K + 1e-6)


class DS_Strict:  # 严格 DS 规则
    def __init__(self):
        self.ds_vec = []

    def process(self, bba_vec):
        if not self.ds_vec:
            self.ds_vec = bba_vec.copy()
            return

        a1, b1 = self.ds_vec
        a2, b2 = bba_vec

        # 计算冲突度 K
        K = a1 * b2 + b1 * a2
        norm = (1 - K) if abs(1 - K) > 1e-6 else 1e-6

        # 融合结果
        A = (a1 * a2) / norm
        B = (b1 * b2) / norm

        self.ds_vec = [A, B]


# -------------------------------
# 测试数据
# -------------------------------
bba_list = [
    [0.9, 0.1],  # 偏向 A
    [0.8, 0.2],  # 仍然偏向 A
    [0.2, 0.8],  # 转向支持 B
    [0.5, 0.5]   # 模糊证据
]

dot_model = DS_DotProduct()
strict_model = DS_Strict()

results_dot = []
results_strict = []

for bba in bba_list:
    dot_model.process(bba.copy())
    strict_model.process(bba.copy())
    results_dot.append(dot_model.ds_vec.copy())
    results_strict.append(strict_model.ds_vec.copy())

# -------------------------------
# 打印结果
# -------------------------------
print("点积法结果：")
for i, res in enumerate(results_dot):
    print(f"Step {i+1}: {res}")

print("\n严格 DS 结果：")
for i, res in enumerate(results_strict):
    print(f"Step {i+1}: {res}")

# -------------------------------
# 可视化
# -------------------------------
results_dot = np.array(results_dot)
results_strict = np.array(results_strict)
steps = np.arange(1, len(bba_list)+1)

plt.figure(figsize=(10,5))

# 点积法
plt.subplot(1,2,1)
plt.plot(steps, results_dot[:,0], marker='o', label='Belief in A')
plt.plot(steps, results_dot[:,1], marker='s', label='Belief in B')
plt.title("Dot-Product Fusion")
plt.xlabel("Step")
plt.ylabel("Belief value")
plt.legend()
plt.grid(True)

# 严格 DS
plt.subplot(1,2,2)
plt.plot(steps, results_strict[:,0], marker='o', label='Belief in A')
plt.plot(steps, results_strict[:,1], marker='s', label='Belief in B')
plt.title("Strict DS Fusion")
plt.xlabel("Step")
plt.ylabel("Belief value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
