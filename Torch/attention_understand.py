import numpy as np

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

print("\n=== Step 0: 输入序列（3 个 token，每个维度为 4） ===")
x = np.array([
    [1.0, 0.0, 1.0, 0.0],   # token A
    [0.0, 2.0, 0.0, 1.0],   # token B
    [1.0, 1.0, 1.0, 1.0],   # token C
])
print(x)

# 随机初始化 Q,K,V 的线性变换矩阵
W_Q = np.random.randn(4, 4)
W_K = np.random.randn(4, 4)
W_V = np.random.randn(4, 4)

print("\n=== Step 1: 线性变换，生成 Q,K,V ===")
Q = x @ W_Q
K = x @ W_K
V = x @ W_V
print("Q =\n", Q)
print("K =\n", K)
print("V =\n", V)

print("\n=== Step 2: 注意力分数（score = Q·K^T） ===")
scores = Q @ K.T
print(scores)

print("\n=== Step 3: 对每个 token 做 softmax，得到注意力权重 ===")
attn_weights = np.array([softmax(row) for row in scores])
print(attn_weights)

print("\n=== Step 4: 加权求和输出（attention output = weights·V） ===")
out = attn_weights @ V
print(out)

print("\n=== Step 5: 解释 ===")
print("""
• Q 表示“我要关注哪些模式”
• K 表示“每个 token 具有什么模式”
• Q·Kᵀ 表示匹配程度（越大越相关）
• softmax 转成概率（注意力权重）
• 权重 × V 实现信息融合

你现在看到的 out 就是 self-attention 的输出。
""")
