import torch
import math
import matplotlib.pyplot as plt

# ====== 定义位置编码函数 ======
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    for i in range(0, d_model, 2):
        pe[:, i] = torch.sin(position[:, 0] * div_term[i // 2])
        if i + 1 < d_model:
            pe[:, i+1] = torch.cos(position[:, 0] * div_term[i // 2])

    return pe

# ====== 可视化 ======
seq_len = 50
d_model = 8

pe = positional_encoding(seq_len, d_model)

plt.figure(figsize=(14, 7))

for i in range(d_model):
    plt.plot(pe[:, i].numpy(), label=f"dim {i}")

plt.title("Positional Encoding 波形（每个维度一条曲线）")
plt.xlabel("Position (pos)")
plt.ylabel("Value (sin/cos)")
plt.legend()
plt.grid(True)
plt.show()
