import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads
        self.qkv = nn.Linear(embed_size, embed_size * 3) # 将输入的最后一个维度乘以3，生成Q/K/V
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        N, seq_len, _ = x.shape
        print(f"输入 x: {x.shape}")  # (N, seq_len, embed_size)

        # 1️⃣ 一次线性变换，生成Q/K/V
        qkv = self.qkv(x)
        print(f"线性层输出 qkv: {qkv.shape}")  # (N, seq_len, 3*embed_size)

        # 2️⃣ reshape 为 (N, seq_len, 3, heads, head_dim)
        qkv = qkv.reshape(N, seq_len, 3, self.heads, self.head_dim)
        print(f"reshape 后 qkv: {qkv.shape}")  # (N, seq_len, 3, heads, head_dim)

        # 3️⃣ permute 调整维度顺序
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")  # (N, heads, seq_len, head_dim)

        # 4️⃣ 计算注意力分数 Q × K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # 批量矩阵乘法
        print(f"注意力得分 scores: {scores.shape}")  # (N, heads, seq_len, seq_len)

        # 5️⃣ softmax 得到注意力权重
        attention = F.softmax(scores, dim=-1)
        print(f"注意力权重 attention: {attention.shape}")  # (N, heads, seq_len, seq_len)

        # 6️⃣ 加权求和 Attention × V
        out = torch.matmul(attention, v)
        print(f"加权求和 out: {out.shape}")  # (N, heads, seq_len, head_dim)

        # 7. 拼接多头输出
        out = out.transpose(1, 2).reshape(N, seq_len, -1)
        print(f"拼接后 out: {out.shape}")  # (N, seq_len, embed_size)

        # 8. 输出线性层
        out = self.fc_out(out)
        print(f"输出层 out: {out.shape}")  # (N, seq_len, embed_size)
        print("-" * 60)
        return out


# 一次注意力机制前向传播
embed_size = 3
heads = 3
seq_len = 2
batch_size = 1

x = torch.randn(batch_size, seq_len, embed_size)
print(f"输入 x 的全部内容: {x.tolist()}")
model = SimpleSelfAttention(embed_size, heads)

out = model(x)

# test_data = [((1,2,3), (4,5,6)), ((7,8,9), (10,11,12))]
# test_data = torch.tensor(test_data)
# print(f"test_data shape: {test_data.shape}")
# print(f"test_data: {test_data.tolist()}")
# test_data2 = test_data.reshape(2, -1)
# print(f"test_data2 shape: {test_data2.shape}")
# print(f"test_data2: {test_data2.tolist()}")

