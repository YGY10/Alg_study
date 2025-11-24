
# 任务目标：判断序列是否单调递增
# 给定长度为 10 的序列（例如 [0.1, 0.3, 0.5, 0.7, ...]），
# 模型需要输出：
# 1 → 单调递增
# 0 → 否则


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    def forward(self, input):
        # 获取输入的形状
        N, seq_len, _ = input.shape
        # 求qkv
        qkv = self.qkv(input)
        # 取出q k v 
        qkv = qkv.reshape(N, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 转化为概率
        attention = torch.softmax(scores, dim=-1)
        # 加权 [N, heads, seq_len, heads_dim]
        out = torch.matmul(attention, v) 
        # 转回原始形状
        out.reshape(1,2).reshape(N, seq_len, -1)
        # 输出
        out = self.fc_out(out)
        return out
    
# Encoder
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size) 
        self.feedfoward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )
    
    def forward(self, input):
        attention_out  = self.attention(input)
        output = self.norm1(input + attention_out)
        ff_out = self.feedfoward(output)
        output = self.norm2(output + ff_out)
        return output
    
# 位置编码
class PositionalEncoderBlock(nn.Module):
    def __init__(self, embed_size, max_len, scale):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: [max_len, 1]
        # 1 / ((10000 / e_s) ^ (2i))
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0 / embed_size)))
        pe[:, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, input):
        return input + self.pe[:, :input.size(1)]

# 分类模型
class TransformerClassifier(nn.Module):