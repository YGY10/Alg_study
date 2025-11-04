import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------
# 自注意力
# ---------------------------
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads # 多头注意力头数
        self.head_dim = embed_size // heads # 每个头处理的维度
        # Q：查询  K：键 V：值
        self.qkv = nn.Linear(embed_size, embed_size * 3)  # 利用线性层实现Q，K，V的线性变换
        self.fc_out = nn.Linear(embed_size, embed_size) # 输出层
        
    def forward(self, x):
        N, seq_len, _ = x.shape # 获取输入的形状, N: 批次大小, seq_len: 序列长度， x shape: [N, seq_len, embed_size]
        # 生成qkv shape: [N, seq_len, 3 * embed_size], reshape 后：[N, seq_len, 3, heads, heads_dim]
        qkv = self.qkv(x).reshape(N, seq_len, 3, self.heads, self.head_dim) 
        # 分开q, k, v: 每个的shape为 [N, heads, seq_len, heads_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # 计算注意力分数，根据公式 q k^t / sqrt(h_d)， 将k变为 [1, N, heads, heads_dim, seq_len], 与q相乘；scores shape: [N, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 用 softmax转为概率分布
        attention = F.softmax(scores, dim=-1)
        # 加权求和 shape: [N, heads, seq_len, heads_dim]
        out = torch.matmul(attention, v)
        # transpose之后: [N, seq_len, heads, heads_dim], reshape后： [N, seq_len, heads * heads_dim]
        out = out.transpose(1, 2).reshape(N, seq_len, -1)
        # 线性输出
        out = self.fc_out(out)
        return out
  
# ---------------------------
# Transformer Encoder Block
# ---------------------------
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = SimpleSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ---------------------------
# 位置编码
# ---------------------------
class LightPositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000, scale=0.1):
        super().__init__()
        pe = torch.zeros(max_len, embed_size) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 得到一个shape为(max_Len, 1)的张量，内容为0-max_len-1
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           (-math.log(10000.0) / embed_size)) # 这段其实对应《Attention Is All You Need》中位置编码公式中的括号里面部分的分母：1 / (10000 ^ (2i / dmodel))
        pe[:, 0::2] = torch.sin(position * div_term) * scale # 下面这两行就是位置编码公式
        pe[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe.unsqueeze(0)) # 将位置编码注册为模型参数，这样在训练过程中就不需要更新了, 同时在前面加一个维度，方便和输入的形状匹配
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # 将位置编码添加到输入中

# ---------------------------
# 回归模型
# ---------------------------
class ImprovedTransformerRegressor(nn.Module):
    def __init__(self, seq_len=10, embed_size=32, num_layers=2, heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(1, embed_size),
            nn.LayerNorm(embed_size)
        )
        self.pos_encoding = LightPositionalEncoding(embed_size, scale=0.1)
        self.layers = nn.ModuleList(
            [SimpleTransformerBlock(embed_size, heads) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x).squeeze(-1)

# ---------------------------
# 数据生成
# ---------------------------
def generate_improved_data(num_samples=5000, seq_len=10, max_value=10):
    """
    生成更合理的数据
    目标：输入序列 -> 累积和序列
    """
    # 生成输入序列
    X = torch.rand(num_samples, seq_len) * max_value # 生成 0-max_value范围内的随机数
    
    # 计算累积和
    Y = torch.cumsum(X, dim=1)
    
    # 统一归一化到[0,1]范围
    X_normalized = X / max_value
    Y_normalized = Y / (max_value * seq_len)  # 最大可能的累积和，这样Y最后的范围也被归一化到[0, 1]的范围
    
    print(f"数据统计:")
    print(f"X范围: [{X.min():.2f}, {X.max():.2f}] -> 归一化: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    print(f"Y范围: [{Y.min():.2f}, {Y.max():.2f}] -> 归一化: [{Y_normalized.min():.3f}, {Y_normalized.max():.3f}]")
    
    return X_normalized, Y_normalized, max_value

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # ✅ 修正：使用正确的参数名
    max_value = 5  # 输入数据的最大值
    seq_len = 10
    embed_size = 32
    num_layers = 1 # 注意力层数
    heads = 4
    X, Y, max_value = generate_improved_data(5000, seq_len, max_value)
    train_X, test_X = X[:4500], X[4500:]
    train_Y, test_Y = Y[:4500], Y[4500:]
    
    # 模型和优化器
    model = ImprovedTransformerRegressor(seq_len, embed_size, num_layers, heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("\n开始训练...")
    for epoch in range(1000):
        model.train()
        pred = model(train_X.unsqueeze(-1))
        loss = criterion(pred, train_Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 50 == 0:
            with torch.no_grad():
                test_loss = criterion(model(test_X.unsqueeze(-1)), test_Y)
            print(f"Epoch {epoch:03d} | Train={loss.item():.4f} | Test={test_loss.item():.4f}")

    # ✅ 修正：正确的验证推理
    print("\n" + "="*50)
    print("验证推理结果:")
    
    # 测试序列 [1,0,3,4,0,2,3,1,1,0]
    sample_x_original = torch.tensor([[1,0,3,4,0,2,3,1,1,0]], dtype=torch.float32)
    
    # ✅ 修正：使用相同的归一化方式
    sample_x_normalized = sample_x_original / max_value
    
    with torch.no_grad():
        pred_y_normalized = model(sample_x_normalized.unsqueeze(-1))
    
    # ✅ 修正：反归一化得到真实预测值
    pred_y_denormalized = pred_y_normalized * (max_value * seq_len)  # 乘以最大可能累积和
    
    # ✅ 修正：计算真实累积和用于对比
    true_cumsum = torch.cumsum(sample_x_original, dim=1)
    
    print(f"输入序列: {sample_x_original[0].tolist()}")
    print(f"真实累积和: {[f'{v:.1f}' for v in true_cumsum[0]]}")
    print(f"预测累积和: {[f'{v:.1f}' for v in pred_y_denormalized[0]]}")
    
    # ✅ 修正：计算误差
    mae = torch.mean(torch.abs(pred_y_denormalized - true_cumsum))
    print(f"平均绝对误差(MAE): {mae.item():.4f}")
    
    # ✅ 导出模型（取消注释以使用）
    # model.eval()
    # example_input = torch.randn(1, 10, 1)
    # traced_model = torch.jit.trace(model, example_input)
    # traced_model.save("transformer_regressor.pt")
    # print("\n✅ 模型已导出为 transformer_regressor.pt")