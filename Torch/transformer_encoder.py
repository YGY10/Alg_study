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
        self.heads = heads
        self.head_dim = embed_size // heads
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        N, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(N, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, N, heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(N, seq_len, -1)
        out = self.fc_out(out)
        return out  # ⚠️ TorchScript 不支持返回 tuple (去掉注意力输出)


# ---------------------------
# Transformer Block
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
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------
# 回归模型（用于导出）
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
            nn.Linear(16, 1)
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
def generate_improved_data(num_samples=5000, seq_len=10):
    X = torch.rand(num_samples, seq_len) * 15 + 1
    Y = torch.cumsum(X, dim=1) / 10.0
    X = X / 15.0
    return X, Y


# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    X, Y = generate_improved_data(5000, 10)
    train_X, test_X = X[:4500], X[4500:]
    train_Y, test_Y = Y[:4500], Y[4500:]

    model = ImprovedTransformerRegressor(seq_len=10, embed_size=32, num_layers=2, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("🚀 开始训练")
    for epoch in range(300):
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

    # ✅ 保存 TorchScript 模型（用于 C++）
    # ✅ 保存 TorchScript 模型（支持 GPU）
    model.eval().to("cuda")

    # ⚠️ 注意：样例输入也必须在 GPU 上
    example_input = torch.randn(1, 10, 1, device="cuda")

    # 跟踪（trace）模型
    traced = torch.jit.trace(model, example_input)
    traced.save("transformer_regressor_gpu.pt")

    print("\n✅ 模型已导出: transformer_regressor_gpu.pt (支持CUDA)")


    # ✅ 简单验证推理
    sample_x = torch.tensor([[1,0,3,4,0,2,7,8,1,0]], dtype=torch.float32) / 15.0
    with torch.no_grad():
        pred_y = model(sample_x.unsqueeze(-1))
    print("输入:", sample_x[0].tolist())
    print("预测输出:", [f"{v:.2f}" for v in pred_y[0]])
