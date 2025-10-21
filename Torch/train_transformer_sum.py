import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder

# 数据生成（方案一：全局归一化）
def generate_data(num_samples=1000, seq_len=10):
    X = torch.randint(1, 10, (num_samples, seq_len)).float()
    Y = torch.cumsum(X, dim=1)
    Y_max = Y.max()  # ✅ 保存全局最大值
    Y = Y / Y_max    # ✅ 用统一归一化系数
    return X / 10.0, Y, Y_max

# 生成数据
X, Y, Y_max = generate_data(2000)
train_X, train_Y = X[:1800], Y[:1800]
test_X, test_Y = X[1800:], Y[1800:]

# 模型定义
class TransformerRegressor(nn.Module):
    def __init__(self, embed_size=32, num_layers=2, heads=2, forward_expansion=4):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_size)
        self.encoder = TransformerEncoder(embed_size, num_layers, heads, forward_expansion)
        self.output_proj = nn.Linear(embed_size, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x.squeeze(-1)

# 初始化
model = TransformerRegressor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练
for epoch in range(500):
    optimizer.zero_grad()
    x = train_X.unsqueeze(-1)
    y = train_Y
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            test_pred = model(test_X.unsqueeze(-1))
            test_loss = criterion(test_pred, test_Y)
        print(f"Epoch {epoch:03d} | train_loss={loss.item():.4f} | test_loss={test_loss.item():.4f}")

# 测试（⚠️ 用相同的归一化系数 Y_max）
with torch.no_grad():
    sample_x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
    pred_y = model(sample_x.unsqueeze(-1))
    true_y = torch.cumsum(sample_x, dim=1) / Y_max  # ✅ 用训练时相同归一化
print("输入：", sample_x)
print("预测输出：", pred_y)
print("真实输出：", true_y)
