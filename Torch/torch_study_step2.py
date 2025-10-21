import  torch
import  torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(3, 4)

# 定义Q K V的线性变换
W_Q = torch.randn(4, 4)
W_K = torch.randn(4, 4)
W_V = torch.randn(4, 4)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# 计算注意力
d_k = Q.shape[-1]
scores = Q @ K.T / d_k ** 0.5
attn = F.softmax(scores, dim=-1)
output = attn @ V

print("Input X:\n", X)
print("Attention Weights:\n", attn)
print("Output:\n", output)