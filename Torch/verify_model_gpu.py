import torch

model = torch.jit.load("transformer_regressor.pt", map_location="cuda")
x = torch.randn(1, 10, 1, device="cuda")
y = model(x)
print("输出:", y)
print("设备:", next(model.parameters()).device)
