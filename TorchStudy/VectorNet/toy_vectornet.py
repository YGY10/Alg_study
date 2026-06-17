import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use("Agg")  # 不使用图形窗口，只保存图片
import matplotlib.pyplot as plt

# ============================================================
# 1. Toy Dataset
# ============================================================


class ToyVectorNetDataset(Dataset):
    """
    生成一个极简轨迹预测数据集。

    场景：
      - 三条车道：y = -3.5, 0.0, 3.5
      - 目标车在其中一条车道上向前行驶
      - 输入：目标车历史轨迹 + 三条车道线
      - 输出：目标车未来轨迹

    目标：
      让你理解 VectorNet 的输入组织方式。
    """

    def __init__(
        self,
        num_samples=2000,
        num_segments=8,
        future_steps=5,
        dt=0.5,
        seed=0,
    ):
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.future_steps = future_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.lane_y_values = [-3.5, 0.0, 3.5]

    def __len__(self):
        return self.num_samples

    def polyline_to_vectors(self, pts, type_onehot):
        """
        把一条 polyline 点序列转换成 vector segment。

        pts:
          shape = [N, 2]

        输出：
          vectors: [num_segments, 6]
          mask:    [num_segments]
        """
        vectors = []

        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]

            vectors.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    type_onehot[0],
                    type_onehot[1],
                ]
            )

        feat_dim = 6
        out = np.zeros((self.num_segments, feat_dim), dtype=np.float32)
        mask = np.zeros((self.num_segments,), dtype=np.float32)

        n = min(len(vectors), self.num_segments)
        if n > 0:
            out[:n] = np.asarray(vectors[:n], dtype=np.float32)
            mask[:n] = 1.0

        return out, mask

    def __getitem__(self, idx):
        # 随机选择目标车在哪条车道
        lane_idx = self.rng.integers(0, 3)
        lane_y = self.lane_y_values[lane_idx]

        # 随机速度
        speed = self.rng.uniform(4.0, 10.0)

        # 目标车历史轨迹：当前点在 x=0，过去点在负 x
        # 速度越快，历史点间距越大
        history_steps = 6
        hist_pts = []

        for i in range(history_steps):
            t = -(history_steps - 1 - i) * self.dt
            x = speed * t
            y = lane_y + self.rng.normal(0.0, 0.05)
            hist_pts.append([x, y])

        hist_pts = np.asarray(hist_pts, dtype=np.float32)

        polylines = []
        masks = []

        # polyline 0：目标车历史轨迹
        # type = [is_agent, is_lane]
        p, m = self.polyline_to_vectors(hist_pts, [1.0, 0.0])
        polylines.append(p)
        masks.append(m)

        # polyline 1~3：三条车道线
        lane_x = np.linspace(-20.0, 40.0, self.num_segments + 1)

        for ly in self.lane_y_values:
            lane_pts = np.stack(
                [lane_x, np.full_like(lane_x, ly)],
                axis=1,
            ).astype(np.float32)

            p, m = self.polyline_to_vectors(lane_pts, [0.0, 1.0])
            polylines.append(p)
            masks.append(m)

        # 未来轨迹真值
        future = []
        for k in range(1, self.future_steps + 1):
            t = k * self.dt
            x = speed * t
            y = lane_y
            future.append([x, y])

        future = np.asarray(future, dtype=np.float32)

        polylines = np.stack(polylines, axis=0)  # [P, V, F]
        masks = np.stack(masks, axis=0)  # [P, V]

        return (
            torch.tensor(polylines, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(future, dtype=torch.float32),
        )


# ============================================================
# 2. SubGraph Encoder
# ============================================================


class SubGraphEncoder(nn.Module):
    """
    VectorNet 的第一层核心：
    对每条 polyline 内部的 vector segments 做编码。

    输入：
      x:    [B, P, V, F]
      mask: [B, P, V]

    输出：
      polyline_feature: [B, P, D]
    """

    def __init__(self, in_dim=6, hidden_dim=64):
        super().__init__()

        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def masked_max_pool(self, h, mask):
        """
        h:
          [B, P, V, D]

        mask:
          [B, P, V]

        return:
          [B, P, D]
        """
        h = h.masked_fill(mask[..., None] == 0.0, -1e9)
        pooled = h.max(dim=2).values
        pooled = torch.where(
            pooled < -1e8,
            torch.zeros_like(pooled),
            pooled,
        )
        return pooled

    def forward(self, x, mask):
        # 每个 vector segment 先过 MLP
        h = self.local_mlp(x)  # [B, P, V, D]

        # 每条 polyline 内部 max pooling
        polyline_global = self.masked_max_pool(h, mask)  # [B, P, D]

        # 把整条 polyline 的全局特征拼回每个 vector 上
        B, P, V, D = h.shape
        global_expand = polyline_global[:, :, None, :].expand(B, P, V, D)

        h_fused = torch.cat([h, global_expand], dim=-1)

        h_fused = self.fusion_mlp(h_fused)

        # 再 pool 一次，得到每条 polyline 的 embedding
        polyline_feature = self.masked_max_pool(h_fused, mask)

        return polyline_feature


# ============================================================
# 3. Global Graph Encoder
# ============================================================


class GlobalGraphEncoder(nn.Module):
    """
    VectorNet 的第二层核心：
    对所有 polyline 之间做全局交互。

    这里为了好理解，用一个简单的 fully-connected graph 聚合：
      - 每条 polyline 先过 MLP
      - 对所有 polyline 做 max pooling 得到 scene feature
      - 再把 scene feature 拼回每条 polyline
      - 输出交互后的 polyline feature

    这不是完整论文实现，但足够理解 VectorNet 的思想。
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, polyline_feature):
        """
        polyline_feature:
          [B, P, D]
        """
        h = self.node_mlp(polyline_feature)

        # 全局场景特征
        scene_global = h.max(dim=1).values  # [B, D]

        B, P, D = h.shape
        scene_expand = scene_global[:, None, :].expand(B, P, D)

        h = torch.cat([h, scene_expand], dim=-1)
        h = self.fusion_mlp(h)

        return h


# ============================================================
# 4. Toy VectorNet
# ============================================================


class ToyVectorNet(nn.Module):
    """
    极简 VectorNet：

      vector segments
          ↓
      SubGraph Encoder
          ↓
      Polyline Embeddings
          ↓
      Global Graph Encoder
          ↓
      Target Agent Feature
          ↓
      Decoder
          ↓
      Future Trajectory
    """

    def __init__(self, in_dim=6, hidden_dim=64, future_steps=5):
        super().__init__()

        self.future_steps = future_steps

        self.subgraph = SubGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
        )

        self.global_graph = GlobalGraphEncoder(
            hidden_dim=hidden_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, x, mask):
        """
        x:
          [B, P, V, F]

        mask:
          [B, P, V]
        """
        polyline_feature = self.subgraph(x, mask)  # [B, P, D]
        global_feature = self.global_graph(polyline_feature)  # [B, P, D]

        # polyline 0 是目标车历史轨迹
        target_feature = global_feature[:, 0, :]  # [B, D]

        pred = self.decoder(target_feature)
        pred = pred.view(-1, self.future_steps, 2)

        return pred


# ============================================================
# 5. Train
# ============================================================


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ToyVectorNetDataset(
        num_samples=3000,
        num_segments=8,
        future_steps=5,
        dt=0.5,
        seed=0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    model = ToyVectorNet(
        in_dim=6,
        hidden_dim=64,
        future_steps=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total_loss = 0.0

        for x, mask, y in dataloader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            pred = model(x, mask)

            loss = ((pred - y) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch={epoch:02d}, loss={avg_loss:.6f}")

    return model, dataset, device


# ============================================================
# 6. Visualize one sample
# ============================================================


def visualize_one_sample(model, dataset, device):
    model.eval()

    x, mask, y = dataset[0]

    with torch.no_grad():
        pred = (
            model(
                x[None].to(device),
                mask[None].to(device),
            )[0]
            .cpu()
            .numpy()
        )

    y = y.numpy()
    x_np = x.numpy()
    mask_np = mask.numpy()

    plt.figure(figsize=(8, 6))

    # 画输入 polyline
    num_polylines = x_np.shape[0]

    for pidx in range(num_polylines):
        for vidx in range(x_np.shape[1]):
            if mask_np[pidx, vidx] < 0.5:
                continue

            x0, y0, x1, y1, is_agent, is_lane = x_np[pidx, vidx]

            if is_agent > 0.5:
                label = "target history" if vidx == 0 else None
                linewidth = 3
            else:
                label = "lane centerline" if pidx == 1 and vidx == 0 else None
                linewidth = 1

            plt.plot(
                [x0, x1],
                [y0, y1],
                "-o",
                linewidth=linewidth,
                markersize=3,
                label=label,
            )

    # 画 GT future
    plt.plot(
        y[:, 0],
        y[:, 1],
        "o-",
        linewidth=3,
        label="GT future",
    )

    # 画 Pred future
    plt.plot(
        pred[:, 0],
        pred[:, 1],
        "x--",
        linewidth=3,
        label="Pred future",
    )

    plt.axhline(0.0, linestyle=":", linewidth=1)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Toy VectorNet: input polylines and predicted future")
    plt.xlabel("x forward [m]")
    plt.ylabel("y lateral [m]")
    plt.savefig("toy_vectornet_result.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved figure to toy_vectornet_result.png")


if __name__ == "__main__":
    model, dataset, device = train()
    visualize_one_sample(model, dataset, device)
