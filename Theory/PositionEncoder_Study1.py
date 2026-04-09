# 比较普通位置编码和旋转位置编码的区别
import math
import torch
import matplotlib.pyplot as plt
from typing import Tuple


# =========================================================
# 1. 普通位置编码（Absolute Positional Encoding）
# =========================================================
def sinusoidal_position_encoding(seq_len: int, dim: int) -> torch.Tensor:
    """
    经典 Transformer 正弦位置编码
    shape: [seq_len, dim]
    """
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =========================================================
# 2. RoPE（旋转位置编码）
# =========================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    把最后一维两两成对旋转:
    [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    """
    x_odd = x[..., 0::2]
    x_even = x[..., 1::2]

    x_rot = torch.stack([-x_even, x_odd], dim=-1)
    return x_rot.flatten(start_dim=-2)


def rope_angles(seq_len: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 RoPE 需要的 cos/sin
    返回 shape: [seq_len, dim]
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    inv_freq = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    angles = position * inv_freq  # [seq_len, dim/2]

    cos_part = torch.cos(angles)
    sin_part = torch.sin(angles)

    cos_full = torch.zeros(seq_len, dim, dtype=torch.float32)
    sin_full = torch.zeros(seq_len, dim, dtype=torch.float32)

    cos_full[:, 0::2] = cos_part
    cos_full[:, 1::2] = cos_part
    sin_full[:, 0::2] = sin_part
    sin_full[:, 1::2] = sin_part

    return cos_full, sin_full


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对输入 x 应用 RoPE
    x shape:   [seq_len, dim]
    cos shape: [seq_len, dim]
    sin shape: [seq_len, dim]
    """
    return x * cos + rotate_half(x) * sin


# =========================================================
# 3. 构造一个最小 attention 例子
# =========================================================
def build_toy_input(seq_len: int, dim: int) -> torch.Tensor:
    """
    构造一个简单输入
    这里直接随机生成 token embedding
    """
    torch.manual_seed(0)
    return torch.randn(seq_len, dim)


def build_qk(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造 Q, K
    """
    torch.manual_seed(1)
    w_q = torch.randn(dim, dim)
    w_k = torch.randn(dim, dim)

    q = x @ w_q
    k = x @ w_k
    return q, k


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    计算 attention score 矩阵
    shape: [seq_len, seq_len]
    """
    d = q.shape[-1]
    return (q @ k.T) / math.sqrt(d)


# =========================================================
# 4. 可视化工具
# =========================================================
def plot_heatmap(mat: torch.Tensor, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(mat.detach().numpy(), aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()


def plot_position_encoding_lines(
    pe: torch.Tensor, title: str, dims_to_show: int = 6
) -> None:
    seq_len, dim = pe.shape
    show_dim = min(dims_to_show, dim)

    plt.figure(figsize=(10, 5))
    for i in range(show_dim):
        plt.plot(pe[:, i].numpy(), label=f"dim {i}")
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_2d_points(points: torch.Tensor, title: str) -> None:
    """
    points shape: [seq_len, 2]
    """
    xs = points[:, 0].numpy()
    ys = points[:, 1].numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys)

    for i in range(points.shape[0]):
        plt.annotate(str(i), (xs[i], ys[i]))

    plt.title(title)
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()


# =========================================================
# 5. 主逻辑：对比普通 PE 和 RoPE
# =========================================================
def main() -> None:
    seq_len = 16
    dim = 8

    # -----------------------------------------------------
    # A. 原始 token embedding
    # -----------------------------------------------------
    x = build_toy_input(seq_len, dim)

    print("=" * 80)
    print("原始输入 X:")
    print(x)
    print()

    # -----------------------------------------------------
    # B. 普通位置编码
    # -----------------------------------------------------
    pe_abs = sinusoidal_position_encoding(seq_len, dim)
    x_abs = x + pe_abs

    print("=" * 80)
    print("普通位置编码 PE:")
    print(pe_abs)
    print()

    # 普通位置编码下的 Q/K
    q_abs, k_abs = build_qk(x_abs, dim)
    scores_abs = attention_scores(q_abs, k_abs)

    # -----------------------------------------------------
    # C. RoPE
    # -----------------------------------------------------
    # 注意：RoPE 不直接 x + pe
    # 而是先得到 Q/K，再对 Q/K 施加旋转
    q_plain, k_plain = build_qk(x, dim)
    cos, sin = rope_angles(seq_len, dim)
    q_rope = apply_rope(q_plain, cos, sin)
    k_rope = apply_rope(k_plain, cos, sin)
    scores_rope = attention_scores(q_rope, k_rope)

    print("=" * 80)
    print("RoPE 的 cos:")
    print(cos)
    print()

    print("=" * 80)
    print("RoPE 的 sin:")
    print(sin)
    print()

    # -----------------------------------------------------
    # D. 打印几组对比
    # -----------------------------------------------------
    print("=" * 80)
    print("普通位置编码 attention score:")
    print(scores_abs)
    print()

    print("=" * 80)
    print("RoPE attention score:")
    print(scores_rope)
    print()

    print("=" * 80)
    print("普通位置编码 vs RoPE 的 score 差值:")
    print(scores_rope - scores_abs)
    print()

    # -----------------------------------------------------
    # E. 可视化 1：普通位置编码曲线
    # -----------------------------------------------------
    plot_position_encoding_lines(
        pe_abs,
        title="Absolute Positional Encoding (first several dimensions)",
        dims_to_show=8,
    )

    # -----------------------------------------------------
    # F. 可视化 2：RoPE 的 cos / sin 曲线
    # -----------------------------------------------------
    plot_position_encoding_lines(
        cos,
        title="RoPE cos values (first several dimensions)",
        dims_to_show=8,
    )

    plot_position_encoding_lines(
        sin,
        title="RoPE sin values (first several dimensions)",
        dims_to_show=8,
    )

    # -----------------------------------------------------
    # G. 可视化 3：看前两个维度下，普通 PE 的位置点
    # -----------------------------------------------------
    plot_2d_points(
        pe_abs[:, :2], title="Absolute Positional Encoding in 2D (dim0, dim1)"
    )

    # -----------------------------------------------------
    # H. 可视化 4：看 RoPE 对 Q 的旋转效果（前两个维度）
    # -----------------------------------------------------
    plot_2d_points(q_plain[:, :2], title="Original Q in 2D before RoPE (dim0, dim1)")

    plot_2d_points(q_rope[:, :2], title="Q in 2D after RoPE (dim0, dim1)")

    # -----------------------------------------------------
    # I. 可视化 5：attention score 热力图
    # -----------------------------------------------------
    plot_heatmap(scores_abs, "Attention Scores with Absolute Positional Encoding")
    plot_heatmap(scores_rope, "Attention Scores with RoPE")
    plot_heatmap(scores_rope - scores_abs, "RoPE - Absolute PE (Score Difference)")

    # -----------------------------------------------------
    # J. 额外说明打印
    # -----------------------------------------------------
    print("=" * 80)
    print("结论：")
    print("1. 普通位置编码：先 x + pe，再算 Q/K。")
    print("2. RoPE：先算 Q/K，再按位置对 Q/K 做旋转。")
    print("3. 普通位置编码把'位置'加到输入里。")
    print("4. RoPE把'位置关系'直接注入 attention 的点积结构里。")
    print("=" * 80)
    plt.show()


if __name__ == "__main__":
    main()
