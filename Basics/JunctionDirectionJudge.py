from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict


# ===============================
# 基础数据结构
# ===============================


@dataclass
class Vec2D:
    x: float
    y: float


Lane = List[Vec2D]


def quintic_poly_coeff(x0, y0, x1, y1):
    """
    y(x) = a0 + a1 x + a2 x^2 + a3 x^3 + a4 x^4 + a5 x^5
    约束：
      y(x0)=y0, y'(x0)=0, y''(x0)=0
      y(x1)=y1, y'(x1)=0, y''(x1)=0
    """
    A = np.array(
        [
            [1, x0, x0**2, x0**3, x0**4, x0**5],
            [0, 1, 2 * x0, 3 * x0**2, 4 * x0**3, 5 * x0**4],
            [0, 0, 2, 6 * x0, 12 * x0**2, 20 * x0**3],
            [1, x1, x1**2, x1**3, x1**4, x1**5],
            [0, 1, 2 * x1, 3 * x1**2, 4 * x1**3, 5 * x1**4],
            [0, 0, 2, 6 * x1, 12 * x1**2, 20 * x1**3],
        ]
    )
    b = np.array([y0, 0.0, 0.0, y1, 0.0, 0.0])
    return np.linalg.solve(A, b)


def generate_lane_quintic(
    p_in: Vec2D,
    p_out: Vec2D,
    num_points: int = 50,
) -> Lane:
    coeff = quintic_poly_coeff(p_in.x, p_in.y, p_out.x, p_out.y)
    xs = np.linspace(p_in.x, p_out.x, num_points)

    lane = []
    for x in xs:
        y = (
            coeff[0]
            + coeff[1] * x
            + coeff[2] * x**2
            + coeff[3] * x**3
            + coeff[4] * x**4
            + coeff[5] * x**5
        )
        lane.append(Vec2D(float(x), float(y)))

    return lane


def generate_lane_arc(
    p_in: Vec2D,
    p_out: Vec2D,
    num_points: int = 50,
) -> Lane:
    """
    工程安全版圆弧：
    - 起点切向必须指向终点
    - 若不满足，直接退化为直线
    """

    x0, y0 = p_in.x, p_in.y
    x1, y1 = p_out.x, p_out.y

    # 期望行驶方向（起点切向）
    vx = x1 - x0
    vy = y1 - y0
    v_norm = np.hypot(vx, vy)
    if v_norm < 1e-6:
        return [Vec2D(x0, y0) for _ in range(num_points)]

    tx = vx / v_norm
    ty = vy / v_norm

    # 左 / 右法向
    normals = [
        np.array([-ty, tx]),  # 左转
        np.array([ty, -tx]),  # 右转
    ]

    for n in normals:
        denom = 2 * (vx * n[0] + vy * n[1])
        if abs(denom) < 1e-6:
            continue

        R = (vx * vx + vy * vy) / denom
        if R <= 0:
            continue

        cx = x0 + R * n[0]
        cy = y0 + R * n[1]
        R = abs(R)

        # 起止角
        theta0 = np.arctan2(y0 - cy, x0 - cx)
        theta1 = np.arctan2(y1 - cy, x1 - cx)
        if theta1 < theta0:
            theta1 += 2 * np.pi

        thetas = np.linspace(theta0, theta1, num_points)

        # 生成圆弧
        lane = []
        for th in thetas:
            x = cx + R * np.cos(th)
            y = cy + R * np.sin(th)
            lane.append(Vec2D(float(x), float(y)))

        # ===== 关键校验：起点切向是否向前 =====
        dx0 = lane[1].x - lane[0].x
        dy0 = lane[1].y - lane[0].y
        if dx0 * tx + dy0 * ty <= 0:
            continue  # ❌ 起步方向不对，丢弃这个圆弧

        return lane

    # 所有圆弧都不合法 → 退化为直线
    return generate_lane_line(p_in, p_out, num_points)


def generate_lane_line(p_in: Vec2D, p_out: Vec2D, num_points: int = 50) -> Lane:
    xs = np.linspace(p_in.x, p_out.x, num_points)
    ys = np.linspace(p_in.y, p_out.y, num_points)
    return [Vec2D(float(x), float(y)) for x, y in zip(xs, ys)]


def generate_all_lanes(in_points, out_points):
    lanes = []
    for p_in in in_points:
        for p_out in out_points:
            lanes.append(generate_lane_line(p_in, p_out))
    return lanes


# ===============================
# overlap 区间工具（统一来源）
# ===============================


def get_overlap_interval(lane_i: Lane, lane_j: Lane):
    xi = np.array([p.x for p in lane_i])
    xj = np.array([p.x for p in lane_j])

    x_start = max(xi.min(), xj.min())
    x_end = min(xi.max(), xj.max())

    if x_end <= x_start:
        return None
    return float(x_start), float(x_end)


def crop_lane_by_x(lane: Lane, x_start: float, x_end: float) -> Lane:
    xs = np.array([p.x for p in lane])
    ys = np.array([p.y for p in lane])

    mask = (xs >= x_start) & (xs <= x_end)
    if mask.sum() < 2:
        return []

    return [Vec2D(float(x), float(y)) for x, y in zip(xs[mask], ys[mask])]


def align_lane_to_origin(lane: Lane) -> Lane:
    if not lane:
        return []
    x0, y0 = lane[0].x, lane[0].y
    return [Vec2D(p.x - x0, p.y - y0) for p in lane]


# 距离：裁剪 + 平移 + RMS
def lane_distance_overlap_aligned(
    lane_i: Lane,
    lane_j: Lane,
    num_samples: int = 30,
    large_penalty: float = 1e6,
) -> float:
    interval = get_overlap_interval(lane_i, lane_j)
    if interval is None:
        return large_penalty

    x_start, x_end = interval

    lane_i_crop = crop_lane_by_x(lane_i, x_start, x_end)
    lane_j_crop = crop_lane_by_x(lane_j, x_start, x_end)

    if len(lane_i_crop) < 2 or len(lane_j_crop) < 2:
        return large_penalty

    lane_i_align = align_lane_to_origin(lane_i_crop)
    lane_j_align = align_lane_to_origin(lane_j_crop)

    xi = np.array([p.x for p in lane_i_align])
    yi = np.array([p.y for p in lane_i_align])
    xj = np.array([p.x for p in lane_j_align])
    yj = np.array([p.y for p in lane_j_align])

    L = min(xi.max(), xj.max())
    if L <= 0:
        return large_penalty

    xs = np.linspace(0.0, L, num_samples)
    yi_interp = np.interp(xs, xi, yi)
    yj_interp = np.interp(xs, xj, yj)

    dy = yi_interp - yj_interp
    return float(np.sqrt(np.mean(dy**2)))


def compute_distance_matrix_overlap_aligned(lanes, num_samples=30):
    N = len(lanes)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = lane_distance_overlap_aligned(lanes[i], lanes[j], num_samples)
            D[i, j] = d
            D[j, i] = d
    return D


# 重叠区间矩阵
def compute_overlap_interval_matrix(lanes: List[Lane]):
    N = len(lanes)
    M = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            M[i][j] = get_overlap_interval(lanes[i], lanes[j])
    return M


def print_overlap_interval_matrix(overlap_matrix):
    print("\n重叠区间矩阵:")
    for i, row in enumerate(overlap_matrix):
        items = []
        for v in row:
            if v is None:
                items.append("None".ljust(14))
            else:
                items.append(f"({v[0]:5.1f},{v[1]:5.1f})")
        print(f"lane {i}: ", "  ".join(items))


# DBSCAN
def cluster_lanes_dbscan_overlap_aligned(
    lanes: List[Lane],
    eps: float,
    min_samples: int,
    num_samples: int,
):
    dist_matrix = compute_distance_matrix_overlap_aligned(lanes, num_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters, dist_matrix


def find_dominant_clusters(clusters):
    counts = {k: len(v) for k, v in clusters.items() if k != -1}
    if not counts:
        return []
    max_cnt = max(counts.values())
    return [k for k, v in counts.items() if v == max_cnt]


def plot_lanes(lanes, in_points, out_points, current_in_idx):
    plt.figure(figsize=(8, 6))
    for lane in lanes:
        plt.plot([p.x for p in lane], [p.y for p in lane], "r", alpha=0.4)

    for i, p in enumerate(in_points):
        plt.scatter(p.x, p.y, c="orange" if i == current_in_idx else "blue", s=120)

    for p in out_points:
        plt.scatter(p.x, p.y, c="green", s=80)

    plt.axis("equal")
    plt.grid(True)
    plt.title("All Lane Connections")


def plot_dominant_clusters(lanes, clusters, dominant_labels):
    colors = ["red", "blue", "orange", "purple"]
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(dominant_labels):
        for idx in clusters[label]:
            lane = lanes[idx]
            plt.plot(
                [p.x for p in lane],
                [p.y for p in lane],
                color=colors[i % len(colors)],
                linewidth=2,
            )
    plt.axis("equal")
    plt.grid(True)
    plt.title("Dominant Lane Clusters (Aligned Overlap Distance)")


# ===============================
# main
# ===============================

if __name__ == "__main__":
    in_points = [
        Vec2D(0, 3.5),
        Vec2D(5, 0),
        Vec2D(1, -3.50),
    ]

    out_points = [
        Vec2D(30, 3.5),
        Vec2D(27, 1.5),
        Vec2D(29.5, -1.5),
        Vec2D(10, -10),
    ]

    lane1 = generate_lane_line(in_points[0], out_points[0])
    lane2 = generate_lane_line(in_points[0], out_points[1])
    lane3 = generate_lane_line(in_points[1], out_points[1])
    lane4 = generate_lane_line(in_points[1], out_points[2])
    lane5 = generate_lane_line(in_points[2], out_points[2])
    lane6 = generate_lane_arc(in_points[2], out_points[3])

    lanes = [
        lane1,
        lane2,
        lane3,
        lane4,
        lane5,
        lane6,
    ]

    clusters, dist_matrix = cluster_lanes_dbscan_overlap_aligned(
        lanes,
        eps=0.8,
        min_samples=2,
        num_samples=30,
    )

    dominant_labels = find_dominant_clusters(clusters)

    print("Clusters detail:")
    for label, idxs in clusters.items():
        print(f"cluster {label}: {idxs}")

    print("\n距离矩阵:")
    np.set_printoptions(precision=3, suppress=True)
    print(dist_matrix)

    overlap_matrix = compute_overlap_interval_matrix(lanes)
    print_overlap_interval_matrix(overlap_matrix)

    plot_lanes(lanes, in_points, out_points, current_in_idx=0)
    plot_dominant_clusters(lanes, clusters, dominant_labels)

    plt.show()
