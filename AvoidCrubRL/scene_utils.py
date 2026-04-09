import numpy as np


def build_end_decay_window(s):
    u = (s - s[0]) / max(s[-1] - s[0], 1e-6)
    return np.sin(np.pi * u) ** 2


def build_turn_path(r=10.0, straight_len=20.0, n_line=120, n_arc=120):
    y1 = np.linspace(0.0, straight_len, n_line)
    x1 = np.zeros_like(y1)

    theta = np.linspace(np.pi, np.pi / 2.0, n_arc)
    x2 = r + r * np.cos(theta)
    y2 = straight_len + r * np.sin(theta)

    x3 = np.linspace(r, r + straight_len, n_line)
    y3 = np.full_like(x3, straight_len + r)

    x = np.concatenate([x1, x2[1:], x3[1:]])
    y = np.concatenate([y1, y2[1:], y3[1:]])
    return x, y


def build_boundary_from_points(points, num_per_seg=40):
    pts = np.asarray(points, dtype=float)
    assert len(pts) >= 2, "至少需要两个点"

    xs = []
    ys = []

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]

        if i < len(pts) - 2:
            t = np.linspace(0.0, 1.0, num_per_seg, endpoint=False)
        else:
            t = np.linspace(0.0, 1.0, num_per_seg, endpoint=True)

        x_seg = p0[0] * (1.0 - t) + p1[0] * t
        y_seg = p0[1] * (1.0 - t) + p1[1] * t

        xs.append(x_seg)
        ys.append(y_seg)

    bx = np.concatenate(xs)
    by = np.concatenate(ys)
    return bx, by
