import numpy as np


def compute_s(x, y):
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def compute_yaw_from_xy(x, y):
    n = len(x)
    yaw = np.zeros(n)

    for i in range(n):
        if i == 0:
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
        elif i == n - 1:
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
        else:
            dx = 0.5 * (x[i + 1] - x[i - 1])
            dy = 0.5 * (y[i + 1] - y[i - 1])

        yaw[i] = np.arctan2(dy, dx)

    return yaw


def compute_normals_from_yaw(yaw):
    nx = -np.sin(yaw)
    ny = np.cos(yaw)
    return nx, ny


def apply_local_offset(x, y, yaw, offset_l):
    nx, ny = compute_normals_from_yaw(yaw)
    x_new = x + nx * offset_l
    y_new = y + ny * offset_l
    return x_new, y_new


def resample_polyline_by_s(x, y, num=200):
    s = compute_s(x, y)
    s_new = np.linspace(s[0], s[-1], num)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new, s_new


def point_to_segment_projection(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return ax, ay, np.hypot(px - ax, py - ay), 0.0

    t = (apx * abx + apy * aby) / ab2
    t = np.clip(t, 0.0, 1.0)

    qx = ax + t * abx
    qy = ay + t * aby
    dist = np.hypot(px - qx, py - qy)
    return qx, qy, dist, t


def point_to_polyline_projection(px, py, line_x, line_y):
    best_dist = np.inf
    best_qx = None
    best_qy = None

    for i in range(len(line_x) - 1):
        qx, qy, dist, _ = point_to_segment_projection(
            px, py, line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]
        )
        if dist < best_dist:
            best_dist = dist
            best_qx = qx
            best_qy = qy

    return best_qx, best_qy, best_dist


def compute_min_distance_to_boundary(path_x, path_y, boundary_x, boundary_y):
    dists = np.zeros(len(path_x))
    for i in range(len(path_x)):
        _, _, dist = point_to_polyline_projection(
            path_x[i], path_y[i], boundary_x, boundary_y
        )
        dists[i] = dist
    return dists
