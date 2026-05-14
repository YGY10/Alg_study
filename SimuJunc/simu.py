import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def interp_polyline(points, ds=0.5):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return points.copy()

    s_new = np.arange(0.0, s[-1] + ds, ds)
    x_new = np.interp(s_new, s, points[:, 0])
    y_new = np.interp(s_new, s, points[:, 1])
    return np.column_stack([x_new, y_new])


def polyline_s(path):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return np.array([0.0])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(path):
    s = polyline_s(path)
    return float(s[-1])


def point_at_s(path, s_query):
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return np.array([0.0, 0.0])
    if len(path) == 1:
        return path[0].copy()

    s = polyline_s(path)
    s_query = np.clip(s_query, 0.0, s[-1])
    x = np.interp(s_query, s, path[:, 0])
    y = np.interp(s_query, s, path[:, 1])
    return np.array([x, y])


def project_point_to_polyline_s(path, pt):
    path = np.asarray(path, dtype=float)
    pt = np.asarray(pt, dtype=float)

    if len(path) == 0:
        return 0.0, pt.copy()
    if len(path) == 1:
        return 0.0, path[0].copy()

    s = polyline_s(path)

    best_dist2 = np.inf
    best_s = 0.0
    best_proj = path[0].copy()

    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]
        v = p1 - p0
        vv = np.dot(v, v)
        if vv < 1e-12:
            continue

        t = np.dot(pt - p0, v) / vv
        t = np.clip(t, 0.0, 1.0)

        proj = p0 + t * v
        dist2 = np.sum((pt - proj) ** 2)

        if dist2 < best_dist2:
            best_dist2 = dist2
            best_proj = proj
            seg_len = np.linalg.norm(v)
            best_s = s[i] + t * seg_len

    return best_s, best_proj


def cut_polyline_from_projection(path, ego):
    path = np.asarray(path, dtype=float)
    ego = np.asarray(ego, dtype=float)

    if len(path) == 0:
        return np.asarray([ego], dtype=float)

    s_path = polyline_s(path)
    s_proj, proj = project_point_to_polyline_s(path, ego)

    remain_mask = s_path > s_proj + 1e-6
    remain = path[remain_mask]

    out = [ego.copy()]
    if np.linalg.norm(proj - ego) > 1e-6:
        out.append(proj.copy())

    for p in remain:
        if np.linalg.norm(p - out[-1]) > 1e-6:
            out.append(p.copy())

    return np.asarray(out, dtype=float)


def concat_paths(paths):
    merged = []
    for p in paths:
        if p is None or len(p) == 0:
            continue
        p = np.asarray(p, dtype=float)

        if len(merged) == 0:
            merged.extend(p.tolist())
        else:
            for i in range(len(p)):
                if np.linalg.norm(np.asarray(merged[-1]) - p[i]) > 1e-6:
                    merged.append(p[i].tolist())

    if len(merged) == 0:
        return np.zeros((0, 2))

    return np.asarray(merged, dtype=float)


def build_scene():
    real_pre_ctrl = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
        ],
        dtype=float,
    )
    real_pre = interp_polyline(real_pre_ctrl, ds=0.5)

    real_post_ctrl = np.array(
        [
            [2.0, 70.0],
            [2.0, 85.0],
            [2.0, 100.0],
        ],
        dtype=float,
    )
    real_post = interp_polyline(real_post_ctrl, ds=0.5)

    map_ctrl = np.array(
        [
            [-1.5, 0.0],
            [-1.0, 20.0],
            [0.5, 45.0],
            [1.5, 70.0],
            [2.7, 100.0],
        ],
        dtype=float,
    )
    light_map = interp_polyline(map_ctrl, ds=0.5)

    return real_pre, real_post, light_map


def build_reference(ego, real_pre, real_post, light_map, visible_dist=30.0):
    post_start = real_post[0]
    dist_to_post_start = np.linalg.norm(ego - post_start)
    post_visible = dist_to_post_start <= visible_dist

    s_pre, _ = project_point_to_polyline_s(real_pre, ego)
    pre_total = polyline_length(real_pre)
    pre_forward = cut_polyline_from_projection(real_pre, ego)

    if not post_visible:
        if s_pre < pre_total - 0.5:
            pre_end = real_pre[-1]
            s_map_join, _ = project_point_to_polyline_s(light_map, pre_end)

            s_light = polyline_s(light_map)
            s_samples = np.arange(s_map_join, s_light[-1] + 0.5, 0.5)
            x = np.interp(s_samples, s_light, light_map[:, 0])
            y = np.interp(s_samples, s_light, light_map[:, 1])
            map_after_pre_end = np.column_stack([x, y])

            ref = concat_paths([pre_forward, map_after_pre_end])
        else:
            ref = cut_polyline_from_projection(light_map, ego)

        switch_point = None

    else:
        pre_end = real_pre[-1]
        s_map_from, _ = project_point_to_polyline_s(light_map, pre_end)
        s_map_to, _ = project_point_to_polyline_s(light_map, post_start)

        if s_map_to < s_map_from:
            s_map_to = s_map_from

        s_light = polyline_s(light_map)
        s_samples = np.arange(s_map_from, s_map_to + 0.5, 0.5)
        x = np.interp(s_samples, s_light, light_map[:, 0])
        y = np.interp(s_samples, s_light, light_map[:, 1])
        map_mid = np.column_stack([x, y])

        if s_pre < pre_total - 0.5:
            ref = concat_paths([pre_forward, map_mid, real_post])
        else:
            map_to_post = concat_paths([map_mid, real_post])
            ref = cut_polyline_from_projection(map_to_post, ego)

        switch_point = post_start.copy()

    ref = cut_polyline_from_projection(ref, ego)
    return ref, post_visible, switch_point


def main():
    real_pre, real_post, light_map = build_scene()

    ego = real_pre[0].copy()

    dt = 0.1
    v = 4.0
    ds_move = v * dt
    total_frames = 400

    ego_history = []
    ref_history = []
    visible_history = []
    switch_history = []

    for _ in range(total_frames):
        ref, post_visible, switch_point = build_reference(
            ego, real_pre, real_post, light_map, visible_dist=30.0
        )

        ego_history.append(ego.copy())
        ref_history.append(ref.copy())
        visible_history.append(post_visible)
        switch_history.append(None if switch_point is None else switch_point.copy())

        ref_len = polyline_length(ref)
        if ref_len < 1e-3:
            break

        ego = point_at_s(ref, ds_move)

        if np.linalg.norm(ego - real_post[-1]) < 1.0:
            ego_history.append(ego.copy())
            ref_history.append(ref.copy())
            visible_history.append(post_visible)
            switch_history.append(None if switch_point is None else switch_point.copy())
            break

    fig, ax = plt.subplots(figsize=(12, 16))

    def update(frame):
        ax.clear()

        ego_now = ego_history[frame]
        ref_now = ref_history[frame]
        post_visible = visible_history[frame]
        switch_point = switch_history[frame]

        ax.plot(
            real_pre[:, 0],
            real_pre[:, 1],
            linewidth=3,
            label="real_lane_before_intersection",
        )
        ax.plot(
            real_post[:, 0],
            real_post[:, 1],
            linewidth=3,
            label="real_lane_after_intersection",
        )
        ax.plot(
            light_map[:, 0], light_map[:, 1], "--", linewidth=2, label="light_map_lane"
        )
        ax.plot(ref_now[:, 0], ref_now[:, 1], linewidth=3, label="current_reference")

        hist = np.asarray(ego_history[: frame + 1])
        ax.plot(hist[:, 0], hist[:, 1], linewidth=2, label="ego_path")

        ax.scatter([ego_now[0]], [ego_now[1]], s=80, marker="o", label="ego")
        ax.scatter(
            [real_post[0, 0]],
            [real_post[0, 1]],
            s=80,
            marker="x",
            label="post_lane_start",
        )

        if switch_point is not None:
            ax.scatter(
                [switch_point[0]],
                [switch_point[1]],
                s=100,
                marker="*",
                label="switch_to_real_post",
            )

        dist_to_post = np.linalg.norm(ego_now - real_post[0])
        ax.set_title(
            f"frame={frame} | dist_to_post_start={dist_to_post:.1f} m | "
            f"post_visible={'YES' if post_visible else 'NO'}"
        )

        ax.set_aspect("equal")
        x, y = ego_now
        ax.set_xlim(x - 6, x + 6)
        ax.set_ylim(y - 10, y + 20)
        ax.grid(True)
        ax.legend(loc="upper left")

    ani = FuncAnimation(fig, update, frames=len(ego_history), interval=80, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
