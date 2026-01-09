import csv
import matplotlib.pyplot as plt
import numpy as np


def load_blocks(csv_path):
    """
    解析格式：
    time, 0.200
    header...
    data...
    """
    blocks = []
    current = None
    header = None

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # ---- time 行 ----
            if row[0].strip() == "time":
                if current is not None:
                    blocks.append(current)
                current = {"time": float(row[1]), "rows": []}
                header = None
                continue

            # ---- 表头 ----
            if row[0].strip() == "plan_id":
                header = [c.strip() for c in row]
                continue

            # ---- 数据 ----
            if header is not None and current is not None:
                data = {}
                for k, v in zip(header, row):
                    try:
                        data[k] = float(v)
                    except ValueError:
                        data[k] = v
                current["rows"].append(data)

    if current is not None:
        blocks.append(current)

    return blocks


def plot_time_window(csv_path, t_start, t_end, draw_ego=True):
    blocks = load_blocks(csv_path)

    # ---- 选时间窗口 ----
    sel = [b for b in blocks if t_start <= b["time"] <= t_end]
    if not sel:
        raise ValueError("No blocks in given time window")

    sel.sort(key=lambda b: b["time"])

    # ---- 颜色映射 ----
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(sel)))

    plt.figure(figsize=(8, 6))

    for b, c in zip(sel, colors):
        xs = [r["x"] for r in b["rows"]]
        ys = [r["y"] for r in b["rows"]]

        plt.plot(xs, ys, color=c, alpha=0.8, label=f"t={b['time']:.2f}s")

        if draw_ego:
            ego_x = b["rows"][0]["ego_x"]
            ego_y = b["rows"][0]["ego_y"]
            plt.scatter(ego_x, ego_y, color=c, marker="x")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Planned trajectories in time window [{t_start}, {t_end}] s")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    CSV_PATH = "st_frenet_traj_log.csv"

    # ===== 示例：画 0 ~ 1 秒内所有规划轨迹 =====
    plot_time_window(
        CSV_PATH,
        t_start=0.2,
        t_end=0.5,
        draw_ego=True,
    )
