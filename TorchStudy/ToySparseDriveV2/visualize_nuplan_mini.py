from __future__ import annotations

import argparse
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


DEFAULT_DB_ROOT = Path("nuplan_dataset/nuplan-v1.1_mini/data/cache/mini")
DEFAULT_OUTPUT_DIR = Path("outputs/nuplan_mini_vis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a nuPlan mini tagged scenario directly from SQLite db files."
    )
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--tag", type=str, default="following_lane_with_slow_lead")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--db-index", type=int, default=-1)
    parser.add_argument("--history-sec", type=float, default=4.0)
    parser.add_argument("--future-sec", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--list-tags", action="store_true")
    parser.add_argument("--max-dbs", type=int, default=0)
    return parser.parse_args()


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def find_db_paths(db_root: Path, max_dbs: int = 0) -> list[Path]:
    db_paths = sorted(db_root.glob("*.db"))
    if max_dbs > 0:
        db_paths = db_paths[:max_dbs]
    if not db_paths:
        raise FileNotFoundError(f"No .db files found under {db_root}")
    return db_paths


def token_hex(token: bytes | None) -> str:
    return token.hex() if token is not None else ""


def list_tag_counts(db_paths: list[Path]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for db_path in db_paths:
        with sqlite3.connect(db_path) as con:
            for tag_type, count in con.execute(
                "SELECT type, COUNT(*) FROM scenario_tag GROUP BY type"
            ):
                counts[str(tag_type)] = counts.get(str(tag_type), 0) + int(count)
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def find_tag_instances(
    db_paths: list[Path],
    tag_type: str,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for db_path in db_paths:
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                """
                SELECT
                    scenario_tag.token,
                    scenario_tag.lidar_pc_token,
                    scenario_tag.agent_track_token,
                    lidar_pc.timestamp,
                    log.location,
                    log.map_version,
                    log.logfile
                FROM scenario_tag
                JOIN lidar_pc ON lidar_pc.token = scenario_tag.lidar_pc_token
                JOIN lidar ON lidar.token = lidar_pc.lidar_token
                JOIN log ON log.token = lidar.log_token
                WHERE scenario_tag.type = ?
                ORDER BY lidar_pc.timestamp
                """,
                (tag_type,),
            ).fetchall()
            for row in rows:
                instances.append(
                    {
                        "db_path": db_path,
                        "tag_token": row[0],
                        "lidar_pc_token": row[1],
                        "agent_track_token": row[2],
                        "timestamp": int(row[3]),
                        "location": str(row[4]),
                        "map_version": str(row[5]),
                        "logfile": str(row[6]),
                    }
                )
    return instances


def load_lidar_samples(
    con: sqlite3.Connection,
    center_timestamp: int,
    history_sec: float,
    future_sec: float,
) -> list[dict[str, Any]]:
    start_ts = center_timestamp - int(history_sec * 1_000_000)
    end_ts = center_timestamp + int(future_sec * 1_000_000)
    rows = con.execute(
        """
        SELECT
            lidar_pc.token,
            lidar_pc.timestamp,
            ego_pose.x,
            ego_pose.y,
            ego_pose.qw,
            ego_pose.qx,
            ego_pose.qy,
            ego_pose.qz,
            ego_pose.vx,
            ego_pose.vy
        FROM lidar_pc
        JOIN ego_pose ON ego_pose.token = lidar_pc.ego_pose_token
        WHERE lidar_pc.timestamp BETWEEN ? AND ?
        ORDER BY lidar_pc.timestamp
        """,
        (start_ts, end_ts),
    ).fetchall()
    samples = []
    for row in rows:
        yaw = yaw_from_quaternion(float(row[4]), float(row[5]), float(row[6]), float(row[7]))
        speed = math.hypot(float(row[8]), float(row[9]))
        samples.append(
            {
                "token": row[0],
                "timestamp": int(row[1]),
                "x": float(row[2]),
                "y": float(row[3]),
                "yaw": yaw,
                "speed": speed,
            }
        )
    return samples


def load_boxes(
    con: sqlite3.Connection,
    lidar_pc_tokens: list[bytes],
) -> list[dict[str, Any]]:
    if not lidar_pc_tokens:
        return []
    placeholders = ",".join("?" for _ in lidar_pc_tokens)
    rows = con.execute(
        f"""
        SELECT
            lidar_box.lidar_pc_token,
            lidar_box.track_token,
            lidar_box.x,
            lidar_box.y,
            lidar_box.width,
            lidar_box.length,
            lidar_box.vx,
            lidar_box.vy,
            lidar_box.yaw,
            category.name
        FROM lidar_box
        JOIN track ON track.token = lidar_box.track_token
        JOIN category ON category.token = track.category_token
        WHERE lidar_box.lidar_pc_token IN ({placeholders})
        """,
        lidar_pc_tokens,
    ).fetchall()
    boxes = []
    for row in rows:
        boxes.append(
            {
                "lidar_pc_token": row[0],
                "track_token": row[1],
                "x": float(row[2]),
                "y": float(row[3]),
                "width": float(row[4]),
                "length": float(row[5]),
                "vx": float(row[6]),
                "vy": float(row[7]),
                "yaw": float(row[8]),
                "category": str(row[9]),
            }
        )
    return boxes


def load_current_tags(
    con: sqlite3.Connection,
    lidar_pc_token: bytes,
) -> list[tuple[str, str]]:
    rows = con.execute(
        """
        SELECT type, agent_track_token
        FROM scenario_tag
        WHERE lidar_pc_token = ?
        ORDER BY type
        """,
        (lidar_pc_token,),
    ).fetchall()
    return [(str(tag_type), token_hex(agent_token)) for tag_type, agent_token in rows]


def transform_to_local(
    xy: np.ndarray,
    origin_xy: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    delta = xy - origin_xy[None, :]
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return np.stack([x, y], axis=-1)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    *,
    color: str,
    alpha: float,
    label: str | None = None,
) -> None:
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label=label,
        zorder=5,
    )
    transform = (
        matplotlib.transforms.Affine2D()
        .rotate(yaw)
        .translate(x, y)
        + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)


def visualize_instance(
    instance: dict[str, Any],
    tag_type: str,
    rank: int,
    args: argparse.Namespace,
) -> Path:
    db_path: Path = instance["db_path"]
    with sqlite3.connect(db_path) as con:
        samples = load_lidar_samples(
            con,
            int(instance["timestamp"]),
            args.history_sec,
            args.future_sec,
        )
        if not samples:
            raise RuntimeError(f"No lidar samples around timestamp: {instance['timestamp']}")
        lidar_tokens = [sample["token"] for sample in samples]
        boxes = load_boxes(con, lidar_tokens)
        current_tags = load_current_tags(con, instance["lidar_pc_token"])

    center_index = min(
        range(len(samples)),
        key=lambda index: abs(samples[index]["timestamp"] - int(instance["timestamp"])),
    )
    center = samples[center_index]
    center_xy = np.asarray([center["x"], center["y"]], dtype=np.float32)
    center_yaw = float(center["yaw"])

    sample_by_token = {sample["token"]: sample for sample in samples}
    current_boxes = [
        box for box in boxes if box["lidar_pc_token"] == instance["lidar_pc_token"]
    ]
    agent_token = instance["agent_track_token"]
    target_boxes = [
        box for box in boxes if agent_token and box["track_token"] == agent_token
    ]

    ego_xy = np.asarray([[sample["x"], sample["y"]] for sample in samples], dtype=np.float32)
    ego_local = transform_to_local(ego_xy, center_xy, center_yaw)
    center_time = int(center["timestamp"])
    times = np.asarray(
        [(sample["timestamp"] - center_time) / 1_000_000.0 for sample in samples],
        dtype=np.float32,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax_global, ax_local, ax_speed = axes

    ax_global.plot(ego_xy[:, 0], ego_xy[:, 1], color="#2ca02c", lw=2.2, label="ego history/future")
    ax_global.scatter([center_xy[0]], [center_xy[1]], color="#2ca02c", s=70, zorder=8, label="ego current")
    ax_local.plot(ego_local[:, 1], ego_local[:, 0], color="#2ca02c", lw=2.2, label="ego local")
    ax_local.scatter([0.0], [0.0], color="#2ca02c", s=70, zorder=8, label="ego current")

    for box in current_boxes:
        color = "#6f6f6f"
        alpha = 0.35
        if agent_token and box["track_token"] == agent_token:
            color = "#d62728"
            alpha = 0.75
        draw_box(
            ax_global,
            box["x"],
            box["y"],
            box["yaw"],
            box["length"],
            box["width"],
            color=color,
            alpha=alpha,
        )
        local_xy = transform_to_local(
            np.asarray([[box["x"], box["y"]]], dtype=np.float32),
            center_xy,
            center_yaw,
        )[0]
        draw_box(
            ax_local,
            float(local_xy[1]),
            float(local_xy[0]),
            box["yaw"] - center_yaw,
            box["length"],
            box["width"],
            color=color,
            alpha=alpha,
        )

    if target_boxes:
        target_by_time = sorted(
            target_boxes,
            key=lambda box: sample_by_token[box["lidar_pc_token"]]["timestamp"],
        )
        target_xy = np.asarray(
            [[box["x"], box["y"]] for box in target_by_time],
            dtype=np.float32,
        )
        target_local = transform_to_local(target_xy, center_xy, center_yaw)
        ax_global.plot(target_xy[:, 0], target_xy[:, 1], color="#d62728", lw=2.0, label="tagged agent")
        ax_local.plot(target_local[:, 1], target_local[:, 0], color="#d62728", lw=2.0, label="tagged agent")

    ax_speed.plot(times, [sample["speed"] for sample in samples], color="#9467bd", marker=".")
    ax_speed.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("ego speed [m/s]")
    ax_speed.grid(True, alpha=0.25)

    tags_text = ", ".join(tag for tag, _ in current_tags[:10])
    title = (
        f"tag={tag_type} | rank={rank} | db={db_path.name}\n"
        f"location={instance['location']} map={instance['map_version']} | "
        f"ego_speed={center['speed']:.2f} m/s | tags={tags_text}"
    )
    fig.suptitle(title, fontsize=11)

    for ax, title_part in [(ax_global, "global"), (ax_local, "ego frame")]:
        ax.set_title(title_part)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    ax_global.set_xlabel("global x [m]")
    ax_global.set_ylabel("global y [m]")
    ax_local.set_xlabel("y left [m]")
    ax_local.set_ylabel("x forward [m]")
    ax_local.set_xlim(-40, 40)
    ax_local.set_ylim(-20, 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = tag_type.replace("/", "_")
    output_path = args.output_dir / f"nuplan_{safe_tag}_{rank:04d}_{db_path.stem}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    db_paths = find_db_paths(args.db_root, args.max_dbs)
    if args.db_index >= 0:
        db_paths = [db_paths[args.db_index]]

    if args.list_tags:
        for tag_type, count in list_tag_counts(db_paths):
            print(f"{tag_type}: {count}")
        return

    instances = find_tag_instances(db_paths, args.tag)
    if not instances:
        raise ValueError(f"No instances found for tag: {args.tag}")
    if args.index < 0 or args.index >= len(instances):
        raise IndexError(f"--index {args.index} out of range: {len(instances)}")

    instance = instances[args.index]
    output_path = visualize_instance(instance, args.tag, args.index, args)
    print(f"tag: {args.tag}")
    print(f"instances: {len(instances)}")
    print(f"selected_index: {args.index}")
    print(f"db_path: {instance['db_path']}")
    print(f"timestamp: {instance['timestamp']}")
    print(f"location: {instance['location']}")
    print(f"map_version: {instance['map_version']}")
    print(f"agent_track_token: {token_hex(instance['agent_track_token'])}")
    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
