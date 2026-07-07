from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT_DIR = Path("outputs/nuplan_bypass_scan")
DEFAULT_SOURCE_TAGS = (
    "following_lane_with_slow_lead",
    "near_multiple_vehicles",
    "near_long_vehicle",
    "behind_long_vehicle",
    "near_trafficcone_on_driveable",
    "near_barrier_on_driveable",
    "near_construction_zone_sign",
    "stationary_in_traffic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine nuPlan mini for static/slow obstacle bypass-like ego behavior."
    )
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dbs", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--visualize-top-k", type=int, default=30)
    parser.add_argument("--max-candidates-per-db", type=int, default=300)
    parser.add_argument("--dedupe-sec", type=float, default=1.0)
    parser.add_argument("--history-sec", type=float, default=3.0)
    parser.add_argument("--future-sec", type=float, default=8.0)
    parser.add_argument("--source-tag", action="append", default=[])
    parser.add_argument("--min-ego-speed", type=float, default=2.0)
    parser.add_argument("--object-min-x", type=float, default=3.0)
    parser.add_argument("--object-max-x", type=float, default=30.0)
    parser.add_argument("--object-max-y", type=float, default=5.0)
    parser.add_argument("--max-object-speed", type=float, default=1.0)
    parser.add_argument("--min-lateral-shift", type=float, default=1.2)
    parser.add_argument("--min-future-progress", type=float, default=12.0)
    parser.add_argument("--min-pass-margin", type=float, default=3.0)
    parser.add_argument("--max-heading-change", type=float, default=1.2)
    return parser.parse_args()


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def find_db_paths(db_root: Path, max_dbs: int) -> list[Path]:
    db_paths = sorted(db_root.glob("*.db"))
    if max_dbs > 0:
        db_paths = db_paths[:max_dbs]
    if not db_paths:
        raise FileNotFoundError(f"No .db files found under {db_root}")
    return db_paths


def token_hex(token: bytes | None) -> str:
    return token.hex() if token is not None else ""


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


def load_candidate_tags(
    con: sqlite3.Connection,
    source_tags: list[str],
    *,
    dedupe_sec: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in source_tags)
    rows = con.execute(
        f"""
        SELECT
            scenario_tag.type,
            scenario_tag.lidar_pc_token,
            scenario_tag.agent_track_token,
            lidar_pc.timestamp
        FROM scenario_tag
        JOIN lidar_pc ON lidar_pc.token = scenario_tag.lidar_pc_token
        WHERE scenario_tag.type IN ({placeholders})
        ORDER BY lidar_pc.timestamp
        """,
        source_tags,
    ).fetchall()
    candidates: dict[int, dict[str, Any]] = {}
    bucket_us = max(1, int(dedupe_sec * 1_000_000))
    for tag_type, lidar_pc_token, agent_track_token, timestamp in rows:
        bucket = int(timestamp) // bucket_us
        item = candidates.setdefault(
            bucket,
            {
                "lidar_pc_token": lidar_pc_token,
                "agent_track_token": agent_track_token,
                "timestamp": int(timestamp),
                "tags": set(),
            },
        )
        item["tags"].add(str(tag_type))
        if item["agent_track_token"] is None and agent_track_token is not None:
            item["agent_track_token"] = agent_track_token
    values = sorted(candidates.values(), key=lambda item: int(item["timestamp"]))
    if max_candidates > 0:
        values = values[:max_candidates]
    return values


def load_ego_samples(
    con: sqlite3.Connection,
    center_timestamp: int,
    history_sec: float,
    future_sec: float,
) -> list[dict[str, Any]]:
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
        (
            center_timestamp - int(history_sec * 1_000_000),
            center_timestamp + int(future_sec * 1_000_000),
        ),
    ).fetchall()
    samples = []
    for row in rows:
        yaw = yaw_from_quaternion(float(row[4]), float(row[5]), float(row[6]), float(row[7]))
        samples.append(
            {
                "token": row[0],
                "timestamp": int(row[1]),
                "x": float(row[2]),
                "y": float(row[3]),
                "yaw": yaw,
                "speed": float(math.hypot(float(row[8]), float(row[9]))),
            }
        )
    return samples


def load_boxes_for_tokens(
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
    return [
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
            "speed": float(math.hypot(float(row[6]), float(row[7]))),
        }
        for row in rows
    ]


def score_candidate(
    db_path: Path,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as con:
        ego_samples = load_ego_samples(
            con,
            int(candidate["timestamp"]),
            args.history_sec,
            args.future_sec,
        )
        if len(ego_samples) < 10:
            return None
        tokens = [sample["token"] for sample in ego_samples]
        boxes = load_boxes_for_tokens(con, tokens)
        log = con.execute(
            """
            SELECT log.location, log.map_version, log.logfile
            FROM lidar_pc
            JOIN lidar ON lidar.token = lidar_pc.lidar_token
            JOIN log ON log.token = lidar.log_token
            WHERE lidar_pc.token = ?
            """,
            (candidate["lidar_pc_token"],),
        ).fetchone()

    center_index = min(
        range(len(ego_samples)),
        key=lambda index: abs(ego_samples[index]["timestamp"] - int(candidate["timestamp"])),
    )
    center = ego_samples[center_index]
    if center["speed"] < args.min_ego_speed:
        return None

    center_xy = np.asarray([center["x"], center["y"]], dtype=np.float32)
    center_yaw = float(center["yaw"])
    ego_xy = np.asarray([[sample["x"], sample["y"]] for sample in ego_samples], dtype=np.float32)
    ego_local = transform_to_local(ego_xy, center_xy, center_yaw)
    future_local = ego_local[center_index + 1 :]
    future_samples = ego_samples[center_index + 1 :]
    if len(future_local) < 5:
        return None

    future_progress = float(np.max(future_local[:, 0]))
    lateral_shift = float(np.max(np.abs(future_local[:, 1])))
    heading_change = float(
        np.max(
            np.abs(
                angle_wrap(
                    np.asarray([sample["yaw"] for sample in future_samples], dtype=np.float32)
                    - center_yaw
                )
            )
        )
    )
    if future_progress < args.min_future_progress:
        return None
    if lateral_shift < args.min_lateral_shift:
        return None
    if heading_change > args.max_heading_change:
        return None

    current_boxes = [
        box for box in boxes if box["lidar_pc_token"] == candidate["lidar_pc_token"]
    ]
    if not current_boxes:
        return None
    box_xy = np.asarray([[box["x"], box["y"]] for box in current_boxes], dtype=np.float32)
    box_local = transform_to_local(box_xy, center_xy, center_yaw)
    front_mask = (
        (box_local[:, 0] >= args.object_min_x)
        & (box_local[:, 0] <= args.object_max_x)
        & (np.abs(box_local[:, 1]) <= args.object_max_y)
        & np.asarray(
            [box["speed"] <= args.max_object_speed for box in current_boxes],
            dtype=bool,
        )
    )
    if not bool(front_mask.any()):
        return None

    object_indices = np.flatnonzero(front_mask)
    front_cost = box_local[front_mask, 0] + 2.0 * np.abs(box_local[front_mask, 1])
    object_index = int(object_indices[np.argmin(front_cost)])
    object_box = current_boxes[object_index]
    object_local = box_local[object_index]

    pass_margin = future_progress - float(object_local[0])
    if pass_margin < args.min_pass_margin:
        return None

    object_future_boxes = [
        box for box in boxes if box["track_token"] == object_box["track_token"]
    ]
    if object_future_boxes:
        object_future_xy = np.asarray(
            [[box["x"], box["y"]] for box in object_future_boxes],
            dtype=np.float32,
        )
        object_future_local = transform_to_local(object_future_xy, center_xy, center_yaw)
        object_future_max_x = float(np.max(object_future_local[:, 0]))
        dynamic_pass_margin = future_progress - object_future_max_x
    else:
        object_future_max_x = float(object_local[0])
        dynamic_pass_margin = pass_margin

    score = (
        2.0 * min(lateral_shift / 4.0, 2.0)
        + 2.0 * min(pass_margin / 20.0, 2.0)
        + 1.5 * min(max(0.0, center["speed"] - object_box["speed"]) / 4.0, 2.0)
        + 1.0 * min(dynamic_pass_margin / 15.0, 2.0)
        + 1.0 * min(max(0.0, args.object_max_y - abs(float(object_local[1]))) / args.object_max_y, 1.0)
    )

    return {
        "score": score,
        "db_path": str(db_path),
        "db_name": db_path.name,
        "timestamp": int(candidate["timestamp"]),
        "lidar_pc_token": token_hex(candidate["lidar_pc_token"]),
        "object_track_token": token_hex(object_box["track_token"]),
        "tags": "+".join(sorted(candidate["tags"])),
        "location": str(log[0]) if log else "",
        "map_version": str(log[1]) if log else "",
        "logfile": str(log[2]) if log else "",
        "ego_speed": float(center["speed"]),
        "object_category": str(object_box["category"]),
        "object_speed": float(object_box["speed"]),
        "object_x": float(object_local[0]),
        "object_y": float(object_local[1]),
        "future_progress": future_progress,
        "lateral_shift": lateral_shift,
        "heading_change": heading_change,
        "pass_margin": pass_margin,
        "object_future_max_x": object_future_max_x,
        "dynamic_pass_margin": dynamic_pass_margin,
    }


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


def visualize_candidate(row: dict[str, Any], rank: int, args: argparse.Namespace) -> Path:
    db_path = Path(str(row["db_path"]))
    center_timestamp = int(row["timestamp"])
    lidar_pc_token = bytes.fromhex(str(row["lidar_pc_token"]))
    object_track_token = bytes.fromhex(str(row["object_track_token"]))
    with sqlite3.connect(db_path) as con:
        ego_samples = load_ego_samples(
            con,
            center_timestamp,
            args.history_sec,
            args.future_sec,
        )
        boxes = load_boxes_for_tokens(con, [sample["token"] for sample in ego_samples])

    center_index = min(
        range(len(ego_samples)),
        key=lambda index: abs(ego_samples[index]["timestamp"] - center_timestamp),
    )
    center = ego_samples[center_index]
    center_xy = np.asarray([center["x"], center["y"]], dtype=np.float32)
    center_yaw = float(center["yaw"])

    ego_xy = np.asarray([[sample["x"], sample["y"]] for sample in ego_samples], dtype=np.float32)
    ego_local = transform_to_local(ego_xy, center_xy, center_yaw)
    times = np.asarray(
        [(sample["timestamp"] - center_timestamp) / 1_000_000.0 for sample in ego_samples],
        dtype=np.float32,
    )

    current_boxes = [box for box in boxes if box["lidar_pc_token"] == lidar_pc_token]
    object_boxes = [box for box in boxes if box["track_token"] == object_track_token]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax_global, ax_local, ax_speed = axes
    ax_global.plot(ego_xy[:, 0], ego_xy[:, 1], color="#2ca02c", lw=2.4, label="ego")
    ax_global.scatter([center_xy[0]], [center_xy[1]], color="#2ca02c", s=70, zorder=8)
    ax_local.plot(ego_local[:, 1], ego_local[:, 0], color="#2ca02c", lw=2.4, label="ego")
    ax_local.scatter([0.0], [0.0], color="#2ca02c", s=70, zorder=8)

    for box in current_boxes:
        is_object = box["track_token"] == object_track_token
        color = "#d62728" if is_object else "#777777"
        alpha = 0.75 if is_object else 0.28
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

    if object_boxes:
        object_xy = np.asarray([[box["x"], box["y"]] for box in object_boxes], dtype=np.float32)
        object_local = transform_to_local(object_xy, center_xy, center_yaw)
        ax_global.plot(object_xy[:, 0], object_xy[:, 1], color="#d62728", lw=2.0, label="object")
        ax_local.plot(object_local[:, 1], object_local[:, 0], color="#d62728", lw=2.0, label="object")

    ax_speed.plot(times, [sample["speed"] for sample in ego_samples], color="#9467bd", marker=".", label="ego")
    ax_speed.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_speed.axhline(float(row["object_speed"]), color="#d62728", linestyle=":", linewidth=1.4, label="object speed")
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(loc="best")

    for ax, title in [(ax_global, "global"), (ax_local, "ego frame")]:
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    ax_global.set_xlabel("global x [m]")
    ax_global.set_ylabel("global y [m]")
    ax_local.set_xlabel("y left [m]")
    ax_local.set_ylabel("x forward [m]")
    ax_local.set_xlim(-35, 35)
    ax_local.set_ylim(-15, 80)

    fig.suptitle(
        f"rank={rank} score={float(row['score']):.2f} | tags={row['tags']} | "
        f"ego={float(row['ego_speed']):.1f} object={row['object_category']}:{float(row['object_speed']):.1f} | "
        f"lat={float(row['lateral_shift']):.1f} pass={float(row['pass_margin']):.1f}"
    )

    vis_dir = args.output_dir / "candidates"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / f"bypass_{rank:03d}_{db_path.stem}_{center_timestamp}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_tags = args.source_tag or list(DEFAULT_SOURCE_TAGS)
    db_paths = find_db_paths(args.db_root, args.max_dbs)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for db_index, db_path in enumerate(db_paths, start=1):
        with sqlite3.connect(db_path) as con:
            tag_candidates = load_candidate_tags(
                con,
                source_tags,
                dedupe_sec=args.dedupe_sec,
                max_candidates=args.max_candidates_per_db,
            )
        for candidate in tag_candidates:
            key = (str(db_path), int(candidate["timestamp"]))
            if key in seen:
                continue
            seen.add(key)
            row = score_candidate(db_path, candidate, args)
            if row is not None:
                rows.append(row)
        if db_index % 10 == 0:
            print(f"scanned dbs={db_index}/{len(db_paths)} candidates={len(rows)}")

    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    rows = rows[: args.top_k]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    csv_path = args.output_dir / "bypass_candidates.csv"
    write_csv(csv_path, rows)
    for rank, row in enumerate(rows[: args.visualize_top_k], start=1):
        output_path = visualize_candidate(row, rank, args)
        print(f"saved candidate {rank:03d}: {output_path}")

    print(f"dbs_scanned: {len(db_paths)}")
    print(f"source_tags: {','.join(source_tags)}")
    print(f"bypass_candidates: {len(rows)}")
    print(f"saved csv to: {csv_path}")


if __name__ == "__main__":
    main()
