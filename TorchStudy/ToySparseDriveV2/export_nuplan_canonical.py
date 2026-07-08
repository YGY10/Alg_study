from __future__ import annotations
import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from shapely import from_wkb

DEFAULT_DB_ROOT = Path("nuplan_dataset/nuplan-v1.1_mini/data/cache/mini")
DEFAULT_OUTPUT_DIR = Path("outputs/nuplan_canonical/episodes")
DEFAULT_MAP_ROOT = Path("nuplan_dataset/maps")
MAP_VERSION_TO_DIR = {
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dbs", type=int, default=0)
    parser.add_argument("--max-scenes-per-db", type=int, default=0)
    parser.add_argument("--min-duration-sec", type=float, default=6.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    return parser.parse_args()


def yaw_from_quaternion(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def token_hex(token):
    return token.hex() if token is not None else ""


def normalize_category(name: str) -> str:
    name = name.lower()
    if "vehicle" in name or "car" in name or "truck" in name or "bus" in name:
        return "vehicle"
    if "pedestrian" in name:
        return "pedestrian"
    if "bicycle" in name or "cyclist" in name:
        return "bicycle"
    if "traffic_cone" in name or "cone" in name:
        return "traffic_cone"
    if "barrier" in name:
        return "barrier"
    return "generic_static"


def gpkg_geom_to_shapely(blob: bytes):
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        return None
    flags = blob[3]
    envelope_code = (flags >> 1) & 0x07
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    offset = 8 + envelope_sizes.get(envelope_code, 0)
    return from_wkb(blob[offset:])


def lonlat_to_utm_xy(lon: float, lat: float) -> tuple[float, float]:
    # WGS84 transverse Mercator, enough for nuPlan local map visualization/training.
    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    k0 = 0.9996
    zone = int((lon + 180.0) / 6.0) + 1
    lon_origin = (zone - 1) * 6.0 - 180.0 + 3.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon_origin_rad = math.radians(lon_origin)
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    n = a / math.sqrt(1.0 - ecc_sq * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = ecc_prime_sq * math.cos(lat_rad) ** 2
    a_term = math.cos(lat_rad) * (lon_rad - lon_origin_rad)
    m = a * (
        (1.0 - ecc_sq / 4.0 - 3.0 * ecc_sq**2 / 64.0 - 5.0 * ecc_sq**3 / 256.0) * lat_rad
        - (3.0 * ecc_sq / 8.0 + 3.0 * ecc_sq**2 / 32.0 + 45.0 * ecc_sq**3 / 1024.0) * math.sin(2.0 * lat_rad)
        + (15.0 * ecc_sq**2 / 256.0 + 45.0 * ecc_sq**3 / 1024.0) * math.sin(4.0 * lat_rad)
        - (35.0 * ecc_sq**3 / 3072.0) * math.sin(6.0 * lat_rad)
    )
    x = k0 * n * (
        a_term
        + (1.0 - t + c) * a_term**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * ecc_prime_sq) * a_term**5 / 120.0
    ) + 500000.0
    y = k0 * (
        m
        + n * math.tan(lat_rad) * (
            a_term**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * a_term**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * ecc_prime_sq) * a_term**6 / 720.0
        )
    )
    if lat < 0:
        y += 10000000.0
    return x, y


def line_geom_to_utm_path(geom) -> list[list[float]]:
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "MultiLineString":
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
    else:
        coords = list(geom.exterior.coords) if hasattr(geom, "exterior") else []
    xy = [lonlat_to_utm_xy(float(lon), float(lat)) for lon, lat, *_ in coords]
    if len(xy) < 2:
        return []
    yaws = []
    for i, point in enumerate(xy):
        if i + 1 < len(xy):
            nxt = xy[i + 1]
            dx = nxt[0] - point[0]
            dy = nxt[1] - point[1]
        else:
            prev = xy[i - 1]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
        yaws.append(math.atan2(dy, dx))
    return [[float(x), float(y), float(yaw)] for (x, y), yaw in zip(xy, yaws)]


def polygon_geom_to_utm_rings(geom) -> list[list[list[float]]]:
    if geom is None:
        return []
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        return []

    rings: list[list[list[float]]] = []
    for polygon in polygons:
        ring = []
        for lon, lat, *_ in polygon.exterior.coords:
            x, y = lonlat_to_utm_xy(float(lon), float(lat))
            ring.append([float(x), float(y)])
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def map_path_for_scene(map_root: Path, scene: dict[str, Any]) -> Path | None:
    relative = MAP_VERSION_TO_DIR.get(str(scene.get("map_version", "")))
    if relative is None:
        return None
    path = map_root / relative
    return path if path.is_file() else None


def route_candidates_for_roadblock(map_con: sqlite3.Connection, roadblock_id: int) -> list[list[list[float]]]:
    candidates: list[list[list[float]]] = []
    lane_rows = map_con.execute(
        "SELECT lane_fid FROM lanes_polygons WHERE lane_group_fid = ? ORDER BY lane_index, lane_fid",
        (roadblock_id,),
    ).fetchall()
    for (lane_fid,) in lane_rows:
        for (geom_blob,) in map_con.execute(
            "SELECT geom FROM baseline_paths WHERE lane_fid = ?",
            (lane_fid,),
        ).fetchall():
            path = line_geom_to_utm_path(gpkg_geom_to_shapely(geom_blob))
            if len(path) >= 2:
                candidates.append(path)

    connector_rows = map_con.execute(
        "SELECT fid FROM lane_connectors WHERE lane_group_connector_fid = ? ORDER BY fid",
        (roadblock_id,),
    ).fetchall()
    for (connector_fid,) in connector_rows:
        for (geom_blob,) in map_con.execute(
            "SELECT geom FROM baseline_paths WHERE lane_connector_fid = ?",
            (connector_fid,),
        ).fetchall():
            path = line_geom_to_utm_path(gpkg_geom_to_shapely(geom_blob))
            if len(path) >= 2:
                candidates.append(path)
    return candidates


def route_polygons_for_roadblock(
    map_con: sqlite3.Connection,
    roadblock_id: int,
) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    for table in ("lane_groups_polygons", "lane_group_connectors"):
        rows = map_con.execute(
            f"SELECT geom FROM {table} WHERE fid = ?",
            (roadblock_id,),
        ).fetchall()
        for (geom_blob,) in rows:
            polygons.extend(polygon_geom_to_utm_rings(gpkg_geom_to_shapely(geom_blob)))
    return polygons


def build_route_map_features(
    map_con: sqlite3.Connection,
    roadblock_ids: list[str],
) -> dict[str, list]:
    route_polygons: list[list[list[float]]] = []
    lane_centerlines: list[list[list[float]]] = []
    for raw_id in roadblock_ids:
        try:
            roadblock_id = int(raw_id)
        except ValueError:
            continue
        route_polygons.extend(route_polygons_for_roadblock(map_con, roadblock_id))
        lane_centerlines.extend(route_candidates_for_roadblock(map_con, roadblock_id))
    return {
        "route_polygons": route_polygons,
        "lane_centerlines": lane_centerlines,
    }


def reverse_path(path: list[list[float]]) -> list[list[float]]:
    reversed_path = [point[:] for point in reversed(path)]
    for i, point in enumerate(reversed_path):
        if i + 1 < len(reversed_path):
            nxt = reversed_path[i + 1]
            point[2] = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
        elif i > 0:
            point[2] = reversed_path[i - 1][2]
    return reversed_path


def path_endpoint_distance(path: list[list[float]], xy: tuple[float, float]) -> float:
    return math.hypot(path[0][0] - xy[0], path[0][1] - xy[1])


def build_route_from_map(
    map_con: sqlite3.Connection,
    roadblock_ids: list[str],
    start_xy: tuple[float, float],
) -> list[list[float]]:
    route: list[list[float]] = []
    current_xy = start_xy
    for raw_id in roadblock_ids:
        try:
            roadblock_id = int(raw_id)
        except ValueError:
            continue
        candidates = route_candidates_for_roadblock(map_con, roadblock_id)
        oriented_candidates = []
        for candidate in candidates:
            oriented_candidates.append(candidate)
            oriented_candidates.append(reverse_path(candidate))
        if not oriented_candidates:
            continue
        best = min(oriented_candidates, key=lambda path: path_endpoint_distance(path, current_xy))
        if route and math.hypot(best[0][0] - route[-1][0], best[0][1] - route[-1][1]) < 1.0:
            route.extend(best[1:])
        else:
            route.extend(best)
        current_xy = (route[-1][0], route[-1][1])
    return route


def find_db_paths(db_root: Path, max_dbs: int) -> list[Path]:
    paths = sorted(db_root.glob("*.db"))
    if max_dbs > 0:
        paths = paths[:max_dbs]
    if not paths:
        raise FileNotFoundError(f"No .db files under {db_root}")
    return paths


def load_scenes(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute("""
        SELECT
            scene.token,
            scene.name,
            scene.goal_ego_pose_token,
            scene.roadblock_ids,
            log.location,
            log.map_version,
            log.logfile
        FROM scene
        JOIN log ON log.token = scene.log_token
        ORDER BY scene.name
        """).fetchall()

    return [
        {
            "token": row[0],
            "name": str(row[1]),
            "goal_ego_pose_token": row[2],
            "roadblock_ids": str(row[3] or ""),
            "location": str(row[4]),
            "map_version": str(row[5]),
            "logfile": str(row[6]),
        }
        for row in rows
    ]


def load_ego_samples(con, scene_token) -> list[dict[str, Any]]:
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
            ego_pose.vy,
            ego_pose.acceleration_x,
            ego_pose.acceleration_y,
            ego_pose.angular_rate_z
        FROM lidar_pc
        JOIN ego_pose ON ego_pose.token = lidar_pc.ego_pose_token
        WHERE lidar_pc.scene_token = ?
        ORDER BY lidar_pc.timestamp
        """,
        (scene_token,),
    ).fetchall()

    samples = []
    if not rows:
        return samples

    t0 = int(rows[0][1])
    for row in rows:
        yaw = yaw_from_quaternion(
            float(row[4]), float(row[5]), float(row[6]), float(row[7])
        )
        vx = float(row[8])
        vy = float(row[9])
        ax = float(row[10])
        ay = float(row[11])
        samples.append(
            {
                "lidar_pc_token": row[0],
                "timestamp": int(row[1]),
                "t": (int(row[1]) - t0) / 1_000_000.0,
                "x": float(row[2]),
                "y": float(row[3]),
                "yaw": yaw,
                "speed": math.hypot(vx, vy),
                "accel": math.hypot(ax, ay),
                "yaw_rate": float(row[12]),
            }
        )
    return samples


def load_agents(
    con: sqlite3.Connection,
    lidar_pc_tokens: list[bytes],
    token_to_time: dict[bytes, float],
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
        ORDER BY lidar_box.track_token, lidar_box.lidar_pc_token
        """,
        lidar_pc_tokens,
    ).fetchall()

    agents_by_track: dict[bytes, dict[str, Any]] = {}

    for row in rows:
        lidar_pc_token = row[0]
        track_token = row[1]
        if lidar_pc_token not in token_to_time:
            continue

        category = normalize_category(str(row[9]))
        length = float(row[5])
        width = float(row[4])

        agent = agents_by_track.setdefault(
            track_token,
            {
                "track_id": token_hex(track_token),
                "type": category,
                "size": [length, width],
                "states": [],
            },
        )

        # Keep the first observed category/size. nuPlan boxes can have tiny noise;
        # canonical format wants one stable size per track.
        agent["states"].append(
            [
                float(token_to_time[lidar_pc_token]),
                float(row[2]),  # x
                float(row[3]),  # y
                float(row[8]),  # yaw
                float(row[6]),  # vx
                float(row[7]),  # vy
                1.0,  # valid
            ]
        )

    agents = list(agents_by_track.values())
    for agent in agents:
        agent["states"].sort(key=lambda state: state[0])

    return agents


def load_tags(
    con: sqlite3.Connection,
    lidar_pc_tokens: list[bytes],
) -> list[str]:
    if not lidar_pc_tokens:
        return []

    placeholders = ",".join("?" for _ in lidar_pc_tokens)
    rows = con.execute(
        f"""
        SELECT DISTINCT type
        FROM scenario_tag
        WHERE lidar_pc_token IN ({placeholders})
        ORDER BY type
        """,
        lidar_pc_tokens,
    ).fetchall()

    return [str(row[0]) for row in rows]


def load_goal_xy(
    con: sqlite3.Connection,
    goal_ego_pose_token: bytes | None,
    fallback_xy: list[float],
) -> list[float]:
    if goal_ego_pose_token is None:
        return fallback_xy

    row = con.execute(
        """
        SELECT x, y
        FROM ego_pose
        WHERE token = ?
        """,
        (goal_ego_pose_token,),
    ).fetchone()

    if row is None:
        return fallback_xy

    return [float(row[0]), float(row[1])]


def build_episode(
    db_path: Path,
    scene: dict[str, Any],
    ego_samples: list[dict[str, Any]],
    agents: list[dict[str, Any]],
    tags: list[str],
    goal_xy: list[float],
    route_path: list[list[float]] | None = None,
    route_source: str = "ego_path",
    map_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(ego_samples) < 2:
        raise ValueError(
            f"Scene has too few ego samples: {db_path.name}:{scene['name']}"
        )

    dt_values = [
        ego_samples[i + 1]["t"] - ego_samples[i]["t"]
        for i in range(len(ego_samples) - 1)
    ]
    dt_values = [dt for dt in dt_values if dt > 1.0e-6]
    dt = float(sorted(dt_values)[len(dt_values) // 2]) if dt_values else 0.1

    ego_history = [
        [
            float(sample["x"]),
            float(sample["y"]),
            float(sample["yaw"]),
            float(sample["speed"]),
            float(sample["accel"]),
            float(sample["yaw_rate"]),
        ]
        for sample in ego_samples
    ]

    if route_path is None or len(route_path) < 2:
        route_path = [
            [
                float(sample["x"]),
                float(sample["y"]),
                float(sample["yaw"]),
            ]
            for sample in ego_samples
        ]
        route_source = "ego_path"

    roadblock_ids = [
        item for item in str(scene.get("roadblock_ids", "")).split() if item
    ]

    return {
        "source": "nuplan",
        "scene_id": f"{db_path.stem}:{scene['name']}",
        "db_name": db_path.name,
        "location": scene["location"],
        "map_version": scene["map_version"],
        "logfile": scene["logfile"],
        "scenario_tags": tags,
        "dt": dt,
        "ego_history": ego_history,
        "agents": agents,
        "map": map_data or {},
        "route_path": route_path,
        "goal_xy": goal_xy,
        "nuplan": {
            "scene_token": token_hex(scene["token"]),
            "goal_ego_pose_token": token_hex(scene["goal_ego_pose_token"]),
            "roadblock_ids": roadblock_ids,
            "route_source": route_source,
        },
    }


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def export_db(
    db_path: Path,
    output_dir: Path,
    *,
    max_scenes_per_db: int,
    min_duration_sec: float,
    stride: int,
    map_root: Path,
) -> tuple[int, int]:
    exported = 0
    skipped = 0
    with sqlite3.connect(db_path) as con:
        scenes = load_scenes(con)
        if max_scenes_per_db > 0:
            scenes = scenes[:max_scenes_per_db]

        for scene in scenes:
            ego_samples = load_ego_samples(con, scene["token"])
            if stride > 1:
                ego_samples = ego_samples[::stride]
            if len(ego_samples) < 2:
                skipped += 1
                continue

            duration_sec = float(ego_samples[-1]["t"] - ego_samples[0]["t"])
            if duration_sec < min_duration_sec:
                skipped += 1
                continue

            lidar_pc_tokens = [sample["lidar_pc_token"] for sample in ego_samples]
            token_to_time = {
                sample["lidar_pc_token"]: float(sample["t"])
                for sample in ego_samples
            }
            agents = load_agents(con, lidar_pc_tokens, token_to_time)
            tags = load_tags(con, lidar_pc_tokens)
            fallback_goal = [
                float(ego_samples[-1]["x"]),
                float(ego_samples[-1]["y"]),
            ]
            goal_xy = load_goal_xy(
                con,
                scene["goal_ego_pose_token"],
                fallback_goal,
            )
            roadblock_ids = [
                item for item in str(scene.get("roadblock_ids", "")).split() if item
            ]
            route_path = None
            route_source = "ego_path"
            map_data: dict[str, Any] = {}
            map_path = map_path_for_scene(map_root, scene)
            if map_path is not None and roadblock_ids:
                with sqlite3.connect(map_path) as map_con:
                    map_data = build_route_map_features(map_con, roadblock_ids)
                    route_path = build_route_from_map(
                        map_con,
                        roadblock_ids,
                        start_xy=(float(ego_samples[0]["x"]), float(ego_samples[0]["y"])),
                    )
                if route_path and len(route_path) >= 2:
                    route_source = "roadblock_baseline_path"

            episode = build_episode(
                db_path=db_path,
                scene=scene,
                ego_samples=ego_samples,
                agents=agents,
                tags=tags,
                goal_xy=goal_xy,
                route_path=route_path,
                route_source=route_source,
                map_data=map_data,
            )

            output_path = output_dir / (
                safe_filename(f"{db_path.stem}_{scene['name']}") + ".json"
            )
            output_path.write_text(
                json.dumps(episode, indent=2),
                encoding="utf-8",
            )
            exported += 1

    return exported, skipped


def main() -> None:
    args = parse_args()
    db_paths = find_db_paths(args.db_root, args.max_dbs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_exported = 0
    total_skipped = 0
    for db_index, db_path in enumerate(db_paths, start=1):
        exported, skipped = export_db(
            db_path=db_path,
            output_dir=args.output_dir,
            max_scenes_per_db=args.max_scenes_per_db,
            min_duration_sec=args.min_duration_sec,
            stride=max(1, int(args.stride)),
            map_root=args.map_root,
        )
        total_exported += exported
        total_skipped += skipped
        print(
            f"[{db_index}/{len(db_paths)}] {db_path.name}: "
            f"exported={exported} skipped={skipped}"
        )

    print(f"dbs: {len(db_paths)}")
    print(f"episodes_exported: {total_exported}")
    print(f"scenes_skipped: {total_skipped}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()

