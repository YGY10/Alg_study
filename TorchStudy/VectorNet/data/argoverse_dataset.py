from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from data.line import Line, LineKind, SceneSample
from data.argoverse_map import ArgoverseMap


class ArgoverseForecastingDataset(Dataset):
    """
    Argoverse 1 motion forecasting dataset 的第一版读取器。

    当前版本只读取轨迹，不读取 HD map。

    每个 sample:
      - AGENT 的前 history_steps 帧作为目标车历史轨迹
      - AGENT 的后 future_steps 帧作为监督 future
      - 其他车辆的前 history_steps 帧作为上下文 agent polyline
    """

    def __init__(
        self,
        data_dir: str | Path,
        history_steps: int = 20,
        future_steps: int = 30,
        max_samples: int | None = None,
        map_dir: str | Path | None = None,
        use_map: bool = False,
        lane_radius: float = 60.0,
        max_lanes: int | None = 32,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_map = use_map
        self.lane_radius = lane_radius
        self.max_lanes = max_lanes

        if self.use_map:
            if map_dir is None:
                raise ValueError("map_dir must be provided when use_map=True.")
            self.map = ArgoverseMap(map_dir)
        else:
            self.map = None

        self.csv_paths = sorted(self.data_dir.glob("*.csv"))

        if max_samples is not None:
            self.csv_paths = self.csv_paths[:max_samples]

        if len(self.csv_paths) == 0:
            raise ValueError(f"No csv files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.csv_paths)

    def __getitem__(self, idx: int) -> SceneSample:
        csv_path = self.csv_paths[idx]
        rows = self._read_csv(csv_path)
        city_name = str(rows[0]["city_name"])
        timestamps = sorted({row["timestamp"] for row in rows})
        required_steps = self.history_steps + self.future_steps

        if len(timestamps) < required_steps:
            raise ValueError(
                f"{csv_path} has only {len(timestamps)} timestamps, "
                f"but requires {required_steps}."
            )

        history_timestamps = set(timestamps[: self.history_steps])
        future_timestamps = set(
            timestamps[self.history_steps : self.history_steps + self.future_steps]
        )

        target_track_id = self._find_target_track_id(rows)

        target_history = self._get_track_points(
            rows=rows,
            track_id=target_track_id,
            timestamp_set=history_timestamps,
        )

        target_future = self._get_track_points(
            rows=rows,
            track_id=target_track_id,
            timestamp_set=future_timestamps,
        )

        if target_history.shape[0] != self.history_steps:
            raise ValueError(
                f"{csv_path} target history length is {target_history.shape[0]}, "
                f"expected {self.history_steps}."
            )

        if target_future.shape[0] != self.future_steps:
            raise ValueError(
                f"{csv_path} target future length is {target_future.shape[0]}, "
                f"expected {self.future_steps}."
            )

        # 以目标车最后一个历史点为原点，并把目标车当前朝向旋转到 +x 轴。
        origin = target_history[-1].copy()
        heading = self._compute_heading(target_history)

        target_history = self._normalize_points(
            points=target_history,
            origin=origin,
            heading=heading,
        )

        target_future = self._normalize_points(
            points=target_future,
            origin=origin,
            heading=heading,
        )

        lines: List[Line] = []
        line_id = 0

        target_agent_id = 0

        lines.append(
            Line(
                points=target_history,
                kind=LineKind.AGENT_HISTORY,
                line_id=line_id,
                agent_id=target_agent_id,
                is_target=True,
            )
        )
        line_id += 1

        other_track_ids = sorted(
            {
                row["track_id"]
                for row in rows
                if row["track_id"] != target_track_id
                and row["timestamp"] in history_timestamps
            }
        )

        next_agent_id = 1

        for track_id in other_track_ids:
            points = self._get_track_points(
                rows=rows,
                track_id=track_id,
                timestamp_set=history_timestamps,
            )

            # 有些其他车只出现 1 帧，无法构成 Line，因为 Line 至少需要 2 个点。
            if points.shape[0] < 2:
                continue

            points = self._normalize_points(
                points=points,
                origin=origin,
                heading=heading,
            )

            lines.append(
                Line(
                    points=points,
                    kind=LineKind.AGENT_HISTORY,
                    line_id=line_id,
                    agent_id=next_agent_id,
                    is_target=False,
                )
            )

            line_id += 1
            next_agent_id += 1

        if self.use_map:
            assert self.map is not None

            lane_centerlines = self.map.get_lane_centerlines_near(
                city_name=city_name,
                center=origin,
                radius=self.lane_radius,
            )
            lane_centerlines = self._select_nearest_lanes(
                lanes=lane_centerlines,
                center=origin,
            )

            for lane in lane_centerlines:
                normalized_lane = self._normalize_points(
                    points=lane,
                    origin=origin,
                    heading=heading,
                )

                # 有些 lane 在局部坐标下仍然很长，第一版先保留原始点。
                # Line 至少需要 2 个点。
                if normalized_lane.shape[0] < 2:
                    continue

                lines.append(
                    Line(
                        points=normalized_lane,
                        kind=LineKind.LANE_LINE,
                        line_id=line_id,
                    )
                )
                line_id += 1

        return SceneSample(
            lines=lines,
            target_agent_id=target_agent_id,
            future=target_future,
        )

    def _read_csv(self, csv_path: Path) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rows.append(
                    {
                        "timestamp": float(row["TIMESTAMP"]),
                        "track_id": row["TRACK_ID"],
                        "object_type": row["OBJECT_TYPE"],
                        "x": float(row["X"]),
                        "y": float(row["Y"]),
                        "city_name": row["CITY_NAME"],
                    }
                )

        return rows

    def _find_target_track_id(self, rows: List[Dict[str, object]]) -> str:
        for row in rows:
            if row["object_type"] == "AGENT":
                return str(row["track_id"])

        raise ValueError("No AGENT track found.")

    def _compute_heading(self, points: np.ndarray) -> float:
        """
        用目标车最后两个历史点估计当前朝向。
        """
        if points.shape[0] < 2:
            raise ValueError("Need at least 2 points to compute heading.")

        direction = points[-1] - points[-2]
        norm = float(np.linalg.norm(direction))

        if norm < 1e-6:
            return 0.0

        return float(np.arctan2(direction[1], direction[0]))

    def _normalize_points(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """
        平移到 origin，再旋转 -heading。

        原始坐标系:
            任意城市全局坐标。

        归一化后:
            目标车当前位置在 (0, 0)
            目标车当前朝向大致对齐 +x 方向。
        """
        translated = points - origin

        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)

        rotation = np.array(
            [
                [cos_h, -sin_h],
                [sin_h, cos_h],
            ],
            dtype=np.float32,
        )

        return translated @ rotation.T

    def _select_nearest_lanes(
        self,
        lanes: List[np.ndarray],
        center: np.ndarray,
    ) -> List[np.ndarray]:
        if self.max_lanes is None or len(lanes) <= self.max_lanes:
            return lanes

        center = np.asarray(center, dtype=np.float32)

        def lane_distance(lane: np.ndarray) -> float:
            distances = np.linalg.norm(lane - center, axis=1)
            return float(np.min(distances))

        return sorted(lanes, key=lane_distance)[: self.max_lanes]

    def _get_track_points(
        self,
        rows: List[Dict[str, object]],
        track_id: str,
        timestamp_set: set[float],
    ) -> np.ndarray:
        track_rows = [
            row
            for row in rows
            if row["track_id"] == track_id and row["timestamp"] in timestamp_set
        ]

        track_rows.sort(key=lambda row: row["timestamp"])

        points = [[float(row["x"]), float(row["y"])] for row in track_rows]

        return np.asarray(points, dtype=np.float32)
