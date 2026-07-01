from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np


class ArgoverseMap:
    """
    读取 Argoverse 1 vector map XML。

    第一版只解析 lane centerline:
      - XML 里的 node 保存点坐标
      - way 表示一条 lane polyline
      - way 里的 nd ref 引用 node id
    """

    def __init__(self, map_dir: str | Path) -> None:
        self.map_dir = Path(map_dir)

        self.city_to_xml_path = {
            "MIA": self.map_dir / "pruned_argoverse_MIA_10316_vector_map.xml",
            "PIT": self.map_dir / "pruned_argoverse_PIT_10314_vector_map.xml",
        }

        self.city_lane_centerlines: Dict[str, List[np.ndarray]] = {}

        for city_name, xml_path in self.city_to_xml_path.items():
            self.city_lane_centerlines[city_name] = self._load_city_lanes(xml_path)

    def get_lane_centerlines_near(
        self,
        city_name: str,
        center: np.ndarray,
        radius: float = 60.0,
    ) -> List[np.ndarray]:
        """
        取目标车附近的 lane centerlines。

        city_name:
            "MIA" 或 "PIT"

        center:
            [2]，目标车当前全局坐标。

        radius:
            搜索半径，单位米。
        """
        if city_name not in self.city_lane_centerlines:
            raise ValueError(f"Unknown city_name: {city_name}")

        center = np.asarray(center, dtype=np.float32)

        nearby_lanes: List[np.ndarray] = []

        for lane in self.city_lane_centerlines[city_name]:
            distances = np.linalg.norm(lane - center, axis=1)

            if np.min(distances) <= radius:
                nearby_lanes.append(lane)

        return nearby_lanes

    def _load_city_lanes(self, xml_path: Path) -> List[np.ndarray]:
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)

        root = ET.parse(xml_path).getroot()

        node_xy: Dict[str, np.ndarray] = {}

        for child in root:
            if child.tag != "node":
                continue

            node_id = child.attrib["id"]
            x = float(child.attrib["x"])
            y = float(child.attrib["y"])

            node_xy[node_id] = np.asarray([x, y], dtype=np.float32)

        lane_centerlines: List[np.ndarray] = []

        for child in root:
            if child.tag != "way":
                continue

            points = []

            for item in child:
                if item.tag != "nd":
                    continue

                node_id = item.attrib["ref"]
                points.append(node_xy[node_id])

            if len(points) >= 2:
                lane_centerlines.append(np.stack(points, axis=0).astype(np.float32))

        return lane_centerlines
