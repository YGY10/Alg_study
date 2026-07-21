from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim2d.map.types import LaneType

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WorldLaneGeometry:
    """真实世界中的一条车道几何，拓扑 ID 继承地图层。"""

    entity_id: str
    map_lane_id: str
    lane_type: LaneType

    centerline: FloatArray
    left_boundary: FloatArray
    right_boundary: FloatArray

    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "centerline",
            "left_boundary",
            "right_boundary",
        ):
            array = np.asarray(
                getattr(self, name),
                dtype=np.float64,
            )

            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(
                    f"{name} must have shape [N, 2], got {array.shape}"
                )

            if array.shape[0] < 2:
                raise ValueError(
                    f"{name} must contain at least two points"
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{name} must contain only finite values"
                )

            object.__setattr__(
                self,
                name,
                array.copy(),
            )

        object.__setattr__(
            self,
            "predecessor_ids",
            tuple(self.predecessor_ids),
        )
        object.__setattr__(
            self,
            "successor_ids",
            tuple(self.successor_ids),
        )
