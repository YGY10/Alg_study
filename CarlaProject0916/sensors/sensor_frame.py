from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CameraFrame:
    name: str
    frame_id: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int
    sensor_type: str = "rgb"


@dataclass
class MultiCameraFrame:
    frames: Dict[str, Optional[CameraFrame]]
    timestamp: float
    semantic_frames: Optional[Dict[str, Optional[CameraFrame]]] = None
    depth_frames: Optional[Dict[str, Optional[CameraFrame]]] = None
    frame_id: Optional[int] = None
    synchronized: bool = False
