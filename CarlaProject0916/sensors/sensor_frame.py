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


@dataclass
class MultiCameraFrame:
    frames: Dict[str, Optional[CameraFrame]]
    timestamp: float