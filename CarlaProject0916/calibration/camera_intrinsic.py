import math
import numpy as np


def compute_camera_intrinsic(width, height, fov_deg):
    """
    CARLA camera fov is horizontal field of view.

    K:
        [fx  0 cx]
        [ 0 fy cy]
        [ 0  0  1]
    """
    fov_rad = math.radians(fov_deg)

    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K


def print_intrinsic(name, K):
    print(f"\n[CALIB] Camera '{name}' intrinsic K:")
    print(K)