from __future__ import annotations
import numpy as np

NUM_ANCHORS = 21
NUM_POINTS = 31
FORWARD_DISTANCE = 30.0
MAX_LATERAL_OFFSET = 12.0


def build_trajectory_anchors(
    num_anchors: int = NUM_ANCHORS,
    num_points: int = NUM_POINTS,
    forward_distance: float = FORWARD_DISTANCE,
    max_lateral_offset: float = MAX_LATERAL_OFFSET,
) -> np.ndarray:
    """
    Build fixed smooth path anchors
    Coordinate definition:
        x: longitudinal / forward, positive in front of ego.
        y: lateral, positive to the left of ego.

    Returns:
        anchors: [num_anchors, num_points, 2], each point is [x, y].
    """
    terminal_y = np.linspace(
        -max_lateral_offset,
        max_lateral_offset,
        num_anchors,
        dtype=np.float32,
    )
    progress = np.linspace(0.0, 1.0, num_points, dtype=np.float32)

    smooth_ratio = 3.0 * progress**2 - 2 * progress**3

    anchors = []
    for end_y in terminal_y:
        x = forward_distance * progress
        y = end_y * smooth_ratio
        trajectory = np.stack([x, y], axis=-1)
        anchors.append(trajectory)
    return np.stack(anchors, axis=0)


if __name__ == "__main__":
    anchors = build_trajectory_anchors()

    print("anchors shape:", anchors.shape)
    print("rightmost end point:", anchors[0, -1])
    print("straight end point:", anchors[len(anchors) // 2, -1])
    print("leftmost end point:", anchors[-1, -1])
