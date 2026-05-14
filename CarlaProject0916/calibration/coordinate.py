import math
import numpy as np


def deg2rad(deg):
    return deg * math.pi / 180.0


def rotation_matrix_roll_pitch_yaw(roll_deg, pitch_deg, yaw_deg):
    """
    CARLA-like rotation matrix.

    Coordinate convention:
        x forward
        y right
        z up

    Rotation order:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Important:
        In CARLA/UE style, negative pitch means looking downward.
        Therefore pitch=-10deg makes local x-axis point slightly toward -z.
    """
    roll = deg2rad(roll_deg)
    pitch = deg2rad(pitch_deg)
    yaw = deg2rad(yaw_deg)

    cr = math.cos(roll)
    sr = math.sin(roll)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    # CARLA pitch convention:
    # local x axis after pitch is [cos(pitch), 0, sin(pitch)].
    # pitch=-10deg => z component is negative => looking down.
    Ry = np.array(
        [
            [cp, 0.0, -sp],
            [0.0, 1.0, 0.0],
            [sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


def make_transform_matrix(R, t):
    """
    Make 4x4 homogeneous transform.

    p_parent = R @ p_child + t
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def inverse_transform(T):
    """
    Invert 4x4 rigid transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def transform_points(T, points):
    """
    Transform Nx3 points by 4x4 matrix.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim == 1:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])

    transformed = (T @ points_h.T).T
    return transformed[:, :3]


def carla_camera_to_optical_points(points_carla_camera):
    """
    Convert CARLA camera local frame to optical camera frame.

    CARLA camera local:
        x forward
        y right
        z up

    Optical camera frame:
        x right
        y down
        z forward

    Mapping:
        x_opt = y_carla
        y_opt = -z_carla
        z_opt = x_carla
    """
    p = np.asarray(points_carla_camera, dtype=np.float64)

    if p.ndim == 1:
        p = p.reshape(1, 3)

    x_opt = p[:, 1]
    y_opt = -p[:, 2]
    z_opt = p[:, 0]

    return np.stack([x_opt, y_opt, z_opt], axis=1)


def print_transform(name, T):
    print(f"\n[CALIB] {name}:")
    print(T)