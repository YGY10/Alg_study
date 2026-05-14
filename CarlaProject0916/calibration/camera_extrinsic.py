import numpy as np

from calibration.coordinate import (
    rotation_matrix_roll_pitch_yaw,
    make_transform_matrix,
    inverse_transform,
)


def compute_camera_extrinsic(camera_config):
    """
    Compute camera extrinsic from config.

    Config rotation order:
        rotation = [pitch, yaw, roll]

    Returns:
        T_vehicle_camera:
            p_vehicle = T_vehicle_camera @ p_camera

        T_camera_vehicle:
            p_camera = T_camera_vehicle @ p_vehicle

    注意：
    这里先建立的是 CARLA sensor actor local 坐标系：
        x forward
        y right
        z up

    后面做 IPM 投影时，需要再处理 camera optical frame：
        x right
        y down
        z forward

    所以这里先不要急着把它用于像素投影。
    """
    loc = camera_config["location"]
    rot = camera_config["rotation"]

    pitch_deg = rot[0]
    yaw_deg = rot[1]
    roll_deg = rot[2]

    R_vehicle_camera = rotation_matrix_roll_pitch_yaw(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )

    t_vehicle_camera = np.array(
        [loc[0], loc[1], loc[2]],
        dtype=np.float64,
    )

    T_vehicle_camera = make_transform_matrix(
        R_vehicle_camera,
        t_vehicle_camera,
    )

    T_camera_vehicle = inverse_transform(T_vehicle_camera)

    return T_vehicle_camera, T_camera_vehicle


def print_extrinsic(name, T_vehicle_camera, T_camera_vehicle):
    print(f"\n[CALIB] Camera '{name}' T_vehicle_camera:")
    print(T_vehicle_camera)

    print(f"\n[CALIB] Camera '{name}' T_camera_vehicle:")
    print(T_camera_vehicle)