# config/camera_config.py

# CARLA vehicle local coordinate:
# x: forward
# y: right
# z: up
#
# Rotation:
# pitch: down is negative
# yaw:
#   front       = 0
#   front_right = 45
#   right       = 90
#   rear_right  = 135
#   rear        = 180
#   rear_left   = -135
#   left        = -90
#   front_left  = -45
# roll: usually 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_SENSOR_TICK = 0.05  # 20Hz target

# 默认中远距 FOV
CAMERA_FOV = 95.0


def make_camera_config(
    location,
    rotation,
    fov,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    sensor_tick=CAMERA_SENSOR_TICK,
):
    return {
        "width": width,
        "height": height,
        "fov": fov,
        "sensor_tick": sensor_tick,
        "location": location,
        "rotation": rotation,
    }


CAMERA_CONFIGS = {
    # =========================
    # Front sector
    # front 保留中远距能力
    # front_left / front_right 加强近场覆盖
    # =========================
    "front": make_camera_config(
        location=[1.9, 0.0, 1.9],
        rotation=[-8.0, 0.0, 0.0],
        fov=95.0,
    ),

    "front_right": make_camera_config(
        location=[1.3, 0.95, 1.65],
        rotation=[-18.0, 45.0, 0.0],
        fov=120.0,
    ),

    "front_left": make_camera_config(
        location=[1.3, -0.95, 1.65],
        rotation=[-18.0, -45.0, 0.0],
        fov=120.0,
    ),

    # =========================
    # Side sector
    # left / right 强化自车左右近场
    # =========================
    "right": make_camera_config(
        location=[0.0, 1.25, 1.45],
        rotation=[-30.0, 90.0, 0.0],
        fov=140.0,
    ),

    "left": make_camera_config(
        location=[0.0, -1.25, 1.45],
        rotation=[-30.0, -90.0, 0.0],
        fov=140.0,
    ),

    # =========================
    # Rear sector
    # rear / rear_left / rear_right 偏近场一些
    # =========================
    "rear_right": make_camera_config(
        location=[-1.3, 0.95, 1.65],
        rotation=[-20.0, 135.0, 0.0],
        fov=120.0,
    ),

    "rear": make_camera_config(
        location=[-1.9, 0.0, 1.75],
        rotation=[-15.0, 180.0, 0.0],
        fov=110.0,
    ),

    "rear_left": make_camera_config(
        location=[-1.3, -0.95, 1.65],
        rotation=[-20.0, -135.0, 0.0],
        fov=120.0,
    ),
}


# 显示顺序：2 x 4
CAMERA_DISPLAY_ORDER = [
    "front_left",
    "front",
    "front_right",
    "right",
    "rear_right",
    "rear",
    "rear_left",
    "left",
]