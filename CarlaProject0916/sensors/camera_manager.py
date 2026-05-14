import time

from config.camera_config import CAMERA_CONFIGS
from sensors.camera_sensor import CameraSensor
from sensors.sensor_frame import MultiCameraFrame


class CameraManager:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.cameras = {}

    def spawn_all(self):
        for name, cfg in CAMERA_CONFIGS.items():
            camera = CameraSensor(
                world=self.world,
                ego_vehicle=self.ego_vehicle,
                name=name,
                config=cfg,
            )
            camera.spawn()
            self.cameras[name] = camera

        print(f"[INFO] Total cameras spawned: {len(self.cameras)}")

    def get_latest_frames(self):
        frames = {}

        for name, camera in self.cameras.items():
            frames[name] = camera.get_latest_frame()

        return MultiCameraFrame(
            frames=frames,
            timestamp=time.time(),
        )

    def destroy_all(self):
        for camera in self.cameras.values():
            camera.destroy()

        self.cameras.clear()