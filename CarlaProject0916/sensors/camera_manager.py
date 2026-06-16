import time

from config.camera_config import (
    CAMERA_CONFIGS,
    ENABLE_DEPTH_CAMERA,
    ENABLE_SEMANTIC_CAMERA,
)
from sensors.camera_sensor import CameraSensor
from sensors.sensor_frame import MultiCameraFrame


class CameraManager:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.cameras = {}
        self.semantic_cameras = {}
        self.depth_cameras = {}

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

            if ENABLE_SEMANTIC_CAMERA:
                semantic_camera = CameraSensor(
                    world=self.world,
                    ego_vehicle=self.ego_vehicle,
                    name=name,
                    config=cfg,
                    sensor_type="semantic",
                )
                semantic_camera.spawn()
                self.semantic_cameras[name] = semantic_camera

            if ENABLE_DEPTH_CAMERA:
                depth_camera = CameraSensor(
                    world=self.world,
                    ego_vehicle=self.ego_vehicle,
                    name=name,
                    config=cfg,
                    sensor_type="depth",
                )
                depth_camera.spawn()
                self.depth_cameras[name] = depth_camera

        print(
            f"[INFO] Total cameras spawned: "
            f"rgb={len(self.cameras)}, "
            f"semantic={len(self.semantic_cameras)}, "
            f"depth={len(self.depth_cameras)}"
        )

    def get_latest_frames(self):
        frames = self._get_latest_from_group(self.cameras)
        semantic_frames = self._get_latest_from_group(self.semantic_cameras)
        depth_frames = self._get_latest_from_group(self.depth_cameras)

        frame_ids = [
            frame.frame_id
            for frame in frames.values()
            if frame is not None
        ]

        return MultiCameraFrame(
            frames=frames,
            timestamp=time.time(),
            semantic_frames=semantic_frames,
            depth_frames=depth_frames,
            frame_id=max(frame_ids) if frame_ids else None,
            synchronized=False,
        )

    def get_synchronized_frames(self):
        """
        Return frames with a common CARLA frame_id when possible.

        If a full RGB/semantic/depth intersection is unavailable during startup,
        this falls back to latest frames so visualization can still begin.
        """
        self._drain_all_latest()

        groups = [self.cameras]
        if self.semantic_cameras:
            groups.append(self.semantic_cameras)
        if self.depth_cameras:
            groups.append(self.depth_cameras)

        common_ids = None
        for group in groups:
            group_common = self._common_frame_ids(group)
            if not group_common:
                return self.get_latest_frames()

            if common_ids is None:
                common_ids = group_common
            else:
                common_ids &= group_common

        if not common_ids:
            return self.get_latest_frames()

        frame_id = max(common_ids)

        return MultiCameraFrame(
            frames=self._get_group_by_frame_id(self.cameras, frame_id),
            timestamp=time.time(),
            semantic_frames=self._get_group_by_frame_id(
                self.semantic_cameras,
                frame_id,
            ),
            depth_frames=self._get_group_by_frame_id(
                self.depth_cameras,
                frame_id,
            ),
            frame_id=frame_id,
            synchronized=True,
        )

    def destroy_all(self):
        for group in (self.cameras, self.semantic_cameras, self.depth_cameras):
            for camera in group.values():
                camera.destroy()

        self.cameras.clear()
        self.semantic_cameras.clear()
        self.depth_cameras.clear()

    @staticmethod
    def _get_latest_from_group(group):
        frames = {}
        for name, camera in group.items():
            frames[name] = camera.get_latest_frame()
        return frames

    def _drain_all_latest(self):
        for group in (self.cameras, self.semantic_cameras, self.depth_cameras):
            for camera in group.values():
                camera.get_latest_frame()

    @staticmethod
    def _common_frame_ids(group):
        common = None
        for camera in group.values():
            ids = camera.get_available_frame_ids()
            if common is None:
                common = ids
            else:
                common &= ids
        return common or set()

    @staticmethod
    def _get_group_by_frame_id(group, frame_id):
        return {
            name: camera.get_frame_by_id(frame_id)
            for name, camera in group.items()
        }
