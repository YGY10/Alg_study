# sensors/camera_sensor.py

import queue
import time
import threading

import carla
import numpy as np

from sensors.sensor_frame import CameraFrame


class CameraSensor:
    def __init__(self, world, ego_vehicle, name, config):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.name = name
        self.config = config

        self.actor = None

        # 仍然保留 queue，但不再依赖 queue 是否为空决定显示黑图
        self.frame_queue = queue.Queue(maxsize=1)

        # 关键：保存上一帧有效图像
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()

        self.total_received = 0
        self.total_dropped = 0
        self.last_report_time = time.time()

    def spawn(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", str(self.config["width"]))
        camera_bp.set_attribute("image_size_y", str(self.config["height"]))
        camera_bp.set_attribute("fov", str(self.config["fov"]))
        camera_bp.set_attribute("sensor_tick", str(self.config["sensor_tick"]))

        loc = self.config["location"]
        rot = self.config["rotation"]

        transform = carla.Transform(
            carla.Location(x=loc[0], y=loc[1], z=loc[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        self.actor = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.ego_vehicle,
        )

        self.actor.listen(self._on_image)

        print(
            f"[INFO] Camera '{self.name}' spawned: "
            f"loc={loc}, rot={rot}, "
            f"size={self.config['width']}x{self.config['height']}, "
            f"fov={self.config['fov']}, "
            f"tick={self.config['sensor_tick']}"
        )

    def _on_image(self, image):
        frame_bgr = self._carla_image_to_bgr(image)

        camera_frame = CameraFrame(
            name=self.name,
            frame_id=image.frame,
            timestamp=time.time(),
            image=frame_bgr,
            width=image.width,
            height=image.height,
        )

        self.total_received += 1

        # 关键：无论 queue 情况如何，都保存 latest_frame
        with self.latest_frame_lock:
            self.latest_frame = camera_frame

        # queue 只用于“最新帧通知”，满了就丢旧帧
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.total_dropped += 1
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(camera_frame)
        except queue.Full:
            self.total_dropped += 1

        # 偶尔打印一次，不要每帧刷屏
        now = time.time()
        if now - self.last_report_time > 5.0:
            print(
                f"[CAMERA DEBUG] {self.name}: "
                f"received={self.total_received}, "
                f"dropped={self.total_dropped}, "
                f"latest_frame={image.frame}"
            )
            self.last_report_time = now

    @staticmethod
    def _carla_image_to_bgr(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # CARLA RGB camera raw_data is BGRA.
        # OpenCV uses BGR, so直接取前三通道即可。
        bgr = array[:, :, :3].copy()
        return bgr

    def get_latest_frame(self):
        """
        返回最新有效帧。

        以前的问题：
            queue 被主线程取空后，如果下一轮还没有新 callback，
            就返回 None，viewer 会显示黑图，造成“眨眼”。

        现在：
            queue 有新帧就取新帧；
            queue 没新帧就返回 latest_frame；
            只有程序刚启动、从来没收到过图时才返回 None。
        """
        try:
            frame = self.frame_queue.get_nowait()

            with self.latest_frame_lock:
                self.latest_frame = frame

            return frame

        except queue.Empty:
            with self.latest_frame_lock:
                return self.latest_frame

    def has_frame(self):
        with self.latest_frame_lock:
            return self.latest_frame is not None

    def destroy(self):
        if self.actor is not None:
            print(f"[INFO] Destroying camera '{self.name}'.")
            try:
                self.actor.stop()
            except Exception as e:
                print(f"[WARN] Failed to stop camera '{self.name}': {e}")

            try:
                self.actor.destroy()
            except Exception as e:
                print(f"[WARN] Failed to destroy camera '{self.name}': {e}")

            self.actor = None