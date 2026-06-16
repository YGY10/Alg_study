# sensors/camera_sensor.py

import queue
import time
import threading

import carla
import numpy as np

from sensors.sensor_frame import CameraFrame


class CameraSensor:
    def __init__(self, world, ego_vehicle, name, config, sensor_type="rgb"):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.name = name
        self.config = config
        self.sensor_type = sensor_type

        self.actor = None

        # 仍然保留 queue，但不再依赖 queue 是否为空决定显示黑图
        self.frame_queue = queue.Queue(maxsize=1)

        # 关键：保存上一帧有效图像
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.frame_buffer = {}
        self.max_buffer_size = 8

        self.total_received = 0
        self.total_dropped = 0
        self.last_report_time = time.time()
        self.semantic_channel_index = None

    def spawn(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find(self._get_blueprint_id())

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
            f"[INFO] Camera '{self.name}' spawned: type={self.sensor_type}, "
            f"loc={loc}, rot={rot}, "
            f"size={self.config['width']}x{self.config['height']}, "
            f"fov={self.config['fov']}, "
            f"tick={self.config['sensor_tick']}"
        )

    def _get_blueprint_id(self):
        if self.sensor_type == "rgb":
            return "sensor.camera.rgb"
        if self.sensor_type == "semantic":
            return "sensor.camera.semantic_segmentation"
        if self.sensor_type == "depth":
            return "sensor.camera.depth"
        raise ValueError(f"[CameraSensor] Unsupported sensor_type: {self.sensor_type}")

    def _on_image(self, image):
        frame_image = self._convert_carla_image(image)

        camera_frame = CameraFrame(
            name=self.name,
            frame_id=image.frame,
            timestamp=time.time(),
            image=frame_image,
            width=image.width,
            height=image.height,
            sensor_type=self.sensor_type,
        )

        self.total_received += 1

        # 关键：无论 queue 情况如何，都保存 latest_frame
        with self.latest_frame_lock:
            self.latest_frame = camera_frame
            self.frame_buffer[camera_frame.frame_id] = camera_frame
            self._trim_frame_buffer_locked()

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

    def _carla_image_to_semantic_label(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # CARLA semantic raw_data is normally BGRA with the semantic tag in R.
        # Some setups/converters make this easy to misread, so auto-pick the
        # channel that contains the most plausible semantic IDs.
        if self.semantic_channel_index is None:
            self.semantic_channel_index = self._select_semantic_channel(array)
            channel_name = ("B", "G", "R")[self.semantic_channel_index]
            print(
                f"[SEMANTIC] camera={self.name}, "
                f"selected_channel={channel_name}"
            )

        return array[:, :, self.semantic_channel_index].copy()

    @staticmethod
    def _select_semantic_channel(array):
        best_idx = 2
        best_score = -1

        known_ids = np.arange(1, 23, dtype=np.uint8)
        driving_ids = np.array([6, 7, 8], dtype=np.uint8)

        for idx in range(3):
            channel = array[:, :, idx]
            known_count = int(np.count_nonzero(np.isin(channel, known_ids)))
            driving_count = int(np.count_nonzero(np.isin(channel, driving_ids)))
            nonzero_count = int(np.count_nonzero(channel))
            score = known_count + 5 * driving_count + min(nonzero_count, 1000)

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    @staticmethod
    def _carla_image_to_depth_meters(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4)).astype(np.float32)

        b = array[:, :, 0]
        g = array[:, :, 1]
        r = array[:, :, 2]

        normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (
            256.0 * 256.0 * 256.0 - 1.0
        )
        return normalized * 1000.0

    def _convert_carla_image(self, image):
        if self.sensor_type == "rgb":
            return self._carla_image_to_bgr(image)
        if self.sensor_type == "semantic":
            return self._carla_image_to_semantic_label(image)
        if self.sensor_type == "depth":
            return self._carla_image_to_depth_meters(image)
        raise ValueError(f"[CameraSensor] Unsupported sensor_type: {self.sensor_type}")

    def _trim_frame_buffer_locked(self):
        if len(self.frame_buffer) <= self.max_buffer_size:
            return

        old_ids = sorted(self.frame_buffer.keys())[: -self.max_buffer_size]
        for frame_id in old_ids:
            self.frame_buffer.pop(frame_id, None)

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

    def get_available_frame_ids(self):
        with self.latest_frame_lock:
            return set(self.frame_buffer.keys())

    def get_frame_by_id(self, frame_id):
        with self.latest_frame_lock:
            return self.frame_buffer.get(frame_id)

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
