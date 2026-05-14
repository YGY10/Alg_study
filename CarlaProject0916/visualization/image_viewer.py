# visualization/image_viewer.py

import math
import cv2
import numpy as np

from config.camera_config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_CONFIGS,
    CAMERA_DISPLAY_ORDER,
)
from utils.image_utils import draw_text, make_black_image


class MultiCameraViewer:
    def __init__(self, window_name="CARLA Multi Camera View"):
        self.window_name = window_name

        self.display_order = [
            name for name in CAMERA_DISPLAY_ORDER if name in CAMERA_CONFIGS
        ]

        # 如果 DISPLAY_ORDER 漏了某些 camera，自动补上
        for name in CAMERA_CONFIGS.keys():
            if name not in self.display_order:
                self.display_order.append(name)

        self.num_cameras = len(self.display_order)

        # 8 camera 默认 2x4
        if self.num_cameras <= 4:
            self.rows = 2
            self.cols = 2
        else:
            self.rows = 2
            self.cols = int(math.ceil(self.num_cameras / self.rows))

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(
            self.window_name,
            CAMERA_WIDTH * self.cols,
            CAMERA_HEIGHT * self.rows,
        )

        print(
            f"[VIEW] MultiCameraViewer: "
            f"{self.num_cameras} cameras, layout={self.rows}x{self.cols}, "
            f"order={self.display_order}"
        )

    def show(self, multi_camera_frame, ego_speed_kmh=0.0, main_fps=0.0):
        cells = []

        for name in self.display_order:
            cell = self._get_or_black(multi_camera_frame, name)
            cells.append(cell)

        # 不足格子补黑图
        total_cells = self.rows * self.cols
        while len(cells) < total_cells:
            cells.append(
                make_black_image(
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                    text="EMPTY",
                )
            )

        row_images = []

        for r in range(self.rows):
            row_cells = cells[r * self.cols:(r + 1) * self.cols]
            row_images.append(np.hstack(row_cells))

        canvas = np.vstack(row_images)

        draw_text(
            canvas,
            f"speed={ego_speed_kmh:.2f} km/h | main_fps={main_fps:.2f}",
            pos=(20, 30),
            color=(0, 255, 255),
        )

        cv2.imshow(self.window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        return key

    def _get_or_black(self, multi_camera_frame, name):
        camera_frame = multi_camera_frame.frames.get(name)

        if camera_frame is None:
            image = make_black_image(
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                text=f"{name}: NO IMAGE",
            )
            return image

        image = camera_frame.image.copy()

        # 保证尺寸一致
        if image.shape[1] != CAMERA_WIDTH or image.shape[0] != CAMERA_HEIGHT:
            image = cv2.resize(
                image,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        draw_text(
            image,
            f"{name} | frame={camera_frame.frame_id}",
            pos=(10, 30),
            color=(0, 255, 0),
            scale=0.55,
            thickness=2,
        )

        return image

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass