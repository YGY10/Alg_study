# visualization/bev_viewer.py

import cv2
import numpy as np

from utils.image_utils import draw_text


class BEVViewer:
    """
    单独显示 BEV 图像。

    例如：
        Surround BEV
    """

    def __init__(self, window_name="Surround BEV"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, bev_image, fps=0.0, title="Surround BEV"):
        if bev_image is None:
            return -1

        canvas = bev_image.copy()

        draw_text(
            canvas,
            f"{title} | fps={fps:.2f}",
            pos=(10, 25),
            color=(0, 255, 255),
            scale=0.6,
            thickness=2,
        )

        cv2.imshow(self.window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass


class BEVCompareViewer:
    """
    左右对比显示：

        左：原始 Surround BEV
        右：BEV Perception Debug

    这样方便判断感知结果是否和原始 BEV 对齐。
    """

    def __init__(
        self,
        window_name="BEV Compare",
        target_height=800,
    ):
        self.window_name = window_name
        self.target_height = int(target_height)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(
        self,
        surround_bev,
        perception_debug_image,
        fps=0.0,
        title_left="Surround BEV",
        title_right="Perception Debug",
    ):
        if surround_bev is None or perception_debug_image is None:
            return -1

        left = self._ensure_bgr(surround_bev)
        right = self._ensure_bgr(perception_debug_image)

        left = self._resize_by_height(left, self.target_height)
        right = self._resize_by_height(right, self.target_height)

        # 保证两张图高度一致
        h = min(left.shape[0], right.shape[0])
        left = left[:h, :, :]
        right = right[:h, :, :]

        # 在两张图上分别写标题
        left_canvas = left.copy()
        right_canvas = right.copy()

        draw_text(
            left_canvas,
            title_left,
            pos=(15, 35),
            color=(0, 255, 255),
            scale=0.8,
            thickness=2,
        )

        # draw_text(
        #     right_canvas,
        #     title_right,
        #     pos=(15, 35),
        #     color=(0, 255, 255),
        #     scale=0.8,
        #     thickness=2,
        # )

        draw_text(
            left_canvas,
            f"fps={fps:.2f}",
            pos=(15, 70),
            color=(0, 255, 255),
            scale=0.6,
            thickness=2,
        )

        # 中间加一条分隔线
        separator = np.zeros((h, 6, 3), dtype=np.uint8)
        separator[:, :, :] = (30, 30, 30)

        canvas = np.hstack([left_canvas, separator, right_canvas])

        cv2.imshow(self.window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        return key

    @staticmethod
    def _ensure_bgr(image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return image[:, :, :3].copy()

        return image.copy()

    @staticmethod
    def _resize_by_height(image, target_height):
        h, w = image.shape[:2]

        if h == target_height:
            return image

        scale = target_height / float(h)
        target_width = int(round(w * scale))

        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass