# visualization/perception_viewer.py

import cv2


class PerceptionViewer:
    def __init__(self, window_name="BEV Perception Debug"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, debug_image):
        if debug_image is None:
            return -1

        cv2.imshow(self.window_name, debug_image)
        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        cv2.destroyWindow(self.window_name)