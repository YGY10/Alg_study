# bev/bev_stitcher.py

import numpy as np
import cv2


class BEVStitcher:
    def __init__(self, bev_grid, camera_configs):
        self.bev_grid = bev_grid
        self.camera_configs = camera_configs

        self.angle_weight_maps = self._build_angle_weight_maps()
        self.distance_weight = self._build_distance_weight()

        # 保存最近一次 stitch 的融合权重，用于 perception 的 observed_mask
        self.last_weight_sum = None
        self.last_observed_mask = None

    @staticmethod
    def _wrap_angle_deg(angle):
        return (angle + 180.0) % 360.0 - 180.0

    def _smooth_angle_weight(
        self,
        angle_deg,
        center_deg,
        half_span_deg,
        fade_deg,
    ):
        """
        Soft angular weight around camera yaw direction.

        angle_deg:
            BEV point angle in ego frame.
            front=0, right=90, rear=180/-180, left=-90

        center_deg:
            camera yaw direction.
        """
        diff = self._wrap_angle_deg(angle_deg - center_deg)
        diff_abs = np.abs(diff)

        w = np.ones_like(angle_deg, dtype=np.float32)

        inner = half_span_deg
        outer = half_span_deg + fade_deg

        mid_mask = (diff_abs > inner) & (diff_abs < outer)
        out_mask = diff_abs >= outer

        t = np.zeros_like(angle_deg, dtype=np.float32)
        t[mid_mask] = (diff_abs[mid_mask] - inner) / max(fade_deg, 1e-6)

        w[mid_mask] = 0.5 * (1.0 + np.cos(np.pi * t[mid_mask]))
        w[out_mask] = 0.0

        return w.astype(np.float32)

    def _build_angle_weight_maps(self):
        """
        Build N-camera soft direction weights.

        Vehicle coordinate:
            x forward
            y right

        Angle:
            front = 0 deg
            right = 90 deg
            rear = 180 / -180 deg
            left = -90 deg
        """
        points = self.bev_grid.generate_ground_points_vehicle()
        x = points[:, 0]
        y = points[:, 1]

        angle = np.degrees(np.arctan2(y, x))

        bev_w, bev_h = self.bev_grid.get_size()

        maps = {}

        print("[BEV] N-camera soft angle weight maps:")

        for camera_name, cfg in self.camera_configs.items():
            yaw_deg = float(cfg["rotation"][1])
            fov_deg = float(cfg["fov"])

            # FOV 越大，主权重区域越大
            # 对 100° FOV：
            #   half_span = 45
            #   fade = 30
            half_span_deg = max(25.0, min(0.45 * fov_deg, 60.0))
            fade_deg = max(20.0, min(0.30 * fov_deg, 40.0))

            w = self._smooth_angle_weight(
                angle_deg=angle,
                center_deg=yaw_deg,
                half_span_deg=half_span_deg,
                fade_deg=fade_deg,
            )

            w = w.reshape(bev_h, bev_w).astype(np.float32)

            maps[camera_name] = w

            ratio = float(np.count_nonzero(w > 0.01)) / float(w.size)
            print(
                f"  {camera_name:>12s}: "
                f"yaw={yaw_deg:7.1f}, "
                f"fov={fov_deg:6.1f}, "
                f"half={half_span_deg:5.1f}, "
                f"fade={fade_deg:5.1f}, "
                f"active_ratio={ratio:.3f}"
            )

        return maps

    def _build_distance_weight(self):
        """
        Distance confidence weight in BEV space.

        Far area is less reliable because IPM stretches near horizon.
        But for 80m lane perception, do not suppress far area too hard.
        """
        points = self.bev_grid.generate_ground_points_vehicle()
        x = points[:, 0]
        y = points[:, 1]

        r = np.sqrt(x * x + y * y)

        near_m = 0.0
        far_m = 85.0

        t = (r - near_m) / max(far_m - near_m, 1e-6)
        t = np.clip(t, 0.0, 1.0)

        # 更温和的远距衰减
        w = 1.0 - 0.35 * (t ** 1.3)
        w = np.clip(w, 0.45, 1.0)

        bev_w, bev_h = self.bev_grid.get_size()
        w = w.reshape(bev_h, bev_w).astype(np.float32)

        print(
            f"[BEV] Distance weight built: "
            f"min={w.min():.3f}, max={w.max():.3f}, mean={w.mean():.3f}"
        )

        return w

    def _make_soft_valid_mask(self, valid_mask, bev_image):
        if valid_mask is None:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
            hard = (gray > 3).astype(np.uint8)
        else:
            hard = valid_mask.astype(np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)
        hard = cv2.erode(hard, kernel, iterations=1)

        soft = cv2.GaussianBlur(
            hard.astype(np.float32),
            ksize=(31, 31),
            sigmaX=0,
        )

        soft = np.clip(soft, 0.0, 1.0)
        return soft.astype(np.float32)

    def _inpaint_small_holes(self, image, weight_sum):
        covered = (weight_sum > 1e-4).astype(np.uint8) * 255

        kernel = np.ones((9, 9), dtype=np.uint8)
        closed = cv2.morphologyEx(covered, cv2.MORPH_CLOSE, kernel)

        hole_mask = cv2.bitwise_and(closed, cv2.bitwise_not(covered))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            hole_mask,
            connectivity=8,
        )

        filtered = np.zeros_like(hole_mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= 2000:
                filtered[labels == i] = 255

        if np.any(filtered > 0):
            image = cv2.inpaint(image, filtered, 3, cv2.INPAINT_TELEA)

        return image

    def stitch(self, bev_images, valid_masks, confidence_maps=None):
        """
        N-camera soft-blending surround BEV stitch.

        Args:
            bev_images:
                dict camera_name -> bev image

            valid_masks:
                dict camera_name -> valid mask from IPMProjector

            confidence_maps:
                dict camera_name -> confidence map from IPMProjector

        Return:
            stitched BEV image
        """
        bev_w, bev_h = self.bev_grid.get_size()

        acc_img = np.zeros((bev_h, bev_w, 3), dtype=np.float32)
        acc_w = np.zeros((bev_h, bev_w), dtype=np.float32)

        for camera_name in self.camera_configs.keys():
            bev = bev_images.get(camera_name)
            valid = valid_masks.get(camera_name)

            if bev is None:
                continue

            if bev.shape[0] != bev_h or bev.shape[1] != bev_w:
                bev = cv2.resize(
                    bev,
                    (bev_w, bev_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            bev_f = bev.astype(np.float32)

            soft_valid = self._make_soft_valid_mask(valid, bev)

            angle_w = self.angle_weight_maps.get(camera_name)
            if angle_w is None:
                continue

            if confidence_maps is not None:
                conf = confidence_maps.get(camera_name)

                if conf is None:
                    conf = np.ones((bev_h, bev_w), dtype=np.float32)
                else:
                    if conf.shape[0] != bev_h or conf.shape[1] != bev_w:
                        conf = cv2.resize(
                            conf,
                            (bev_w, bev_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    conf = np.clip(conf.astype(np.float32), 0.0, 1.0)
            else:
                conf = np.ones((bev_h, bev_w), dtype=np.float32)

            # N-camera final weight
            weight = soft_valid * angle_w * self.distance_weight * conf

            # 轻微压低边缘
            weight = np.power(np.clip(weight, 0.0, 1.0), 1.10).astype(np.float32)

            acc_img += bev_f * weight[:, :, None]
            acc_w += weight

        # 保存融合权重，后续 perception 会用它判断哪些区域是真实观测区域
        self.last_weight_sum = acc_w.copy()

        # observed mask：有足够投影覆盖的 BEV 区域
        # 阈值不要太高，否则远处会被误杀；先用 0.03
        observed_mask = (acc_w > 0.03).astype(np.uint8) * 255

        # 做一点形态学，去掉毛刺，补小洞
        kernel = np.ones((5, 5), dtype=np.uint8)
        observed_mask = cv2.morphologyEx(
            observed_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        observed_mask = cv2.morphologyEx(
            observed_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        self.last_observed_mask = observed_mask

        acc_w_safe = np.maximum(acc_w, 1e-6)
        stitched = acc_img / acc_w_safe[:, :, None]
        stitched = np.clip(stitched, 0, 255).astype(np.uint8)

        uncovered = acc_w <= 1e-4
        stitched[uncovered] = 0

        stitched = self._inpaint_small_holes(stitched, acc_w)

        return stitched



    def draw_debug_grid(self, bev_bgr):
        """
        在 BEV 图上画调试信息：
        1) ego origin
        2) x=0 横线
        3) y=0 竖线
        4) ego footprint
        5) 前后左右文字
        """
        if bev_bgr is None:
            return bev_bgr

        vis = bev_bgr.copy()
        h, w = vis.shape[:2]

        # =========================
        # 1. ego 原点像素位置
        # =========================
        ego_row, ego_col = self.bev_grid.vehicle_xy_to_bev_pixel(0.0, 0.0)
        ego_row = int(round(ego_row))
        ego_col = int(round(ego_col))

        # 防御一下，避免越界
        ego_row = max(0, min(h - 1, ego_row))
        ego_col = max(0, min(w - 1, ego_col))

        # 只打印一次，方便你确认 ego 在图里的位置
        if not hasattr(self, "_printed_ego_pixel"):
            print(
                f"[BEV DEBUG] ego pixel: row={ego_row}, col={ego_col}, "
                f"image_h={h}, image_w={w}"
            )
            self._printed_ego_pixel = True

        # =========================
        # 2. x=0 横线（黄色）
        # =========================
        cv2.line(
            vis,
            (0, ego_row),
            (w - 1, ego_row),
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # =========================
        # 3. y=0 竖线（黄色）
        # =========================
        cv2.line(
            vis,
            (ego_col, 0),
            (ego_col, h - 1),
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # =========================
        # 4. ego origin（红点）
        # =========================
        cv2.circle(vis, (ego_col, ego_row), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            "ego origin",
            (ego_col + 8, ego_row - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # =========================
        # 5. ego footprint（红框）
        #    这里用一个常见轿车尺寸：
        #    长 4.8m，宽 2.0m
        #    如果你后面想更真实，可以改成从 config 里读
        # =========================
        ego_length_m = 4.8
        ego_width_m = 2.0

        half_l = ego_length_m * 0.5
        half_w = ego_width_m * 0.5

        # 车体四个角（vehicle 坐标系）
        # x 前进方向，y 左正右负/或右正左负取决于你的约定
        # 这里只要和 vehicle_xy_to_bev_pixel 一致即可
        corners_xy = [
            (+half_l, -half_w),  # front-right
            (+half_l, +half_w),  # front-left
            (-half_l, +half_w),  # rear-left
            (-half_l, -half_w),  # rear-right
        ]

        corners_px = []
        for x_m, y_m in corners_xy:
            r, c = self.bev_grid.vehicle_xy_to_bev_pixel(x_m, y_m)
            r = int(round(r))
            c = int(round(c))
            corners_px.append([c, r])   # OpenCV 用 (x, y) = (col, row)

        corners_px = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))

        # 红色轮廓
        cv2.polylines(vis, [corners_px], isClosed=True, color=(0, 0, 255), thickness=2)

        # 朝前方向的小箭头
        front_center_x = half_l
        front_center_y = 0.0
        fr, fc = self.bev_grid.vehicle_xy_to_bev_pixel(front_center_x, front_center_y)
        fr = int(round(fr))
        fc = int(round(fc))

        cv2.arrowedLine(
            vis,
            (ego_col, ego_row),
            (fc, fr),
            (0, 0, 255),
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )

        cv2.putText(
            vis,
            "ego footprint",
            (ego_col + 8, ego_row + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # =========================
        # 6. 标一下前后左右
        # =========================
        cv2.putText(
            vis,
            "FRONT",
            (ego_col + 12, max(25, ego_row - 80)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "REAR",
            (ego_col + 12, min(h - 15, ego_row + 80)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "LEFT",
            (max(10, ego_col - 90), ego_row - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "RIGHT",
            (min(w - 90, ego_col + 15), ego_row - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return vis