# perception/bev_perception.py

import cv2
import numpy as np

from perception.base_perception import BasePerception
from perception.perception_output import (
    BEVPerceptionInput,
    BEVPerceptionOutput,
)
from utils.image_utils import draw_text


CARLA_LABEL_ROAD_LINES = 6
CARLA_LABEL_ROADS = 7
CARLA_LABEL_SIDEWALKS = 8
CARLA_LABEL_PEDESTRIANS = 4
CARLA_LABEL_VEHICLES = 10
CARLA_LABEL_GROUND = 14
CARLA_ROAD_SURFACE_LABELS = [CARLA_LABEL_ROADS, CARLA_LABEL_GROUND]
CARLA_NON_ROAD_LABELS = {
    0,
    1,   # building
    2,   # fence
    4,   # pedestrian
    5,   # pole
    6,   # road line
    8,   # sidewalk
    9,   # vegetation
    10,  # vehicle
    11,  # wall
    12,  # traffic sign
    13,  # sky
    15,  # bridge
    16,  # rail track
    17,  # guard rail
    18,  # traffic light
    20,  # dynamic
    21,  # water
    22,  # terrain
}
CARLA_SEMANTIC_COLORS = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (100, 40, 40),
    3: (55, 90, 80),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (255, 255, 255),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45, 60, 150),
    22: (145, 170, 100),
}


class BEVPerception(BasePerception):
    def __init__(self):
        # ============================================================
        # White road marking threshold
        # ============================================================
        # 白色路面标记阈值：收紧，减少浅灰路面/建筑误检
        self.white_lower = np.array([0, 0, 175], dtype=np.uint8)
        self.white_upper = np.array([180, 55, 255], dtype=np.uint8)

        # white mask 局部亮度增强参数
        self.white_tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (17, 17),
        )
        self.white_tophat_thresh = 18

        # white mask 形状过滤参数
        self.white_min_area = 18
        self.white_max_area = 8000
        self.white_min_aspect = 1.8

        # ============================================================
        # Yellow road marking threshold
        # ============================================================
        self.yellow_lower = np.array([15, 60, 80], dtype=np.uint8)
        self.yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        # ============================================================
        # Edge
        # ============================================================
        self.canny_low = 60
        self.canny_high = 150

        # ============================================================
        # Morphology kernels
        # ============================================================
        self.kernel_small = np.ones((3, 3), dtype=np.uint8)
        self.kernel_mid = np.ones((5, 5), dtype=np.uint8)
        self.kernel_large = np.ones((11, 11), dtype=np.uint8)

        # ============================================================
        # Ego vehicle model
        # ============================================================
        self.ego_length_m = 4.8
        self.ego_width_m = 2.0

        # near unknown 外扩范围
        # 这块不是 ego 本体，而是车身周围相机不可见/低可信区域
        self.near_unknown_margin_m = 0.45

    def process(self, perception_input: BEVPerceptionInput) -> BEVPerceptionOutput:
        bev_image = perception_input.bev_image

        if bev_image is None:
            raise ValueError("[BEVPerception] Input bev_image is None.")

        if len(bev_image.shape) != 3 or bev_image.shape[2] != 3:
            raise ValueError(
                f"[BEVPerception] Expected BGR image with shape HxWx3, "
                f"got {bev_image.shape}"
            )

        image = bev_image.copy()

        if perception_input.semantic_bev is not None:
            return self._process_semantic_bev(
                image=image,
                semantic_bev=perception_input.semantic_bev,
                depth_bev=perception_input.depth_bev,
                perception_input=perception_input,
            )

        # 0. observed / coverage mask
        observed_mask = self._prepare_observed_mask(
            observed_mask=perception_input.observed_mask,
            image_shape=image.shape[:2],
        )

        # 1. ego footprint / near unknown mask
        ego_footprint_mask, near_unknown_mask = self._build_ego_and_unknown_masks(
            image_bgr=image,
            bev_grid=perception_input.bev_grid,
        )

        # 2. 原始 road surface mask
        road_surface_raw = self._extract_road_surface_mask(
            image_bgr=image,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
        )

        # 2.1 几何覆盖约束：低可信/无观测区域不能作为 road
        road_surface_raw = cv2.bitwise_and(road_surface_raw, observed_mask)

        # 3. road 后处理
        road_surface_mask = self._postprocess_road_mask(
            road_mask=road_surface_raw,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
        )

        road_surface_mask = cv2.bitwise_and(road_surface_mask, observed_mask)

        # 4. 颜色阈值
        white_mask_raw = self._extract_white_mask(image)
        yellow_mask_raw = self._extract_yellow_mask(image)

        # 5. 边缘检测
        edge_mask_raw = self._extract_edge_mask(image)

        # 6. 只保留 observed + road 内部，并排除 ego / unknown
        valid_perception_mask = road_surface_mask.copy()
        valid_perception_mask = cv2.bitwise_and(valid_perception_mask, observed_mask)
        valid_perception_mask[ego_footprint_mask > 0] = 0
        valid_perception_mask[near_unknown_mask > 0] = 0

        white_mask = cv2.bitwise_and(white_mask_raw, valid_perception_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask_raw, valid_perception_mask)
        edge_mask = cv2.bitwise_and(edge_mask_raw, valid_perception_mask)

        # 7. lane / road marking
        lane_candidate_mask = self._build_lane_candidate_mask(
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            edge_mask=edge_mask,
            valid_mask=valid_perception_mask,
        )

        road_marking_mask = self._build_road_marking_mask(
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            valid_mask=valid_perception_mask,
        )

        # 8. drivable 第一版直接用 road surface
        drivable_candidate_mask = road_surface_mask.copy()

        # 9. debug
        debug_image = self._make_debug_image(
            image=image,
            observed_mask=observed_mask,
            road_surface_raw=road_surface_raw,
            road_surface_mask=road_surface_mask,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
            white_mask_raw=white_mask_raw,
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            edge_mask=edge_mask,
            lane_candidate_mask=lane_candidate_mask,
            road_marking_mask=road_marking_mask,
            ego_speed_kmh=perception_input.ego_speed_kmh,
        )

        return BEVPerceptionOutput(
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            edge_mask=edge_mask,
            lane_candidate_mask=lane_candidate_mask,
            road_marking_mask=road_marking_mask,
            drivable_candidate_mask=drivable_candidate_mask,
            debug_image=debug_image,
            road_surface_mask=road_surface_mask,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
            observed_mask=observed_mask,
            depth_bev=perception_input.depth_bev,
        )

    # ============================================================
    # CARLA semantic supervision path
    # ============================================================
    def _process_semantic_bev(
        self,
        image,
        semantic_bev,
        depth_bev,
        perception_input,
    ):
        h, w = image.shape[:2]

        semantic = semantic_bev.copy()
        if semantic.shape[0] != h or semantic.shape[1] != w:
            semantic = cv2.resize(
                semantic,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        observed_mask = self._prepare_observed_mask(
            observed_mask=perception_input.observed_mask,
            image_shape=image.shape[:2],
        )

        ego_footprint_mask, near_unknown_mask = self._build_ego_and_unknown_masks(
            image_bgr=image,
            bev_grid=perception_input.bev_grid,
        )

        adaptive_road_labels = self._infer_road_surface_labels(
            semantic=semantic,
            bev_grid=perception_input.bev_grid,
        )
        road_surface_mask = self._labels_to_mask(semantic, adaptive_road_labels)
        road_marking_mask = self._labels_to_mask(semantic, [CARLA_LABEL_ROAD_LINES])
        sidewalk_mask = self._labels_to_mask(semantic, [CARLA_LABEL_SIDEWALKS])
        vehicle_mask = self._labels_to_mask(semantic, [CARLA_LABEL_VEHICLES])
        pedestrian_mask = self._labels_to_mask(semantic, [CARLA_LABEL_PEDESTRIANS])

        road_surface_mask = cv2.bitwise_and(road_surface_mask, observed_mask)
        road_marking_mask = cv2.bitwise_and(road_marking_mask, observed_mask)
        sidewalk_mask = cv2.bitwise_and(sidewalk_mask, observed_mask)
        vehicle_mask = cv2.bitwise_and(vehicle_mask, observed_mask)
        pedestrian_mask = cv2.bitwise_and(pedestrian_mask, observed_mask)

        for mask in (
            road_surface_mask,
            road_marking_mask,
            sidewalk_mask,
            vehicle_mask,
            pedestrian_mask,
        ):
            mask[ego_footprint_mask > 0] = 0
            mask[near_unknown_mask > 0] = 0

        road_context = cv2.dilate(
            road_surface_mask,
            self.kernel_large,
            iterations=2,
        )
        road_marking_mask = cv2.bitwise_and(road_marking_mask, road_context)

        road_surface_mask = cv2.morphologyEx(
            road_surface_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )
        road_marking_mask = cv2.morphologyEx(
            road_marking_mask,
            cv2.MORPH_CLOSE,
            self.kernel_small,
            iterations=1,
        )
        road_marking_mask = self._filter_components_by_area(
            road_marking_mask,
            min_area=8,
            keep_top_k=None,
        )

        vehicle_mask = self._filter_components_by_area(
            vehicle_mask,
            min_area=12,
            keep_top_k=20,
        )
        pedestrian_mask = self._filter_components_by_area(
            pedestrian_mask,
            min_area=6,
            keep_top_k=20,
        )

        lane_candidate_mask = road_marking_mask.copy()
        white_mask = road_marking_mask.copy()
        yellow_mask = np.zeros_like(road_marking_mask)
        edge_mask = self._extract_edge_mask(image)
        edge_mask = cv2.bitwise_and(edge_mask, observed_mask)

        drivable_candidate_mask = cv2.bitwise_or(
            road_surface_mask,
            road_marking_mask,
        )

        debug_image = self._make_semantic_debug_image(
            image=image,
            observed_mask=observed_mask,
            semantic=semantic,
            road_surface_mask=road_surface_mask,
            road_marking_mask=road_marking_mask,
            sidewalk_mask=sidewalk_mask,
            vehicle_mask=vehicle_mask,
            pedestrian_mask=pedestrian_mask,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
            depth_bev=depth_bev,
            road_labels=adaptive_road_labels,
            ego_speed_kmh=perception_input.ego_speed_kmh,
        )

        return BEVPerceptionOutput(
            white_mask=white_mask,
            yellow_mask=yellow_mask,
            edge_mask=edge_mask,
            lane_candidate_mask=lane_candidate_mask,
            road_marking_mask=road_marking_mask,
            drivable_candidate_mask=drivable_candidate_mask,
            debug_image=debug_image,
            road_surface_mask=road_surface_mask,
            ego_footprint_mask=ego_footprint_mask,
            near_unknown_mask=near_unknown_mask,
            observed_mask=observed_mask,
            semantic_bev=semantic,
            depth_bev=depth_bev,
            sidewalk_mask=sidewalk_mask,
            vehicle_mask=vehicle_mask,
            pedestrian_mask=pedestrian_mask,
        )

    @staticmethod
    def _labels_to_mask(label_image, label_ids):
        mask = np.isin(label_image, label_ids).astype(np.uint8) * 255
        return mask

    def _infer_road_surface_labels(self, semantic, bev_grid):
        """
        CARLA maps are not perfectly consistent about road surface tags.
        Town10HD often uses label 3 (Other) for visible asphalt patches, so
        infer the local drivable surface from a small ego-front sample.
        """
        road_labels = set(CARLA_ROAD_SURFACE_LABELS)

        h, w = semantic.shape[:2]
        if bev_grid is None:
            row_center = int(h * 0.75)
            col_center = w // 2
            row_radius = max(8, int(h * 0.08))
            col_radius = max(8, int(w * 0.08))
        else:
            row_center, col_center = bev_grid.vehicle_xy_to_bev_pixel(4.0, 0.0)
            row_center = int(round(row_center))
            col_center = int(round(col_center))
            row_radius = max(8, int(round(5.0 / bev_grid.resolution)))
            col_radius = max(8, int(round(3.0 / bev_grid.resolution)))

        r1 = max(0, row_center - row_radius)
        r2 = min(h, row_center + row_radius)
        c1 = max(0, col_center - col_radius)
        c2 = min(w, col_center + col_radius)

        patch = semantic[r1:r2, c1:c2]
        if patch.size == 0:
            return sorted(road_labels)

        labels, counts = np.unique(patch, return_counts=True)
        pairs = sorted(
            zip(labels.tolist(), counts.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

        for label_id, count in pairs:
            label_id = int(label_id)
            if label_id in CARLA_NON_ROAD_LABELS:
                continue
            if count < 20:
                continue
            road_labels.add(label_id)
            break

        return sorted(road_labels)

    # ============================================================
    # Observed / coverage mask
    # ============================================================
    def _prepare_observed_mask(self, observed_mask, image_shape):
        """
        准备 observed mask。

        observed_mask 来自 BEVStitcher.last_observed_mask。
        如果没有传入，就默认整图 observed，保证旧流程不崩。
        """
        h, w = image_shape

        if observed_mask is None:
            return np.ones((h, w), dtype=np.uint8) * 255

        mask = observed_mask.copy()

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = (mask > 0).astype(np.uint8) * 255

        # 轻微闭运算，避免 observed mask 太碎
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        return mask

    # ============================================================
    # Ego / unknown masks
    # ============================================================
    def _build_ego_and_unknown_masks(self, image_bgr, bev_grid):
        """
        生成：
            ego_footprint_mask: 自车真实 footprint
            near_unknown_mask : 自车附近不可见/低可信区域

        注意：
            near_unknown 不是路，也不是障碍。
            后续 road/lane 都要排除它。
        """
        h, w = image_bgr.shape[:2]

        ego_mask = np.zeros((h, w), dtype=np.uint8)
        unknown_mask = np.zeros((h, w), dtype=np.uint8)

        if bev_grid is None:
            # fallback：没有 bev_grid 时，只能按图像中心近似
            ego_col = w // 2
            ego_row = int(h * 0.8)

            box_w = 20
            box_h = 48
            x1 = max(0, ego_col - box_w // 2)
            x2 = min(w - 1, ego_col + box_w // 2)
            y1 = max(0, ego_row - box_h // 2)
            y2 = min(h - 1, ego_row + box_h // 2)

            cv2.rectangle(ego_mask, (x1, y1), (x2, y2), 255, -1)

            kernel = np.ones((11, 11), dtype=np.uint8)
            unknown_mask = cv2.dilate(ego_mask, kernel, iterations=1)
            return ego_mask, unknown_mask

        # ego footprint
        ego_poly = self._vehicle_box_polygon_px(
            bev_grid=bev_grid,
            length_m=self.ego_length_m,
            width_m=self.ego_width_m,
        )
        cv2.fillPoly(ego_mask, [ego_poly], 255)

        # near unknown = ego footprint 外扩 margin
        unknown_poly = self._vehicle_box_polygon_px(
            bev_grid=bev_grid,
            length_m=self.ego_length_m + 2.0 * self.near_unknown_margin_m,
            width_m=self.ego_width_m + 2.0 * self.near_unknown_margin_m,
        )
        cv2.fillPoly(unknown_mask, [unknown_poly], 255)

        # 再结合真实黑洞：
        # 只在 ego 周围一定半径内，把黑色无观测区也纳入 unknown
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        black = (gray < 8).astype(np.uint8) * 255

        ego_row, ego_col = bev_grid.vehicle_xy_to_bev_pixel(0.0, 0.0)
        ego_row = int(round(ego_row))
        ego_col = int(round(ego_col))

        near_radius_m = 4.0
        near_radius_px = max(1, int(round(near_radius_m / bev_grid.resolution)))

        near_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            near_circle,
            (ego_col, ego_row),
            near_radius_px,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

        local_black = cv2.bitwise_and(black, near_circle)
        unknown_mask = cv2.bitwise_or(unknown_mask, local_black)

        unknown_mask = cv2.morphologyEx(
            unknown_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        return ego_mask, unknown_mask

    def _vehicle_box_polygon_px(self, bev_grid, length_m, width_m):
        half_l = length_m * 0.5
        half_w = width_m * 0.5

        corners_xy = [
            (+half_l, -half_w),
            (+half_l, +half_w),
            (-half_l, +half_w),
            (-half_l, -half_w),
        ]

        pts = []
        for x_m, y_m in corners_xy:
            row, col = bev_grid.vehicle_xy_to_bev_pixel(x_m, y_m)
            pts.append([int(round(col)), int(round(row))])

        return np.array(pts, dtype=np.int32)

    # ============================================================
    # Road mask
    # ============================================================
    def _extract_road_surface_mask(
        self,
        image_bgr,
        ego_footprint_mask,
        near_unknown_mask,
    ):
        """
        自适应 road surface mask。

        方法：
            1. 在 ego 前后附近选参考区域
            2. 估计路面 HSV 颜色
            3. 找颜色接近的区域
            4. 排除 ego / unknown
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        ego_x = w // 2
        ego_y = int(h * 0.8)

        # 更稳：参考区域放在自车前后附近，而不是完全覆盖车身中心
        ref_w = max(30, int(w * 0.12))
        ref_h = max(30, int(h * 0.12))

        x1 = max(0, ego_x - ref_w)
        x2 = min(w, ego_x + ref_w)
        y1 = max(0, ego_y - ref_h * 2)
        y2 = min(h, ego_y + ref_h)

        ref_hsv = hsv[y1:y2, x1:x2]
        ref_bgr = image_bgr[y1:y2, x1:x2]

        ref_invalid = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        ref_invalid |= ego_footprint_mask[y1:y2, x1:x2] > 0
        ref_invalid |= near_unknown_mask[y1:y2, x1:x2] > 0

        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

        valid_ref = (
            (ref_gray > 30)
            & (ref_gray < 220)
            & (ref_hsv[:, :, 1] < 130)
            & (~ref_invalid)
        )

        if np.count_nonzero(valid_ref) < 80:
            road = self._fallback_gray_road_mask(image_bgr)
        else:
            s_med = np.median(ref_hsv[:, :, 1][valid_ref])
            v_med = np.median(ref_hsv[:, :, 2][valid_ref])

            s = hsv[:, :, 1].astype(np.float32)
            v = hsv[:, :, 2].astype(np.float32)

            ds = np.abs(s - float(s_med))
            dv = np.abs(v - float(v_med))

            # 比之前略收紧，减少建筑/墙面进入 road
            road = (
                (ds < 45.0)
                & (dv < 65.0)
                & (v > 25.0)
                & (v < 235.0)
                & (s < 145.0)
            ).astype(np.uint8) * 255

        # 排除 ego 和 near unknown
        road[ego_footprint_mask > 0] = 0
        road[near_unknown_mask > 0] = 0

        road = cv2.morphologyEx(
            road,
            cv2.MORPH_OPEN,
            self.kernel_small,
            iterations=1,
        )

        road = cv2.morphologyEx(
            road,
            cv2.MORPH_CLOSE,
            self.kernel_large,
            iterations=2,
        )

        return road

    def _fallback_gray_road_mask(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        road = (
            (s < 125)
            & (v > 35)
            & (v < 225)
        ).astype(np.uint8) * 255

        road = cv2.morphologyEx(
            road,
            cv2.MORPH_CLOSE,
            self.kernel_large,
            iterations=2,
        )

        return road

    def _postprocess_road_mask(
        self,
        road_mask,
        ego_footprint_mask,
        near_unknown_mask,
    ):
        """
        road 后处理：
            1. 排除 ego / unknown
            2. 去小碎块
            3. 只保留较大区域
            4. 适度闭运算补洞
        """
        road = road_mask.copy()

        road[ego_footprint_mask > 0] = 0
        road[near_unknown_mask > 0] = 0

        road = cv2.morphologyEx(
            road,
            cv2.MORPH_OPEN,
            self.kernel_small,
            iterations=1,
        )

        road = self._filter_components_by_area(
            road,
            min_area=150,
            keep_top_k=8,
        )

        road = cv2.morphologyEx(
            road,
            cv2.MORPH_CLOSE,
            self.kernel_large,
            iterations=1,
        )

        road[ego_footprint_mask > 0] = 0
        road[near_unknown_mask > 0] = 0

        return road

    # ============================================================
    # Lane / markings
    # ============================================================
    def _extract_white_mask(self, image_bgr):
        """
        提取白色路面标记候选。

        相比单纯 HSV 阈值，这里多做三件事：
            1. HSV 收紧，减少浅灰路面误检
            2. V 通道 top-hat，保留“比周围更亮”的局部结构
            3. 连通域形状过滤，去掉大块墙面/人行道
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. 基础白色候选：低饱和 + 高亮度
        hsv_white = cv2.inRange(
            hsv,
            self.white_lower,
            self.white_upper,
        )

        # 2. 局部亮度 top-hat
        # 车道线/斑马线/箭头通常是局部亮结构；
        # 大片浅灰路面虽然亮，但 top-hat 响应不会太强。
        v = hsv[:, :, 2]
        opened = cv2.morphologyEx(
            v,
            cv2.MORPH_OPEN,
            self.white_tophat_kernel,
            iterations=1,
        )
        tophat = cv2.subtract(v, opened)

        _, tophat_mask = cv2.threshold(
            tophat,
            self.white_tophat_thresh,
            255,
            cv2.THRESH_BINARY,
        )

        # 3. 梯度约束
        # 白线边界通常有梯度；大片平坦亮区域梯度较弱。
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0.0)

        _, grad_mask = cv2.threshold(
            grad,
            22,
            255,
            cv2.THRESH_BINARY,
        )

        # 4. 融合：
        # HSV 是基础颜色约束；
        # top-hat/gradient 至少满足一个，避免大片白色区域全进来。
        local_structure = cv2.bitwise_or(tophat_mask, grad_mask)
        white_mask = cv2.bitwise_and(hsv_white, local_structure)

        # 5. 形态学去噪和连接
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            self.kernel_small,
            iterations=1,
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        # 6. 连通域形状过滤
        white_mask = self._filter_white_components_by_shape(white_mask)

        return white_mask

    def _filter_white_components_by_shape(self, mask):
        """
        白色候选连通域过滤。

        保留：
            - 细长线段
            - 小面积路面标记
            - 斑马线条块

        过滤：
            - 大块墙面/人行道
            - 面积过小噪声
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        out = np.zeros_like(mask)

        for i in range(1, num_labels):
            comp_x = stats[i, cv2.CC_STAT_LEFT]
            comp_y = stats[i, cv2.CC_STAT_TOP]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.white_min_area:
                continue

            if area > self.white_max_area:
                continue

            long_side = max(comp_w, comp_h)
            short_side = max(1, min(comp_w, comp_h))
            aspect = long_side / float(short_side)

            fill_ratio = area / float(max(1, comp_w * comp_h))

            # 细长结构：典型车道线
            is_line_like = aspect >= self.white_min_aspect and fill_ratio < 0.75

            # 小块结构：箭头碎片、停止线碎片、短虚线、斑马线块
            is_small_marking = area < 900 and fill_ratio < 0.95

            if is_line_like or is_small_marking:
                out[labels == i] = 255

        return out

    def _extract_yellow_mask(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            self.yellow_lower,
            self.yellow_upper,
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            self.kernel_small,
            iterations=1,
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        return yellow_mask

    def _extract_edge_mask(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edge = cv2.Canny(
            blur,
            self.canny_low,
            self.canny_high,
        )

        edge = cv2.dilate(
            edge,
            self.kernel_small,
            iterations=1,
        )

        return edge

    def _build_lane_candidate_mask(
        self,
        white_mask,
        yellow_mask,
        edge_mask,
        valid_mask,
    ):
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        edge_and_color = cv2.bitwise_and(edge_mask, color_mask)

        lane_mask = cv2.bitwise_or(color_mask, edge_and_color)
        lane_mask = cv2.bitwise_and(lane_mask, valid_mask)

        lane_mask = cv2.morphologyEx(
            lane_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        lane_mask = self._filter_components_by_area(
            lane_mask,
            min_area=20,
            keep_top_k=None,
        )

        return lane_mask

    def _build_road_marking_mask(
        self,
        white_mask,
        yellow_mask,
        valid_mask,
    ):
        road_marking_mask = cv2.bitwise_or(white_mask, yellow_mask)
        road_marking_mask = cv2.bitwise_and(road_marking_mask, valid_mask)

        road_marking_mask = cv2.morphologyEx(
            road_marking_mask,
            cv2.MORPH_CLOSE,
            self.kernel_mid,
            iterations=1,
        )

        road_marking_mask = self._filter_components_by_area(
            road_marking_mask,
            min_area=20,
            keep_top_k=None,
        )

        return road_marking_mask

    @staticmethod
    def _filter_components_by_area(mask, min_area=20, keep_top_k=None):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        comps = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append((i, area))

        comps.sort(key=lambda item: item[1], reverse=True)

        if keep_top_k is not None:
            comps = comps[:keep_top_k]

        out = np.zeros_like(mask)

        for label_id, _ in comps:
            out[labels == label_id] = 255

        return out

    # ============================================================
    # Debug image
    # ============================================================
    def _make_debug_image(
        self,
        image,
        observed_mask,
        road_surface_raw,
        road_surface_mask,
        ego_footprint_mask,
        near_unknown_mask,
        white_mask_raw,
        white_mask,
        yellow_mask,
        edge_mask,
        lane_candidate_mask,
        road_marking_mask,
        ego_speed_kmh=0.0,
    ):
        debug = image.copy()
        debug = (debug.astype(np.float32) * 0.50).astype(np.uint8)

        # observed 外的区域压暗，方便看哪些区域本来就是低可信投影
        unobserved_pixels = observed_mask == 0
        debug[unobserved_pixels] = (
            debug[unobserved_pixels].astype(np.float32) * 0.25
        ).astype(np.uint8)

        # raw road: 暗紫色，只显示被过滤掉的 road 候选
        raw_road_only = (road_surface_raw > 0) & (road_surface_mask == 0)
        debug[raw_road_only] = (120, 70, 120)

        # final road: 暗绿色
        road_pixels = road_surface_mask > 0
        debug[road_pixels] = (
            0.70 * debug[road_pixels]
            + 0.30 * np.array([0, 130, 0], dtype=np.float32)
        ).astype(np.uint8)

        # near unknown: 深灰
        unknown_pixels = near_unknown_mask > 0
        debug[unknown_pixels] = (40, 40, 40)

        # ego footprint: 红色
        ego_pixels = ego_footprint_mask > 0
        debug[ego_pixels] = (0, 0, 255)

        # raw white 被 road mask 过滤掉的地方：紫色
        raw_white_only = (white_mask_raw > 0) & (white_mask == 0)
        debug[raw_white_only] = (180, 120, 180)

        # final white
        debug[white_mask > 0] = (255, 255, 255)

        # yellow
        debug[yellow_mask > 0] = (0, 255, 255)

        # edge 蓝色弱提示
        edge_pixels = edge_mask > 0
        debug[edge_pixels] = (
            0.7 * debug[edge_pixels] + 0.3 * np.array([255, 0, 0])
        ).astype(np.uint8)

        # lane contours
        contours, _ = cv2.findContours(
            lane_candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            debug,
            contours,
            -1,
            (0, 255, 0),
            1,
        )

        draw_text(
            debug,
            f"BEV Perception | speed={ego_speed_kmh:.2f} km/h",
            pos=(15, 35),
            color=(0, 255, 255),
            scale=0.8,
            thickness=2,
        )

        draw_text(
            debug,
            "observed=normal | unobserved=dark | road=dark-green | "
            "filtered=purple | unknown=gray | ego=red | lane=green",
            pos=(15, 65),
            color=(0, 255, 255),
            scale=0.5,
            thickness=1,
        )

        return debug

    def _make_semantic_debug_image(
        self,
        image,
        observed_mask,
        semantic,
        road_surface_mask,
        road_marking_mask,
        sidewalk_mask,
        vehicle_mask,
        pedestrian_mask,
        ego_footprint_mask,
        near_unknown_mask,
        depth_bev,
        road_labels,
        ego_speed_kmh=0.0,
    ):
        debug = np.zeros_like(image)
        debug[observed_mask > 0] = (18, 18, 18)

        unobserved_pixels = observed_mask == 0
        debug[unobserved_pixels] = (12, 12, 12)

        road_pixels = road_surface_mask > 0
        debug[road_pixels] = (50, 170, 50)

        sidewalk_pixels = sidewalk_mask > 0
        debug[sidewalk_pixels] = (180, 65, 170)

        debug[road_marking_mask > 0] = (255, 255, 255)
        debug[vehicle_mask > 0] = (0, 165, 255)
        debug[pedestrian_mask > 0] = (255, 0, 255)
        debug[near_unknown_mask > 0] = (40, 40, 40)
        debug[ego_footprint_mask > 0] = (0, 0, 255)

        contours, _ = cv2.findContours(
            road_marking_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)

        depth_text = "depth=off"
        if depth_bev is not None:
            depth_valid = depth_bev[(depth_bev > 1e-3) & (observed_mask > 0)]
            if depth_valid.size > 0:
                depth_text = (
                    f"depth mean={float(np.mean(depth_valid)):.1f}m "
                    f"max={float(np.max(depth_valid)):.1f}m"
                )

        draw_text(
            debug,
            f"Semantic BEV | speed={ego_speed_kmh:.1f} km/h | {depth_text}",
            pos=(10, 24),
            color=(0, 255, 255),
            scale=0.45,
            thickness=1,
        )

        draw_text(
            debug,
            f"road labels={list(road_labels)} | line white | sidewalk pink | ego red",
            pos=(10, 44),
            color=(0, 255, 255),
            scale=0.35,
            thickness=1,
        )

        label_count = int(np.count_nonzero(semantic))
        draw_text(
            debug,
            f"semantic pixels={label_count}",
            pos=(10, 62),
            color=(0, 255, 255),
            scale=0.35,
            thickness=1,
        )

        return debug

    @staticmethod
    def _colorize_semantic(semantic):
        h, w = semantic.shape[:2]
        color = np.zeros((h, w, 3), dtype=np.uint8)

        for label_id, bgr in CARLA_SEMANTIC_COLORS.items():
            color[semantic == label_id] = bgr

        unknown = (semantic != 0) & ~np.isin(
            semantic,
            list(CARLA_SEMANTIC_COLORS.keys()),
        )
        color[unknown] = (255, 255, 0)

        return color
