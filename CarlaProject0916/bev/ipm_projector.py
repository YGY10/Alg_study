import cv2
import numpy as np

from calibration.camera_intrinsic import compute_camera_intrinsic
from calibration.camera_extrinsic import compute_camera_extrinsic
from calibration.coordinate import (
    transform_points,
    carla_camera_to_optical_points,
)


class IPMProjector:
    def __init__(self, camera_name, camera_config, bev_grid):
        self.camera_name = camera_name
        self.camera_config = camera_config
        self.bev_grid = bev_grid

        self.width = int(camera_config["width"])
        self.height = int(camera_config["height"])
        self.fov = float(camera_config["fov"])

        self.K = compute_camera_intrinsic(
            width=self.width,
            height=self.height,
            fov_deg=self.fov,
        )

        self.T_vehicle_camera, self.T_camera_vehicle = compute_camera_extrinsic(
            camera_config
        )

        self.map_x = None
        self.map_y = None
        self.valid_mask = None
        self.confidence_map = None
        self.expected_depth_map = None
        self.expected_range_map = None

        self._build_remap_table()

    def _build_remap_table(self):
        """
        Build inverse sampling table:

        BEV pixel
            -> vehicle ground point
            -> camera actor local
            -> optical camera frame
            -> image pixel

        同时构建 confidence_map，用来降低不可信区域权重：
            1. 图像太靠上，权重低
            2. 图像太靠左右边缘，权重低
            3. 距离相机太远，权重低
        """
        points_vehicle = self.bev_grid.generate_ground_points_vehicle()

        # vehicle -> CARLA camera local
        points_camera_carla = transform_points(
            self.T_camera_vehicle,
            points_vehicle,
        )

        # CARLA camera local -> optical frame
        points_optical = carla_camera_to_optical_points(points_camera_carla)

        x = points_optical[:, 0]
        y = points_optical[:, 1]
        z = points_optical[:, 2]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        eps = 1e-6
        valid_z = z > eps

        u = np.zeros_like(z, dtype=np.float64)
        v = np.zeros_like(z, dtype=np.float64)

        u[valid_z] = fx * x[valid_z] / z[valid_z] + cx
        v[valid_z] = fy * y[valid_z] / z[valid_z] + cy

        valid_uv = (
            valid_z
            & (u >= 0.0)
            & (u < self.width - 1)
            & (v >= 0.0)
            & (v < self.height - 1)
        )

        bev_w, bev_h = self.bev_grid.get_size()

        self.map_x = u.reshape(bev_h, bev_w).astype(np.float32)
        self.map_y = v.reshape(bev_h, bev_w).astype(np.float32)
        self.valid_mask = valid_uv.reshape(bev_h, bev_w)
        self.expected_depth_map = z.reshape(bev_h, bev_w).astype(np.float32)
        self.expected_range_map = np.sqrt(
            x * x + y * y + z * z
        ).reshape(bev_h, bev_w).astype(np.float32)

        self.confidence_map = self._build_confidence_map(
            u=u,
            v=v,
            z=z,
            valid_uv=valid_uv,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        valid_ratio = float(np.count_nonzero(self.valid_mask)) / float(self.valid_mask.size)
        conf_mean = float(self.confidence_map[self.valid_mask].mean()) if np.any(self.valid_mask) else 0.0

        print(
            f"[IPM] camera={self.camera_name}, "
            f"bev_size={bev_w}x{bev_h}, "
            f"valid_ratio={valid_ratio:.3f}, "
            f"conf_mean={conf_mean:.3f}"
        )

    def _build_confidence_map(self, u, v, z, valid_uv, bev_h, bev_w):
        """
        Build confidence map for IPM sampling.

        直觉：
        - 图像越靠下，越可能是地面，权重越高
        - 图像越靠左右边缘，畸变越大，权重越低
        - 点越远，IPM 越容易被拉伸，权重越低
        """
        conf = np.zeros_like(z, dtype=np.float64)

        if not np.any(valid_uv):
            return conf.reshape(bev_h, bev_w).astype(np.float32)

        # 1. vertical confidence
        # v 越大越靠近图像底部，越可信。
        # 对图像上半部大幅降权。
        v_norm = np.clip(v / max(self.height - 1, 1), 0.0, 1.0)

        # 这个曲线可调：
        # v_norm < 0.35 基本不可信
        # v_norm > 0.75 高可信
        vertical_conf = np.clip((v_norm - 0.32) / 0.58, 0.0, 1.0)
        vertical_conf = vertical_conf ** 1.15

        # 2. horizontal confidence
        # 图像左右边缘畸变大，中心更可信
        u_center_dist = np.abs(u - self.width * 0.5) / max(self.width * 0.5, 1)
        horizontal_conf = np.clip(1.0 - 0.45 * (u_center_dist ** 1.5), 0.35, 1.0)

        # 3. depth confidence
        # z 是 optical frame forward depth。
        # 太远的地面点降低权重。
        near_z = 1.0
        far_z = 60.0
        z_norm = np.clip((z - near_z) / max(far_z - near_z, 1e-6), 0.0, 1.0)
        depth_conf = 1.0 - 0.65 * (z_norm ** 1.5)
        depth_conf = np.clip(depth_conf, 0.25, 1.0)

        total_conf = vertical_conf * horizontal_conf * depth_conf

        conf[valid_uv] = total_conf[valid_uv]

        conf = conf.reshape(bev_h, bev_w).astype(np.float32)

        # 轻微平滑，避免 confidence 边界过硬
        conf = cv2.GaussianBlur(conf, (9, 9), 0)

        conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
        return conf

    def project(
        self,
        image,
        interpolation=cv2.INTER_LINEAR,
        invalid_value=0,
    ):
        """
        Project camera image/mask/depth to BEV image.
        """
        if image is None:
            return None

        bev = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=invalid_value,
        )

        bev[~self.valid_mask] = 0
        return bev

    def project_rgb(self, image_bgr):
        return self.project(
            image_bgr,
            interpolation=cv2.INTER_LINEAR,
            invalid_value=(0, 0, 0),
        )

    def project_label(self, label_image):
        return self.project(
            label_image,
            interpolation=cv2.INTER_NEAREST,
            invalid_value=0,
        )

    def project_depth(self, depth_image):
        return self.project(
            depth_image,
            interpolation=cv2.INTER_LINEAR,
            invalid_value=0,
        )

    def get_valid_mask(self):
        return self.valid_mask

    def get_confidence_map(self):
        return self.confidence_map

    def get_ground_consistency_mask(
        self,
        depth_bev,
        abs_tolerance_m=1.2,
        rel_tolerance=0.08,
    ):
        """
        Keep BEV pixels whose measured depth matches the expected ground depth.

        This suppresses vertical objects/buildings that otherwise get smeared by
        ground-plane IPM.
        """
        if depth_bev is None:
            return self.valid_mask.copy()

        depth = depth_bev.astype(np.float32)
        expected_z = self.expected_depth_map
        expected_range = self.expected_range_map

        if depth.shape != expected_z.shape:
            depth = cv2.resize(
                depth,
                (expected_z.shape[1], expected_z.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        tolerance_z = np.maximum(abs_tolerance_m, rel_tolerance * expected_z)
        tolerance_range = np.maximum(abs_tolerance_m, rel_tolerance * expected_range)

        matches_z = np.abs(depth - expected_z) <= tolerance_z
        matches_range = np.abs(depth - expected_range) <= tolerance_range

        mask = (
            self.valid_mask
            & (depth > 1e-3)
            & (expected_z > 1e-3)
            & (matches_z | matches_range)
        )

        return mask
