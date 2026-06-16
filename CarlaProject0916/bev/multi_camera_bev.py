from bev.ipm_projector import IPMProjector
from config.camera_config import ENABLE_DEPTH_GROUND_FILTER


class MultiCameraBEV:
    def __init__(self, camera_configs, bev_grid):
        self.camera_configs = camera_configs
        self.bev_grid = bev_grid
        self.projectors = {}
        self.last_ground_ratios = {}

        self._build_projectors()

    def _build_projectors(self):
        for camera_name, camera_config in self.camera_configs.items():
            projector = IPMProjector(
                camera_name=camera_name,
                camera_config=camera_config,
                bev_grid=self.bev_grid,
            )
            self.projectors[camera_name] = projector

        print(f"[BEV] MultiCameraBEV projectors created: {list(self.projectors.keys())}")

    def project_all(self, multi_camera_frame):
        """
        Return:
            bev_images: dict[str, np.ndarray]
            valid_masks: dict[str, np.ndarray]
            confidence_maps: dict[str, np.ndarray]
        """
        bev_images = {}
        valid_masks = {}
        confidence_maps = {}

        for camera_name, projector in self.projectors.items():
            camera_frame = multi_camera_frame.frames.get(camera_name)

            if camera_frame is None:
                bev_images[camera_name] = None
                valid_masks[camera_name] = projector.get_valid_mask()
                confidence_maps[camera_name] = projector.get_confidence_map()
                continue

            bev = projector.project_rgb(camera_frame.image)

            bev_images[camera_name] = bev
            valid_masks[camera_name] = projector.get_valid_mask()
            confidence_maps[camera_name] = projector.get_confidence_map()

        return bev_images, valid_masks, confidence_maps

    def project_all_modalities(self, multi_camera_frame):
        """
        Project RGB, semantic label, and depth frames with the same camera geometry.

        Return:
            dict with keys:
                rgb_images, semantic_images, depth_images,
                valid_masks, confidence_maps
        """
        rgb_images = {}
        semantic_images = {}
        depth_images = {}
        valid_masks = {}
        ground_masks = {}
        confidence_maps = {}

        semantic_frames = multi_camera_frame.semantic_frames or {}
        depth_frames = multi_camera_frame.depth_frames or {}

        for camera_name, projector in self.projectors.items():
            rgb_frame = multi_camera_frame.frames.get(camera_name)
            semantic_frame = semantic_frames.get(camera_name)
            depth_frame = depth_frames.get(camera_name)

            depth_bev = (
                None
                if depth_frame is None
                else projector.project_depth(depth_frame.image)
            )
            if ENABLE_DEPTH_GROUND_FILTER:
                ground_mask = projector.get_ground_consistency_mask(depth_bev)
            else:
                ground_mask = projector.get_valid_mask()

            rgb_images[camera_name] = (
                None
                if rgb_frame is None
                else projector.project_rgb(rgb_frame.image)
            )
            semantic_images[camera_name] = (
                None
                if semantic_frame is None
                else projector.project_label(semantic_frame.image)
            )
            depth_images[camera_name] = depth_bev

            if (
                ENABLE_DEPTH_GROUND_FILTER
                and rgb_images[camera_name] is not None
                and depth_bev is not None
            ):
                rgb_images[camera_name][~ground_mask] = 0

            if (
                ENABLE_DEPTH_GROUND_FILTER
                and semantic_images[camera_name] is not None
                and depth_bev is not None
            ):
                semantic_images[camera_name][~ground_mask] = 0

            valid_masks[camera_name] = projector.get_valid_mask()
            ground_masks[camera_name] = ground_mask
            self.last_ground_ratios[camera_name] = (
                float(ground_mask.sum()) / float(ground_mask.size)
                if ground_mask is not None and ground_mask.size > 0
                else 0.0
            )
            confidence_maps[camera_name] = projector.get_confidence_map()

        return {
            "rgb_images": rgb_images,
            "semantic_images": semantic_images,
            "depth_images": depth_images,
            "valid_masks": valid_masks,
            "ground_masks": ground_masks,
            "confidence_maps": confidence_maps,
        }
