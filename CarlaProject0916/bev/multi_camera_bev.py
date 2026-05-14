from bev.ipm_projector import IPMProjector


class MultiCameraBEV:
    def __init__(self, camera_configs, bev_grid):
        self.camera_configs = camera_configs
        self.bev_grid = bev_grid
        self.projectors = {}

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

            bev = projector.project(camera_frame.image)

            bev_images[camera_name] = bev
            valid_masks[camera_name] = projector.get_valid_mask()
            confidence_maps[camera_name] = projector.get_confidence_map()

        return bev_images, valid_masks, confidence_maps