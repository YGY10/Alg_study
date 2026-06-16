import json
from pathlib import Path

import cv2
import numpy as np


class BEVFrameRecorder:
    def __init__(self, output_dir, every_n_frames=5):
        self.output_dir = Path(output_dir)
        self.every_n_frames = max(1, int(every_n_frames))
        self.count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def maybe_record(
        self,
        multi_camera_frame,
        surround_bev,
        semantic_bev,
        depth_bev,
        perception_output,
        ego_speed_kmh,
        bev_grid,
    ):
        self.count += 1
        if self.count % self.every_n_frames != 0:
            return

        frame_id = multi_camera_frame.frame_id
        if frame_id is None:
            frame_id = self.count

        stem = f"frame_{int(frame_id):08d}"

        if surround_bev is not None:
            cv2.imwrite(str(self.output_dir / f"{stem}_rgb_bev.png"), surround_bev)

        if semantic_bev is not None:
            cv2.imwrite(
                str(self.output_dir / f"{stem}_semantic_bev.png"),
                semantic_bev.astype(np.uint8),
            )

        if depth_bev is not None:
            np.save(self.output_dir / f"{stem}_depth_bev.npy", depth_bev)

        if perception_output is not None and perception_output.debug_image is not None:
            cv2.imwrite(
                str(self.output_dir / f"{stem}_perception_debug.png"),
                perception_output.debug_image,
            )

        metadata = {
            "frame_id": int(frame_id),
            "timestamp": float(multi_camera_frame.timestamp),
            "synchronized": bool(multi_camera_frame.synchronized),
            "ego_speed_kmh": float(ego_speed_kmh),
            "bev": {
                "x_min": bev_grid.x_min,
                "x_max": bev_grid.x_max,
                "y_min": bev_grid.y_min,
                "y_max": bev_grid.y_max,
                "resolution": bev_grid.resolution,
                "width": bev_grid.width,
                "height": bev_grid.height,
            },
        }

        with open(self.output_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
