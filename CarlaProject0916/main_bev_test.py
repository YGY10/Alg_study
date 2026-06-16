# main_bev_test.py

import time
import cv2
import numpy as np

from config.carla_config import (
    BEV_RECORD_DIR,
    BEV_RECORD_EVERY_N_FRAMES,
    ENABLE_BEV_RECORDING,
    MAX_SIMULATION_FRAMES,
    MAX_SIMULATION_SECONDS,
    SYNCHRONOUS_MODE,
    FIXED_DELTA_SECONDS,
    WINDOW_NAME,
)

from config.camera_config import CAMERA_CONFIGS
from config.camera_config import ENABLE_DEPTH_GROUND_FILTER
from config.bev_config import (
    BEV_X_MIN,
    BEV_X_MAX,
    BEV_Y_MIN,
    BEV_Y_MAX,
    BEV_RESOLUTION,
)

from carla_core.carla_client import create_carla_client
from carla_core.world_manager import WorldManager
from carla_core.ego_vehicle import EgoVehicle

from sensors.camera_manager import CameraManager
from visualization.image_viewer import MultiCameraViewer
from visualization.bev_viewer import BEVViewer, BEVCompareViewer
from visualization.perception_viewer import PerceptionViewer

from control.simple_vehicle_control import SimpleVehicleControl
from utils.bev_frame_recorder import BEVFrameRecorder
from utils.fps_counter import FPSCounter

from calibration.camera_intrinsic import compute_camera_intrinsic, print_intrinsic
from calibration.camera_extrinsic import compute_camera_extrinsic, print_extrinsic

from bev.bev_grid import BEVGrid
from bev.multi_camera_bev import MultiCameraBEV
from bev.bev_stitcher import BEVStitcher

from perception.bev_perception import BEVPerception
from perception.perception_output import BEVPerceptionInput


PRINT_CAMERA_CALIBRATION = False
SEMANTIC_LABEL_NAMES = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "Sidewalk",
    9: "Vegetation",
    10: "Vehicle",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain",
}


def print_camera_calibration():
    print("\n" + "=" * 80)
    print("[INFO] Camera calibration check")
    print("=" * 80)

    for cam_name, cam_cfg in CAMERA_CONFIGS.items():
        K = compute_camera_intrinsic(
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fov_deg=cam_cfg["fov"],
        )

        T_vehicle_camera, T_camera_vehicle = compute_camera_extrinsic(cam_cfg)

        print_intrinsic(cam_name, K)
        print_extrinsic(cam_name, T_vehicle_camera, T_camera_vehicle)


def print_semantic_diagnostics(multi_frame, semantic_bev, ground_ratios=None):
    raw_summary = format_raw_semantic_summary(multi_frame.semantic_frames)
    if semantic_bev is None:
        print(
            f"[DIAG] frame={multi_frame.frame_id}, "
            f"sync={multi_frame.synchronized}, semantic_bev=None, "
            f"ground={format_ground_ratios(ground_ratios)}, "
            f"raw={raw_summary}"
        )
        return

    values, counts = np.unique(semantic_bev, return_counts=True)
    pairs = sorted(
        zip(values.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    summary = []
    for label_id, count in pairs[:10]:
        name = SEMANTIC_LABEL_NAMES.get(int(label_id), f"label_{label_id}")
        summary.append(f"{int(label_id)}:{name}={int(count)}")

    road_count = int(
        np.count_nonzero((semantic_bev == 3) | (semantic_bev == 7) | (semantic_bev == 14))
    )
    line_count = int(np.count_nonzero(semantic_bev == 6))
    sidewalk_count = int(np.count_nonzero(semantic_bev == 8))

    print(
        f"[DIAG] frame={multi_frame.frame_id}, "
        f"sync={multi_frame.synchronized}, "
        f"road={road_count}, line={line_count}, sidewalk={sidewalk_count}, "
        f"ground={format_ground_ratios(ground_ratios)}, "
        f"raw={raw_summary}, "
        f"top_labels=[{'; '.join(summary)}]"
    )


def format_ground_ratios(ground_ratios):
    if not ground_ratios:
        return "none"

    values = list(ground_ratios.values())
    if not values:
        return "none"

    mean_ratio = float(np.mean(values))
    min_ratio = float(np.min(values))
    max_ratio = float(np.max(values))
    return f"mean={mean_ratio:.3f},min={min_ratio:.3f},max={max_ratio:.3f}"


def format_raw_semantic_summary(semantic_frames):
    if not semantic_frames:
        return "none"

    parts = []
    for name in ("front", "front_left", "front_right"):
        frame = semantic_frames.get(name)
        if frame is None:
            continue

        labels = frame.image
        other = int(np.count_nonzero(labels == 3))
        road = int(np.count_nonzero(labels == 7))
        ground = int(np.count_nonzero(labels == 14))
        line = int(np.count_nonzero(labels == 6))
        sidewalk = int(np.count_nonzero(labels == 8))
        parts.append(f"{name}:o{other}/r{road}/g{ground}/l{line}/s{sidewalk}")

    return ",".join(parts) if parts else "none"


def main():
    client, world = create_carla_client()

    world_manager = WorldManager(
        world=world,
        synchronous_mode=SYNCHRONOUS_MODE,
        fixed_delta_seconds=FIXED_DELTA_SECONDS,
    )

    ego = EgoVehicle(world)
    camera_manager = None
    image_viewer = None
    bev_viewer = None
    perception_viewer = None
    bev_compare_viewer = None
    controller = None
    recorder = None

    main_fps_counter = FPSCounter("main_loop")
    bev_fps_counter = FPSCounter("bev_loop")
    last_diag_time = 0.0
    loop_start_time = None
    simulation_frame_count = 0

    try:
        world_manager.setup()

        vehicle = ego.spawn()
        controller = SimpleVehicleControl(vehicle)

        camera_manager = CameraManager(
            world=world,
            ego_vehicle=vehicle,
        )
        camera_manager.spawn_all()

        if PRINT_CAMERA_CALIBRATION:
            print_camera_calibration()

        bev_grid = BEVGrid(
            x_min=BEV_X_MIN,
            x_max=BEV_X_MAX,
            y_min=BEV_Y_MIN,
            y_max=BEV_Y_MAX,
            resolution=BEV_RESOLUTION,
        )

        print(
            f"[BEV] grid created: "
            f"x=[{BEV_X_MIN}, {BEV_X_MAX}], "
            f"y=[{BEV_Y_MIN}, {BEV_Y_MAX}], "
            f"res={BEV_RESOLUTION}, "
            f"size={bev_grid.get_size()}"
        )

        multi_camera_bev = MultiCameraBEV(
            camera_configs=CAMERA_CONFIGS,
            bev_grid=bev_grid,
        )

        bev_stitcher = BEVStitcher(
            bev_grid=bev_grid,
            camera_configs=CAMERA_CONFIGS,
        )

        bev_perception = BEVPerception()
        if ENABLE_BEV_RECORDING:
            recorder = BEVFrameRecorder(
                output_dir=BEV_RECORD_DIR,
                every_n_frames=BEV_RECORD_EVERY_N_FRAMES,
            )

        image_viewer = MultiCameraViewer(WINDOW_NAME)
        bev_viewer = BEVViewer("Surround BEV")
        perception_viewer = PerceptionViewer("BEV Perception Debug")
        bev_compare_viewer = BEVCompareViewer("BEV Compare", target_height=800)

        # 先 tick 几帧，让 sensors 开始回调
        for _ in range(10):
            world_manager.tick()
            time.sleep(0.01)

        print("[INFO] 8-camera surround BEV perception test started.")
        print("[INFO] Press 'q' in any image window to quit.")
        if MAX_SIMULATION_FRAMES is not None:
            print(f"[INFO] Auto-stop after {MAX_SIMULATION_FRAMES} simulation frames.")
        if MAX_SIMULATION_SECONDS is not None:
            print(f"[INFO] Auto-stop after {MAX_SIMULATION_SECONDS} seconds.")

        loop_start_time = time.time()
        while True:
            simulation_frame_count += 1
            controller.apply_forward(throttle=0.35, steer=0.0)

            world_manager.tick()

            multi_frame = camera_manager.get_synchronized_frames()

            ego_speed_kmh = ego.get_speed_kmh()
            main_fps = main_fps_counter.update()

            key1 = image_viewer.show(
                multi_camera_frame=multi_frame,
                ego_speed_kmh=ego_speed_kmh,
                main_fps=main_fps,
            )

            bev_outputs = multi_camera_bev.project_all_modalities(multi_frame)

            bev_images = bev_outputs["rgb_images"]
            semantic_bev_images = bev_outputs["semantic_images"]
            depth_bev_images = bev_outputs["depth_images"]
            valid_masks = bev_outputs["valid_masks"]
            ground_masks = bev_outputs["ground_masks"]
            confidence_maps = bev_outputs["confidence_maps"]

            stitch_masks = (
                ground_masks
                if ENABLE_DEPTH_GROUND_FILTER and ground_masks
                else valid_masks
            )

            surround_bev = bev_stitcher.stitch(
                bev_images=bev_images,
                valid_masks=stitch_masks,
                confidence_maps=confidence_maps,
            )

            semantic_bev = None
            if any(image is not None for image in semantic_bev_images.values()):
                semantic_bev = bev_stitcher.stitch_labels(
                    label_images=semantic_bev_images,
                    valid_masks=stitch_masks,
                    confidence_maps=confidence_maps,
                )

            depth_bev = None
            if any(image is not None for image in depth_bev_images.values()):
                depth_bev = bev_stitcher.stitch_depth(
                    depth_images=depth_bev_images,
                    valid_masks=stitch_masks,
                    confidence_maps=confidence_maps,
                )

            now = time.time()
            if now - last_diag_time > 3.0:
                print_semantic_diagnostics(
                    multi_frame,
                    semantic_bev,
                    ground_ratios=multi_camera_bev.last_ground_ratios,
                )
                last_diag_time = now

            surround_bev_debug = bev_stitcher.draw_debug_grid(surround_bev)

            bev_fps = bev_fps_counter.update()

            key2 = bev_viewer.show(
                surround_bev_debug,
                fps=bev_fps,
                title="8-Camera Surround BEV",
            )

            perception_input = BEVPerceptionInput(
                bev_image=surround_bev,
                timestamp=time.time(),
                ego_speed_kmh=ego_speed_kmh,
                bev_grid=bev_grid,
                observed_mask=bev_stitcher.last_observed_mask,
                semantic_bev=semantic_bev,
                depth_bev=depth_bev,
            )

            perception_output = bev_perception.process(perception_input)

            if recorder is not None:
                recorder.maybe_record(
                    multi_camera_frame=multi_frame,
                    surround_bev=surround_bev,
                    semantic_bev=semantic_bev,
                    depth_bev=depth_bev,
                    perception_output=perception_output,
                    ego_speed_kmh=ego_speed_kmh,
                    bev_grid=bev_grid,
                )

            key3 = perception_viewer.show(
                perception_output.debug_image
            )

            key4 = bev_compare_viewer.show(
                surround_bev=surround_bev,
                perception_debug_image=perception_output.debug_image,
                fps=bev_fps,
                title_left="8-Camera Surround BEV",
                title_right="Perception Debug",
            )

            if (
                key1 == ord("q")
                or key2 == ord("q")
                or key3 == ord("q")
                or key4 == ord("q")
            ):
                print("[INFO] Quit requested.")
                break

            if (
                MAX_SIMULATION_FRAMES is not None
                and simulation_frame_count >= int(MAX_SIMULATION_FRAMES)
            ):
                print(
                    f"[INFO] Reached MAX_SIMULATION_FRAMES="
                    f"{MAX_SIMULATION_FRAMES}."
                )
                break

            if (
                MAX_SIMULATION_SECONDS is not None
                and loop_start_time is not None
                and time.time() - loop_start_time >= float(MAX_SIMULATION_SECONDS)
            ):
                print(
                    f"[INFO] Reached MAX_SIMULATION_SECONDS="
                    f"{MAX_SIMULATION_SECONDS}."
                )
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("[INFO] Cleaning up...")

        if controller is not None:
            controller.stop()

        if camera_manager is not None:
            camera_manager.destroy_all()

        ego.destroy()

        world_manager.restore()

        if image_viewer is not None:
            image_viewer.close()

        if bev_viewer is not None:
            bev_viewer.close()

        if perception_viewer is not None:
            perception_viewer.close()

        if bev_compare_viewer is not None:
            bev_compare_viewer.close()

        cv2.destroyAllWindows()

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
