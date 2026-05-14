# main_bev_test.py

import time
import cv2

from config.carla_config import (
    SYNCHRONOUS_MODE,
    FIXED_DELTA_SECONDS,
    WINDOW_NAME,
)

from config.camera_config import CAMERA_CONFIGS
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
from utils.fps_counter import FPSCounter

from calibration.camera_intrinsic import compute_camera_intrinsic, print_intrinsic
from calibration.camera_extrinsic import compute_camera_extrinsic, print_extrinsic

from bev.bev_grid import BEVGrid
from bev.multi_camera_bev import MultiCameraBEV
from bev.bev_stitcher import BEVStitcher

from perception.bev_perception import BEVPerception
from perception.perception_output import BEVPerceptionInput


PRINT_CAMERA_CALIBRATION = False


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

    main_fps_counter = FPSCounter("main_loop")
    bev_fps_counter = FPSCounter("bev_loop")

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

        while True:
            controller.apply_forward(throttle=0.35, steer=0.0)

            world_manager.tick()

            multi_frame = camera_manager.get_latest_frames()

            ego_speed_kmh = ego.get_speed_kmh()
            main_fps = main_fps_counter.update()

            key1 = image_viewer.show(
                multi_camera_frame=multi_frame,
                ego_speed_kmh=ego_speed_kmh,
                main_fps=main_fps,
            )

            bev_images, valid_masks, confidence_maps = multi_camera_bev.project_all(
                multi_frame
            )

            surround_bev = bev_stitcher.stitch(
                bev_images=bev_images,
                valid_masks=valid_masks,
                confidence_maps=confidence_maps,
            )

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
            )

            perception_output = bev_perception.process(perception_input)

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