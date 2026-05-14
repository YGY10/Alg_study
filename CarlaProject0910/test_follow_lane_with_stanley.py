# test_follow_lane_with_stanley.py
import math
import carla

from simulator.carla_env import CarlaEnv
from control.stanley_controller import StanleyController


def build_reference_path_from_waypoints(waypoints):
    """
    把 CARLA waypoint 序列转成 reference path
    每个点包含 x, y, yaw, s
    """
    ref_path = []
    accumulated_s = 0.0

    for i, wp in enumerate(waypoints):
        loc = wp.transform.location
        yaw_deg = wp.transform.rotation.yaw
        yaw = math.radians(yaw_deg)

        if i > 0:
            prev = ref_path[-1]
            ds = math.hypot(loc.x - prev["x"], loc.y - prev["y"])
            accumulated_s += ds

        ref_path.append(
            {
                "x": loc.x,
                "y": loc.y,
                "yaw": yaw,
                "s": accumulated_s,
            }
        )

    return ref_path


def draw_reference_path(
    world, ref_path, target_idx=None, nearest_idx=None, life_time=0.1
):
    debug = world.debug

    for i, pt in enumerate(ref_path):
        loc = carla.Location(x=pt["x"], y=pt["y"], z=0.5)

        if target_idx is not None and i == target_idx:
            color = carla.Color(255, 255, 0)  # 黄色：lookahead目标点
            size = 0.16
        elif nearest_idx is not None and i == nearest_idx:
            color = carla.Color(0, 255, 255)  # 青色：最近点
            size = 0.14
        else:
            color = carla.Color(255, 0, 0)  # 红色：普通参考点
            size = 0.10

        debug.draw_point(
            loc,
            size=size,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )

        if i < len(ref_path) - 1:
            next_pt = ref_path[i + 1]
            next_loc = carla.Location(x=next_pt["x"], y=next_pt["y"], z=0.5)
            debug.draw_line(
                loc,
                next_loc,
                thickness=0.06,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
                persistent_lines=False,
            )


def set_spectator_follow(world, ego, mode="top"):
    spectator = world.get_spectator()
    tf = ego.get_transform()

    if mode == "top":
        spectator.set_transform(
            carla.Transform(
                tf.location + carla.Location(z=25.0),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        )
    else:
        yaw_rad = math.radians(tf.rotation.yaw)
        back_dist = 8.0
        height = 4.0

        cam_x = tf.location.x - back_dist * math.cos(yaw_rad)
        cam_y = tf.location.y - back_dist * math.sin(yaw_rad)
        cam_z = tf.location.z + height

        spectator.set_transform(
            carla.Transform(
                carla.Location(cam_x, cam_y, cam_z),
                carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw, roll=0.0),
            )
        )


def simple_speed_control(target_speed, current_speed):
    """
    最简单纵向控制：
    低速阶段更温和，减少大转角起步时卡死的概率
    """
    speed_error = target_speed - current_speed

    # 低速保护：起步阶段不要给太大油门
    if current_speed < 1.0:
        return 0.20, 0.0

    if speed_error > 1.0:
        throttle = 0.40
        brake = 0.0
    elif speed_error > 0.3:
        throttle = 0.25
        brake = 0.0
    elif speed_error > -0.5:
        throttle = 0.10
        brake = 0.0
    else:
        throttle = 0.0
        brake = 0.20

    return throttle, brake


def choose_spawn_point(spawn_points, index=0):
    if not spawn_points:
        raise RuntimeError("No spawn points found.")
    index = max(0, min(index, len(spawn_points) - 1))
    return spawn_points[index]


def main():
    env = CarlaEnv(
        host="localhost",
        port=2000,
        timeout=10.0,
        town="Town03",
        dt=0.05,
    )

    try:
        print("[INFO] Connecting to CARLA...")
        env.connect()
        print(f"[INFO] Connected map: {env.map.name}")

        print("[INFO] Setting synchronous mode...")
        env.setup_sync_mode()

        spawn_points = env.get_spawn_points()
        print(f"[INFO] Spawn points: {len(spawn_points)}")

        # 可以尝试换不同spawn点，比如 10、20、30
        spawn_point = choose_spawn_point(spawn_points, index=0)

        ego = env.spawn_ego_vehicle(
            blueprint_filter="vehicle.tesla.model3",
            spawn_point=spawn_point,
        )
        print(f"[INFO] Ego spawned: id={ego.id}")

        controller = StanleyController(
            k=1.0,
            ks=1.0,
            wheelbase=2.8,
            max_steer_rad=0.61,  # 约35度
            lookahead_base=3.0,
            lookahead_gain=0.5,
            low_speed_steer_scale=0.6,
            low_speed_threshold=1.0,
            low_speed_max_steer_rad=0.30,  # 新增：低速最大转角约17度
        )

        target_speed = 3.0  # m/s，先保守一点
        print("[INFO] Start Stanley lane following...")

        for i in range(300):
            env.tick()

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw = math.radians(ego_tf.rotation.yaw)

            vel = ego.get_velocity()
            ego_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # 1) 当前 waypoint
            cur_wp = env.get_nearest_waypoint(ego_loc)
            if cur_wp is None:
                print(f"[WARN][{i:03d}] current waypoint is None")
                continue

            # 2) 向前采样局部中心线
            wps = env.sample_lane_centerline(
                start_waypoint=cur_wp,
                step=2.0,
                num_points=40,
            )
            if len(wps) < 2:
                print(f"[WARN][{i:03d}] not enough waypoints")
                continue

            # 3) 转成 reference path
            ref_path = build_reference_path_from_waypoints(wps)

            # 额外算最近点，只用于画图
            nearest_idx = controller.find_nearest_index(
                ego_x=ego_loc.x,
                ego_y=ego_loc.y,
                ref_path=ref_path,
            )

            # 4) Stanley 横向控制
            steer_rad, target_idx, cte, heading_error = controller.compute_control(
                ego_x=ego_loc.x,
                ego_y=ego_loc.y,
                ego_yaw=ego_yaw,
                ego_speed=ego_speed,
                ref_path=ref_path,
            )
            steer_cmd = controller.steer_rad_to_carla(steer_rad)

            # 5) 简单纵向控制
            throttle_cmd, brake_cmd = simple_speed_control(
                target_speed=target_speed,
                current_speed=ego_speed,
            )

            # 6) 发控制
            control = carla.VehicleControl(
                throttle=throttle_cmd,
                steer=steer_cmd,
                brake=brake_cmd,
            )
            ego.apply_control(control)

            # 7) 画参考线
            draw_reference_path(
                env.world,
                ref_path,
                target_idx=target_idx,
                nearest_idx=nearest_idx,
                life_time=0.10,
            )

            # 8) spectator 跟随
            set_spectator_follow(env.world, ego, mode="top")

            print(
                f"[{i:03d}] "
                f"speed={ego_speed:.2f} m/s, "
                f"throttle={throttle_cmd:.2f}, "
                f"brake={brake_cmd:.2f}, "
                f"steer={steer_cmd:.2f}, "
                f"cte={cte:.2f} m, "
                f"heading_err={math.degrees(heading_error):.2f} deg, "
                f"road_id={cur_wp.road_id}, lane_id={cur_wp.lane_id}, "
                f"target_idx={target_idx}, nearest_idx={nearest_idx}"
            )

        print("[INFO] Test finished.")

    finally:
        print("[INFO] Cleaning up...")
        env.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
