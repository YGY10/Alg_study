import carla
import time
import math
import matplotlib.pyplot as plt


TARGET_MAP = "Town10HD"

# ===== 参考线偏移参数 =====
OFFSET = 1.0
RECOVER_DIST = 30.0


def get_available_maps(client):
    maps = client.get_available_maps()
    print("\nAvailable maps:")
    for i, m in enumerate(maps):
        print(f"  [{i}] {m}")
    return maps


def ensure_map(client, target):
    world = client.get_world()
    current = world.get_map().name

    print(f"Current map: {current}")

    maps = get_available_maps(client)

    if target not in current:
        print(f"\nLoading map {target} ...")
        world = client.load_world(target)
        time.sleep(2.0)

    print(f"Using map: {world.get_map().name}")
    return world


def sample_path(start_wp, step=1.0, max_len=200):
    pts = []
    wp = start_wp

    for _ in range(max_len):
        loc = wp.transform.location
        yaw = wp.transform.rotation.yaw

        pts.append((loc.x, loc.y, yaw))

        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]

    return pts


def build_light_map(gt_path):
    lm = []
    for i, (x, y, yaw) in enumerate(gt_path):
        if i < 30:
            offset = OFFSET
        elif i < 60:
            offset = OFFSET
        else:
            ratio = min(1.0, (i - 60) / RECOVER_DIST)
            offset = OFFSET * (1.0 - ratio)

        rad = math.radians(yaw)
        dx = -math.sin(rad) * offset
        dy = math.cos(rad) * offset

        lm.append((x + dx, y + dy))

    return lm


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = ensure_map(client, TARGET_MAP)
    amap = world.get_map()

    spawn_points = amap.get_spawn_points()
    print("spawn_points num =", len(spawn_points))

    spawn_tf = spawn_points[0]
    spawn_tf.location.z += 0.5

    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(bp, spawn_tf)

    # ✅ 关键：关闭物理，防止跳动
    vehicle.set_simulate_physics(False)

    try:
        ego_wp = amap.get_waypoint(spawn_tf.location)
        gt_path = sample_path(ego_wp, step=1.0, max_len=150)
        lm_path = build_light_map(gt_path)

        # ===== 可视化 =====
        xs = [p[0] for p in gt_path]
        ys = [p[1] for p in gt_path]

        lm_x = [p[0] for p in lm_path]
        lm_y = [p[1] for p in lm_path]

        plt.figure()
        plt.plot(xs, ys, label="GT")
        plt.plot(lm_x, lm_y, "--", label="LightMap")
        plt.legend()
        plt.axis("equal")
        plt.title("Reference Switch Demo")

        # ===== 车辆沿轨迹移动 =====
        for i in range(len(gt_path)):
            x, y, yaw = gt_path[i]

            # ✅ 关键：用 waypoint 获取真实高度
            wp_now = amap.get_waypoint(
                carla.Location(x=x, y=y, z=0),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            z = wp_now.transform.location.z + 0.05

            tf = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=yaw),
            )

            vehicle.set_transform(tf)

            # 绘制路径
            world.debug.draw_point(
                carla.Location(x=x, y=y, z=z + 0.5),
                size=0.05,
                color=carla.Color(0, 255, 0),
                life_time=0.2,
            )

            time.sleep(0.05)

        plt.show()

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    main()
