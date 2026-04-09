import carla
import matplotlib.pyplot as plt


def sample_lane_forward(wp, step=1.0, n=60):
    points = []
    cur = wp
    for _ in range(n):
        loc = cur.transform.location
        points.append((loc.x, loc.y))

        next_wps = cur.next(step)
        if not next_wps:
            break
        cur = next_wps[0]
    return points


def collect_lateral_lanes(center_wp):
    lanes = []

    # 先加当前车道
    lanes.append(center_wp)

    # 向左扩展
    cur = center_wp
    while True:
        nxt = cur.get_left_lane()
        if nxt is None:
            break
        if nxt.lane_type != carla.LaneType.Driving:
            break
        lanes.append(nxt)
        cur = nxt

    # 向右扩展
    cur = center_wp
    while True:
        nxt = cur.get_right_lane()
        if nxt is None:
            break
        if nxt.lane_type != carla.LaneType.Driving:
            break
        lanes.append(nxt)
        cur = nxt

    return lanes


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    carla_map = world.get_map()

    blueprint = world.get_blueprint_library().filter("vehicle.*")[0]
    spawn_point = carla_map.get_spawn_points()[0]

    ego = world.try_spawn_actor(blueprint, spawn_point)
    if ego is None:
        raise RuntimeError("spawn failed")

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location

    print("ego location:", ego_loc)

    # spectator 拉到车顶
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(ego_loc + carla.Location(z=40), carla.Rotation(pitch=-90))
    )

    center_wp = carla_map.get_waypoint(
        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    all_lanes = collect_lateral_lanes(center_wp)
    print("collected lane count:", len(all_lanes))
    for i, wp in enumerate(all_lanes):
        print(
            f"lane[{i}]: road_id={wp.road_id}, lane_id={wp.lane_id}, lane_width={wp.lane_width:.2f}"
        )

    plt.figure(figsize=(8, 8))

    for i, lane_wp in enumerate(all_lanes):
        pts = sample_lane_forward(lane_wp, step=1.0, n=80)
        if not pts:
            continue
        xs, ys = zip(*pts)
        plt.plot(xs, ys, label=f"lane_id={lane_wp.lane_id}")

    plt.scatter([ego_loc.x], [ego_loc.y], s=80, label="ego")

    plt.axis("equal")
    plt.legend()
    plt.title("All lateral driving lanes around ego")
    plt.show()

    input("Press Enter to destroy ego and exit...")
    ego.destroy()


if __name__ == "__main__":
    main()
