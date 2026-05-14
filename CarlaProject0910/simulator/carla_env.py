# simulator/carla_env.py
import math
import carla


class CarlaEnv:
    def __init__(self, host="localhost", port=2000, timeout=10.0, town=None, dt=0.05):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.dt = dt
        self.town = town

        self.client = None
        self.world = None
        self.map = None
        self.blueprint_library = None

        self.ego_vehicle = None
        self.actors = []

        self.original_settings = None

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        if self.town is not None:
            self.world = self.client.load_world(self.town)
        else:
            self.world = self.client.get_world()

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    def setup_sync_mode(self):
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

        self.world.tick()

    def restore_settings(self):
        if self.world is not None and self.original_settings is not None:
            self.world.apply_settings(self.original_settings)

    def get_spawn_points(self):
        return self.map.get_spawn_points()

    def spawn_ego_vehicle(
        self, blueprint_filter="vehicle.tesla.model3", spawn_point=None
    ):
        bp = self.blueprint_library.find(blueprint_filter)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "ego")

        if spawn_point is None:
            spawn_points = self.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found in current map.")
            spawn_point = spawn_points[0]

        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.ego_vehicle = vehicle
        self.actors.append(vehicle)
        self.world.tick()
        return vehicle

    def tick(self):
        return self.world.tick()

    def get_vehicle_state(self, vehicle=None):
        vehicle = vehicle or self.ego_vehicle
        if vehicle is None:
            raise RuntimeError("No vehicle available.")

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        accel = vehicle.get_acceleration()

        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        state = {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "yaw_deg": transform.rotation.yaw,
            "vx": velocity.x,
            "vy": velocity.y,
            "vz": velocity.z,
            "ax": accel.x,
            "ay": accel.y,
            "az": accel.z,
            "speed": speed,
        }
        return state

    def get_nearest_waypoint(
        self, location, project_to_road=True, lane_type=carla.LaneType.Driving
    ):
        waypoint = self.map.get_waypoint(
            location,
            project_to_road=project_to_road,
            lane_type=lane_type,
        )
        return waypoint

    def sample_lane_centerline(self, start_waypoint, step=2.0, num_points=30):
        """
        从 start_waypoint 开始，沿当前车道中心线向前采样。
        返回 waypoint 列表，包含起点。
        """
        if start_waypoint is None:
            return []

        waypoints = [start_waypoint]
        current_wp = start_waypoint

        for _ in range(num_points - 1):
            next_wps = current_wp.next(step)
            if not next_wps:
                break

            current_wp = next_wps[0]
            waypoints.append(current_wp)

        return waypoints

    def destroy_all_actors(self):
        for actor in reversed(self.actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception as e:
                print(f"[WARN] Failed to destroy actor: {e}")
        self.actors = []
        self.ego_vehicle = None

    def close(self):
        self.destroy_all_actors()
        self.restore_settings()
