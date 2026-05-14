import random
import carla

from config.carla_config import EGO_VEHICLE_TYPE, EGO_ROLE_NAME


class EgoVehicle:
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.spawn_transform = None

    def spawn(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(EGO_VEHICLE_TYPE)[0]
        vehicle_bp.set_attribute("role_name", EGO_ROLE_NAME)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("[ERROR] No spawn points found in current map.")

        indices = list(range(len(spawn_points)))
        random.shuffle(indices)

        for idx in indices:
            spawn_point = spawn_points[idx]
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if vehicle is not None:
                self.vehicle = vehicle
                self.spawn_transform = spawn_point

                print(
                    f"[INFO] Ego spawned at index={idx}, "
                    f"location={spawn_point.location}"
                )
                return self.vehicle

        raise RuntimeError("[ERROR] Failed to spawn ego vehicle. All spawn points occupied.")

    def get_actor(self):
        if self.vehicle is None:
            raise RuntimeError("[ERROR] Ego vehicle has not been spawned.")
        return self.vehicle

    def destroy(self):
        if self.vehicle is not None:
            print("[INFO] Destroying ego vehicle.")
            self.vehicle.destroy()
            self.vehicle = None

    def get_speed_kmh(self):
        if self.vehicle is None:
            return 0.0

        v = self.vehicle.get_velocity()
        speed_mps = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        return speed_mps * 3.6