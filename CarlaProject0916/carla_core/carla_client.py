import carla

from config.carla_config import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT


def create_carla_client():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    world = client.get_world()

    print(f"[INFO] Connected to CARLA: {CARLA_HOST}:{CARLA_PORT}")
    print(f"[INFO] Current map: {world.get_map().name}")

    return client, world