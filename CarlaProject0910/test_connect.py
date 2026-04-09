import carla


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Connected to CARLA")
    print("Map name:", carla_map.name)


if __name__ == "__main__":
    main()
