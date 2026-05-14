class WorldManager:
    def __init__(self, world, synchronous_mode=True, fixed_delta_seconds=0.05):
        self.world = world
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.original_settings = None

    def setup(self):
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.fixed_delta_seconds = self.fixed_delta_seconds

        self.world.apply_settings(settings)

        print(
            "[INFO] World settings applied: "
            f"synchronous={settings.synchronous_mode}, "
            f"fixed_delta_seconds={settings.fixed_delta_seconds}"
        )

    def tick(self):
        return self.world.tick()

    def restore(self):
        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
            print("[INFO] World settings restored.")