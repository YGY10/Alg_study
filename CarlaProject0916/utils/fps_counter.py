import time


class FPSCounter:
    def __init__(self, name="fps"):
        self.name = name
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if dt > 1e-6:
            self.fps = 1.0 / dt

        return self.fps