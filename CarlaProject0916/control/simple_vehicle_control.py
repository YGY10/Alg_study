import carla


class SimpleVehicleControl:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def apply_forward(self, throttle=0.35, steer=0.0):
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=0.0,
            hand_brake=False,
            reverse=False,
        )
        self.vehicle.apply_control(control)

    def stop(self):
        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            reverse=False,
        )
        self.vehicle.apply_control(control)