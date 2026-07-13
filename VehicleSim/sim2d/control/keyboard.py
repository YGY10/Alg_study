from __future__ import annotations

from dataclasses import dataclass

from sim2d.control.manual import ManualInputState


def _move_towards(
    current: float,
    target: float,
    max_delta: float,
) -> float:
    """以不超过 max_delta 的速度向 target 移动。"""
    if max_delta < 0.0:
        raise ValueError(f"max_delta must be non-negative, got {max_delta}")

    if current < target:
        return min(
            current + max_delta,
            target,
        )

    if current > target:
        return max(
            current - max_delta,
            target,
        )

    return target


@dataclass(frozen=True)
class KeyboardControlConfig:
    """
    键盘驾驶输入变化率配置。

    所有 rate 的单位都是“归一化输入每秒”。

    steering_press_rate:
        按住方向键时，方向盘输入变化速度。

    steering_return_rate:
        松开方向键后，方向盘回正速度。

    throttle_rise_rate:
        按住油门键时，油门增加速度。

    throttle_fall_rate:
        松开油门键后，油门释放速度。

    brake_rise_rate:
        按住刹车键时，刹车增加速度。

    brake_fall_rate:
        松开刹车键后，刹车释放速度。
    """

    steering_press_rate: float = 1.8
    steering_return_rate: float = 2.4

    throttle_rise_rate: float = 1.5
    throttle_fall_rate: float = 3.0

    brake_rise_rate: float = 4.0
    brake_fall_rate: float = 5.0

    def validate(self) -> None:
        values = {
            "steering_press_rate": (self.steering_press_rate),
            "steering_return_rate": (self.steering_return_rate),
            "throttle_rise_rate": (self.throttle_rise_rate),
            "throttle_fall_rate": (self.throttle_fall_rate),
            "brake_rise_rate": (self.brake_rise_rate),
            "brake_fall_rate": (self.brake_fall_rate),
        }

        for name, value in values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")


class KeyboardControlState:
    """
    保存键盘按键状态，并逐帧生成平滑驾驶输入。

    方向约定：
        left_pressed=True  → steering 向 +1 变化
        right_pressed=True → steering 向 *1 变化

    纵向约定：
        throttle_pressed=True → throttle 向 1 变化
        brake_pressed=True    → brake 向 1 变化
    """

    def __init__(
        self,
        config: KeyboardControlConfig | None = None,
    ) -> None:
        self.config = config if config is not None else KeyboardControlConfig()

        self.config.validate()

        self.left_pressed = False
        self.right_pressed = False
        self.throttle_pressed = False
        self.brake_pressed = False

        self._steering = 0.0
        self._throttle = 0.0
        self._brake = 0.0

    @property
    def input_state(self) -> ManualInputState:
        return ManualInputState(
            steering=self._steering,
            throttle=self._throttle,
            brake=self._brake,
        )

    def reset(self) -> None:
        self.left_pressed = False
        self.right_pressed = False
        self.throttle_pressed = False
        self.brake_pressed = False

        self._steering = 0.0
        self._throttle = 0.0
        self._brake = 0.0

    def update(
        self,
        dt: float,
    ) -> ManualInputState:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._update_steering(dt)
        self._update_throttle(dt)
        self._update_brake(dt)

        return self.input_state

    def _update_steering(
        self,
        dt: float,
    ) -> None:
        if self.left_pressed and not self.right_pressed:
            target = 1.0
            rate = self.config.steering_press_rate

        elif self.right_pressed and not self.left_pressed:
            target = -1.0
            rate = self.config.steering_press_rate

        else:
            target = 0.0
            rate = self.config.steering_return_rate

        self._steering = _move_towards(
            current=self._steering,
            target=target,
            max_delta=rate * dt,
        )

    def _update_throttle(
        self,
        dt: float,
    ) -> None:
        if self.throttle_pressed:
            target = 1.0
            rate = self.config.throttle_rise_rate
        else:
            target = 0.0
            rate = self.config.throttle_fall_rate

        self._throttle = _move_towards(
            current=self._throttle,
            target=target,
            max_delta=rate * dt,
        )

    def _update_brake(
        self,
        dt: float,
    ) -> None:
        if self.brake_pressed:
            target = 1.0
            rate = self.config.brake_rise_rate
        else:
            target = 0.0
            rate = self.config.brake_fall_rate

        self._brake = _move_towards(
            current=self._brake,
            target=target,
            max_delta=rate * dt,
        )
