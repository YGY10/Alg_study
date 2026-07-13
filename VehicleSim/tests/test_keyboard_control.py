import pytest

from sim2d.control import (
    KeyboardControlConfig,
    KeyboardControlState,
    ManualInputState,
)


@pytest.fixture
def config() -> KeyboardControlConfig:
    return KeyboardControlConfig(
        steering_press_rate=2.0,
        steering_return_rate=4.0,
        throttle_rise_rate=1.0,
        throttle_fall_rate=2.0,
        brake_rise_rate=5.0,
        brake_fall_rate=10.0,
    )


@pytest.fixture
def keyboard(
    config: KeyboardControlConfig,
) -> KeyboardControlState:
    return KeyboardControlState(config=config)


def test_initial_state_is_neutral(
    keyboard: KeyboardControlState,
) -> None:
    assert keyboard.input_state == ManualInputState()


def test_left_key_increases_left_steering(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.left_pressed = True

    state = keyboard.update(dt=0.1)

    assert state.steering == pytest.approx(0.2)


def test_right_key_increases_right_steering(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.right_pressed = True

    state = keyboard.update(dt=0.1)

    assert state.steering == pytest.approx(-0.2)


def test_steering_is_clamped_at_full_scale(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.right_pressed = True

    state = keyboard.update(dt=10.0)

    assert state.steering == pytest.approx(-1.0)


def test_steering_returns_to_center(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.right_pressed = True
    keyboard.update(dt=0.25)

    keyboard.right_pressed = False
    state = keyboard.update(dt=0.1)

    assert state.steering == pytest.approx(-0.1)


def test_both_steering_keys_return_to_center(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.right_pressed = True
    keyboard.update(dt=0.25)

    keyboard.left_pressed = True
    state = keyboard.update(dt=0.1)

    assert state.steering == pytest.approx(-0.1)


def test_throttle_rises_smoothly(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.throttle_pressed = True

    state = keyboard.update(dt=0.25)

    assert state.throttle == pytest.approx(0.25)


def test_throttle_is_clamped_at_one(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.throttle_pressed = True

    state = keyboard.update(dt=10.0)

    assert state.throttle == pytest.approx(1.0)


def test_throttle_falls_when_released(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.throttle_pressed = True
    keyboard.update(dt=0.5)

    keyboard.throttle_pressed = False
    state = keyboard.update(dt=0.1)

    assert state.throttle == pytest.approx(0.3)


def test_brake_rises_quickly(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.brake_pressed = True

    state = keyboard.update(dt=0.1)

    assert state.brake == pytest.approx(0.5)


def test_brake_falls_when_released(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.brake_pressed = True
    keyboard.update(dt=0.1)

    keyboard.brake_pressed = False
    state = keyboard.update(dt=0.02)

    assert state.brake == pytest.approx(0.3)


def test_reset_clears_keys_and_values(
    keyboard: KeyboardControlState,
) -> None:
    keyboard.left_pressed = True
    keyboard.throttle_pressed = True
    keyboard.brake_pressed = True

    keyboard.update(dt=0.2)
    keyboard.reset()

    assert keyboard.input_state == ManualInputState()

    assert keyboard.left_pressed is False
    assert keyboard.right_pressed is False
    assert keyboard.throttle_pressed is False
    assert keyboard.brake_pressed is False


def test_update_rejects_non_positive_dt(
    keyboard: KeyboardControlState,
) -> None:
    with pytest.raises(ValueError):
        keyboard.update(dt=0.0)

    with pytest.raises(ValueError):
        keyboard.update(dt=-0.1)


@pytest.mark.parametrize(
    "field_name",
    [
        "steering_press_rate",
        "steering_return_rate",
        "throttle_rise_rate",
        "throttle_fall_rate",
        "brake_rise_rate",
        "brake_fall_rate",
    ],
)
def test_config_rejects_non_positive_rate(
    field_name: str,
) -> None:
    values = {
        "steering_press_rate": 1.0,
        "steering_return_rate": 1.0,
        "throttle_rise_rate": 1.0,
        "throttle_fall_rate": 1.0,
        "brake_rise_rate": 1.0,
        "brake_fall_rate": 1.0,
    }

    values[field_name] = 0.0

    config = KeyboardControlConfig(**values)

    with pytest.raises(ValueError):
        KeyboardControlState(config=config)
