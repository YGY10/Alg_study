import math

import pytest

from sim2d.control import (
    ManualControlMapper,
    ManualInputState,
)
from sim2d.types import (
    VehicleConfig,
    VehicleControl,
)


@pytest.fixture
def vehicle_config() -> VehicleConfig:
    return VehicleConfig(
        length=4.6,
        width=1.85,
        wheel_base=2.7,
        acceleration_min=-3.0,
        acceleration_max=2.0,
        steering_min=-0.45,
        steering_max=0.45,
        speed_min=0.0,
        speed_max=15.0,
    )


@pytest.fixture
def mapper(
    vehicle_config: VehicleConfig,
) -> ManualControlMapper:
    return ManualControlMapper(vehicle_config=vehicle_config)


def test_neutral_input_returns_zero_control(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(ManualInputState())

    assert result == VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )


def test_full_throttle_maps_to_max_acceleration(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            throttle=1.0,
        )
    )

    assert result.acceleration == pytest.approx(2.0)


def test_half_throttle_maps_proportionally(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            throttle=0.5,
        )
    )

    assert result.acceleration == pytest.approx(1.0)


def test_full_brake_maps_to_min_acceleration(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            brake=1.0,
        )
    )

    assert result.acceleration == pytest.approx(-3.0)


def test_brake_has_priority_over_throttle(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            throttle=1.0,
            brake=0.5,
        )
    )

    assert result.acceleration == pytest.approx(-1.5)


def test_full_left_steering(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            steering=1.0,
        )
    )

    assert result.steering == pytest.approx(0.45)


def test_full_right_steering(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            steering=-1.0,
        )
    )

    assert result.steering == pytest.approx(-0.45)


def test_partial_steering_maps_proportionally(
    mapper: ManualControlMapper,
) -> None:
    result = mapper.map(
        ManualInputState(
            steering=0.5,
        )
    )

    assert result.steering == pytest.approx(0.225)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("steering", -1.1),
        ("steering", 1.1),
        ("throttle", -0.1),
        ("throttle", 1.1),
        ("brake", -0.1),
        ("brake", 1.1),
    ],
)
def test_invalid_input_range_raises(
    mapper: ManualControlMapper,
    field_name: str,
    value: float,
) -> None:
    values = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    }

    values[field_name] = value

    input_state = ManualInputState(**values)

    with pytest.raises(ValueError):
        mapper.map(input_state)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_input_raises(
    mapper: ManualControlMapper,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        mapper.map(
            ManualInputState(
                steering=value,
            )
        )
