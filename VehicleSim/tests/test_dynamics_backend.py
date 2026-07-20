from __future__ import annotations

import pytest

from sim2d.core.dynamics import KinematicBicycleModel
from sim2d.core.environment import (
    DrivingEnv,
    EnvironmentConfig,
)
from sim2d.dynamics.base import DynamicsBackend
from sim2d.dynamics.kinematic_bicycle import (
    KinematicBicycleBackend,
)
from sim2d.types import (
    GoalState,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


def make_vehicle_config() -> VehicleConfig:
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


def make_state() -> VehicleState:
    return VehicleState(
        x=1.0,
        y=2.0,
        yaw=0.2,
        speed=3.0,
    )


def make_control() -> VehicleControl:
    return VehicleControl(
        acceleration=0.5,
        steering=0.1,
    )


def assert_state_close(
    actual: VehicleState,
    expected: VehicleState,
) -> None:
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)
    assert actual.yaw == pytest.approx(expected.yaw)
    assert actual.speed == pytest.approx(expected.speed)


def test_backend_requires_reset_before_step() -> None:
    backend = KinematicBicycleBackend(
        vehicle_config=make_vehicle_config(),
    )

    with pytest.raises(
        RuntimeError,
        match="reset",
    ):
        backend.step(
            control=make_control(),
            dt=0.1,
        )


def test_backend_requires_reset_before_get_state() -> None:
    backend = KinematicBicycleBackend(
        vehicle_config=make_vehicle_config(),
    )

    with pytest.raises(
        RuntimeError,
        match="reset",
    ):
        backend.get_state()


def test_backend_reset_sets_state() -> None:
    backend = KinematicBicycleBackend(
        vehicle_config=make_vehicle_config(),
    )
    initial_state = make_state()

    returned_state = backend.reset(initial_state)

    assert_state_close(
        returned_state,
        initial_state,
    )
    assert_state_close(
        backend.get_state(),
        initial_state,
    )

    # reset() 应保存独立 dataclass 实例。
    assert returned_state is not initial_state


def test_backend_matches_original_model() -> None:
    config = make_vehicle_config()
    initial_state = make_state()
    control = make_control()
    dt = 0.1

    expected = KinematicBicycleModel(
        config=config,
    ).step(
        state=initial_state,
        control=control,
        dt=dt,
    )

    backend = KinematicBicycleBackend(
        vehicle_config=config,
    )
    backend.reset(initial_state)

    actual = backend.step(
        control=control,
        dt=dt,
    )

    assert_state_close(
        actual,
        expected,
    )
    assert_state_close(
        backend.get_state(),
        expected,
    )


def test_backend_rejects_non_positive_dt() -> None:
    backend = KinematicBicycleBackend(
        vehicle_config=make_vehicle_config(),
    )
    backend.reset(make_state())

    with pytest.raises(
        ValueError,
        match="dt must be positive",
    ):
        backend.step(
            control=make_control(),
            dt=0.0,
        )


class FakeDynamicsBackend(DynamicsBackend):
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.state: VehicleState | None = None

    def reset(
        self,
        initial_state: VehicleState,
    ) -> VehicleState:
        self.reset_count += 1
        self.state = initial_state
        return initial_state

    def step(
        self,
        control: VehicleControl,
        dt: float,
    ) -> VehicleState:
        del control
        del dt

        if self.state is None:
            raise RuntimeError("Fake backend has not been reset")

        self.step_count += 1

        self.state = VehicleState(
            x=self.state.x + 7.0,
            y=self.state.y,
            yaw=self.state.yaw,
            speed=self.state.speed,
        )
        return self.state

    def get_state(self) -> VehicleState:
        if self.state is None:
            raise RuntimeError("Fake backend has not been reset")

        return self.state


def test_environment_uses_injected_backend() -> None:
    backend = FakeDynamicsBackend()

    environment = DrivingEnv(
        vehicle_config=make_vehicle_config(),
        environment_config=EnvironmentConfig(
            dt=0.05,
            max_time=20.0,
        ),
        dynamics_backend=backend,
    )

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=1.0,
    )

    goal = GoalState(
        state=VehicleState(
            x=100.0,
            y=0.0,
            yaw=0.0,
            speed=0.0,
        ),
        position_tolerance=0.5,
        yaw_tolerance=0.1,
        speed_tolerance=0.1,
    )

    environment.reset(
        initial_state=initial_state,
        goal=goal,
        obstacles=(),
    )

    result = environment.step(
        VehicleControl(
            acceleration=0.0,
            steering=0.0,
        )
    )

    assert backend.reset_count == 1
    assert backend.step_count == 1
    assert result.observation.ego.x == pytest.approx(7.0)
    assert environment.state.x == pytest.approx(7.0)
