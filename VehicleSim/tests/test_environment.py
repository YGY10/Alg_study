import numpy as np
import pytest

from sim2d import (
    CircleObstacle,
    GoalState,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)
from sim2d.core.environment import (
    DrivingEnv,
    EnvironmentConfig,
)


@pytest.fixture
def vehicle_config():
    return VehicleConfig(
        length=4.0,
        width=2.0,
        wheel_base=2.5,
        acceleration_min=-3.0,
        acceleration_max=2.0,
        steering_min=-0.5,
        steering_max=0.5,
        speed_min=0.0,
        speed_max=15.0,
    )


@pytest.fixture
def environment_config():
    return EnvironmentConfig(
        dt=0.1,
        max_time=1.0,
        collision_reward=-100.0,
        goal_reward=100.0,
        step_reward=-0.01,
        progress_reward_weight=1.0,
    )


@pytest.fixture
def env(
    vehicle_config,
    environment_config,
):
    return DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=environment_config,
    )


def make_initial_state() -> VehicleState:
    return VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=2.0,
    )


def make_goal(
    x: float = 10.0,
    speed: float = 2.0,
) -> GoalState:
    return GoalState(
        state=VehicleState(
            x=x,
            y=0.0,
            yaw=0.0,
            speed=speed,
        ),
        position_tolerance=0.2,
        yaw_tolerance=0.1,
        speed_tolerance=0.2,
    )


def test_reset_returns_initial_observation(env):
    initial_state = make_initial_state()
    goal = make_goal()

    observation = env.reset(
        initial_state=initial_state,
        goal=goal,
        obstacles=(),
    )

    assert observation.time == pytest.approx(0.0)
    assert observation.frame == 0
    assert observation.ego == initial_state
    assert observation.goal == goal
    assert observation.obstacles == ()

    assert env.is_done is False
    assert env.frame == 0
    assert env.time == pytest.approx(0.0)


def test_step_advances_one_fixed_time_step(env):
    env.reset(
        initial_state=make_initial_state(),
        goal=make_goal(),
        obstacles=(),
    )

    result = env.step(
        VehicleControl(
            acceleration=0.0,
            steering=0.0,
        )
    )

    assert result.observation.frame == 1
    assert result.observation.time == pytest.approx(0.1)

    assert result.observation.ego.x == pytest.approx(0.2)
    assert result.observation.ego.y == pytest.approx(0.0)
    assert result.observation.ego.speed == pytest.approx(2.0)

    assert result.terminated is False
    assert result.truncated is False


def test_progress_toward_goal_gives_positive_component(env):
    env.reset(
        initial_state=make_initial_state(),
        goal=make_goal(),
        obstacles=(),
    )

    result = env.step(
        VehicleControl(
            acceleration=0.0,
            steering=0.0,
        )
    )

    # 前进 0.2 m，进度奖励约为 0.2，
    # 再减去每步惩罚 0.01。
    assert result.reward == pytest.approx(0.19)


def test_collision_terminates_episode(
    vehicle_config,
    environment_config,
):
    collision_env = DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=environment_config,
    )

    obstacle = CircleObstacle(
        obstacle_id="obs_collision",
        x=2.4,
        y=0.0,
        radius=0.3,
    )

    collision_env.reset(
        initial_state=make_initial_state(),
        goal=make_goal(),
        obstacles=(obstacle,),
    )

    result = collision_env.step(
        VehicleControl(
            acceleration=0.0,
            steering=0.0,
        )
    )

    assert result.terminated is True
    assert result.truncated is False

    assert result.info["collision"] is True
    assert result.info["goal_reached"] is False
    assert result.info["min_clearance"] <= 0.0

    assert result.reward < -90.0


def test_goal_reached_terminates_episode(
    vehicle_config,
    environment_config,
):
    goal_env = DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=environment_config,
    )

    goal_env.reset(
        initial_state=VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=1.0,
        ),
        goal=GoalState(
            state=VehicleState(
                x=0.1,
                y=0.0,
                yaw=0.0,
                speed=1.0,
            ),
            position_tolerance=0.05,
            yaw_tolerance=0.1,
            speed_tolerance=0.1,
        ),
        obstacles=(),
    )

    result = goal_env.step(
        VehicleControl(
            acceleration=0.0,
            steering=0.0,
        )
    )

    assert result.observation.ego.x == pytest.approx(0.1)
    assert result.terminated is True
    assert result.truncated is False

    assert result.info["goal_reached"] is True
    assert result.info["collision"] is False

    assert result.reward > 90.0


def test_timeout_sets_truncated(
    vehicle_config,
):
    timeout_env = DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=EnvironmentConfig(
            dt=0.1,
            max_time=0.3,
        ),
    )

    timeout_env.reset(
        initial_state=VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=0.0,
        ),
        goal=make_goal(),
        obstacles=(),
    )

    action = VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )

    result = timeout_env.step(action)
    assert result.truncated is False

    result = timeout_env.step(action)
    assert result.truncated is False

    result = timeout_env.step(action)

    assert result.terminated is False
    assert result.truncated is True
    assert result.info["timeout"] is True


def test_step_after_done_raises_error(
    vehicle_config,
):
    timeout_env = DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=EnvironmentConfig(
            dt=0.1,
            max_time=0.1,
        ),
    )

    timeout_env.reset(
        initial_state=VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=0.0,
        ),
        goal=make_goal(),
        obstacles=(),
    )

    action = VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )

    timeout_env.step(action)

    with pytest.raises(RuntimeError):
        timeout_env.step(action)


def test_snapshot_contains_history(env):
    env.reset(
        initial_state=make_initial_state(),
        goal=make_goal(),
        obstacles=(),
    )

    action = VehicleControl(
        acceleration=0.0,
        steering=0.0,
    )

    env.step(action)
    env.step(action)

    snapshot = env.get_snapshot()

    assert snapshot.frame == 2
    assert snapshot.time == pytest.approx(0.2)

    assert snapshot.history_trajectory.shape == (3, 4)

    assert np.allclose(
        snapshot.history_trajectory[:, 0],
        np.array([0.0, 0.2, 0.4]),
    )


def test_set_planner_debug_is_visible_in_snapshot(env):
    env.reset(
        initial_state=make_initial_state(),
        goal=make_goal(),
        obstacles=(),
    )

    trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0, 2.0],
            [0.4, 0.1, 0.05, 2.0],
        ],
        dtype=np.float64,
    )

    env.set_planner_debug(
        planned_trajectory=trajectory,
        debug={
            "cost": 12.5,
            "iterations": 8,
        },
    )

    snapshot = env.get_snapshot()

    assert snapshot.planned_trajectory is not None

    assert np.allclose(
        snapshot.planned_trajectory,
        trajectory,
    )

    assert snapshot.debug["cost"] == pytest.approx(12.5)
    assert snapshot.debug["iterations"] == 8


def test_environment_requires_reset(
    vehicle_config,
    environment_config,
):
    uninitialized_env = DrivingEnv(
        vehicle_config=vehicle_config,
        environment_config=environment_config,
    )

    with pytest.raises(RuntimeError):
        uninitialized_env.get_observation()

    with pytest.raises(RuntimeError):
        uninitialized_env.get_snapshot()

    with pytest.raises(RuntimeError):
        uninitialized_env.step(
            VehicleControl(
                acceleration=0.0,
                steering=0.0,
            )
        )


def test_invalid_environment_config():
    with pytest.raises(ValueError):
        DrivingEnv(
            vehicle_config=VehicleConfig(),
            environment_config=EnvironmentConfig(
                dt=0.0,
                max_time=10.0,
            ),
        )