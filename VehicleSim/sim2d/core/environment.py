from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim2d.core.collision import (
    has_collision,
    minimum_clearance,
)
from sim2d.dynamics.base import (
    DynamicsBackend,
)
from sim2d.dynamics.kinematic_bicycle import (
    KinematicBicycleBackend,
)
from sim2d.types import (
    GoalState,
    Observation,
    Obstacle,
    SimulationSnapshot,
    StepResult,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    仿真环境配置。

    dt:
        固定仿真步长，单位秒。

    max_time:
        单个 episode 最大仿真时间，单位秒。

    collision_reward:
        碰撞时奖励。

    goal_reward:
        到达目标时奖励。

    step_reward:
        每执行一步的基础奖励。

    progress_reward_weight:
        接近目标时的奖励权重。
    """

    dt: float = 0.05
    max_time: float = 20.0

    collision_reward: float = -100.0
    goal_reward: float = 100.0
    step_reward: float = -0.01
    progress_reward_weight: float = 1.0

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        if self.max_time <= 0.0:
            raise ValueError(f"max_time must be positive, got {self.max_time}")


@dataclass
class DrivingEnv:
    """
    二维自动驾驶仿真环境。

    当前包含：
        固定时间步
        运动学自行车模型
        静态障碍物
        碰撞终止
        目标到达终止
        超时截断
        状态历史
        GUI 快照
        规划预测轨迹
        完整参考路径

    注意：
        环境核心不依赖 PySide6 或任何 GUI 库。

        planned_trajectory 和 reference_path
        只用于显示、调试和记录。

        环境不会使用它们推进车辆。
    """

    vehicle_config: VehicleConfig
    environment_config: EnvironmentConfig

    dynamics_backend: DynamicsBackend | None = field(
        default=None,
        repr=False,
    )

    _dynamics: DynamicsBackend = field(
        init=False,
        repr=False,
    )

    _ego_state: VehicleState | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _initial_state: VehicleState | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _goal: GoalState | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _obstacles: tuple[Obstacle, ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    _time: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    _frame: int = field(
        default=0,
        init=False,
        repr=False,
    )

    _terminated: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    _truncated: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    _collision: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    _goal_reached: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    _timeout: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    _min_clearance: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _history: list[VehicleState] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    _planned_trajectory: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _reference_path: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _debug: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.vehicle_config.validate()
        self.environment_config.validate()

        if self.dynamics_backend is None:
            self._dynamics = KinematicBicycleBackend(
                vehicle_config=self.vehicle_config,
            )
        else:
            self._dynamics = self.dynamics_backend

    def reset(
        self,
        initial_state: VehicleState,
        goal: GoalState,
        obstacles: tuple[Obstacle, ...] = (),
    ) -> Observation:
        """
        重置仿真环境。

        参数：
            initial_state:
                自车初始状态。

            goal:
                目标状态及允许误差。

            obstacles:
                当前场景中的静态障碍物。

        返回：
            初始 Observation。
        """
        goal.validate()

        for obstacle in obstacles:
            obstacle.validate()

        reset_state = self._dynamics.reset(initial_state)

        self._initial_state = reset_state
        self._ego_state = reset_state
        self._goal = goal
        self._obstacles = tuple(obstacles)

        self._time = 0.0
        self._frame = 0

        self._terminated = False
        self._truncated = False

        self._collision = has_collision(
            state=reset_state,
            config=self.vehicle_config,
            obstacles=self._obstacles,
        )

        self._goal_reached = self._check_goal_reached(reset_state)

        self._timeout = False

        self._min_clearance = minimum_clearance(
            state=reset_state,
            config=self.vehicle_config,
            obstacles=self._obstacles,
        )

        self._history = [reset_state]

        self._planned_trajectory = None
        self._reference_path = None
        self._debug = {}

        if self._collision:
            self._terminated = True

        if self._goal_reached:
            self._terminated = True

        return self.get_observation()

    def step(
        self,
        control: VehicleControl,
    ) -> StepResult:
        """
        仿真推进一个固定时间步。

        下一状态仅由车辆动力学模型决定，
        不直接使用规划轨迹或参考路径。
        """
        self._require_initialized()

        if self._terminated or self._truncated:
            raise RuntimeError(
                "Cannot call step() after episode has ended. "
                "Call reset() before stepping again."
            )

        assert self._ego_state is not None

        previous_distance = self._distance_to_goal(self._ego_state)

        next_state = self._dynamics.step(
            control=control,
            dt=self.environment_config.dt,
        )

        self._ego_state = next_state

        self._time += self.environment_config.dt

        self._frame += 1

        self._history.append(next_state)

        self._collision = has_collision(
            state=next_state,
            config=self.vehicle_config,
            obstacles=self._obstacles,
        )

        self._min_clearance = minimum_clearance(
            state=next_state,
            config=self.vehicle_config,
            obstacles=self._obstacles,
        )

        self._goal_reached = self._check_goal_reached(next_state)

        self._timeout = self._time >= self.environment_config.max_time

        self._terminated = self._collision or self._goal_reached

        self._truncated = self._timeout and not self._terminated

        current_distance = self._distance_to_goal(next_state)

        reward = self._compute_reward(
            previous_distance=previous_distance,
            current_distance=current_distance,
        )

        info = {
            "collision": self._collision,
            "goal_reached": self._goal_reached,
            "timeout": self._timeout,
            "min_clearance": self._min_clearance,
            "frame": self._frame,
            "time": self._time,
            "control": control,
        }

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def get_observation(self) -> Observation:
        """
        获取提供给规划器的当前观测。
        """
        self._require_initialized()

        assert self._ego_state is not None
        assert self._goal is not None

        return Observation(
            time=self._time,
            frame=self._frame,
            ego=self._ego_state,
            obstacles=self._obstacles,
            goal=self._goal,
        )

    def get_snapshot(self) -> SimulationSnapshot:
        """
        获取提供给 GUI 的只读快照。
        """
        self._require_initialized()

        assert self._ego_state is not None

        return SimulationSnapshot(
            frame=self._frame,
            time=self._time,
            ego=self._ego_state,
            obstacles=self._obstacles,
            planned_trajectory=(self._copy_optional_array(self._planned_trajectory)),
            reference_path=(self._copy_optional_array(self._reference_path)),
            history_trajectory=(self._history_as_array()),
            collision=self._collision,
            min_clearance=self._min_clearance,
            debug=dict(self._debug),
        )

    def set_planner_debug(
        self,
        planned_trajectory: np.ndarray | None,
        reference_path: np.ndarray | None = None,
        debug: dict[str, Any] | None = None,
    ) -> None:
        """
        保存规划器输出，供 GUI 显示和日志调试。

        planned_trajectory:
            当前规划周期的闭环预测状态轨迹。
            至少应包含 x、y 两列。

        reference_path:
            规划器生成的完整参考路径。
            至少应包含 x、y 两列。

        debug:
            规划器调试信息。

        注意：
            环境不会使用 planned_trajectory 或
            reference_path 推进车辆。
        """
        self._planned_trajectory = self._validate_and_copy_path_array(
            value=planned_trajectory,
            name="planned_trajectory",
        )

        self._reference_path = self._validate_and_copy_path_array(
            value=reference_path,
            name="reference_path",
        )

        self._debug = {} if debug is None else dict(debug)

    @property
    def is_done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def time(self) -> float:
        return self._time

    @property
    def state(self) -> VehicleState:
        self._require_initialized()

        assert self._ego_state is not None

        return self._ego_state

    def _compute_reward(
        self,
        previous_distance: float,
        current_distance: float,
    ) -> float:
        """
        第一版简单奖励函数。

        奖励组成：
            每步基础惩罚
            接近目标的进度奖励
            碰撞惩罚
            到达目标奖励
        """
        reward = self.environment_config.step_reward

        progress = previous_distance - current_distance

        reward += self.environment_config.progress_reward_weight * progress

        if self._collision:
            reward += self.environment_config.collision_reward

        if self._goal_reached:
            reward += self.environment_config.goal_reward

        return float(reward)

    def _check_goal_reached(
        self,
        state: VehicleState,
    ) -> bool:
        assert self._goal is not None

        target = self._goal.state

        position_error = math.hypot(
            state.x - target.x,
            state.y - target.y,
        )

        yaw_error = abs(self._normalize_angle(state.yaw - target.yaw))

        speed_error = abs(state.speed - target.speed)

        return (
            position_error <= self._goal.position_tolerance
            and yaw_error <= self._goal.yaw_tolerance
            and speed_error <= self._goal.speed_tolerance
        )

    def _distance_to_goal(
        self,
        state: VehicleState,
    ) -> float:
        assert self._goal is not None

        target = self._goal.state

        return math.hypot(
            state.x - target.x,
            state.y - target.y,
        )

    def _history_as_array(
        self,
    ) -> np.ndarray:
        return np.array(
            [
                [
                    state.x,
                    state.y,
                    state.yaw,
                    state.speed,
                ]
                for state in self._history
            ],
            dtype=np.float64,
        )

    def _require_initialized(
        self,
    ) -> None:
        if self._ego_state is None or self._goal is None:
            raise RuntimeError("Environment has not been reset. " "Call reset() first.")

    @staticmethod
    def _validate_and_copy_path_array(
        *,
        value: np.ndarray | None,
        name: str,
    ) -> np.ndarray | None:
        """
        校验并复制轨迹或参考路径数组。

        要求：
            二维数组
            至少包含 x、y 两列
            所有元素必须是有限数值
        """
        if value is None:
            return None

        array = np.asarray(
            value,
            dtype=np.float64,
        )

        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, " f"got ndim={array.ndim}")

        if array.shape[1] < 2:
            raise ValueError(f"{name} must contain at least " "x and y columns")

        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only " "finite values")

        return array.copy()

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _copy_optional_array(
        value: np.ndarray | None,
    ) -> np.ndarray | None:
        if value is None:
            return None

        return value.copy()
