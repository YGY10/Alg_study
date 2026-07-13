from __future__ import annotations

from abc import ABC, abstractmethod

from sim2d.types import Observation, PlanResult


class Planner(ABC):
    """所有规划器的统一接口。"""

    @abstractmethod
    def reset(self) -> None:
        """场景复位时清理规划器内部状态。"""
        raise NotImplementedError

    @abstractmethod
    def plan(self, observation: Observation) -> PlanResult:
        """根据当前观测生成控制量和规划轨迹。"""
        raise NotImplementedError
