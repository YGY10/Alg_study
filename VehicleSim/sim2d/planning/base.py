from __future__ import annotations

from abc import ABC, abstractmethod

from sim2d.perception import PlanningInput
from sim2d.types import PlanResult


class Planner(ABC):
    """所有规划器的统一接口。"""

    @abstractmethod
    def reset(self) -> None:
        """场景复位时清理规划器内部状态。"""
        raise NotImplementedError

    @abstractmethod
    def plan(self, planning_input: PlanningInput) -> PlanResult:
        """
        根据地图先验和局部感知生成控制量与规划轨迹。

        PlanningInput 保留旧 Observation 的 time/frame/ego/obstacles/goal
        字段，因此现有规划器可渐进迁移。
        """
        raise NotImplementedError
