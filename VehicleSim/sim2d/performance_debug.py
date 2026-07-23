from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from sim2d.gui.main_window import MainWindow
from sim2d.perception import ground_truth as ground_truth_module
from sim2d.perception.ground_truth import GroundTruthLocalPerception
from sim2d.planning.spatiotemporal_planner.cost import SpatiotemporalCost
from sim2d.planning.spatiotemporal_planner.optimizer import SpatiotemporalOptimizer
from sim2d.planning.spatiotemporal_planner.rollout import TrajectoryRollout
from sim2d.planning.spatiotemporal_planner.spatiotemporal_planner import (
    SpatiotemporalPlanner,
)

_INSTALLED = False
_ACTIVE_OPTIMIZER: SpatiotemporalOptimizer | None = None
_LAST_LANE_SCAN_MS = 0.0


def _milliseconds(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def install() -> None:
    """安装轻量级同步耗时采样，不改变规划和感知计算结果。"""
    global _INSTALLED
    if _INSTALLED:
        return

    _install_perception_timing()
    _install_optimizer_timing()
    _install_planner_timing()
    _install_cycle_timing()
    _INSTALLED = True


def _install_perception_timing() -> None:
    original_lane_scan = ground_truth_module.perceive_lane_corridors
    original_measure = GroundTruthLocalPerception._measure

    def timed_lane_scan(*args, **kwargs):
        global _LAST_LANE_SCAN_MS
        started = perf_counter()
        result = original_lane_scan(*args, **kwargs)
        _LAST_LANE_SCAN_MS = _milliseconds(started)
        return result

    def timed_measure(self, *args, **kwargs):
        global _LAST_LANE_SCAN_MS
        _LAST_LANE_SCAN_MS = 0.0
        started = perf_counter()
        snapshot = original_measure(self, *args, **kwargs)
        total_ms = _milliseconds(started)
        debug = dict(snapshot.debug)
        debug["timing_ms"] = {
            "perception_total": total_ms,
            "lane_scan": _LAST_LANE_SCAN_MS,
            "objects_signals_and_packaging": max(
                total_ms - _LAST_LANE_SCAN_MS,
                0.0,
            ),
        }
        return replace(snapshot, debug=debug)

    ground_truth_module.perceive_lane_corridors = timed_lane_scan
    GroundTruthLocalPerception._measure = timed_measure


def _install_optimizer_timing() -> None:
    original_optimize = SpatiotemporalOptimizer.optimize
    original_gradient = SpatiotemporalOptimizer._finite_difference_gradient
    original_line_search = SpatiotemporalOptimizer._line_search
    original_rollout = TrajectoryRollout.rollout_from_ego
    original_cost = SpatiotemporalCost.evaluate

    def timed_rollout(self, *args, **kwargs):
        started = perf_counter()
        result = original_rollout(self, *args, **kwargs)
        optimizer = _ACTIVE_OPTIMIZER
        if optimizer is not None:
            perf = optimizer._performance_timing
            perf["rollout_ms"] += _milliseconds(started)
            perf["rollout_count"] += 1
        return result

    def timed_cost(self, *args, **kwargs):
        started = perf_counter()
        result = original_cost(self, *args, **kwargs)
        optimizer = _ACTIVE_OPTIMIZER
        if optimizer is not None:
            perf = optimizer._performance_timing
            perf["cost_ms"] += _milliseconds(started)
            perf["cost_count"] += 1
        return result

    def timed_gradient(self, *args, **kwargs):
        started = perf_counter()
        result = original_gradient(self, *args, **kwargs)
        self._performance_timing["finite_difference_gradient_ms"] += _milliseconds(
            started
        )
        self._performance_timing["gradient_call_count"] += 1
        return result

    def timed_line_search(self, *args, **kwargs):
        started = perf_counter()
        result = original_line_search(self, *args, **kwargs)
        self._performance_timing["line_search_ms"] += _milliseconds(started)
        self._performance_timing["line_search_call_count"] += 1
        return result

    def timed_optimize(self, *args, **kwargs):
        global _ACTIVE_OPTIMIZER
        self._performance_timing = {
            "optimizer_total_ms": 0.0,
            "finite_difference_gradient_ms": 0.0,
            "line_search_ms": 0.0,
            "rollout_ms": 0.0,
            "cost_ms": 0.0,
            "rollout_count": 0,
            "cost_count": 0,
            "gradient_call_count": 0,
            "line_search_call_count": 0,
        }
        previous_active = _ACTIVE_OPTIMIZER
        _ACTIVE_OPTIMIZER = self
        started = perf_counter()
        try:
            result = original_optimize(self, *args, **kwargs)
        finally:
            total_ms = _milliseconds(started)
            _ACTIVE_OPTIMIZER = previous_active

        self._performance_timing["optimizer_total_ms"] = total_ms
        known_ms = (
            self._performance_timing["rollout_ms"]
            + self._performance_timing["cost_ms"]
        )
        self._performance_timing["optimizer_other_ms"] = max(
            total_ms - known_ms,
            0.0,
        )

        debug = dict(result.debug)
        debug["timing_ms"] = dict(self._performance_timing)
        return replace(result, debug=debug)

    TrajectoryRollout.rollout_from_ego = timed_rollout
    SpatiotemporalCost.evaluate = timed_cost
    SpatiotemporalOptimizer._finite_difference_gradient = timed_gradient
    SpatiotemporalOptimizer._line_search = timed_line_search
    SpatiotemporalOptimizer.optimize = timed_optimize


def _install_planner_timing() -> None:
    original_plan = SpatiotemporalPlanner.plan

    def timed_plan(self, planning_input):
        started = perf_counter()
        result = original_plan(self, planning_input)
        total_ms = _milliseconds(started)
        debug = dict(result.debug)
        optimizer_debug = debug.get("optimization_debug", {})
        optimizer_timing = optimizer_debug.get("timing_ms", {})
        perception_timing = planning_input.perception.debug.get("timing_ms", {})
        debug["timing_ms"] = {
            "planner_total": total_ms,
            "optimizer_total": float(
                optimizer_timing.get("optimizer_total_ms", 0.0)
            ),
            "planner_non_optimizer": max(
                total_ms
                - float(optimizer_timing.get("optimizer_total_ms", 0.0)),
                0.0,
            ),
            "perception_total": float(
                perception_timing.get("perception_total", 0.0)
            ),
            "lane_scan": float(perception_timing.get("lane_scan", 0.0)),
        }
        return replace(result, debug=debug)

    SpatiotemporalPlanner.plan = timed_plan


def _install_cycle_timing() -> None:
    original_advance = MainWindow.advance_one_step

    def timed_advance(self: MainWindow) -> None:
        started = perf_counter()
        original_advance(self)
        cycle_total_ms = _milliseconds(started)

        try:
            snapshot = self.env.get_snapshot()
            debug: dict[str, Any] = dict(snapshot.debug)
            timing = debug.get("timing_ms", {})
            optimizer_debug = debug.get("optimization_debug", {})
            optimizer_timing = optimizer_debug.get("timing_ms", {})

            planner_ms = float(timing.get("planner_total", 0.0))
            perception_ms = float(timing.get("perception_total", 0.0))
            lane_scan_ms = float(timing.get("lane_scan", 0.0))
            optimizer_ms = float(timing.get("optimizer_total", 0.0))
            gradient_ms = float(
                optimizer_timing.get("finite_difference_gradient_ms", 0.0)
            )
            line_search_ms = float(
                optimizer_timing.get("line_search_ms", 0.0)
            )
            rollout_ms = float(optimizer_timing.get("rollout_ms", 0.0))
            cost_ms = float(optimizer_timing.get("cost_ms", 0.0))
            evaluate_count = int(optimizer_timing.get("cost_count", 0))

            self.append_log(
                "PERF "
                f"cycle={cycle_total_ms:.2f}ms "
                f"perception={perception_ms:.2f}ms "
                f"lane_scan={lane_scan_ms:.2f}ms "
                f"planner={planner_ms:.2f}ms "
                f"optimizer={optimizer_ms:.2f}ms "
                f"gradient={gradient_ms:.2f}ms "
                f"line_search={line_search_ms:.2f}ms "
                f"rollout={rollout_ms:.2f}ms "
                f"cost={cost_ms:.2f}ms "
                f"evals={evaluate_count} "
                f"gui_env_other={max(cycle_total_ms - planner_ms, 0.0):.2f}ms"
            )
        except Exception as error:
            self.append_log(
                "PERF_ERROR "
                f"cycle={cycle_total_ms:.2f}ms "
                f"error={type(error).__name__}: {error}"
            )

    MainWindow.advance_one_step = timed_advance


__all__ = ["install"]
