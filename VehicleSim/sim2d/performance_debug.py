from __future__ import annotations

import math
import textwrap
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QPlainTextEdit

from sim2d.gui.main_window import MainWindow
from sim2d.perception import ground_truth as ground_truth_module
from sim2d.perception.ground_truth import GroundTruthLocalPerception
from sim2d.planning.spatiotemporal_planner.cost import SpatiotemporalCost
from sim2d.planning.spatiotemporal_planner.optimizer import SpatiotemporalOptimizer
from sim2d.planning.spatiotemporal_planner.pnc_map import PNCMap
from sim2d.planning.spatiotemporal_planner.rollout import TrajectoryRollout
from sim2d.planning.spatiotemporal_planner.spatiotemporal_planner import (
    SpatiotemporalPlanner,
)

_INSTALLED = False
_ACTIVE_OPTIMIZER: SpatiotemporalOptimizer | None = None
_LAST_LANE_SCAN_MS = 0.0
_LOG_WRAP_WIDTH = 105


class _LogConsoleDeleteFilter(QObject):
    """允许只读日志框通过 Delete/Backspace 删除当前选中文本。"""

    def eventFilter(self, watched, event) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
        ):
            cursor = watched.textCursor()
            if cursor.hasSelection():
                cursor.removeSelectedText()
                watched.setTextCursor(cursor)
                return True
        return super().eventFilter(watched, event)


def _milliseconds(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def install() -> None:
    """安装同步耗时和 PNC Frenet 连续性调试，不改变规划计算结果。"""
    global _INSTALLED
    if _INSTALLED:
        return

    _install_log_console_behavior()
    _install_perception_timing()
    _install_optimizer_timing()
    _install_pnc_debug_capture()
    _install_planner_timing()
    _install_cycle_timing()
    _INSTALLED = True


def _install_log_console_behavior() -> None:
    original_build_ui = MainWindow._build_ui
    original_append_log = MainWindow.append_log

    def build_ui(self: MainWindow) -> None:
        original_build_ui(self)
        self.log_console.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        delete_filter = _LogConsoleDeleteFilter(self.log_console)
        self.log_console.installEventFilter(delete_filter)
        self._log_console_delete_filter = delete_filter
        self.log_console.setToolTip(
            "Ctrl+A 全选，Ctrl+C 复制；选中后按 Delete 或 Backspace 删除"
        )

    def append_log(self: MainWindow, message: str) -> None:
        formatted_lines: list[str] = []
        for paragraph in str(message).splitlines() or [""]:
            leading_length = len(paragraph) - len(paragraph.lstrip(" "))
            leading = paragraph[:leading_length]
            body = paragraph[leading_length:]
            if not body:
                formatted_lines.append(leading)
                continue
            available_width = max(_LOG_WRAP_WIDTH - len(leading), 20)
            wrapped = textwrap.wrap(
                body,
                width=available_width,
                initial_indent=leading,
                subsequent_indent=leading + "  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            formatted_lines.extend(wrapped or [leading])
        original_append_log(self, "\n".join(formatted_lines))

    MainWindow._build_ui = build_ui
    MainWindow.append_log = append_log


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


def _install_pnc_debug_capture() -> None:
    """保存 PNCMap.update 的完整返回值，供 planner 包装器写入日志。"""
    original_update = PNCMap.update

    def debug_update(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        self._performance_debug_last_update = result
        return result

    PNCMap.update = debug_update


def _lane_line_summary(line) -> dict[str, Any]:
    points = np.asarray(line.points, dtype=np.float64)
    if points.shape[0] < 2:
        length = 0.0
    else:
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    distances = np.linalg.norm(points, axis=1)
    nearest_index = int(np.argmin(distances)) if points.shape[0] else -1
    nearest = (
        (float(points[nearest_index, 0]), float(points[nearest_index, 1]))
        if nearest_index >= 0
        else (math.nan, math.nan)
    )
    start = (
        (float(points[0, 0]), float(points[0, 1]))
        if points.shape[0]
        else (math.nan, math.nan)
    )
    end = (
        (float(points[-1, 0]), float(points[-1, 1]))
        if points.shape[0]
        else (math.nan, math.nan)
    )
    return {
        "id": str(line.line_id),
        "points": int(points.shape[0]),
        "length": length,
        "start": start,
        "end": end,
        "nearest_index": nearest_index,
        "nearest": nearest,
        "confidence": float(line.confidence),
    }


def _reference_summary(reference_path: np.ndarray, ego) -> dict[str, Any]:
    path = np.asarray(reference_path, dtype=np.float64)
    if path.shape[0] == 0:
        return {"points": 0}
    ego_xy = np.array([ego.x, ego.y], dtype=np.float64)
    nearest_index = int(np.argmin(np.linalg.norm(path[:, :2] - ego_xy, axis=1)))
    length = float(path[-1, 3] - path[0, 3]) if path.shape[1] >= 4 else 0.0
    return {
        "points": int(path.shape[0]),
        "length": length,
        "start": (float(path[0, 0]), float(path[0, 1])),
        "end": (float(path[-1, 0]), float(path[-1, 1])),
        "nearest_index": nearest_index,
        "nearest": (
            float(path[nearest_index, 0]),
            float(path[nearest_index, 1]),
        ),
        "nearest_yaw": float(path[nearest_index, 2]),
        "nearest_curvature": float(path[nearest_index, 4]),
    }


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
                total_ms - float(optimizer_timing.get("optimizer_total_ms", 0.0)),
                0.0,
            ),
            "perception_total": float(
                perception_timing.get("perception_total", 0.0)
            ),
            "lane_scan": float(perception_timing.get("lane_scan", 0.0)),
        }

        update = getattr(self.pnc_map, "_performance_debug_last_update", None)
        if update is not None:
            debug["pnc_candidate_costs"] = tuple(update.candidate_costs)
            debug["pnc_candidate_continuity"] = tuple(update.candidate_continuity)
            debug["pnc_selected_orientation"] = update.selected_orientation
        debug["pnc_reference_geometry"] = _reference_summary(
            result.reference_path,
            planning_input.ego,
        )
        debug["pnc_lane_line_summaries"] = tuple(
            _lane_line_summary(line)
            for line in planning_input.perception.lane_lines
        )
        return replace(result, debug=debug)

    SpatiotemporalPlanner.plan = timed_plan


def _format_point(point: tuple[float, float]) -> str:
    return f"({point[0]:.2f},{point[1]:.2f})"


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
            perception_other_ms = max(perception_ms - lane_scan_ms, 0.0)
            optimizer_ms = float(timing.get("optimizer_total", 0.0))
            planner_other_ms = max(planner_ms - optimizer_ms, 0.0)
            gradient_ms = float(
                optimizer_timing.get("finite_difference_gradient_ms", 0.0)
            )
            line_search_ms = float(optimizer_timing.get("line_search_ms", 0.0))
            rollout_ms = float(optimizer_timing.get("rollout_ms", 0.0))
            cost_ms = float(optimizer_timing.get("cost_ms", 0.0))
            evaluate_count = int(optimizer_timing.get("cost_count", 0))
            cycle_other_ms = max(
                cycle_total_ms - perception_ms - planner_ms,
                0.0,
            )

            self.append_log(
                "PERF "
                f"cycle={cycle_total_ms:.2f}ms\n"
                "  perception "
                f"total={perception_ms:.2f}ms "
                f"lane_scan={lane_scan_ms:.2f}ms "
                f"other={perception_other_ms:.2f}ms\n"
                "  planner "
                f"total={planner_ms:.2f}ms "
                f"optimizer={optimizer_ms:.2f}ms "
                f"other={planner_other_ms:.2f}ms\n"
                "  optimizer "
                f"gradient={gradient_ms:.2f}ms "
                f"line_search={line_search_ms:.2f}ms "
                f"rollout={rollout_ms:.2f}ms "
                f"cost={cost_ms:.2f}ms "
                f"evals={evaluate_count}\n"
                "  cycle_other "
                f"env_snapshot_render_log={cycle_other_ms:.2f}ms"
            )

            geometry = debug.get("pnc_reference_geometry", {})
            continuity = debug.get("pnc_candidate_continuity", ())
            lane_summaries = debug.get("pnc_lane_line_summaries", ())
            if geometry:
                self.append_log(
                    "PNC_DEBUG "
                    f"selected={debug.get('pnc_reference_id')} "
                    f"orientation={debug.get('pnc_selected_orientation')} "
                    f"history={debug.get('pnc_history_used')} "
                    f"changed={debug.get('reference_changed')} "
                    f"pending={debug.get('pnc_switch_pending_frames')} "
                    f"continuity={debug.get('pnc_continuity_cost')}\n"
                    "  reference "
                    f"points={geometry.get('points')} "
                    f"length={float(geometry.get('length', 0.0)):.2f} "
                    f"start={_format_point(geometry.get('start', (math.nan, math.nan)))} "
                    f"end={_format_point(geometry.get('end', (math.nan, math.nan)))} "
                    f"nearest_i={geometry.get('nearest_index')} "
                    f"nearest={_format_point(geometry.get('nearest', (math.nan, math.nan)))} "
                    f"yaw={float(geometry.get('nearest_yaw', math.nan)):.3f} "
                    f"kappa={float(geometry.get('nearest_curvature', math.nan)):.4f}"
                )

                if continuity:
                    for metrics in continuity:
                        self.append_log(
                            "  PNC_FRENET "
                            f"id={metrics.candidate_id} "
                            f"cost={metrics.cost:.3f} "
                            f"mean_abs_l={metrics.mean_abs_l:.3f}m "
                            f"mean_abs_dyaw={metrics.mean_abs_heading:.3f}rad "
                            f"mean_abs_dkappa={metrics.mean_abs_curvature:.4f} "
                            f"overlap={metrics.overlap_ratio:.3f} "
                            f"nonmono={metrics.nonmonotonic_ratio:.3f} "
                            f"s_span={metrics.projected_s_span:.2f}m "
                            f"samples={metrics.sample_count}"
                        )
                else:
                    self.append_log("  PNC_FRENET none")

                for line in lane_summaries:
                    self.append_log(
                        "  PNC_LANE "
                        f"id={line['id']} points={line['points']} "
                        f"length={line['length']:.2f} "
                        f"start={_format_point(line['start'])} "
                        f"end={_format_point(line['end'])} "
                        f"nearest_i={line['nearest_index']} "
                        f"nearest={_format_point(line['nearest'])} "
                        f"confidence={line['confidence']:.2f}"
                    )
        except Exception as error:
            self.append_log(
                "PERF_ERROR "
                f"cycle={cycle_total_ms:.2f}ms\n"
                f"  error={type(error).__name__}: {error}"
            )

    MainWindow.advance_one_step = timed_advance


__all__ = ["install"]
