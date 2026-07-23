from __future__ import annotations

import math
from typing import Any

from sim2d.gui.main_window import MainWindow

_INSTALLED = False


def _number(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def install() -> None:
    """在每周期日志末尾追加规划速度、走廊和代价诊断。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_advance = MainWindow.advance_one_step

    def advance_with_planning_diagnostics(self: MainWindow) -> None:
        original_advance(self)
        try:
            snapshot = self.env.get_snapshot()
            debug = dict(snapshot.debug)
            optimization_debug = dict(debug.get("optimization_debug", {}))
            diagnostics = dict(
                optimization_debug.get("trajectory_diagnostics", {})
            )
            cost_terms = dict(debug.get("optimization_cost_terms", {}))
            if not diagnostics:
                return

            current_speed = float(snapshot.ego.speed)
            self.append_log(
                "OPT_SPEED "
                f"current={current_speed:.3f}m/s "
                f"planned_min={_number(diagnostics.get('planned_speed_min')):.3f}m/s "
                f"planned_max={_number(diagnostics.get('planned_speed_max')):.3f}m/s "
                f"planned_terminal={_number(diagnostics.get('planned_speed_terminal')):.3f}m/s "
                f"curve_target_min={_number(diagnostics.get('curve_target_speed_min')):.3f}m/s"
            )
            self.append_log(
                "OPT_CORRIDOR "
                f"reference={debug.get('pnc_reference_id')} "
                f"width_min={_number(diagnostics.get('corridor_width_min')):.3f}m "
                f"width_mean={_number(diagnostics.get('corridor_width_mean')):.3f}m "
                f"width_max={_number(diagnostics.get('corridor_width_max')):.3f}m "
                f"max_violation={_number(diagnostics.get('max_footprint_violation'), 0.0):.3f}m "
                f"mean_violation={_number(diagnostics.get('mean_footprint_violation'), 0.0):.3f}m "
                f"violating_states={int(diagnostics.get('violating_state_count', 0))} "
                f"violating_points={int(diagnostics.get('violating_footprint_count', 0))}"
            )
            ordered_terms = (
                "reference",
                "heading",
                "speed",
                "corridor",
                "frenet_progress",
                "lateral_acceleration",
                "steering",
                "steering_rate",
                "acceleration",
                "acceleration_rate",
                "collision",
                "terminal",
            )
            terms = " ".join(
                f"{name}={_number(cost_terms.get(name), 0.0):.3f}"
                for name in ordered_terms
            )
            self.append_log(f"OPT_COST {terms}")
        except Exception as error:
            self.append_log(
                "OPT_DEBUG_ERROR "
                f"{type(error).__name__}: {error}"
            )

    MainWindow.advance_one_step = advance_with_planning_diagnostics
    _INSTALLED = True


__all__ = ["install"]
