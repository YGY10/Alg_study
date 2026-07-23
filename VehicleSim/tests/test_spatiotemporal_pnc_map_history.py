from __future__ import annotations

import numpy as np

from sim2d.perception import PerceivedLaneLine
from sim2d.planning.spatiotemporal_planner import PNCMap, build_reference_lines
from sim2d.types import VehicleState


def _line(line_id: str, points: np.ndarray) -> PerceivedLaneLine:
    return PerceivedLaneLine(
        line_id=line_id,
        points=np.asarray(points, dtype=np.float64),
        confidence=1.0,
    )


def test_right_angle_lines_keep_turn_direction_when_local_x_is_nearly_constant() -> None:
    left = _line(
        "left",
        np.array(
            [
                [-5.0, 1.8],
                [5.0, 1.8],
                [8.0, 1.0],
                [9.0, -1.0],
                [9.0, -8.0],
            ]
        ),
    )
    right = _line(
        "right",
        np.array(
            [
                [-5.0, -1.8],
                [4.0, -1.8],
                [5.5, -2.5],
                [5.5, -8.0],
            ]
        ),
    )

    references = build_reference_lines(
        (left, right),
        minimum_lane_width=2.0,
        maximum_lane_width=6.0,
        output_point_count=81,
    )

    assert len(references) == 1
    path = references[0].reference_path
    assert path[-1, 1] < -6.0
    assert np.max(np.abs(path[:, 4])) > 1e-3


def test_lane_width_is_measured_along_local_normal_on_right_angle_exit() -> None:
    # 弯后两条边界的 y 很接近变化方向，真实宽度主要体现在 x 方向。
    left = _line(
        "left",
        np.array([[0.0, 1.8], [7.0, 1.8], [9.0, 0.0], [9.0, -10.0]]),
    )
    right = _line(
        "right",
        np.array([[0.0, -1.8], [5.4, -1.8], [5.4, -10.0]]),
    )

    references = build_reference_lines(
        (left, right),
        minimum_lane_width=2.0,
        maximum_lane_width=6.0,
        output_point_count=61,
    )

    assert len(references) == 1
    reference = references[0]
    widths = np.linalg.norm(
        reference.left_boundary - reference.right_boundary,
        axis=1,
    )
    assert float(np.median(widths)) > 2.0
    assert float(np.median(widths)) < 6.0


def test_history_keeps_logical_lane_when_ego_drifts_across_boundary() -> None:
    pnc_map = PNCMap(
        switch_confirm_frames=4,
        history_retention_cost=1.5,
        history_hard_limit=4.0,
    )
    x = np.linspace(-5.0, 30.0, 71)

    first_lines = (
        _line("left", np.column_stack((x, np.full_like(x, 1.8)))),
        _line("middle", np.column_stack((x, np.full_like(x, -1.8)))),
        _line("right", np.column_stack((x, np.full_like(x, -5.4)))),
    )
    first = pnc_map.update(
        first_lines,
        world_origin=VehicleState(0.0, 0.0, 0.0, 2.0),
    )
    assert first.selected is not None
    first_world_center_y = float(
        np.median(first.selected.reference_path[:, 1])
    )
    assert abs(first_world_center_y) < 0.2

    # 自车向右漂移 2 m。瞬时几何上右侧车道已经包围局部原点，但历史中的
    # 逻辑本车道在世界坐标仍是 y=0，PNC Map 应继续保持它。
    second_lines = (
        _line("left2", np.column_stack((x, np.full_like(x, 3.8)))),
        _line("middle2", np.column_stack((x, np.full_like(x, 0.2)))),
        _line("right2", np.column_stack((x, np.full_like(x, -3.4)))),
    )
    second = pnc_map.update(
        second_lines,
        world_origin=VehicleState(0.0, -2.0, 0.0, 2.0),
    )

    assert second.selected is not None
    assert second.history_used is True
    selected_world_y = float(
        np.median(second.selected.reference_path[:, 1]) - 2.0
    )
    assert abs(selected_world_y) < 0.3
    assert second.reference_changed is False


def test_reset_clears_reference_history() -> None:
    pnc_map = PNCMap()
    x = np.linspace(-2.0, 10.0, 25)
    lines = (
        _line("left", np.column_stack((x, np.full_like(x, 1.8)))),
        _line("right", np.column_stack((x, np.full_like(x, -1.8)))),
    )
    pnc_map.update(
        lines,
        world_origin=VehicleState(0.0, 0.0, 0.0, 0.0),
    )
    pnc_map.reset()
    result = pnc_map.update(
        lines,
        world_origin=VehicleState(0.0, 0.0, 0.0, 0.0),
    )
    assert result.history_used is False
    assert result.switch_pending_frames == 0
