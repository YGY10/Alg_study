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


def test_duplicate_partial_boundary_does_not_block_full_lane_pair() -> None:
    x = np.linspace(-5.0, 30.0, 71)
    left = _line("left", np.column_stack((x, np.full_like(x, 2.0))))
    right = _line("right", np.column_stack((x, np.full_like(x, -2.2))))

    # 与右边界很接近的短重复片段会插入横向排序，但不应阻断 left-right 配对。
    duplicate_x = np.linspace(8.0, 30.0, 45)
    duplicate = _line(
        "duplicate",
        np.column_stack((duplicate_x, np.full_like(duplicate_x, -2.0))),
    )

    references = build_reference_lines((left, duplicate, right))
    ids = {reference.reference_id for reference in references}

    assert "ref_left__right" in ids


def test_one_frame_invalid_candidates_coasts_on_history() -> None:
    pnc_map = PNCMap()
    x = np.linspace(-5.0, 30.0, 71)
    stable_lines = (
        _line("left", np.column_stack((x, np.full_like(x, 2.0)))),
        _line("right", np.column_stack((x, np.full_like(x, -2.0)))),
    )

    first = pnc_map.update(
        stable_lines,
        world_origin=VehicleState(0.0, 0.0, 0.0, 2.0),
    )
    assert first.selected is not None

    # 当前帧只有一条边界，无法构造 corridor，应短时沿用历史而不是跳到别处。
    second = pnc_map.update(
        (stable_lines[0],),
        world_origin=VehicleState(0.1, 0.0, 0.0, 2.0),
    )

    assert second.selected is not None
    assert second.selected.reference_id == "coast_history"
    assert second.history_used is True
    assert second.reference_changed is False


def test_new_candidate_requires_confirmation_even_when_history_cost_is_large() -> None:
    pnc_map = PNCMap(switch_confirm_frames=3)
    x = np.linspace(-5.0, 30.0, 71)
    first_lines = (
        _line("left0", np.column_stack((x, np.full_like(x, 2.0)))),
        _line("right0", np.column_stack((x, np.full_like(x, -2.0)))),
    )
    first = pnc_map.update(
        first_lines,
        world_origin=VehicleState(0.0, 0.0, 0.0, 2.0),
    )
    assert first.selected is not None

    shifted_lines = (
        _line("left1", np.column_stack((x, np.full_like(x, 8.0)))),
        _line("right1", np.column_stack((x, np.full_like(x, 4.0)))),
    )
    second = pnc_map.update(
        shifted_lines,
        world_origin=VehicleState(0.0, 6.0, 0.0, 2.0),
    )

    # 无论绝对代价多大，都不能在单帧绕过确认窗口直接切换。
    assert second.reference_changed is False
