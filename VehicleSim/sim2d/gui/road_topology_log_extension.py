from __future__ import annotations

from sim2d.gui.main_window import MainWindow
from sim2d.gui import road_layer_extension

_INSTALLED = False


def install() -> None:
    """在道路世界层重建后输出可核验的拓扑统计。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_apply = road_layer_extension._apply_world_map_geometry

    def apply_with_topology_log(window: MainWindow) -> None:
        original_apply(window)
        _append_topology_log(window)

    road_layer_extension._apply_world_map_geometry = apply_with_topology_log
    _INSTALLED = True


def _append_topology_log(window: MainWindow) -> None:
    if window.road_network is None:
        return

    if not window.env.is_initialized:
        return

    map_lane_ids = {
        lane.lane_id
        for lane in window.road_network.lanes
    }

    original_edges = {
        (lane.lane_id, successor_id)
        for lane in window.road_network.lanes
        for successor_id in lane.successor_ids
        if successor_id in map_lane_ids
        and successor_id != lane.lane_id
    }

    world_lanes = window.env.world.state.road_lanes
    world_lane_ids = {
        lane.map_lane_id
        for lane in world_lanes
    }

    final_edges = {
        (lane.map_lane_id, successor_id)
        for lane in world_lanes
        for successor_id in lane.successor_ids
        if successor_id in world_lane_ids
        and successor_id != lane.map_lane_id
    }

    inferred_edges = final_edges - original_edges
    unmatched_lane_ends = sum(
        1
        for lane in world_lanes
        if not lane.successor_ids
    )

    window.append_log(
        "ROAD_TOPOLOGY_REPAIR "
        f"original_edges={len(original_edges)} "
        f"inferred_edges={len(inferred_edges)} "
        f"final_edges={len(final_edges)} "
        f"unmatched_lane_ends={unmatched_lane_ends}"
    )
