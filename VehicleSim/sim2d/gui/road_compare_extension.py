from __future__ import annotations

from PySide6.QtWidgets import QGraphicsItem

from sim2d.gui.simulation_view import SimulationView

_INSTALLED = False


def install() -> None:
    """优化道路对比模式：世界道路完整显示，地图层只显示关键线框。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_set_mode = SimulationView.set_road_layer_mode
    original_render = SimulationView.render_road_network
    original_set_world = SimulationView.set_world_road_lanes

    def set_mode(self: SimulationView, mode: str) -> None:
        original_set_mode(self, mode)
        _refine_visibility(self)

    def render(self: SimulationView, road_network) -> None:
        original_render(self, road_network)
        _refine_visibility(self)

    def set_world(self: SimulationView, lanes) -> None:
        original_set_world(self, lanes)
        _refine_visibility(self)

    SimulationView.set_road_layer_mode = set_mode
    SimulationView.render_road_network = render
    SimulationView.set_world_road_lanes = set_world
    _INSTALLED = True


def _refine_visibility(view: SimulationView) -> None:
    mode = getattr(view, "road_layer_mode", "world")

    for item in getattr(view, "map_road_items", ()):  # type: QGraphicsItem
        if mode == "world":
            item.setVisible(False)
            item.setOpacity(1.0)
            continue

        if mode == "map":
            item.setVisible(True)
            item.setOpacity(1.0)
            continue

        # 对比模式下不重复填充地图路面，只保留边界、中心线等线框。
        is_surface = item.zValue() <= -90.0
        item.setVisible(not is_surface)
        item.setOpacity(0.34)

    for item in getattr(view, "world_road_items", ()):  # type: QGraphicsItem
        item.setVisible(mode in {"world", "compare"})
        item.setOpacity(1.0)
