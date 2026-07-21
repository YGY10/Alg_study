from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem

from sim2d.gui.graphics_items import PIXELS_PER_METER
from sim2d.gui.simulation_view import SimulationView

_INSTALLED = False


def install() -> None:
    """优化道路对比模式：世界道路完整显示，地图层只显示中心线。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_init = SimulationView.__init__
    original_set_mode = SimulationView.set_road_layer_mode
    original_render = SimulationView.render_road_network
    original_set_world = SimulationView.set_world_road_lanes

    def init(self: SimulationView, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        self.map_compare_centerline_items: list[QGraphicsPathItem] = []

    def set_mode(self: SimulationView, mode: str) -> None:
        original_set_mode(self, mode)
        _refine_visibility(self)

    def render(self: SimulationView, road_network) -> None:
        original_render(self, road_network)
        _rebuild_compare_centerlines(self, road_network)
        _refine_visibility(self)

    def set_world(self: SimulationView, lanes) -> None:
        original_set_world(self, lanes)
        _refine_visibility(self)

    SimulationView.__init__ = init
    SimulationView.set_road_layer_mode = set_mode
    SimulationView.render_road_network = render
    SimulationView.set_world_road_lanes = set_world
    _INSTALLED = True


def _clear_compare_centerlines(view: SimulationView) -> None:
    for item in getattr(view, "map_compare_centerline_items", ()):
        if item.scene() is view.graphics_scene:
            view.graphics_scene.removeItem(item)

    view.map_compare_centerline_items = []


def _rebuild_compare_centerlines(
    view: SimulationView,
    road_network,
) -> None:
    _clear_compare_centerlines(view)

    path = QPainterPath()

    for lane in road_network.lanes:
        points = lane.centerline.points
        if points.shape[0] < 2:
            continue

        path.moveTo(
            float(points[0, 0]) * PIXELS_PER_METER,
            -float(points[0, 1]) * PIXELS_PER_METER,
        )

        for point in points[1:]:
            path.lineTo(
                float(point[0]) * PIXELS_PER_METER,
                -float(point[1]) * PIXELS_PER_METER,
            )

    if path.isEmpty():
        return

    item = QGraphicsPathItem(path)
    pen = QPen(
        QColor("#78c7ff"),
        1.8,
        Qt.PenStyle.DashLine,
    )
    pen.setCosmetic(True)
    item.setPen(pen)
    item.setZValue(-60.0)
    item.setOpacity(0.58)
    item.setVisible(False)

    view.graphics_scene.addItem(item)
    view.map_compare_centerline_items.append(item)


def _refine_visibility(view: SimulationView) -> None:
    mode = getattr(view, "road_layer_mode", "world")

    for item in getattr(view, "map_road_items", ()):  # type: QGraphicsItem
        if mode == "map":
            item.setVisible(True)
            item.setOpacity(1.0)
        else:
            item.setVisible(False)
            item.setOpacity(1.0)

    for item in getattr(view, "map_compare_centerline_items", ()):
        item.setVisible(mode == "compare")
        item.setOpacity(0.58)

    for item in getattr(view, "world_road_items", ()):  # type: QGraphicsItem
        item.setVisible(mode in {"world", "compare"})
        item.setOpacity(1.0)
