from __future__ import annotations

import math

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QGraphicsSimpleTextItem

from sim2d.gui.perception_viewer import PerceptionGraphicsView

_INSTALLED = False


def install() -> None:
    """在感知窗口中显示每个目标的实时车辆坐标。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_draw_objects = PerceptionGraphicsView._draw_objects

    def draw_objects(self: PerceptionGraphicsView, snapshot) -> None:
        original_draw_objects(self, snapshot)

        for obj in snapshot.objects:
            center = self._scene_point(obj.x, obj.y)
            distance = math.hypot(obj.x, obj.y)
            label = QGraphicsSimpleTextItem(
                f"{obj.object_id}  x={obj.x:.2f} m  "
                f"y={obj.y:.2f} m  r={distance:.2f} m"
            )
            label.setBrush(QBrush(QColor("#ffd6b8")))
            label.setPos(center.x() + 8.0, center.y() + 20.0)
            self.graphics_scene.addItem(label)

    PerceptionGraphicsView._draw_objects = draw_objects
    _INSTALLED = True


__all__ = ["install"]
