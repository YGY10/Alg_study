from __future__ import annotations

from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtCore import Qt

from sim2d.gui.perception_viewer import PerceptionGraphicsView

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    def draw_lanes(self: PerceptionGraphicsView, snapshot) -> None:
        line_pen = QPen(QColor("#dce9f3"), 1.6)
        line_pen.setCosmetic(True)
        for line in snapshot.lane_lines:
            self._path_item(line.points, line_pen)

    def draw_status(self: PerceptionGraphicsView, snapshot) -> None:
        text = self.graphics_scene.addSimpleText(
            f"感知帧 {snapshot.frame}  "
            f"measurement={snapshot.measurement_time:.2f}s  "
            f"publish={snapshot.publish_time:.2f}s  "
            f"坐标系={snapshot.coordinate_frame}  "
            f"目标={len(snapshot.objects)}  "
            f"交通灯={len(snapshot.traffic_signals)}  "
            f"车道线={len(snapshot.lane_lines)}"
        )
        text.setBrush(QBrush(QColor("#ffffff")))
        rect = self.graphics_scene.sceneRect()
        text.setPos(rect.left() + 8.0, rect.top() + 8.0)

    PerceptionGraphicsView._draw_lanes = draw_lanes
    PerceptionGraphicsView._draw_status = draw_status
    _INSTALLED = True


__all__ = ["install"]
