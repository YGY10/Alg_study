from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsView

from sim2d.gui.perception_viewer import PerceptionGraphicsView

_INSTALLED = False


def install() -> None:
    """让感知视图支持滚轮缩放，并避免每帧自动重置缩放比例。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original_init = PerceptionGraphicsView.__init__

    def init(self: PerceptionGraphicsView, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self._perception_auto_fit_pending = True
        self._perception_last_snapshot = None

    def render_perception(self: PerceptionGraphicsView, snapshot) -> None:
        if snapshot.coordinate_frame != "vehicle":
            raise ValueError(
                "Perception viewer requires vehicle-frame data, got "
                f"{snapshot.coordinate_frame!r}"
            )

        self._perception_last_snapshot = snapshot
        self.graphics_scene.clear()

        debug = snapshot.debug
        front = float(debug.get("forward_range", 60.0))
        rear = float(debug.get("rear_range", 20.0))
        lateral = float(debug.get("lateral_range", 35.0))
        margin = 3.0

        self.graphics_scene.setSceneRect(
            -(lateral + margin) * 8.0,
            -(front + margin) * 8.0,
            2.0 * (lateral + margin) * 8.0,
            (front + rear + 2.0 * margin) * 8.0,
        )

        self._draw_grid(front, rear, lateral)
        self._draw_lanes(snapshot)
        self._draw_objects(snapshot)
        self._draw_signals(snapshot)
        self._draw_ego()
        self._draw_status(snapshot)

        if self._perception_auto_fit_pending:
            QGraphicsView.fitInView(
                self,
                self.graphics_scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self._perception_auto_fit_pending = False

    def wheel_event(self: PerceptionGraphicsView, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        current_scale = abs(self.transform().m11())
        factor = 1.18 if delta > 0 else 1.0 / 1.18
        target_scale = current_scale * factor

        if 0.08 <= target_scale <= 30.0:
            self.scale(factor, factor)

        event.accept()

    def mouse_double_click_event(
        self: PerceptionGraphicsView,
        event,
    ) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._perception_auto_fit_pending = True
            snapshot = self._perception_last_snapshot
            if snapshot is not None:
                render_perception(self, snapshot)
            event.accept()
            return
        QGraphicsView.mouseDoubleClickEvent(self, event)

    PerceptionGraphicsView.__init__ = init
    PerceptionGraphicsView.render_perception = render_perception
    PerceptionGraphicsView.wheelEvent = wheel_event
    PerceptionGraphicsView.mouseDoubleClickEvent = mouse_double_click_event

    _INSTALLED = True


__all__ = ["install"]
