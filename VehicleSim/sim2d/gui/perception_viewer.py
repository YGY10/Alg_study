from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from sim2d.gui.app_icon import apply_application_icon
from sim2d.gui.main_window import MainWindow

_INSTALLED = False
_PIXELS_PER_METER = 8.0


class PerceptionGraphicsView(QGraphicsView):
    """车辆坐标系下的局部感知结果：前方 +x，左侧 +y。"""

    def __init__(self, parent=None) -> None:
        self.graphics_scene = QGraphicsScene(parent)
        super().__init__(self.graphics_scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#111820")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumSize(760, 520)

    def render_perception(self, snapshot) -> None:
        self.graphics_scene.clear()
        debug = snapshot.debug
        front = float(debug.get("forward_range", 60.0))
        rear = float(debug.get("rear_range", 20.0))
        lateral = float(debug.get("lateral_range", 35.0))

        margin = 3.0
        self.graphics_scene.setSceneRect(
            -(rear + margin) * _PIXELS_PER_METER,
            -(lateral + margin) * _PIXELS_PER_METER,
            (front + rear + 2.0 * margin) * _PIXELS_PER_METER,
            2.0 * (lateral + margin) * _PIXELS_PER_METER,
        )

        self._draw_grid(front, rear, lateral)
        self._draw_lanes(snapshot)
        self._draw_objects(snapshot)
        self._draw_signals(snapshot)
        self._draw_ego()
        self._draw_status(snapshot)
        self.fitInView(
            self.graphics_scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    @staticmethod
    def _vehicle_point(snapshot, x: float, y: float) -> QPointF:
        ego = snapshot.ego
        dx = float(x) - ego.x
        dy = float(y) - ego.y
        c = math.cos(ego.yaw)
        s = math.sin(ego.yaw)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        return QPointF(
            longitudinal * _PIXELS_PER_METER,
            -lateral * _PIXELS_PER_METER,
        )

    def _draw_grid(self, front: float, rear: float, lateral: float) -> None:
        grid_pen = QPen(
            QColor(86, 105, 120, 80),
            1.0,
            Qt.PenStyle.DotLine,
        )
        grid_pen.setCosmetic(True)
        for x in range(-int(rear), int(front) + 1, 10):
            self.graphics_scene.addLine(
                x * _PIXELS_PER_METER,
                -lateral * _PIXELS_PER_METER,
                x * _PIXELS_PER_METER,
                lateral * _PIXELS_PER_METER,
                grid_pen,
            )
        for y in range(-int(lateral), int(lateral) + 1, 10):
            self.graphics_scene.addLine(
                -rear * _PIXELS_PER_METER,
                -y * _PIXELS_PER_METER,
                front * _PIXELS_PER_METER,
                -y * _PIXELS_PER_METER,
                grid_pen,
            )

        x_pen = QPen(QColor("#40d7ff"), 2.0)
        y_pen = QPen(QColor("#7cff8f"), 2.0)
        x_pen.setCosmetic(True)
        y_pen.setCosmetic(True)
        self.graphics_scene.addLine(
            -rear * _PIXELS_PER_METER,
            0.0,
            front * _PIXELS_PER_METER,
            0.0,
            x_pen,
        )
        self.graphics_scene.addLine(
            0.0,
            lateral * _PIXELS_PER_METER,
            0.0,
            -lateral * _PIXELS_PER_METER,
            y_pen,
        )

        x_text = self.graphics_scene.addSimpleText("+x 前进")
        x_text.setBrush(QBrush(QColor("#40d7ff")))
        x_text.setPos((front - 8.0) * _PIXELS_PER_METER, 6.0)
        y_text = self.graphics_scene.addSimpleText("+y 左侧")
        y_text.setBrush(QBrush(QColor("#7cff8f")))
        y_text.setPos(6.0, -(lateral - 3.0) * _PIXELS_PER_METER)

    def _draw_lanes(self, snapshot) -> None:
        center_pen = QPen(
            QColor("#62bfff"),
            1.6,
            Qt.PenStyle.DashLine,
        )
        boundary_pen = QPen(QColor("#dce9f3"), 1.4)
        center_pen.setCosmetic(True)
        boundary_pen.setCosmetic(True)

        for lane in snapshot.road_segments:
            for points, pen in (
                (lane.left_boundary, boundary_pen),
                (lane.right_boundary, boundary_pen),
                (lane.centerline, center_pen),
            ):
                if len(points) < 2:
                    continue
                path = QPainterPath(
                    self._vehicle_point(
                        snapshot,
                        points[0][0],
                        points[0][1],
                    )
                )
                for point in points[1:]:
                    path.lineTo(
                        self._vehicle_point(
                            snapshot,
                            point[0],
                            point[1],
                        )
                    )
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                self.graphics_scene.addItem(item)

    def _draw_objects(self, snapshot) -> None:
        for obj in snapshot.objects:
            center = self._vehicle_point(snapshot, obj.x, obj.y)
            yaw = obj.yaw - snapshot.ego.yaw
            if obj.object_type == "circle":
                radius = (
                    0.5
                    * max(obj.length, obj.width)
                    * _PIXELS_PER_METER
                )
                item = QGraphicsEllipseItem(
                    center.x() - radius,
                    center.y() - radius,
                    2.0 * radius,
                    2.0 * radius,
                )
            else:
                half_l = 0.5 * obj.length * _PIXELS_PER_METER
                half_w = 0.5 * obj.width * _PIXELS_PER_METER
                polygon = QPolygonF()
                c = math.cos(yaw)
                s = math.sin(yaw)
                for local_x, local_y in (
                    (half_l, half_w),
                    (half_l, -half_w),
                    (-half_l, -half_w),
                    (-half_l, half_w),
                ):
                    scene_x = center.x() + c * local_x - s * local_y
                    scene_y = center.y() - (
                        s * local_x + c * local_y
                    )
                    polygon.append(QPointF(scene_x, scene_y))
                item = QGraphicsPolygonItem(polygon)

            item.setPen(QPen(QColor("#ff9e57"), 1.8))
            item.setBrush(QBrush(QColor(255, 112, 67, 120)))
            self.graphics_scene.addItem(item)

            label = QGraphicsSimpleTextItem(obj.object_id)
            label.setBrush(QBrush(QColor("#ffd6b8")))
            label.setPos(center.x() + 5.0, center.y() + 5.0)
            self.graphics_scene.addItem(label)

    def _draw_signals(self, snapshot) -> None:
        colors = {
            "red": QColor("#ff5252"),
            "yellow": QColor("#ffd54f"),
            "green": QColor("#57d67a"),
            "off": QColor("#707b86"),
            "unknown": QColor("#d8dde3"),
        }
        for signal in snapshot.traffic_signals:
            center = self._vehicle_point(snapshot, signal.x, signal.y)
            radius = 4.5
            item = QGraphicsEllipseItem(
                center.x() - radius,
                center.y() - radius,
                2.0 * radius,
                2.0 * radius,
            )
            item.setPen(QPen(QColor("#101010"), 1.2))
            item.setBrush(
                QBrush(
                    colors.get(
                        signal.state,
                        colors["unknown"],
                    )
                )
            )
            self.graphics_scene.addItem(item)

    def _draw_ego(self) -> None:
        length = 4.6 * _PIXELS_PER_METER
        width = 1.85 * _PIXELS_PER_METER
        item = QGraphicsRectItem(
            -0.5 * length,
            -0.5 * width,
            length,
            width,
        )
        item.setPen(QPen(QColor("#35f2ff"), 2.0))
        item.setBrush(QBrush(QColor(37, 205, 230, 150)))
        self.graphics_scene.addItem(item)
        self.graphics_scene.addLine(
            0.0,
            0.0,
            0.65 * length,
            0.0,
            QPen(QColor("#ffffff"), 2.0),
        )

    def _draw_status(self, snapshot) -> None:
        text = self.graphics_scene.addSimpleText(
            f"感知帧 {snapshot.frame}  "
            f"measurement={snapshot.measurement_time:.2f}s  "
            f"publish={snapshot.publish_time:.2f}s  "
            f"目标={len(snapshot.objects)}  "
            f"交通灯={len(snapshot.traffic_signals)}  "
            f"车道段={len(snapshot.road_segments)}"
        )
        text.setBrush(QBrush(QColor("#ffffff")))
        rect = self.graphics_scene.sceneRect()
        text.setPos(rect.left() + 8.0, rect.top() + 8.0)


class PerceptionWindow(QWidget):
    closed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(
            "VehicleSim 2D - 感知视图（车辆坐标系）"
        )
        self.resize(980, 680)
        self.view = PerceptionGraphicsView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class PerceptionViewerController:
    def __init__(self, window: MainWindow) -> None:
        self.main_window = window
        self.viewer: PerceptionWindow | None = None

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.viewer is None:
                self.viewer = PerceptionWindow(self.main_window)
                app = QApplication.instance()
                if isinstance(app, QApplication):
                    apply_application_icon(app, self.viewer)
                self.viewer.closed.connect(self._viewer_closed)
            self.viewer.show()
            self.viewer.raise_()
            self.update()
        elif self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _viewer_closed(self) -> None:
        self.viewer = None
        checkbox = getattr(
            self.main_window,
            "show_perception_view_check",
            None,
        )
        if checkbox is not None:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def update(self) -> None:
        if (
            self.viewer is None
            or not self.main_window.env.is_initialized
        ):
            return
        snapshot = self.main_window.env.get_perception_snapshot()
        self.viewer.view.render_perception(snapshot)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_init = MainWindow.__init__
    original_reset = MainWindow.reset_environment
    original_advance = MainWindow.advance_one_step

    def init(self: MainWindow, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        self.perception_viewer_controller = PerceptionViewerController(
            self
        )

        dock = QDockWidget("感知调试", self)
        dock.setObjectName("perception_viewer_dock")
        panel = QWidget(dock)
        layout = QVBoxLayout(panel)
        self.show_perception_view_check = QCheckBox(
            "打开独立感知视图"
        )
        self.show_perception_view_check.setChecked(False)
        layout.addWidget(self.show_perception_view_check)
        layout.addStretch(1)
        dock.setWidget(panel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            dock,
        )
        self.show_perception_view_check.toggled.connect(
            self.perception_viewer_controller.set_enabled
        )
        self.perception_viewer_dock = dock

    def reset_environment(self: MainWindow) -> None:
        original_reset(self)
        self.perception_viewer_controller.update()

    def advance_one_step(self: MainWindow) -> None:
        original_advance(self)
        self.perception_viewer_controller.update()

    MainWindow.__init__ = init
    MainWindow.reset_environment = reset_environment
    MainWindow.advance_one_step = advance_one_step
    _INSTALLED = True
