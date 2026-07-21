from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
)

from sim2d.gui.perception_viewer import (
    PerceptionGraphicsView,
    _PIXELS_PER_METER,
)

_INSTALLED = False


def install() -> None:
    """将感知视图旋转为：车辆前方 +x 向上，车辆左侧 +y 向左。"""
    global _INSTALLED
    if _INSTALLED:
        return

    PerceptionGraphicsView.render_perception = _render_perception
    PerceptionGraphicsView._vehicle_point = staticmethod(_vehicle_point)
    PerceptionGraphicsView._draw_grid = _draw_grid
    PerceptionGraphicsView._draw_objects = _draw_objects
    PerceptionGraphicsView._draw_ego = _draw_ego

    _INSTALLED = True


def _render_perception(self: PerceptionGraphicsView, snapshot) -> None:
    self.graphics_scene.clear()

    debug = snapshot.debug
    front = float(debug.get("forward_range", 60.0))
    rear = float(debug.get("rear_range", 20.0))
    lateral = float(debug.get("lateral_range", 35.0))

    margin = 3.0

    # 横轴表示车辆横向：左侧 +y 在画面左边，右侧 -y 在画面右边。
    # 纵轴表示车辆纵向：前方 +x 在画面上方，后方 -x 在画面下方。
    self.graphics_scene.setSceneRect(
        -(lateral + margin) * _PIXELS_PER_METER,
        -(front + margin) * _PIXELS_PER_METER,
        2.0 * (lateral + margin) * _PIXELS_PER_METER,
        (front + rear + 2.0 * margin) * _PIXELS_PER_METER,
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


def _vehicle_point(snapshot, x: float, y: float) -> QPointF:
    ego = snapshot.ego

    dx = float(x) - ego.x
    dy = float(y) - ego.y

    c = math.cos(ego.yaw)
    s = math.sin(ego.yaw)

    longitudinal = c * dx + s * dy
    lateral = -s * dx + c * dy

    return _local_point(longitudinal, lateral)


def _local_point(longitudinal: float, lateral: float) -> QPointF:
    return QPointF(
        -float(lateral) * _PIXELS_PER_METER,
        -float(longitudinal) * _PIXELS_PER_METER,
    )


def _draw_grid(
    self: PerceptionGraphicsView,
    front: float,
    rear: float,
    lateral: float,
) -> None:
    grid_pen = QPen(
        QColor(86, 105, 120, 80),
        1.0,
        Qt.PenStyle.DotLine,
    )
    grid_pen.setCosmetic(True)

    # 等纵向距离线：车辆前后方向，对应画面水平线。
    for longitudinal in range(-int(rear), int(front) + 1, 10):
        scene_y = -longitudinal * _PIXELS_PER_METER
        self.graphics_scene.addLine(
            -lateral * _PIXELS_PER_METER,
            scene_y,
            lateral * _PIXELS_PER_METER,
            scene_y,
            grid_pen,
        )

    # 等横向距离线：车辆左右方向，对应画面竖直线。
    for lateral_value in range(-int(lateral), int(lateral) + 1, 10):
        scene_x = -lateral_value * _PIXELS_PER_METER
        self.graphics_scene.addLine(
            scene_x,
            -front * _PIXELS_PER_METER,
            scene_x,
            rear * _PIXELS_PER_METER,
            grid_pen,
        )

    x_pen = QPen(QColor("#40d7ff"), 2.0)
    y_pen = QPen(QColor("#7cff8f"), 2.0)
    x_pen.setCosmetic(True)
    y_pen.setCosmetic(True)

    # +x 前进轴：从后方向前方竖直向上。
    self.graphics_scene.addLine(
        0.0,
        rear * _PIXELS_PER_METER,
        0.0,
        -front * _PIXELS_PER_METER,
        x_pen,
    )

    # +y 左侧轴：从右侧指向左侧。
    self.graphics_scene.addLine(
        lateral * _PIXELS_PER_METER,
        0.0,
        -lateral * _PIXELS_PER_METER,
        0.0,
        y_pen,
    )

    x_text = self.graphics_scene.addSimpleText("+x 前进")
    x_text.setBrush(QBrush(QColor("#40d7ff")))
    x_text.setPos(
        6.0,
        -(front - 3.0) * _PIXELS_PER_METER,
    )

    y_text = self.graphics_scene.addSimpleText("+y 左侧")
    y_text.setBrush(QBrush(QColor("#7cff8f")))
    y_text.setPos(
        -(lateral - 3.0) * _PIXELS_PER_METER,
        6.0,
    )


def _draw_objects(self: PerceptionGraphicsView, snapshot) -> None:
    for obj in snapshot.objects:
        center = self._vehicle_point(snapshot, obj.x, obj.y)
        relative_yaw = obj.yaw - snapshot.ego.yaw

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
            half_length = 0.5 * obj.length
            half_width = 0.5 * obj.width
            c = math.cos(relative_yaw)
            s = math.sin(relative_yaw)
            polygon = QPolygonF()

            for local_longitudinal, local_lateral in (
                (half_length, half_width),
                (half_length, -half_width),
                (-half_length, -half_width),
                (-half_length, half_width),
            ):
                rotated_longitudinal = (
                    c * local_longitudinal
                    - s * local_lateral
                )
                rotated_lateral = (
                    s * local_longitudinal
                    + c * local_lateral
                )
                offset = _local_point(
                    rotated_longitudinal,
                    rotated_lateral,
                )
                polygon.append(
                    QPointF(
                        center.x() + offset.x(),
                        center.y() + offset.y(),
                    )
                )

            item = QGraphicsPolygonItem(polygon)

        item.setPen(QPen(QColor("#ff9e57"), 1.8))
        item.setBrush(QBrush(QColor(255, 112, 67, 120)))
        self.graphics_scene.addItem(item)

        label = self.graphics_scene.addSimpleText(obj.object_id)
        label.setBrush(QBrush(QColor("#ffd6b8")))
        label.setPos(center.x() + 5.0, center.y() + 5.0)


def _draw_ego(self: PerceptionGraphicsView) -> None:
    half_length = 0.5 * 4.6 * _PIXELS_PER_METER
    half_width = 0.5 * 1.85 * _PIXELS_PER_METER

    polygon = QPolygonF(
        [
            QPointF(-half_width, -half_length),
            QPointF(half_width, -half_length),
            QPointF(half_width, half_length),
            QPointF(-half_width, half_length),
        ]
    )

    item = QGraphicsPolygonItem(polygon)
    item.setPen(QPen(QColor("#35f2ff"), 2.0))
    item.setBrush(QBrush(QColor(37, 205, 230, 150)))
    self.graphics_scene.addItem(item)

    self.graphics_scene.addLine(
        0.0,
        0.0,
        0.0,
        -1.3 * half_length,
        QPen(QColor("#ffffff"), 2.0),
    )
