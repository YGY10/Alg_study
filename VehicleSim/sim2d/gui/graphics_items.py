from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
)

from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    VehicleConfig,
    VehicleState,
)


PIXELS_PER_METER = 40.0


def world_to_scene(
    x: float,
    y: float,
) -> QPointF:
    """
    世界坐标转换到 Qt Scene 坐标。

    世界坐标：
        +x 向右
        +y 向上

    Qt Scene：
        +x 向右
        +y 向下
    """
    return QPointF(
        x * PIXELS_PER_METER,
        -y * PIXELS_PER_METER,
    )


def scene_to_world(
    point: QPointF,
) -> tuple[float, float]:
    return (
        point.x() / PIXELS_PER_METER,
        -point.y() / PIXELS_PER_METER,
    )


def trajectory_to_path(
    trajectory: np.ndarray | None,
) -> QPainterPath:
    """
    将 [N, >=2] 的轨迹转换为 QPainterPath。
    """
    path = QPainterPath()

    if trajectory is None:
        return path

    array = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    if array.ndim != 2:
        return path

    if array.shape[0] == 0:
        return path

    if array.shape[1] < 2:
        return path

    first = world_to_scene(
        float(array[0, 0]),
        float(array[0, 1]),
    )

    path.moveTo(first)

    for row in array[1:]:
        path.lineTo(
            world_to_scene(
                float(row[0]),
                float(row[1]),
            )
        )

    return path


class VehicleGraphicsItem(QGraphicsPolygonItem):
    """
    鸟瞰车辆图元。

    车辆图元中心对应 VehicleState 的 x、y。
    """

    def __init__(
        self,
        config: VehicleConfig,
    ) -> None:
        self.config = config

        half_length = (
            0.5
            * config.length
            * PIXELS_PER_METER
        )

        half_width = (
            0.5
            * config.width
            * PIXELS_PER_METER
        )

        polygon = QPolygonF(
            [
                QPointF(
                    half_length,
                    -half_width,
                ),
                QPointF(
                    half_length,
                    half_width,
                ),
                QPointF(
                    -half_length,
                    half_width,
                ),
                QPointF(
                    -half_length,
                    -half_width,
                ),
            ]
        )

        super().__init__(polygon)

        self.setBrush(
            QBrush(QColor("#2386c8"))
        )

        self.setPen(
            QPen(
                QColor("#0b355c"),
                2.0,
            )
        )

        self.setZValue(100)

        # 车头方向标志
        marker_radius = 0.10 * PIXELS_PER_METER

        self.front_marker = QGraphicsEllipseItem(
            half_length - 2.5 * marker_radius,
            -marker_radius,
            2.0 * marker_radius,
            2.0 * marker_radius,
            self,
        )

        self.front_marker.setBrush(
            QBrush(QColor("#fff176"))
        )

        self.front_marker.setPen(
            QPen(QColor("#6d6500"), 1.0)
        )

    def set_state(
        self,
        state: VehicleState,
    ) -> None:
        self.setPos(
            world_to_scene(
                state.x,
                state.y,
            )
        )

        # Qt 顺时针为正，世界 yaw 逆时针为正。
        self.setRotation(
            -math.degrees(state.yaw)
        )


class CircleObstacleGraphicsItem(
    QGraphicsEllipseItem
):
    def __init__(
        self,
        obstacle: CircleObstacle,
    ) -> None:
        radius_px = (
            obstacle.radius
            * PIXELS_PER_METER
        )

        super().__init__(
            -radius_px,
            -radius_px,
            2.0 * radius_px,
            2.0 * radius_px,
        )

        self.obstacle_id = obstacle.obstacle_id

        self.setBrush(
            QBrush(QColor("#e85d3f"))
        )

        self.setPen(
            QPen(
                QColor("#7c2114"),
                2.0,
            )
        )

        self.setZValue(50)

        self.update_obstacle(obstacle)

    def update_obstacle(
        self,
        obstacle: CircleObstacle,
    ) -> None:
        radius_px = (
            obstacle.radius
            * PIXELS_PER_METER
        )

        self.setRect(
            -radius_px,
            -radius_px,
            2.0 * radius_px,
            2.0 * radius_px,
        )

        self.setPos(
            world_to_scene(
                obstacle.x,
                obstacle.y,
            )
        )


class BoxObstacleGraphicsItem(
    QGraphicsRectItem
):
    def __init__(
        self,
        obstacle: BoxObstacle,
    ) -> None:
        super().__init__()

        self.obstacle_id = obstacle.obstacle_id

        self.setBrush(
            QBrush(QColor("#d66b39"))
        )

        self.setPen(
            QPen(
                QColor("#6f2e12"),
                2.0,
            )
        )

        self.setZValue(50)

        self.update_obstacle(obstacle)

    def update_obstacle(
        self,
        obstacle: BoxObstacle,
    ) -> None:
        length_px = (
            obstacle.length
            * PIXELS_PER_METER
        )

        width_px = (
            obstacle.width
            * PIXELS_PER_METER
        )

        self.setRect(
            -0.5 * length_px,
            -0.5 * width_px,
            length_px,
            width_px,
        )

        self.setPos(
            world_to_scene(
                obstacle.x,
                obstacle.y,
            )
        )

        self.setRotation(
            -math.degrees(obstacle.yaw)
        )


class GoalGraphicsItem(QGraphicsItem):
    """
    目标点显示。
    """

    def __init__(
        self,
        radius_m: float = 0.6,
    ) -> None:
        super().__init__()

        self.radius_px = (
            radius_m
            * PIXELS_PER_METER
        )

        self.setZValue(40)

    def boundingRect(self) -> QRectF:
        r = self.radius_px

        return QRectF(
            -r,
            -r,
            2.0 * r,
            2.0 * r,
        )

    def paint(
        self,
        painter: QPainter,
        option,
        widget=None,
    ) -> None:
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.setPen(
            QPen(
                QColor("#1b7f3a"),
                3.0,
            )
        )

        painter.setBrush(
            QBrush(QColor("#78d98b"))
        )

        r = self.radius_px

        painter.drawEllipse(
            QPointF(0.0, 0.0),
            r,
            r,
        )

        painter.drawLine(
            QPointF(-r, 0.0),
            QPointF(r, 0.0),
        )

        painter.drawLine(
            QPointF(0.0, -r),
            QPointF(0.0, r),
        )

    def set_state(
        self,
        state: VehicleState,
    ) -> None:
        self.setPos(
            world_to_scene(
                state.x,
                state.y,
            )
        )