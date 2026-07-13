from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from sim2d.gui.graphics_items import (
    BoxObstacleGraphicsItem,
    CircleObstacleGraphicsItem,
    GoalGraphicsItem,
    PIXELS_PER_METER,
    VehicleGraphicsItem,
    trajectory_to_path,
)
from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    GoalState,
    SimulationSnapshot,
    VehicleConfig,
)
import math


class SimulationView(QGraphicsView):
    """
    仿真画布。

    只负责渲染 SimulationSnapshot，
    不负责调用 Environment.step()。
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.vehicle_config = vehicle_config

        self.graphics_scene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.setBackgroundBrush(QBrush(QColor("#526746")))

        self.vehicle_item = VehicleGraphicsItem(config=vehicle_config)

        self.goal_item = GoalGraphicsItem()

        self.history_item = QGraphicsPathItem()
        self.history_item.setPen(
            QPen(
                QColor("#f6d04d"),
                3.0,
            )
        )
        self.history_item.setZValue(20)

        self.planned_item = QGraphicsPathItem()
        self.planned_item.setPen(
            QPen(
                QColor("#43e188"),
                3.0,
                Qt.PenStyle.DashLine,
            )
        )
        self.planned_item.setZValue(25)

        self.graphics_scene.addItem(self.history_item)

        self.graphics_scene.addItem(self.planned_item)

        self.graphics_scene.addItem(self.goal_item)

        self.graphics_scene.addItem(self.vehicle_item)

        self.obstacle_items: dict[
            str,
            QGraphicsItem,
        ] = {}

        self._draw_default_road()

    def _draw_default_road(self) -> None:
        """
        暂时画一条简单直路。

        后续接入 RoadNetwork 后，
        这里改成 render_map(road_network)。
        """
        road_x_min = -5.0
        road_x_max = 35.0
        road_y_min = -4.0
        road_y_max = 4.0

        x = road_x_min * PIXELS_PER_METER
        y = -road_y_max * PIXELS_PER_METER

        width = (road_x_max - road_x_min) * PIXELS_PER_METER

        height = (road_y_max - road_y_min) * PIXELS_PER_METER

        road_item = QGraphicsRectItem(
            x,
            y,
            width,
            height,
        )

        road_item.setBrush(QBrush(QColor("#35393e")))

        road_item.setPen(
            QPen(
                QColor("#1f2225"),
                2.0,
            )
        )

        road_item.setZValue(-100)

        self.graphics_scene.addItem(road_item)

        boundary_pen = QPen(
            QColor("#f4f0dc"),
            3.0,
        )

        center_pen = QPen(
            QColor("#e5d97d"),
            2.0,
            Qt.PenStyle.DashLine,
        )

        self.graphics_scene.addLine(
            x,
            -road_y_min * PIXELS_PER_METER,
            x + width,
            -road_y_min * PIXELS_PER_METER,
            boundary_pen,
        ).setZValue(-90)

        self.graphics_scene.addLine(
            x,
            -road_y_max * PIXELS_PER_METER,
            x + width,
            -road_y_max * PIXELS_PER_METER,
            boundary_pen,
        ).setZValue(-90)

        self.graphics_scene.addLine(
            x,
            0.0,
            x + width,
            0.0,
            center_pen,
        ).setZValue(-90)

        self.graphics_scene.setSceneRect(
            x - 2.0 * PIXELS_PER_METER,
            y - 3.0 * PIXELS_PER_METER,
            width + 4.0 * PIXELS_PER_METER,
            height + 6.0 * PIXELS_PER_METER,
        )

    def render_snapshot(
        self,
        snapshot: SimulationSnapshot,
        goal: GoalState,
    ) -> None:
        self.vehicle_item.set_state(snapshot.ego)

        self.goal_item.set_state(goal.state)

        self.history_item.setPath(trajectory_to_path(snapshot.history_trajectory))

        self.planned_item.setPath(trajectory_to_path(snapshot.planned_trajectory))

        self._update_obstacles(snapshot)

        if snapshot.collision:
            self.vehicle_item.setBrush(QBrush(QColor("#d32f2f")))
        else:
            self.vehicle_item.setBrush(QBrush(QColor("#2386c8")))

    def _update_obstacles(
        self,
        snapshot: SimulationSnapshot,
    ) -> None:
        active_ids: set[str] = set()

        for obstacle in snapshot.obstacles:
            active_ids.add(obstacle.obstacle_id)

            existing = self.obstacle_items.get(obstacle.obstacle_id)

            if isinstance(
                obstacle,
                CircleObstacle,
            ):
                if not isinstance(
                    existing,
                    CircleObstacleGraphicsItem,
                ):
                    if existing is not None:
                        self.graphics_scene.removeItem(existing)

                    existing = CircleObstacleGraphicsItem(obstacle)

                    self.graphics_scene.addItem(existing)

                    self.obstacle_items[obstacle.obstacle_id] = existing

                existing.update_obstacle(obstacle)

            elif isinstance(
                obstacle,
                BoxObstacle,
            ):
                if not isinstance(
                    existing,
                    BoxObstacleGraphicsItem,
                ):
                    if existing is not None:
                        self.graphics_scene.removeItem(existing)

                    existing = BoxObstacleGraphicsItem(obstacle)

                    self.graphics_scene.addItem(existing)

                    self.obstacle_items[obstacle.obstacle_id] = existing

                existing.update_obstacle(obstacle)

        stale_ids = set(self.obstacle_items) - active_ids

        for obstacle_id in stale_ids:
            item = self.obstacle_items.pop(obstacle_id)

            self.graphics_scene.removeItem(item)

    def fit_world(self) -> None:
        """
        适配完整场景，并将屏幕视角逆时针旋转 90°。

        仿真内部世界坐标、车辆坐标、航向角和动力学
        均保持不变，只改变 QGraphicsView 的观察方向。
        """
        scene_rect = self.graphics_scene.sceneRect()

        if scene_rect.isEmpty():
            return

        self.resetTransform()

        # Qt 屏幕 y 轴向下，因此视觉上的逆时针 90°
        # 对应 QGraphicsView.rotate(-90)。
        self.rotate(-90.0)

        self.fitInView(
            scene_rect,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15

        transform = self.transform()

        # 旋转后的变换矩阵中，m11 不再单独代表缩放。
        current_scale = math.hypot(
            transform.m11(),
            transform.m12(),
        )

        if factor > 1.0 and current_scale > 5.0:
            return

        if factor < 1.0 and current_scale < 0.15:
            return

        self.scale(
            factor,
            factor,
        )
