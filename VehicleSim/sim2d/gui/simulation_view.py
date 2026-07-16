from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
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
from sim2d.map import (
    Lane,
    LaneProjection,
    LaneType,
    RoadNetwork,
)
from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    GoalState,
    SimulationSnapshot,
    VehicleConfig,
)


class SimulationView(QGraphicsView):
    world_mouse_moved = Signal(float, float)
    world_point_clicked = Signal(float, float)
    world_pose_selected = Signal(float, float, float)
    lane_projection_changed = Signal(object)

    """
    仿真画布。

    只负责渲染地图和 SimulationSnapshot，
    不负责调用 Environment.step()。

    动态显示三类轨迹：

        reference_path：
            规划器生成的完整空间参考路径。

        planned_trajectory：
            当前规划周期的闭环预测车辆轨迹。

        history_trajectory：
            车辆经过动力学模型实际执行后的历史轨迹。

    静态地图由 RoadNetwork 提供，只在地图发生变化时重绘。
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

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self.vehicle_item = VehicleGraphicsItem(config=vehicle_config)

        self.goal_item = GoalGraphicsItem()

        # 规划器生成的完整参考路径。
        self.reference_item = QGraphicsPathItem()

        reference_pen = QPen(
            QColor("#6dc7e8"),
            2.0,
            Qt.PenStyle.DotLine,
        )

        reference_pen.setCosmetic(True)

        self.reference_item.setPen(reference_pen)

        self.reference_item.setZValue(15)

        # 车辆实际执行历史轨迹。
        self.history_item = QGraphicsPathItem()

        history_pen = QPen(
            QColor("#f6d04d"),
            3.0,
            Qt.PenStyle.SolidLine,
        )

        history_pen.setCosmetic(True)

        self.history_item.setPen(history_pen)

        self.history_item.setZValue(20)

        # 当前闭环预测轨迹。
        self.planned_item = QGraphicsPathItem()

        planned_pen = QPen(
            QColor("#43e188"),
            3.0,
            Qt.PenStyle.DashLine,
        )

        planned_pen.setCosmetic(True)

        self.planned_item.setPen(planned_pen)

        self.planned_item.setZValue(25)

        self.graphics_scene.addItem(self.reference_item)

        self.graphics_scene.addItem(self.history_item)

        self.graphics_scene.addItem(self.planned_item)

        self.graphics_scene.addItem(self.goal_item)

        self.graphics_scene.addItem(self.vehicle_item)

        self.obstacle_items: dict[
            str,
            QGraphicsItem,
        ] = {}

        # 临时默认道路图元。
        self.default_road_items: list[QGraphicsItem] = []

        # 由 RoadNetwork 创建的地图图元。
        self.map_items: list[QGraphicsItem] = []

        self.road_network: RoadNetwork | None = None

        # 地图调试显示开关。
        #
        # 正常模式默认隐藏 lane centerline；
        # 调试模式可通过 set_show_lane_centerlines(True) 显示。
        self.show_lane_centerlines = False

        # False：正常模式，仅显示每条 road 的外轮廓。
        # True：调试模式，显示所有去重后的 lane 边界。
        self.show_lane_boundaries = False

        # 边界线段量化精度，单位 m。
        # 用于识别方向相反或重复采样得到的同一物理边界。
        self.boundary_quantization = 0.01

        self.selection_enabled = False
        self.snap_to_lane = True
        self.snap_distance = 3.0

        self.last_world_point: tuple[float, float] | None = None
        self.last_lane_projection: LaneProjection | None = None

        # 鼠标拖拽选择姿态。
        self.selection_default_yaw = 0.0
        self.selection_drag_active = False
        self.selection_anchor: tuple[float, float] | None = None
        self.selection_anchor_lane_yaw: float | None = None
        self.selection_drag_yaw: float | None = None
        self.selection_drag_threshold = 0.20

        self.candidate_point_item = QGraphicsEllipseItem()
        point_pen = QPen(
            QColor("#ff7043"),
            2.0,
            Qt.PenStyle.SolidLine,
        )
        point_pen.setCosmetic(True)
        self.candidate_point_item.setPen(point_pen)
        self.candidate_point_item.setBrush(QBrush(QColor("#ffcc80")))
        self.candidate_point_item.setZValue(100)
        self.candidate_point_item.setVisible(False)
        self.graphics_scene.addItem(self.candidate_point_item)

        self.candidate_heading_item = QGraphicsLineItem()
        heading_pen = QPen(
            QColor("#ff7043"),
            3.0,
            Qt.PenStyle.SolidLine,
        )
        heading_pen.setCosmetic(True)
        self.candidate_heading_item.setPen(heading_pen)
        self.candidate_heading_item.setZValue(101)
        self.candidate_heading_item.setVisible(False)
        self.graphics_scene.addItem(self.candidate_heading_item)

        self._draw_default_road()

    def set_show_lane_centerlines(
        self,
        visible: bool,
    ) -> None:
        """
        控制是否显示 lane centerline 调试层。

        正常地图模式下应隐藏 lane centerline，
        因为它不是 OpenDRIVE 中的真实 roadMark。

        修改后会重新绘制当前 RoadNetwork。
        """
        visible = bool(visible)

        if self.show_lane_centerlines == visible:
            return

        self.show_lane_centerlines = visible

        if self.road_network is not None:
            self.render_road_network(self.road_network)

    def set_show_lane_boundaries(
        self,
        visible: bool,
    ) -> None:
        """
        控制 lane 边界调试层。

        False：
            正常模式，只显示每条普通 road 的外轮廓。

        True：
            调试模式，显示所有去重后的 lane 左右边界。

        修改后会重新绘制当前 RoadNetwork。
        """
        visible = bool(visible)

        if self.show_lane_boundaries == visible:
            return

        self.show_lane_boundaries = visible

        if self.road_network is not None:
            self.render_road_network(self.road_network)

    def set_selection_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.selection_enabled = bool(enabled)

        self.selection_drag_active = False
        self.selection_anchor = None
        self.selection_anchor_lane_yaw = None
        self.selection_drag_yaw = None

        if self.selection_enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._update_candidate_from_last_mouse()
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
            self._hide_candidate_preview()

    def set_selection_default_yaw(
        self,
        yaw: float,
    ) -> None:
        if not np.isfinite(yaw):
            raise ValueError("selection yaw must be finite")

        self.selection_default_yaw = float(yaw)

    def set_lane_snapping(
        self,
        *,
        enabled: bool,
        max_distance: float | None = None,
    ) -> None:
        if max_distance is not None:
            if not np.isfinite(max_distance):
                raise ValueError("max_distance must be finite")

            if max_distance < 0.0:
                raise ValueError("max_distance must be non-negative")

            self.snap_distance = float(max_distance)

        self.snap_to_lane = bool(enabled)
        self._update_candidate_from_last_mouse()

    def viewport_to_world(
        self,
        viewport_position,
    ) -> tuple[float, float]:
        scene_position = self.mapToScene(viewport_position)

        world_x = scene_position.x() / PIXELS_PER_METER
        world_y = -scene_position.y() / PIXELS_PER_METER

        return float(world_x), float(world_y)

    def candidate_world_state(
        self,
    ) -> tuple[float, float, float | None] | None:
        if self.selection_drag_active and self.selection_anchor is not None:
            yaw = self.selection_drag_yaw

            if yaw is None:
                yaw = self.selection_anchor_lane_yaw

            if yaw is None:
                yaw = self.selection_default_yaw

            return (
                self.selection_anchor[0],
                self.selection_anchor[1],
                yaw,
            )

        if self.snap_to_lane:
            projection = self.last_lane_projection

            if projection is None:
                return None

            return (
                float(projection.point[0]),
                float(projection.point[1]),
                projection.yaw,
            )

        if self.last_world_point is None:
            return None

        return (
            self.last_world_point[0],
            self.last_world_point[1],
            self.selection_default_yaw,
        )

    def _update_candidate_from_world(
        self,
        world_x: float,
        world_y: float,
    ) -> None:
        if not self.selection_enabled:
            self._hide_candidate_preview()
            return

        if self.snap_to_lane:
            if self.road_network is None:
                self._set_lane_projection(None)
                self._hide_candidate_preview()
                return

            projection = self.road_network.nearest_lane_point(
                x=world_x,
                y=world_y,
                lane_types=(LaneType.DRIVING,),
                max_distance=self.snap_distance,
            )

            self._set_lane_projection(projection)

            if projection is None:
                self._hide_candidate_preview()
                return

            self._show_candidate_preview(
                world_x=float(projection.point[0]),
                world_y=float(projection.point[1]),
                yaw=projection.yaw,
            )
            return

        self._set_lane_projection(None)

        self._show_candidate_preview(
            world_x=world_x,
            world_y=world_y,
            yaw=None,
        )

    def _update_candidate_from_last_mouse(
        self,
    ) -> None:
        if self.last_world_point is None:
            self._hide_candidate_preview()
            return

        self._update_candidate_from_world(
            world_x=self.last_world_point[0],
            world_y=self.last_world_point[1],
        )

    def _set_lane_projection(
        self,
        projection: LaneProjection | None,
    ) -> None:
        self.last_lane_projection = projection
        self.lane_projection_changed.emit(projection)

    def _show_candidate_preview(
        self,
        *,
        world_x: float,
        world_y: float,
        yaw: float | None,
    ) -> None:
        point_radius = 0.30

        scene_x = world_x * PIXELS_PER_METER
        scene_y = -world_y * PIXELS_PER_METER
        radius_pixels = point_radius * PIXELS_PER_METER

        self.candidate_point_item.setRect(
            scene_x - radius_pixels,
            scene_y - radius_pixels,
            2.0 * radius_pixels,
            2.0 * radius_pixels,
        )
        self.candidate_point_item.setVisible(True)

        if yaw is None:
            self.candidate_heading_item.setVisible(False)
            return

        heading_length = 3.0

        end_x = world_x + heading_length * math.cos(yaw)
        end_y = world_y + heading_length * math.sin(yaw)

        self.candidate_heading_item.setLine(
            scene_x,
            scene_y,
            end_x * PIXELS_PER_METER,
            -end_y * PIXELS_PER_METER,
        )
        self.candidate_heading_item.setVisible(True)

    def _hide_candidate_preview(
        self,
    ) -> None:
        self.candidate_point_item.setVisible(False)
        self.candidate_heading_item.setVisible(False)

    def _draw_default_road(
        self,
    ) -> None:
        """
        在尚未加载 RoadNetwork 时，绘制临时直路。

        一旦调用 render_road_network()，
        这些图元会被清除。
        """
        self._clear_default_road()

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

        self.default_road_items.append(road_item)

        boundary_pen = QPen(
            QColor("#f4f0dc"),
            3.0,
        )

        boundary_pen.setCosmetic(True)

        center_pen = QPen(
            QColor("#e5d97d"),
            2.0,
            Qt.PenStyle.DashLine,
        )

        center_pen.setCosmetic(True)

        top_boundary_item = self.graphics_scene.addLine(
            x,
            -road_y_min * PIXELS_PER_METER,
            x + width,
            -road_y_min * PIXELS_PER_METER,
            boundary_pen,
        )

        top_boundary_item.setZValue(-90)

        self.default_road_items.append(top_boundary_item)

        bottom_boundary_item = self.graphics_scene.addLine(
            x,
            -road_y_max * PIXELS_PER_METER,
            x + width,
            -road_y_max * PIXELS_PER_METER,
            boundary_pen,
        )

        bottom_boundary_item.setZValue(-90)

        self.default_road_items.append(bottom_boundary_item)

        center_line_item = self.graphics_scene.addLine(
            x,
            0.0,
            x + width,
            0.0,
            center_pen,
        )

        center_line_item.setZValue(-90)

        self.default_road_items.append(center_line_item)

        self.graphics_scene.setSceneRect(
            x - 2.0 * PIXELS_PER_METER,
            y - 3.0 * PIXELS_PER_METER,
            width + 4.0 * PIXELS_PER_METER,
            height + 6.0 * PIXELS_PER_METER,
        )

    def render_road_network(
        self,
        road_network: RoadNetwork,
    ) -> None:
        """
        绘制 VehicleSim 统一 RoadNetwork。

        正常模式：
            所有 lane polygon 填充路面；
            普通道路绘制外轮廓和弱化的车道分隔线；
            junction connector 不绘制左右边界；
            lane centerline 默认隐藏。

        调试模式：
            额外显示 lane centerline。

        地图属于静态内容，因此只应在加载、切换地图或
        修改地图显示选项时调用。
        """
        road_network.validate()

        self._clear_default_road()
        self._clear_map_items()

        self.road_network = road_network
        self.last_lane_projection = None
        self._hide_candidate_preview()

        all_points: list[np.ndarray] = []

        # 第一层：路面 polygon。
        for lane in road_network.lanes:
            self._draw_lane_surface(lane)

            all_points.extend(
                [
                    lane.centerline.points,
                    lane.left_boundary.points,
                    lane.right_boundary.points,
                ]
            )

        # 第二层：普通道路边界，按量化线段去重。
        self._draw_deduplicated_boundaries(road_network.lanes)

        # 第三层：调试中心线。
        if self.show_lane_centerlines:
            self._draw_lane_centerlines(road_network.lanes)

        if not all_points:
            self.graphics_scene.setSceneRect(
                -10.0 * PIXELS_PER_METER,
                -10.0 * PIXELS_PER_METER,
                20.0 * PIXELS_PER_METER,
                20.0 * PIXELS_PER_METER,
            )

            return

        map_points = np.vstack(all_points)

        x_min = float(np.min(map_points[:, 0]))
        x_max = float(np.max(map_points[:, 0]))
        y_min = float(np.min(map_points[:, 1]))
        y_max = float(np.max(map_points[:, 1]))

        margin = 4.0

        scene_x = (x_min - margin) * PIXELS_PER_METER

        scene_y = -(y_max + margin) * PIXELS_PER_METER

        scene_width = (x_max - x_min + 2.0 * margin) * PIXELS_PER_METER

        scene_height = (y_max - y_min + 2.0 * margin) * PIXELS_PER_METER

        self.graphics_scene.setSceneRect(
            scene_x,
            scene_y,
            scene_width,
            scene_height,
        )

    def _draw_lane_surface(
        self,
        lane: Lane,
    ) -> None:
        """
        绘制单条 lane 的填充路面。

        polygon 本身不绘制轮廓，避免相邻 lane 接缝叠加。
        """
        road_surface_item = QGraphicsPathItem(self._lane_surface_path(lane))

        road_surface_item.setBrush(self._lane_surface_brush(lane.lane_type))

        road_surface_item.setPen(QPen(Qt.PenStyle.NoPen))

        road_surface_item.setZValue(-100)

        self.graphics_scene.addItem(road_surface_item)

        self.map_items.append(road_surface_item)

    def _draw_deduplicated_boundaries(
        self,
        lanes: tuple[Lane, ...],
    ) -> None:
        """
        绘制普通道路边界与简化车道分隔线。

        正常模式：
            count == 1：
                绘制道路外轮廓；

            DRIVING / DRIVING：
                绘制较细、较暗的虚线，提示多车道结构；

            DRIVING / SHOULDER、PARKING、BIKE：
                绘制较细实线；

            包含 SIDEWALK：
                绘制较暗实线。

        调试模式：
            显示所有去重后的 lane 左右边界。

        junction connector 始终跳过边界绘制，
        避免路口内部出现大量交叉线。
        """
        segment_records: dict[
            tuple[
                str,
                tuple[
                    tuple[int, int],
                    tuple[int, int],
                ],
            ],
            tuple[
                list[LaneType],
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        for lane in lanes:
            if self._is_junction_connector(lane):
                continue

            road_id = str(
                lane.metadata.get(
                    "road_id",
                    "",
                )
            )

            for boundary in (
                lane.left_boundary.points,
                lane.right_boundary.points,
            ):
                for point_a, point_b in zip(
                    boundary[:-1],
                    boundary[1:],
                ):
                    if np.linalg.norm(point_b - point_a) <= 1e-9:
                        continue

                    segment_key = self._boundary_segment_key(
                        point_a,
                        point_b,
                    )

                    record_key = (
                        road_id,
                        segment_key,
                    )

                    record = segment_records.get(record_key)

                    if record is None:
                        segment_records[record_key] = (
                            [lane.lane_type],
                            point_a.copy(),
                            point_b.copy(),
                        )
                    else:
                        lane_types, first_a, first_b = record

                        lane_types.append(lane.lane_type)

                        segment_records[record_key] = (
                            lane_types,
                            first_a,
                            first_b,
                        )

        paths_by_style: dict[
            str,
            QPainterPath,
        ] = {}

        for (
            lane_types,
            point_a,
            point_b,
        ) in segment_records.values():
            style = self._boundary_style(
                lane_types=lane_types,
                debug=self.show_lane_boundaries,
            )

            if style is None:
                continue

            path = paths_by_style.setdefault(
                style,
                QPainterPath(),
            )

            path.moveTo(
                float(point_a[0]) * PIXELS_PER_METER,
                -float(point_a[1]) * PIXELS_PER_METER,
            )

            path.lineTo(
                float(point_b[0]) * PIXELS_PER_METER,
                -float(point_b[1]) * PIXELS_PER_METER,
            )

        for style, path in paths_by_style.items():
            if path.isEmpty():
                continue

            item = QGraphicsPathItem(path)

            item.setPen(self._boundary_style_pen(style))

            item.setZValue(-80)

            self.graphics_scene.addItem(item)

            self.map_items.append(item)

    @staticmethod
    def _boundary_style(
        *,
        lane_types: list[LaneType],
        debug: bool,
    ) -> str | None:
        """
        根据同一几何线段两侧的 lane 类型选择显示语义。
        """
        if not lane_types:
            return None

        if debug:
            return "debug"

        # 只被一条 lane 使用，视为当前 road 的外轮廓。
        if len(lane_types) == 1:
            return "outer"

        unique_types = set(lane_types)

        if unique_types == {LaneType.DRIVING}:
            return "driving_separator"

        if LaneType.SIDEWALK in unique_types:
            return "sidewalk_separator"

        if LaneType.DRIVING in unique_types:
            return "drivable_edge"

        return "secondary_separator"

    @staticmethod
    def _boundary_style_pen(
        style: str,
    ) -> QPen:
        """
        为正常模式或调试模式创建边界画笔。

        这些线仍是基于 lane 几何生成的临时可视化，
        后续解析 roadMark 后应由真实标线替换。
        """
        if style == "outer":
            pen = QPen(
                QColor("#eee9d7"),
                2.2,
                Qt.PenStyle.SolidLine,
            )

        elif style == "driving_separator":
            pen = QPen(
                QColor("#a9aa9f"),
                1.15,
                Qt.PenStyle.DashLine,
            )

            pen.setDashPattern([5.0, 5.0])

        elif style == "drivable_edge":
            pen = QPen(
                QColor("#c5c3b7"),
                1.25,
                Qt.PenStyle.SolidLine,
            )

        elif style == "sidewalk_separator":
            pen = QPen(
                QColor("#9d9990"),
                1.15,
                Qt.PenStyle.SolidLine,
            )

        elif style == "secondary_separator":
            pen = QPen(
                QColor("#85898b"),
                1.0,
                Qt.PenStyle.SolidLine,
            )

        elif style == "debug":
            pen = QPen(
                QColor("#f4f0dc"),
                1.6,
                Qt.PenStyle.SolidLine,
            )

        else:
            raise ValueError(f"Unknown boundary style: {style}")

        pen.setCosmetic(True)

        return pen

    def _draw_lane_centerlines(
        self,
        lanes: tuple[Lane, ...],
    ) -> None:
        """
        绘制 lane centerline 调试层。

        centerline 仅用于地图调试和规划几何检查，
        不代表真实道路标线。
        """
        paths_by_lane_type: dict[
            LaneType,
            QPainterPath,
        ] = {}

        for lane in lanes:
            points = lane.centerline.points

            path = paths_by_lane_type.setdefault(
                lane.lane_type,
                QPainterPath(),
            )

            path.moveTo(
                float(points[0, 0]) * PIXELS_PER_METER,
                -float(points[0, 1]) * PIXELS_PER_METER,
            )

            for point in points[1:]:
                path.lineTo(
                    float(point[0]) * PIXELS_PER_METER,
                    -float(point[1]) * PIXELS_PER_METER,
                )

        for lane_type, path in paths_by_lane_type.items():
            if path.isEmpty():
                continue

            item = QGraphicsPathItem(path)

            item.setPen(self._centerline_pen(lane_type))

            item.setZValue(-70)

            self.graphics_scene.addItem(item)

            self.map_items.append(item)

    def _boundary_segment_key(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
    ]:
        """
        生成与线段方向无关的量化 key。
        """
        scale = 1.0 / self.boundary_quantization

        quantized_a = (
            int(round(float(point_a[0]) * scale)),
            int(round(float(point_a[1]) * scale)),
        )

        quantized_b = (
            int(round(float(point_b[0]) * scale)),
            int(round(float(point_b[1]) * scale)),
        )

        if quantized_a <= quantized_b:
            return (
                quantized_a,
                quantized_b,
            )

        return (
            quantized_b,
            quantized_a,
        )

    @staticmethod
    def _is_junction_connector(
        lane: Lane,
    ) -> bool:
        return bool(
            lane.metadata.get(
                "is_junction_connector",
                False,
            )
        )

    @staticmethod
    def _lane_surface_path(
        lane: Lane,
    ) -> QPainterPath:
        """
        根据左右边界构造封闭车道路面多边形。
        """
        left_points = lane.left_boundary.points

        right_points = lane.right_boundary.points

        polygon_points = np.vstack(
            [
                left_points,
                right_points[::-1],
            ]
        )

        path = QPainterPath()

        first_x = float(polygon_points[0, 0]) * PIXELS_PER_METER

        first_y = -float(polygon_points[0, 1]) * PIXELS_PER_METER

        path.moveTo(
            first_x,
            first_y,
        )

        for point in polygon_points[1:]:
            path.lineTo(
                float(point[0]) * PIXELS_PER_METER,
                -float(point[1]) * PIXELS_PER_METER,
            )

        path.closeSubpath()

        return path

    @staticmethod
    def _lane_surface_brush(
        lane_type: LaneType,
    ) -> QBrush:
        if lane_type is LaneType.DRIVING:
            color = QColor("#35393e")

        elif lane_type is LaneType.SHOULDER:
            color = QColor("#44484c")

        elif lane_type is LaneType.PARKING:
            color = QColor("#4b4f53")

        elif lane_type is LaneType.BIKE:
            color = QColor("#3f5550")

        elif lane_type is LaneType.SIDEWALK:
            color = QColor("#77736d")

        else:
            color = QColor("#45494d")

        return QBrush(color)

    @staticmethod
    def _boundary_pen(
        lane_type: LaneType,
    ) -> QPen:
        if lane_type is LaneType.DRIVING:
            color = QColor("#f4f0dc")

        elif lane_type is LaneType.SIDEWALK:
            color = QColor("#c4beb4")

        else:
            color = QColor("#aab0b5")

        pen = QPen(
            color,
            2.0,
            Qt.PenStyle.SolidLine,
        )

        pen.setCosmetic(True)

        return pen

    @staticmethod
    def _centerline_pen(
        lane_type: LaneType,
    ) -> QPen:
        if lane_type is LaneType.DRIVING:
            color = QColor("#d3c86f")

        else:
            color = QColor("#9da3a8")

        pen = QPen(
            color,
            1.5,
            Qt.PenStyle.DashLine,
        )

        pen.setCosmetic(True)

        return pen

    def _clear_default_road(
        self,
    ) -> None:
        """
        清除临时默认道路图元。
        """
        for item in self.default_road_items:
            if item.scene() is self.graphics_scene:
                self.graphics_scene.removeItem(item)

        self.default_road_items.clear()

    def _clear_map_items(
        self,
    ) -> None:
        """
        清除上一张 RoadNetwork 地图图元。
        """
        for item in self.map_items:
            if item.scene() is self.graphics_scene:
                self.graphics_scene.removeItem(item)

        self.map_items.clear()

    def render_snapshot(
        self,
        snapshot: SimulationSnapshot,
        goal: GoalState,
    ) -> None:
        """
        使用最新 SimulationSnapshot 更新所有动态图元。
        """
        self.vehicle_item.set_state(snapshot.ego)

        self.goal_item.set_state(goal.state)

        self.reference_item.setPath(trajectory_to_path(snapshot.reference_path))

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

    def fit_world(
        self,
    ) -> None:
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

    def mouseMoveEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        world_x, world_y = self.viewport_to_world(event.position().toPoint())

        self.last_world_point = (
            world_x,
            world_y,
        )

        self.world_mouse_moved.emit(
            world_x,
            world_y,
        )

        if (
            self.selection_enabled
            and self.selection_drag_active
            and self.selection_anchor is not None
        ):
            delta_x = world_x - self.selection_anchor[0]
            delta_y = world_y - self.selection_anchor[1]

            drag_length = math.hypot(
                delta_x,
                delta_y,
            )

            if drag_length >= self.selection_drag_threshold:
                self.selection_drag_yaw = math.atan2(
                    delta_y,
                    delta_x,
                )
            else:
                self.selection_drag_yaw = None

            preview_yaw = self.selection_drag_yaw

            if preview_yaw is None:
                preview_yaw = self.selection_anchor_lane_yaw

            if preview_yaw is None:
                preview_yaw = self.selection_default_yaw

            self._show_candidate_preview(
                world_x=self.selection_anchor[0],
                world_y=self.selection_anchor[1],
                yaw=preview_yaw,
            )

            event.accept()
            return

        self._update_candidate_from_world(
            world_x=world_x,
            world_y=world_y,
        )

        super().mouseMoveEvent(event)

    def mousePressEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        if self.selection_enabled and event.button() == Qt.MouseButton.LeftButton:
            world_x, world_y = self.viewport_to_world(event.position().toPoint())

            self.last_world_point = (
                world_x,
                world_y,
            )

            self._update_candidate_from_world(
                world_x=world_x,
                world_y=world_y,
            )

            candidate = self.candidate_world_state()

            if candidate is None:
                event.accept()
                return

            self.selection_drag_active = True
            self.selection_anchor = (
                candidate[0],
                candidate[1],
            )
            self.selection_anchor_lane_yaw = candidate[2]
            self.selection_drag_yaw = None

            self._show_candidate_preview(
                world_x=candidate[0],
                world_y=candidate[1],
                yaw=candidate[2],
            )

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        if (
            self.selection_enabled
            and self.selection_drag_active
            and event.button() == Qt.MouseButton.LeftButton
        ):
            candidate = self.candidate_world_state()

            self.selection_drag_active = False

            if candidate is not None:
                selected_x = candidate[0]
                selected_y = candidate[1]
                selected_yaw = (
                    candidate[2]
                    if candidate[2] is not None
                    else self.selection_default_yaw
                )

                self.world_pose_selected.emit(
                    selected_x,
                    selected_y,
                    selected_yaw,
                )

                # 保留旧信号，兼容尚未升级的调用方。
                self.world_point_clicked.emit(
                    selected_x,
                    selected_y,
                )

            self.selection_anchor = None
            self.selection_anchor_lane_yaw = None
            self.selection_drag_yaw = None

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(
        self,
        event,
    ) -> None:
        if not self.selection_drag_active:
            self._hide_candidate_preview()

        super().leaveEvent(event)

    def wheelEvent(
        self,
        event,
    ) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15

        transform = self.transform()

        current_scale = math.hypot(
            transform.m11(),
            transform.m12(),
        )

        if factor > 1.0 and current_scale > 5.0:
            event.accept()
            return

        if factor < 1.0 and current_scale < 0.15:
            event.accept()
            return

        self.scale(
            factor,
            factor,
        )

        event.accept()
