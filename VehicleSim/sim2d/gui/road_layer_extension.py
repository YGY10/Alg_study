from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPathItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sim2d.gui.graphics_items import PIXELS_PER_METER
from sim2d.gui.main_window import MainWindow
from sim2d.gui.simulation_view import SimulationView
from sim2d.gui.traffic_signal_graphics import TrafficSignalGraphicsItem
from sim2d.map import LaneType
from sim2d.world import (
    RoadDeformationConfig,
    TrafficLightState,
    WorldLaneGeometry,
)

_INSTALLED = False


def install() -> None:
    """为现有 GUI 安装连续道路形变与图层对比功能。"""
    global _INSTALLED

    if _INSTALLED:
        return

    _install_simulation_view_extension()
    _install_main_window_extension()
    _INSTALLED = True


def _install_simulation_view_extension() -> None:
    original_init = SimulationView.__init__
    original_render_road_network = SimulationView.render_road_network

    def extended_init(
        self: SimulationView,
        *args,
        **kwargs,
    ) -> None:
        original_init(
            self,
            *args,
            **kwargs,
        )

        self.road_layer_mode = "world"
        self.map_road_items: list[QGraphicsItem] = []
        self.world_road_items: list[QGraphicsItem] = []
        self._world_road_signature: tuple[object, ...] | None = None

    def extended_render_road_network(
        self: SimulationView,
        road_network,
    ) -> None:
        original_render_road_network(
            self,
            road_network,
        )

        self.map_road_items = [
            item
            for item in self.map_items
            if not isinstance(
                item,
                TrafficSignalGraphicsItem,
            )
        ]

        _apply_road_layer_visibility(self)

    SimulationView.__init__ = extended_init
    SimulationView.render_road_network = extended_render_road_network
    SimulationView.set_road_layer_mode = _set_road_layer_mode
    SimulationView.set_world_road_lanes = _set_world_road_lanes
    SimulationView.clear_world_road_lanes = _clear_world_road_lanes


def _set_road_layer_mode(
    self: SimulationView,
    mode: str,
) -> None:
    normalized = str(mode).strip().lower()

    if normalized not in {
        "world",
        "map",
        "compare",
    }:
        raise ValueError(
            f"Unsupported road layer mode: {mode!r}"
        )

    self.road_layer_mode = normalized
    _apply_road_layer_visibility(self)


def _apply_road_layer_visibility(
    view: SimulationView,
) -> None:
    mode = getattr(
        view,
        "road_layer_mode",
        "world",
    )

    show_map = mode in {
        "map",
        "compare",
    }
    show_world = mode in {
        "world",
        "compare",
    }

    for item in getattr(
        view,
        "map_road_items",
        (),
    ):
        item.setVisible(show_map)
        item.setOpacity(
            0.28 if mode == "compare" else 1.0
        )

    for item in getattr(
        view,
        "world_road_items",
        (),
    ):
        item.setVisible(show_world)
        item.setOpacity(1.0)


def _set_world_road_lanes(
    self: SimulationView,
    lanes: tuple[WorldLaneGeometry, ...],
) -> None:
    signature: tuple[object, ...] = tuple(
        (
            lane.entity_id,
            lane.centerline.shape[0],
            float(lane.centerline[0, 0]),
            float(lane.centerline[0, 1]),
            float(lane.centerline[-1, 0]),
            float(lane.centerline[-1, 1]),
        )
        for lane in lanes
    )

    if signature == getattr(
        self,
        "_world_road_signature",
        None,
    ):
        _apply_road_layer_visibility(self)
        return

    _clear_world_road_lanes(self)
    self._world_road_signature = signature

    for lane in lanes:
        _draw_world_lane(
            self,
            lane,
        )

    _apply_road_layer_visibility(self)


def _clear_world_road_lanes(
    self: SimulationView,
) -> None:
    for item in getattr(
        self,
        "world_road_items",
        (),
    ):
        if item.scene() is self.graphics_scene:
            self.graphics_scene.removeItem(item)

    self.world_road_items.clear()
    self._world_road_signature = None


def _draw_world_lane(
    view: SimulationView,
    lane: WorldLaneGeometry,
) -> None:
    polygon_points = np.vstack(
        [
            lane.left_boundary,
            lane.right_boundary[::-1],
        ]
    )

    surface_path = _points_to_path(
        polygon_points,
        close=True,
    )

    surface_item = QGraphicsPathItem(
        surface_path
    )
    surface_item.setBrush(
        _world_lane_brush(lane.lane_type)
    )
    surface_item.setPen(
        QPen(Qt.PenStyle.NoPen)
    )
    surface_item.setZValue(-99.0)
    view.graphics_scene.addItem(surface_item)
    view.world_road_items.append(surface_item)

    boundary_path = QPainterPath()

    for boundary in (
        lane.left_boundary,
        lane.right_boundary,
    ):
        _append_points_to_path(
            boundary_path,
            boundary,
        )

    boundary_item = QGraphicsPathItem(
        boundary_path
    )

    boundary_pen = QPen(
        QColor("#f4f0dc"),
        2.2,
        Qt.PenStyle.SolidLine,
    )
    boundary_pen.setCosmetic(True)

    boundary_item.setPen(boundary_pen)
    boundary_item.setZValue(-79.0)
    view.graphics_scene.addItem(boundary_item)
    view.world_road_items.append(boundary_item)


def _points_to_path(
    points: np.ndarray,
    *,
    close: bool,
) -> QPainterPath:
    path = QPainterPath()
    _append_points_to_path(
        path,
        points,
    )

    if close:
        path.closeSubpath()

    return path


def _append_points_to_path(
    path: QPainterPath,
    points: np.ndarray,
) -> None:
    path.moveTo(
        float(points[0, 0]) * PIXELS_PER_METER,
        -float(points[0, 1]) * PIXELS_PER_METER,
    )

    for point in points[1:]:
        path.lineTo(
            float(point[0]) * PIXELS_PER_METER,
            -float(point[1]) * PIXELS_PER_METER,
        )


def _world_lane_brush(
    lane_type: LaneType,
) -> QBrush:
    if lane_type is LaneType.DRIVING:
        color = QColor("#30353a")
    elif lane_type is LaneType.SHOULDER:
        color = QColor("#40464b")
    elif lane_type is LaneType.PARKING:
        color = QColor("#474d52")
    elif lane_type is LaneType.BIKE:
        color = QColor("#39524c")
    elif lane_type is LaneType.SIDEWALK:
        color = QColor("#706c66")
    else:
        color = QColor("#41474c")

    return QBrush(color)


def _install_main_window_extension() -> None:
    original_init = MainWindow.__init__
    original_reset_environment = MainWindow.reset_environment

    def extended_init(
        self: MainWindow,
        *args,
        **kwargs,
    ) -> None:
        original_init(
            self,
            *args,
            **kwargs,
        )

        _build_road_deformation_dock(self)
        _apply_world_map_geometry(self)

    def extended_reset_environment(
        self: MainWindow,
    ) -> None:
        original_reset_environment(self)

        if hasattr(
            self,
            "road_layer_combo",
        ):
            _apply_world_map_geometry(self)

    MainWindow.__init__ = extended_init
    MainWindow.reset_environment = extended_reset_environment


def _build_road_deformation_dock(
    window: MainWindow,
) -> None:
    dock = QDockWidget(
        "道路 Map / World",
        window,
    )
    dock.setObjectName(
        "road_deformation_dock"
    )

    panel = QWidget(dock)
    layout = QVBoxLayout(panel)
    form = QFormLayout()

    window.road_layer_combo = QComboBox()
    window.road_layer_combo.addItem(
        "真实世界道路",
        "world",
    )
    window.road_layer_combo.addItem(
        "地图道路",
        "map",
    )
    window.road_layer_combo.addItem(
        "道路对比显示",
        "compare",
    )

    window.road_offset_x_spin = _make_spin(
        -100.0,
        100.0,
        0.8,
        0.1,
        " m",
    )
    window.road_offset_y_spin = _make_spin(
        -100.0,
        100.0,
        0.4,
        0.1,
        " m",
    )
    window.road_yaw_spin = _make_spin(
        -180.0,
        180.0,
        2.86,
        0.1,
        "°",
    )
    window.road_longitudinal_scale_spin = _make_spin(
        0.5,
        1.5,
        1.0,
        0.005,
        "",
        decimals=4,
    )
    window.road_lateral_scale_spin = _make_spin(
        0.5,
        1.5,
        1.0,
        0.005,
        "",
        decimals=4,
    )
    window.road_local_longitudinal_spin = _make_spin(
        -20.0,
        20.0,
        0.3,
        0.1,
        " m",
    )
    window.road_local_lateral_spin = _make_spin(
        -20.0,
        20.0,
        0.8,
        0.1,
        " m",
    )
    window.road_wavelength_spin = _make_spin(
        5.0,
        1000.0,
        80.0,
        5.0,
        " m",
        decimals=2,
    )

    form.addRow(
        "道路图层",
        window.road_layer_combo,
    )
    form.addRow(
        "整体偏差 x",
        window.road_offset_x_spin,
    )
    form.addRow(
        "整体偏差 y",
        window.road_offset_y_spin,
    )
    form.addRow(
        "整体偏差 yaw",
        window.road_yaw_spin,
    )
    form.addRow(
        "纵向缩放",
        window.road_longitudinal_scale_spin,
    )
    form.addRow(
        "横向缩放",
        window.road_lateral_scale_spin,
    )
    form.addRow(
        "局部纵向形变",
        window.road_local_longitudinal_spin,
    )
    form.addRow(
        "局部横向形变",
        window.road_local_lateral_spin,
    )
    form.addRow(
        "形变波长",
        window.road_wavelength_spin,
    )

    layout.addLayout(form)

    apply_button = QPushButton(
        "应用道路偏差"
    )
    layout.addWidget(apply_button)
    layout.addStretch(1)

    dock.setWidget(panel)
    window.addDockWidget(
        Qt.DockWidgetArea.RightDockWidgetArea,
        dock,
    )

    window.road_layer_combo.currentIndexChanged.connect(
        lambda _: _on_road_layer_changed(window)
    )
    apply_button.clicked.connect(
        lambda: _apply_world_map_geometry(window)
    )

    window.road_deformation_dock = dock


def _make_spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    suffix: str,
    *,
    decimals: int = 3,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(
        minimum,
        maximum,
    )
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def _on_road_layer_changed(
    window: MainWindow,
) -> None:
    mode = window.road_layer_combo.currentData()
    window.simulation_view.set_road_layer_mode(
        mode
    )
    window.append_log(
        f"ROAD_LAYER mode={mode}"
    )


def _road_deformation_config(
    window: MainWindow,
) -> RoadDeformationConfig:
    return RoadDeformationConfig(
        offset_x=window.road_offset_x_spin.value(),
        offset_y=window.road_offset_y_spin.value(),
        yaw_offset=math.radians(
            window.road_yaw_spin.value()
        ),
        longitudinal_scale=(
            window.road_longitudinal_scale_spin.value()
        ),
        lateral_scale=(
            window.road_lateral_scale_spin.value()
        ),
        local_longitudinal_amplitude=(
            window.road_local_longitudinal_spin.value()
        ),
        local_lateral_amplitude=(
            window.road_local_lateral_spin.value()
        ),
        local_wavelength=(
            window.road_wavelength_spin.value()
        ),
    )


def _apply_world_map_geometry(
    window: MainWindow,
) -> None:
    if window.road_network is None:
        window.simulation_view.clear_world_road_lanes()
        return

    if not window.env.is_initialized:
        return

    deformation = _road_deformation_config(
        window
    )

    window.env.world.initialize_map_geometry(
        window.road_network,
        deformation,
        initial_signal_state=TrafficLightState.RED,
    )

    window.simulation_view.set_world_road_lanes(
        window.env.world.state.road_lanes
    )

    window.simulation_view.render_snapshot(
        window.env.get_snapshot(),
        window.goal,
    )

    mode = window.road_layer_combo.currentData()
    window.simulation_view.set_road_layer_mode(
        mode
    )

    window.append_log(
        "ROAD_DEFORMATION_APPLIED "
        f"dx={deformation.offset_x:.3f} "
        f"dy={deformation.offset_y:.3f} "
        f"dyaw={deformation.yaw_offset:.5f} "
        f"scale_s={deformation.longitudinal_scale:.4f} "
        f"scale_d={deformation.lateral_scale:.4f} "
        f"amp_s={deformation.local_longitudinal_amplitude:.3f} "
        f"amp_d={deformation.local_lateral_amplitude:.3f} "
        f"wavelength={deformation.local_wavelength:.3f}"
    )
