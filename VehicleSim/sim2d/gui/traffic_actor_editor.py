from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sim2d.core.environment import DrivingEnv
from sim2d.gui.main_window import MainWindow
from sim2d.world.traffic_actor import (
    TrafficActorType,
    create_traffic_actor,
)

_INSTALLED = False


_TYPE_LABELS = {
    TrafficActorType.PEDESTRIAN.value: "行人",
    TrafficActorType.SMALL_CAR.value: "小车",
    TrafficActorType.LARGE_VEHICLE.value: "大车",
}


def install() -> None:
    """安装代码配置接口和 GUI 交通参与者放置工具。"""
    global _INSTALLED
    if _INSTALLED:
        return

    _install_environment_api()
    _install_main_window_editor()
    _INSTALLED = True


def _install_environment_api() -> None:
    original_post_init = DrivingEnv.__post_init__
    original_reset = DrivingEnv.reset

    def post_init(self: DrivingEnv) -> None:
        original_post_init(self)
        self._configured_traffic_actors = None

    def reset(self: DrivingEnv, *args, **kwargs):
        configured = getattr(self, "_configured_traffic_actors", None)
        if configured is not None:
            if "obstacles" in kwargs:
                kwargs["obstacles"] = tuple(configured)
            elif len(args) >= 3:
                args = (*args[:2], tuple(configured), *args[3:])
            else:
                kwargs["obstacles"] = tuple(configured)
        return original_reset(self, *args, **kwargs)

    def set_traffic_actors(self: DrivingEnv, actors) -> None:
        actors = tuple(actors)
        ids = [actor.obstacle_id for actor in actors]
        if len(ids) != len(set(ids)):
            raise ValueError("traffic actor ids must be unique")
        for actor in actors:
            actor.validate()
        self._configured_traffic_actors = actors

    DrivingEnv.__post_init__ = post_init
    DrivingEnv.reset = reset
    DrivingEnv.set_traffic_actors = set_traffic_actors


def _install_main_window_editor() -> None:
    original_init = MainWindow.__init__

    def init(self: MainWindow, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)

        self._traffic_actor_specs = []
        self._actor_placement_active = False
        self._actor_sequence = 0

        dock = QDockWidget("交通参与者", self)
        dock.setObjectName("traffic_actor_editor_dock")
        panel = QWidget(dock)
        root = QVBoxLayout(panel)

        help_label = QLabel(
            "选择类型和速度后点击“拖拽放置”，\n"
            "在世界视图按下确定位置，拖动确定初始方向。"
        )
        help_label.setWordWrap(True)
        root.addWidget(help_label)

        form = QFormLayout()
        self.traffic_actor_type_combo = QComboBox(panel)
        for actor_type in TrafficActorType:
            self.traffic_actor_type_combo.addItem(
                _TYPE_LABELS[actor_type.value], actor_type.value
            )
        form.addRow("类型", self.traffic_actor_type_combo)

        self.traffic_actor_speed_spin = QDoubleSpinBox(panel)
        self.traffic_actor_speed_spin.setRange(0.0, 40.0)
        self.traffic_actor_speed_spin.setDecimals(2)
        self.traffic_actor_speed_spin.setSuffix(" m/s")
        self.traffic_actor_speed_spin.setValue(0.0)
        form.addRow("速度", self.traffic_actor_speed_spin)

        self.traffic_actor_yaw_spin = QDoubleSpinBox(panel)
        self.traffic_actor_yaw_spin.setRange(-180.0, 180.0)
        self.traffic_actor_yaw_spin.setDecimals(1)
        self.traffic_actor_yaw_spin.setSuffix("°")
        form.addRow("初始方向", self.traffic_actor_yaw_spin)
        root.addLayout(form)

        buttons = QHBoxLayout()
        self.place_traffic_actor_button = QPushButton("拖拽放置", panel)
        self.delete_traffic_actor_button = QPushButton("删除选中", panel)
        buttons.addWidget(self.place_traffic_actor_button)
        buttons.addWidget(self.delete_traffic_actor_button)
        root.addLayout(buttons)

        self.clear_traffic_actors_button = QPushButton("清空全部", panel)
        root.addWidget(self.clear_traffic_actors_button)

        self.traffic_actor_list = QListWidget(panel)
        root.addWidget(self.traffic_actor_list, 1)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.traffic_actor_editor_dock = dock

        self.place_traffic_actor_button.clicked.connect(
            lambda: _begin_placement(self)
        )
        self.delete_traffic_actor_button.clicked.connect(
            lambda: _delete_selected(self)
        )
        self.clear_traffic_actors_button.clicked.connect(
            lambda: _clear_all(self)
        )
        self.simulation_view.world_pose_selected.connect(
            lambda x, y, yaw: _finish_placement(self, x, y, yaw)
        )

        # 使用语义默认场景替换旧的圆形/矩形 demo。
        ego = self.selected_initial_state
        self._traffic_actor_specs = [
            _relative_actor(
                TrafficActorType.PEDESTRIAN,
                "pedestrian_001",
                ego,
                12.0,
                0.0,
                0.0,
                0.0,
            ),
            _relative_actor(
                TrafficActorType.SMALL_CAR,
                "small_car_001",
                ego,
                20.0,
                -2.4,
                0.2,
                0.0,
            ),
        ]
        self._actor_sequence = 2
        _apply_actors(self, reset=True)

    MainWindow.__init__ = init


def _relative_actor(actor_type, actor_id, ego, local_x, local_y, relative_yaw, speed):
    c = math.cos(ego.yaw)
    s = math.sin(ego.yaw)
    return create_traffic_actor(
        actor_type,
        actor_id=actor_id,
        x=ego.x + c * local_x - s * local_y,
        y=ego.y + s * local_x + c * local_y,
        yaw=ego.yaw + relative_yaw,
        speed=speed,
    )


def _begin_placement(window: MainWindow) -> None:
    window._actor_placement_active = True
    default_yaw = math.radians(window.traffic_actor_yaw_spin.value())
    window.simulation_view.set_selection_default_yaw(default_yaw)
    window.simulation_view.set_lane_snapping(enabled=False)
    window.simulation_view.set_selection_enabled(True)
    window.place_traffic_actor_button.setText("请在世界中拖拽…")
    window.append_log("TRAFFIC_ACTOR_PLACE_BEGIN")


def _finish_placement(window: MainWindow, x: float, y: float, dragged_yaw: float) -> None:
    if not window._actor_placement_active:
        return

    window._actor_placement_active = False
    window.simulation_view.set_selection_enabled(False)
    window.place_traffic_actor_button.setText("拖拽放置")

    actor_type = TrafficActorType(window.traffic_actor_type_combo.currentData())
    window._actor_sequence += 1
    actor_id = f"{actor_type.value}_{window._actor_sequence:03d}"

    actor = create_traffic_actor(
        actor_type,
        actor_id=actor_id,
        x=x,
        y=y,
        yaw=dragged_yaw,
        speed=window.traffic_actor_speed_spin.value(),
    )
    window._traffic_actor_specs.append(actor)
    _apply_actors(window, reset=True)
    window.append_log(
        "TRAFFIC_ACTOR_ADDED "
        f"id={actor_id} type={actor_type.value} "
        f"x={x:.2f} y={y:.2f} yaw={dragged_yaw:.3f} "
        f"speed={actor.speed:.2f}"
    )


def _delete_selected(window: MainWindow) -> None:
    row = window.traffic_actor_list.currentRow()
    if row < 0 or row >= len(window._traffic_actor_specs):
        return
    removed = window._traffic_actor_specs.pop(row)
    _apply_actors(window, reset=True)
    window.append_log(f"TRAFFIC_ACTOR_REMOVED id={removed.obstacle_id}")


def _clear_all(window: MainWindow) -> None:
    window._traffic_actor_specs.clear()
    _apply_actors(window, reset=True)
    window.append_log("TRAFFIC_ACTORS_CLEARED")


def _apply_actors(window: MainWindow, *, reset: bool) -> None:
    window.env.set_traffic_actors(tuple(window._traffic_actor_specs))
    _refresh_list(window)
    if reset:
        window.reset_environment()


def _refresh_list(window: MainWindow) -> None:
    window.traffic_actor_list.clear()
    for actor in window._traffic_actor_specs:
        label = _TYPE_LABELS.get(actor.semantic_type, actor.semantic_type)
        window.traffic_actor_list.addItem(
            f"{actor.obstacle_id} | {label} | "
            f"v={actor.speed:.2f} m/s | yaw={math.degrees(actor.yaw):.1f}°"
        )


__all__ = ["install"]
