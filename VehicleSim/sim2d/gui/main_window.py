from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QFocusEvent,
    QFont,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sim2d.control import (
    KeyboardControlState,
    ManualControlMapper,
)
from sim2d.core import (
    DrivingEnv,
    EnvironmentConfig,
)
from sim2d.gui.simulation_view import (
    SimulationView,
)
from sim2d.map import (
    RoadNetwork,
    load_opendrive_road_network,
)
from sim2d.map.route import (
    LaneRoute,
    NoRouteError,
    build_lane_route,
)
from sim2d.planning import (
    BezierPlanner,
)
from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    GoalState,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)

from sim2d.world import TrafficLightState


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.settings = QSettings(
            "VehicleSim",
            "VehicleSim2D",
        )

        self.setWindowTitle("VehicleSim 2D")

        self.resize(
            1400,
            850,
        )

        # 允许窗口自由缩小；左侧内容不足时由滚动区域承载。
        self.setMinimumSize(
            820,
            560,
        )

        self.vehicle_config = VehicleConfig(
            length=4.6,
            width=1.85,
            wheel_base=2.7,
            acceleration_min=-3.0,
            acceleration_max=2.0,
            steering_min=-0.45,
            steering_max=0.45,
            speed_min=0.0,
            speed_max=15.0,
        )

        self.environment_config = EnvironmentConfig(
            dt=0.05,
            max_time=20.0,
        )

        self.env = DrivingEnv(
            vehicle_config=(self.vehicle_config),
            environment_config=(self.environment_config),
        )

        self.planner = BezierPlanner(
            vehicle_config=(self.vehicle_config),
            target_speed=4.0,
            speed_gain=1.5,
            braking_deceleration=2.5,
            path_sample_count=201,
            handle_scale=0.35,
            minimum_handle_length=2.0,
            maximum_handle_length=12.0,
            lookahead_base=2.0,
            lookahead_speed_gain=0.45,
            minimum_lookahead=1.5,
            stop_target_radius=0.10,
            braking_margin=0.20,
            prediction_dt=0.1,
            prediction_steps=50,
        )

        self.manual_control_mapper = ManualControlMapper(
            vehicle_config=(self.vehicle_config),
        )

        self.keyboard_control = KeyboardControlState()

        self.goal = GoalState(
            state=VehicleState(
                x=25.0,
                y=-6.0,
                yaw=0.0,
                speed=1.0,
            ),
            position_tolerance=0.5,
            yaw_tolerance=0.2,
            speed_tolerance=0.5,
        )

        self.simulation_view = SimulationView(
            vehicle_config=(self.vehicle_config),
        )

        self.timer = QTimer(self)

        self.timer.setInterval(int(self.environment_config.dt * 1000.0))

        self.timer.timeout.connect(self.advance_one_step)

        self.running = False

        self.current_map_path: Path | None = None
        self.road_network: RoadNetwork | None = None
        self.current_lane_route: LaneRoute | None = None
        self.map_sample_step = 0.5

        # 场景起点/终点编辑状态。
        #
        # selection_mode:
        #     None    浏览模式
        #     "start" 选择起点
        #     "goal"  选择终点
        self.selection_mode: str | None = None

        self.selected_initial_state = VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=2.0,
        )

        self.selected_goal_state = self.goal.state

        self._build_ui()
        self._create_actions()

        self.simulation_view.world_mouse_moved.connect(self._on_world_mouse_moved)

        self.simulation_view.world_pose_selected.connect(self._on_world_pose_selected)

        self.simulation_view.lane_projection_changed.connect(
            self._on_lane_projection_changed
        )

        self.reset_environment()

        QTimer.singleShot(
            0,
            self._restore_last_opendrive_map,
        )

    def _build_ui(
        self,
    ) -> None:
        root_splitter = QSplitter(Qt.Orientation.Horizontal)

        control_panel = QWidget()

        control_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        control_layout = QVBoxLayout(control_panel)

        title = QLabel("VehicleSim 控制台")

        title.setFont(
            QFont(
                "Sans Serif",
                16,
                QFont.Weight.Bold,
            )
        )

        control_layout.addWidget(title)

        simulation_group = QGroupBox("仿真控制")

        simulation_layout = QVBoxLayout(simulation_group)

        buttons = QHBoxLayout()

        self.run_button = QPushButton("运行")

        self.pause_button = QPushButton("暂停")

        self.step_button = QPushButton("单步")

        self.reset_button = QPushButton("复位")

        buttons.addWidget(self.run_button)

        buttons.addWidget(self.pause_button)

        buttons.addWidget(self.step_button)

        buttons.addWidget(self.reset_button)

        simulation_layout.addLayout(buttons)

        self.run_button.clicked.connect(self.start_simulation)

        self.pause_button.clicked.connect(self.pause_simulation)

        self.step_button.clicked.connect(self.single_step)

        self.reset_button.clicked.connect(self.reset_environment)

        control_form = QFormLayout()

        self.control_mode_combo = QComboBox()

        self.control_mode_combo.addItem(
            "手动控制",
            "manual",
        )

        self.control_mode_combo.addItem(
            "自动规划",
            "auto",
        )

        # 默认自动规划。
        self.control_mode_combo.setCurrentIndex(1)

        self.control_mode_combo.currentIndexChanged.connect(
            self._on_control_mode_changed
        )

        control_form.addRow(
            "控制模式",
            self.control_mode_combo,
        )

        self.manual_input_combo = QComboBox()

        self.manual_input_combo.addItem(
            "精确输入",
            "numeric",
        )

        self.manual_input_combo.addItem(
            "键盘驾驶",
            "keyboard",
        )

        # 默认键盘驾驶。
        self.manual_input_combo.setCurrentIndex(1)

        self.manual_input_combo.currentIndexChanged.connect(
            self._on_manual_input_mode_changed
        )

        control_form.addRow(
            "手动输入方式",
            self.manual_input_combo,
        )

        self.acceleration_spin = QDoubleSpinBox()

        self.acceleration_spin.setRange(
            self.vehicle_config.acceleration_min,
            self.vehicle_config.acceleration_max,
        )

        self.acceleration_spin.setDecimals(2)

        self.acceleration_spin.setSingleStep(0.1)

        self.acceleration_spin.setValue(0.0)

        self.steering_spin = QDoubleSpinBox()

        self.steering_spin.setRange(
            self.vehicle_config.steering_min,
            self.vehicle_config.steering_max,
        )

        self.steering_spin.setDecimals(3)

        self.steering_spin.setSingleStep(0.02)

        self.steering_spin.setValue(0.0)

        control_form.addRow(
            "加速度 m/s²",
            self.acceleration_spin,
        )

        control_form.addRow(
            "前轮转角 rad",
            self.steering_spin,
        )

        self.keyboard_steering_label = QLabel("+0.000")

        self.keyboard_throttle_label = QLabel("0.000")

        self.keyboard_brake_label = QLabel("0.000")

        control_form.addRow(
            "方向盘输入",
            self.keyboard_steering_label,
        )

        control_form.addRow(
            "油门踏板",
            self.keyboard_throttle_label,
        )

        control_form.addRow(
            "制动踏板",
            self.keyboard_brake_label,
        )

        simulation_layout.addLayout(control_form)

        control_layout.addWidget(simulation_group)

        scenario_group = QGroupBox("场景起点 / 终点")

        scenario_layout = QVBoxLayout(scenario_group)

        selection_buttons = QHBoxLayout()

        self.select_start_button = QPushButton("选择起点")
        self.select_goal_button = QPushButton("选择终点")
        self.cancel_selection_button = QPushButton("取消选择")

        selection_buttons.addWidget(self.select_start_button)
        selection_buttons.addWidget(self.select_goal_button)
        selection_buttons.addWidget(self.cancel_selection_button)

        scenario_layout.addLayout(selection_buttons)

        self.select_start_button.clicked.connect(self.begin_start_selection)

        self.select_goal_button.clicked.connect(self.begin_goal_selection)

        self.cancel_selection_button.clicked.connect(self.cancel_point_selection)

        self.snap_to_lane_check = QCheckBox("吸附到可行驶车道")
        self.snap_to_lane_check.setChecked(True)
        self.snap_to_lane_check.toggled.connect(self._on_lane_snap_changed)

        self.snap_distance_spin = QDoubleSpinBox()
        self.snap_distance_spin.setRange(0.0, 100.0)
        self.snap_distance_spin.setDecimals(2)
        self.snap_distance_spin.setSingleStep(0.25)
        self.snap_distance_spin.setValue(3.0)
        self.snap_distance_spin.setSuffix(" m")
        self.snap_distance_spin.valueChanged.connect(
            self._on_lane_snap_distance_changed
        )

        scenario_form = QFormLayout()

        scenario_form.addRow(self.snap_to_lane_check)

        scenario_form.addRow(
            "吸附半径",
            self.snap_distance_spin,
        )

        self.start_x_spin = self._create_coordinate_spin()
        self.start_y_spin = self._create_coordinate_spin()
        self.start_yaw_spin = self._create_yaw_spin()
        self.start_speed_spin = self._create_speed_spin(2.0)

        self.goal_x_spin = self._create_coordinate_spin()
        self.goal_y_spin = self._create_coordinate_spin()
        self.goal_yaw_spin = self._create_yaw_spin()
        self.goal_speed_spin = self._create_speed_spin(1.0)

        scenario_form.addRow("起点 x", self.start_x_spin)
        scenario_form.addRow("起点 y", self.start_y_spin)
        scenario_form.addRow("起点 yaw", self.start_yaw_spin)
        scenario_form.addRow("起点 speed", self.start_speed_spin)

        scenario_form.addRow("终点 x", self.goal_x_spin)
        scenario_form.addRow("终点 y", self.goal_y_spin)
        scenario_form.addRow("终点 yaw", self.goal_yaw_spin)
        scenario_form.addRow("终点 speed", self.goal_speed_spin)

        scenario_layout.addLayout(scenario_form)

        self.apply_scenario_button = QPushButton("应用起点和终点")

        self.apply_scenario_button.clicked.connect(self.apply_scenario_from_inputs)

        scenario_layout.addWidget(self.apply_scenario_button)

        self.selection_status_label = QLabel("当前：浏览模式")
        self.selection_status_label.setWordWrap(True)

        scenario_layout.addWidget(self.selection_status_label)

        control_layout.addWidget(scenario_group)

        status_group = QGroupBox("车辆状态")

        status_form = QFormLayout(status_group)

        self.frame_label = QLabel("0")

        self.time_label = QLabel("0.00 s")

        self.x_label = QLabel("0.00 m")

        self.y_label = QLabel("0.00 m")

        self.yaw_label = QLabel("0.00°")

        self.speed_label = QLabel("0.00 m/s")

        self.clearance_label = QLabel("--")

        self.status_label = QLabel("未开始")

        status_form.addRow(
            "frame",
            self.frame_label,
        )

        status_form.addRow(
            "time",
            self.time_label,
        )

        status_form.addRow(
            "x",
            self.x_label,
        )

        status_form.addRow(
            "y",
            self.y_label,
        )

        status_form.addRow(
            "yaw",
            self.yaw_label,
        )

        status_form.addRow(
            "speed",
            self.speed_label,
        )

        status_form.addRow(
            "clearance",
            self.clearance_label,
        )

        status_form.addRow(
            "状态",
            self.status_label,
        )

        control_layout.addWidget(status_group)

        view_group = QGroupBox("视图")

        view_layout = QVBoxLayout(view_group)

        self.load_map_button = QPushButton("加载 OpenDRIVE 地图")

        self.load_map_button.clicked.connect(self.select_opendrive_map)

        view_layout.addWidget(self.load_map_button)

        self.map_name_label = QLabel("当前地图：默认道路")

        self.map_name_label.setWordWrap(True)

        view_layout.addWidget(self.map_name_label)

        self.show_lane_boundaries_check = QCheckBox("显示全部车道边界（调试）")
        self.show_lane_boundaries_check.setChecked(False)
        self.show_lane_boundaries_check.toggled.connect(
            self._on_show_lane_boundaries_changed
        )

        self.show_lane_centerlines_check = QCheckBox("显示车道中心线（调试）")
        self.show_lane_centerlines_check.setChecked(False)
        self.show_lane_centerlines_check.toggled.connect(
            self._on_show_lane_centerlines_changed
        )

        self.show_traffic_signals_check = QCheckBox("显示交通灯")
        self.show_traffic_signals_check.setChecked(True)
        self.show_traffic_signals_check.toggled.connect(
            self._on_show_traffic_signals_changed
        )

        self.traffic_signal_layer_combo = QComboBox()
        self.traffic_signal_layer_combo.addItem(
            "真实交通灯",
            "world",
        )
        self.traffic_signal_layer_combo.addItem(
            "地图交通灯",
            "map",
        )
        self.traffic_signal_layer_combo.addItem(
            "对比显示",
            "compare",
        )
        self.traffic_signal_layer_combo.setCurrentIndex(0)
        self.traffic_signal_layer_combo.currentIndexChanged.connect(
            self._on_traffic_signal_layer_mode_changed
        )

        self.show_traffic_signal_ids_check = QCheckBox("显示交通灯 ID（调试）")
        self.show_traffic_signal_ids_check.setChecked(False)
        self.show_traffic_signal_ids_check.toggled.connect(
            self._on_show_traffic_signal_ids_changed
        )

        view_layout.addWidget(self.show_lane_boundaries_check)
        view_layout.addWidget(self.show_lane_centerlines_check)
        view_layout.addWidget(self.show_traffic_signals_check)

        traffic_signal_layer_form = QFormLayout()
        traffic_signal_layer_form.addRow(
            "交通灯图层",
            self.traffic_signal_layer_combo,
        )
        view_layout.addLayout(traffic_signal_layer_form)

        view_layout.addWidget(self.show_traffic_signal_ids_check)

        fit_button = QPushButton("适配整个场景")

        fit_button.clicked.connect(self.simulation_view.fit_world)

        view_layout.addWidget(fit_button)

        control_layout.addWidget(view_group)

        self.hint_label = QLabel()

        self.hint_label.setWordWrap(True)

        self.hint_label.setStyleSheet(
            "background: #edf2f7;"
            "border: 1px solid #c6d0da;"
            "padding: 10px;"
            "border-radius: 5px;"
        )

        control_layout.addWidget(self.hint_label)

        control_layout.addStretch(1)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        right_splitter.addWidget(self.simulation_view)

        self.log_console = QPlainTextEdit()

        self.log_console.setReadOnly(True)

        self.log_console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.log_console.setFont(
            QFont(
                "Monospace",
                9,
            )
        )

        self.log_console.setStyleSheet(
            "QPlainTextEdit {"
            "background: #111820;"
            "color: #d9e4ec;"
            "border: 1px solid #45515d;"
            "padding: 6px;"
            "}"
        )

        self.log_console.document().setMaximumBlockCount(500)

        right_splitter.addWidget(self.log_console)

        right_splitter.setStretchFactor(
            0,
            4,
        )

        right_splitter.setStretchFactor(
            1,
            1,
        )

        right_splitter.setSizes(
            [
                650,
                180,
            ]
        )

        control_scroll = QScrollArea()

        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        control_scroll.setMinimumWidth(270)
        control_scroll.setWidget(control_panel)

        root_splitter.addWidget(control_scroll)
        root_splitter.addWidget(right_splitter)

        root_splitter.setStretchFactor(
            0,
            0,
        )

        root_splitter.setStretchFactor(
            1,
            1,
        )

        root_splitter.setSizes(
            [
                360,
                1040,
            ]
        )

        central = QWidget()

        layout = QVBoxLayout(central)

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.addWidget(root_splitter)

        self.setCentralWidget(central)

        self._sync_scenario_inputs_from_states()
        self._update_keyboard_input_display()
        self._update_control_mode_ui()

    def _create_coordinate_spin(
        self,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()

        spin.setRange(
            -1_000_000.0,
            1_000_000.0,
        )

        spin.setDecimals(3)
        spin.setSingleStep(0.25)
        spin.setSuffix(" m")

        return spin

    def _create_yaw_spin(
        self,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()

        spin.setRange(
            -360.0,
            360.0,
        )

        spin.setDecimals(2)
        spin.setSingleStep(5.0)
        spin.setSuffix("°")

        return spin

    def _create_speed_spin(
        self,
        value: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()

        spin.setRange(
            self.vehicle_config.speed_min,
            self.vehicle_config.speed_max,
        )

        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setSuffix(" m/s")
        spin.setValue(value)

        return spin

    def _sync_scenario_inputs_from_states(
        self,
    ) -> None:
        start = self.selected_initial_state
        goal = self.selected_goal_state

        self.start_x_spin.setValue(start.x)
        self.start_y_spin.setValue(start.y)
        self.start_yaw_spin.setValue(math.degrees(start.yaw))
        self.start_speed_spin.setValue(start.speed)

        self.goal_x_spin.setValue(goal.x)
        self.goal_y_spin.setValue(goal.y)
        self.goal_yaw_spin.setValue(math.degrees(goal.yaw))
        self.goal_speed_spin.setValue(goal.speed)

    def begin_start_selection(
        self,
    ) -> None:
        self._begin_point_selection("start")

    def begin_goal_selection(
        self,
    ) -> None:
        self._begin_point_selection("goal")

    def _begin_point_selection(
        self,
        mode: str,
    ) -> None:
        if mode not in {
            "start",
            "goal",
        }:
            raise ValueError(f"Unsupported selection mode: {mode!r}")

        self.pause_simulation()

        self.selection_mode = mode

        self.simulation_view.set_lane_snapping(
            enabled=self.snap_to_lane_check.isChecked(),
            max_distance=self.snap_distance_spin.value(),
        )

        if mode == "start":
            default_yaw = math.radians(self.start_yaw_spin.value())
        else:
            default_yaw = math.radians(self.goal_yaw_spin.value())

        self.simulation_view.set_selection_default_yaw(default_yaw)

        self.simulation_view.set_selection_enabled(True)

        if mode == "start":
            description = "选择起点"
        else:
            description = "选择终点"

        self.selection_status_label.setText(
            f"当前：{description}。左键按下确定位置，拖动确定方向，松开确认；速度使用数值框"
        )

        self.append_log(
            f"POINT_SELECTION_BEGIN mode={mode} "
            f"snap={self.snap_to_lane_check.isChecked()} "
            f"radius={self.snap_distance_spin.value():.2f}"
        )

    def cancel_point_selection(
        self,
    ) -> None:
        previous_mode = self.selection_mode

        self.selection_mode = None

        self.simulation_view.set_selection_enabled(False)

        self.selection_status_label.setText("当前：浏览模式")

        if previous_mode is not None:
            self.append_log(f"POINT_SELECTION_CANCEL mode={previous_mode}")

    def _on_lane_snap_changed(
        self,
        enabled: bool,
    ) -> None:
        self.snap_distance_spin.setEnabled(enabled)

        self.simulation_view.set_lane_snapping(
            enabled=enabled,
            max_distance=self.snap_distance_spin.value(),
        )

    def _on_lane_snap_distance_changed(
        self,
        value: float,
    ) -> None:
        self.simulation_view.set_lane_snapping(
            enabled=self.snap_to_lane_check.isChecked(),
            max_distance=value,
        )

    def _on_world_mouse_moved(
        self,
        x: float,
        y: float,
    ) -> None:
        if self.selection_mode is None:
            self.statusBar().showMessage(f"mouse=({x:.3f}, {y:.3f})")

    def _on_lane_projection_changed(
        self,
        projection,
    ) -> None:
        if self.selection_mode is None:
            return

        if not self.snap_to_lane_check.isChecked():
            return

        if projection is None:
            self.statusBar().showMessage("附近没有满足吸附半径的可行驶车道")

            return

        self.statusBar().showMessage(
            f"lane={projection.lane_id} "
            f"point=({projection.point[0]:.3f}, "
            f"{projection.point[1]:.3f}) "
            f"yaw={math.degrees(projection.yaw):.2f}° "
            f"distance={projection.distance:.3f} m"
        )

    def _on_world_pose_selected(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> None:
        if self.selection_mode is None:
            return

        selected_x = float(x)
        selected_y = float(y)
        selected_yaw = float(yaw)

        if self.selection_mode == "start":
            self.selected_initial_state = VehicleState(
                x=selected_x,
                y=selected_y,
                yaw=selected_yaw,
                speed=self.start_speed_spin.value(),
            )

            description = "START_SELECTED"
            selected_speed = self.start_speed_spin.value()

        elif self.selection_mode == "goal":
            self.selected_goal_state = VehicleState(
                x=selected_x,
                y=selected_y,
                yaw=selected_yaw,
                speed=self.goal_speed_spin.value(),
            )

            description = "GOAL_SELECTED"
            selected_speed = self.goal_speed_spin.value()

        else:
            raise RuntimeError("Invalid point selection mode")

        lane_id = None

        if self.simulation_view.last_lane_projection is not None:
            lane_id = self.simulation_view.last_lane_projection.lane_id

        self._sync_scenario_inputs_from_states()

        self.append_log(
            f"{description} "
            f"x={selected_x:.3f} "
            f"y={selected_y:.3f} "
            f"yaw={selected_yaw:.3f} "
            f"speed={selected_speed:.3f} "
            f"lane={lane_id}"
        )

        self.cancel_point_selection()

    def apply_scenario_from_inputs(
        self,
    ) -> None:
        self.selected_initial_state = VehicleState(
            x=self.start_x_spin.value(),
            y=self.start_y_spin.value(),
            yaw=math.radians(self.start_yaw_spin.value()),
            speed=self.start_speed_spin.value(),
        )

        self.selected_goal_state = VehicleState(
            x=self.goal_x_spin.value(),
            y=self.goal_y_spin.value(),
            yaw=math.radians(self.goal_yaw_spin.value()),
            speed=self.goal_speed_spin.value(),
        )

        self.goal = GoalState(
            state=self.selected_goal_state,
            position_tolerance=(self.goal.position_tolerance),
            yaw_tolerance=(self.goal.yaw_tolerance),
            speed_tolerance=(self.goal.speed_tolerance),
        )

        self.reset_environment()

        self.append_log(
            "SCENARIO_APPLIED "
            f"start=({self.selected_initial_state.x:.3f}, "
            f"{self.selected_initial_state.y:.3f}, "
            f"{self.selected_initial_state.yaw:.3f}) "
            f"goal=({self.selected_goal_state.x:.3f}, "
            f"{self.selected_goal_state.y:.3f}, "
            f"{self.selected_goal_state.yaw:.3f})"
        )

    def _create_actions(
        self,
    ) -> None:
        run_action = QAction(
            "运行/暂停",
            self,
        )

        run_action.setShortcut("Space")

        run_action.triggered.connect(self.toggle_simulation)

        self.addAction(run_action)

        step_action = QAction(
            "单步",
            self,
        )

        step_action.setShortcut("N")

        step_action.triggered.connect(self.single_step)

        self.addAction(step_action)

        reset_action = QAction(
            "复位",
            self,
        )

        reset_action.setShortcut("R")

        reset_action.triggered.connect(self.reset_environment)

        self.addAction(reset_action)

        fit_action = QAction(
            "适配视图",
            self,
        )

        fit_action.setShortcut("F")

        fit_action.triggered.connect(self.simulation_view.fit_world)

        self.addAction(fit_action)

        load_map_action = QAction(
            "加载 OpenDRIVE 地图",
            self,
        )

        load_map_action.setShortcut("Ctrl+O")

        load_map_action.triggered.connect(self.select_opendrive_map)

        self.addAction(load_map_action)

        toggle_lane_boundaries_action = QAction(
            "切换全部车道边界",
            self,
        )

        toggle_lane_boundaries_action.setShortcut("Ctrl+B")

        toggle_lane_boundaries_action.triggered.connect(
            self.show_lane_boundaries_check.toggle
        )

        self.addAction(toggle_lane_boundaries_action)

        toggle_lane_centerlines_action = QAction(
            "切换车道中心线",
            self,
        )

        toggle_lane_centerlines_action.setShortcut("Ctrl+L")

        toggle_lane_centerlines_action.triggered.connect(
            self.show_lane_centerlines_check.toggle
        )

        self.addAction(toggle_lane_centerlines_action)

        select_start_action = QAction(
            "选择起点",
            self,
        )

        select_start_action.setShortcut("Ctrl+1")

        select_start_action.triggered.connect(self.begin_start_selection)

        self.addAction(select_start_action)

        select_goal_action = QAction(
            "选择终点",
            self,
        )

        select_goal_action.setShortcut("Ctrl+2")

        select_goal_action.triggered.connect(self.begin_goal_selection)

        self.addAction(select_goal_action)

    def _on_show_lane_boundaries_changed(
        self,
        enabled: bool,
    ) -> None:
        """
        切换全部 lane 左右边界调试层。
        """
        self.simulation_view.set_show_lane_boundaries(enabled)

        self.append_log("MAP_DEBUG_BOUNDARIES " f"enabled={bool(enabled)}")

    def _on_show_lane_centerlines_changed(
        self,
        enabled: bool,
    ) -> None:
        """
        切换 lane centerline 调试层。
        """
        self.simulation_view.set_show_lane_centerlines(enabled)

        self.append_log("MAP_DEBUG_CENTERLINES " f"enabled={bool(enabled)}")

    def _on_show_traffic_signals_changed(
        self,
        enabled: bool,
    ) -> None:
        enabled = bool(enabled)

        self.simulation_view.set_show_traffic_signals(enabled)

        self.traffic_signal_layer_combo.setEnabled(enabled)
        self.show_traffic_signal_ids_check.setEnabled(enabled)

        self.append_log("TRAFFIC_SIGNALS " f"enabled={enabled}")

    def _on_traffic_signal_layer_mode_changed(
        self,
        _: int,
    ) -> None:
        mode = self.traffic_signal_layer_combo.currentData()

        if mode not in {
            "world",
            "map",
            "compare",
        }:
            raise RuntimeError("Unknown traffic signal layer mode: " f"{mode!r}")

        self.simulation_view.set_traffic_signal_layer_mode(mode)

        self.append_log("TRAFFIC_SIGNAL_LAYER " f"mode={mode}")

    def _on_show_traffic_signal_ids_changed(
        self,
        enabled: bool,
    ) -> None:
        self.simulation_view.set_show_traffic_signal_ids(enabled)

        self.append_log("MAP_TRAFFIC_SIGNAL_IDS " f"enabled={bool(enabled)}")

    def _restore_last_opendrive_map(
        self,
    ) -> None:
        """
        启动后恢复上一次成功加载的 OpenDRIVE 地图。

        恢复失败采用软失败策略：
            只写日志；
            不弹出错误对话框；
            不阻止 GUI 启动；
            保留默认道路。
        """
        saved_path = self.settings.value(
            "map/last_successful_opendrive_path",
            "",
            type=str,
        )

        if not saved_path:
            self.append_log("MAP_RESTORE_SKIPPED " "reason=no_saved_path")
            self.simulation_view.fit_world()
            return

        map_path = Path(saved_path).expanduser()

        if not map_path.is_file():
            self.append_log(
                "MAP_RESTORE_SKIPPED " f"path={map_path} " "reason=file_not_found"
            )
            self.simulation_view.fit_world()
            return

        restored = self.load_opendrive_map(
            map_path,
            show_error_dialog=False,
            remember_on_success=False,
            load_reason="restore",
        )

        if not restored:
            self.simulation_view.fit_world()

    def select_opendrive_map(
        self,
    ) -> None:
        """弹出文件选择框并加载 OpenDRIVE 地图。"""
        self.pause_simulation()
        self.cancel_point_selection()

        saved_directory = self.settings.value(
            "map/last_directory",
            "",
            type=str,
        )

        if saved_directory and Path(saved_directory).expanduser().is_dir():
            initial_directory = str(Path(saved_directory).expanduser().resolve())
        elif self.current_map_path is not None:
            initial_directory = str(self.current_map_path.parent)
        else:
            initial_directory = str(Path.cwd())

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择 OpenDRIVE 地图",
            initial_directory,
            ("OpenDRIVE 地图 (*.xodr);;" "XML 文件 (*.xml);;" "所有文件 (*)"),
        )

        if not file_name:
            return

        self.load_opendrive_map(
            Path(file_name),
            show_error_dialog=True,
            remember_on_success=True,
            load_reason="manual",
        )

    def load_opendrive_map(
        self,
        path: Path,
        *,
        show_error_dialog: bool = True,
        remember_on_success: bool = True,
        load_reason: str = "manual",
    ) -> bool:
        """
        读取、转换并显示一个 OpenDRIVE 地图。

        返回：
            True：
                地图成功加载并完成显示。

            False：
                文件不存在、解析失败或渲染失败。

        remember_on_success：
            仅成功加载后记录为“最后成功地图”。

        show_error_dialog：
            手动加载时通常为 True；
            启动自动恢复时应为 False。
        """
        map_path = Path(path).expanduser().resolve()

        if not map_path.is_file():
            error_message = "OpenDRIVE map file does not exist"

            self.append_log(
                "MAP_LOAD_FAILED "
                f"reason={load_reason} "
                f"path={map_path} "
                f"error=FileNotFoundError: "
                f"{error_message}"
            )

            if show_error_dialog:
                QMessageBox.critical(
                    self,
                    "地图加载失败",
                    ("OpenDRIVE 地图文件不存在：\n" f"{map_path}"),
                )

            return False

        try:
            road_network = load_opendrive_road_network(
                map_path,
                sample_step=self.map_sample_step,
            )

            self.simulation_view.render_road_network(road_network)

        except Exception as error:
            self.append_log(
                "MAP_LOAD_FAILED "
                f"reason={load_reason} "
                f"path={map_path} "
                f"error={type(error).__name__}: "
                f"{error}"
            )

            if show_error_dialog:
                QMessageBox.critical(
                    self,
                    "地图加载失败",
                    (
                        "无法加载 OpenDRIVE 地图：\n"
                        f"{map_path}\n\n"
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                )

            return False

        self.current_map_path = map_path
        self.road_network = road_network

        self.map_name_label.setText(f"当前地图：{map_path.name}")

        self.cancel_point_selection()

        # 新地图加载后，立即用当前起终点重新配置地图参考路径。
        self.reset_environment()
        self.simulation_view.fit_world()

        if remember_on_success:
            self.settings.setValue(
                "map/last_successful_opendrive_path",
                str(map_path),
            )
            self.settings.setValue(
                "map/last_directory",
                str(map_path.parent),
            )
            self.settings.sync()

        topology_edge_count = int(
            road_network.metadata.get(
                "intra_road_topology_edge_count",
                0,
            )
        )

        self.append_log(
            "MAP_LOADED "
            f"reason={load_reason} "
            f"path={map_path} "
            f"lanes={road_network.lane_count} "
            f"traffic_signals={road_network.traffic_signal_count} "
            f"topology_edges={topology_edge_count} "
            f"sample_step={self.map_sample_step:.3f}"
        )

        return True

    def current_control_mode(
        self,
    ) -> str:
        mode = self.control_mode_combo.currentData()

        if mode not in {
            "manual",
            "auto",
        }:
            raise RuntimeError("Unknown control mode: " f"{mode!r}")

        return mode

    def current_manual_input_mode(
        self,
    ) -> str:
        mode = self.manual_input_combo.currentData()

        if mode not in {
            "numeric",
            "keyboard",
        }:
            raise RuntimeError("Unknown manual input mode: " f"{mode!r}")

        return mode

    def _on_control_mode_changed(
        self,
        _: int,
    ) -> None:
        self.keyboard_control.reset()

        self._update_keyboard_input_display()
        self._update_control_mode_ui()

        mode = self.current_control_mode()

        self.append_log(f"CONTROL_MODE mode={mode}")

    def _on_manual_input_mode_changed(
        self,
        _: int,
    ) -> None:
        self.keyboard_control.reset()

        self._update_keyboard_input_display()
        self._update_control_mode_ui()

        mode = self.current_manual_input_mode()

        self.append_log("MANUAL_INPUT_MODE " f"mode={mode}")

    def _update_control_mode_ui(
        self,
    ) -> None:
        control_mode = self.current_control_mode()

        manual_input_mode = self.current_manual_input_mode()

        is_manual = control_mode == "manual"

        is_numeric = is_manual and manual_input_mode == "numeric"

        is_keyboard = is_manual and manual_input_mode == "keyboard"

        self.manual_input_combo.setEnabled(is_manual)

        # 精确输入模式下可编辑。
        # 键盘模式下作为映射后控制量显示。
        self.acceleration_spin.setEnabled(is_numeric)

        self.steering_spin.setEnabled(is_numeric)

        self.keyboard_steering_label.setEnabled(is_keyboard)

        self.keyboard_throttle_label.setEnabled(is_keyboard)

        self.keyboard_brake_label.setEnabled(is_keyboard)

        if control_mode == "auto":
            hint_text = (
                "快捷键：\n"
                "Space：运行/暂停\n"
                "N：单步执行\n"
                "R：复位\n"
                "F：适配视图\n"
                "Ctrl+O：加载地图\n"
                "Ctrl+B：切换全部车道边界\n"
                "Ctrl+L：切换车道中心线\n"
                "Ctrl+1：选择起点\n"
                "Ctrl+2：选择终点\n\n"
                "当前为自动规划模式。\n"
                "lane graph 路线可用时，"
                "优先跟踪地图 reference path；\n"
                "否则回退到 Bézier 参考路径。"
            )

        elif manual_input_mode == "keyboard":
            hint_text = (
                "快捷键：\n"
                "Space：运行/暂停\n"
                "N：单步执行\n"
                "R：复位\n"
                "F：适配视图\n"
                "Ctrl+O：加载地图\n"
                "Ctrl+B：切换全部车道边界\n"
                "Ctrl+L：切换车道中心线\n"
                "Ctrl+1：选择起点\n"
                "Ctrl+2：选择终点\n\n"
                "当前为键盘驾驶模式：\n"
                "W / ↑：油门\n"
                "S / ↓：制动\n"
                "A / ←：左转\n"
                "D / →：右转\n\n"
                "方向盘会逐渐转动，"
                "松开方向键后自动回正。\n"
                "刹车和油门同时输入时，"
                "刹车优先。"
            )

        else:
            hint_text = (
                "快捷键：\n"
                "Space：运行/暂停\n"
                "N：单步执行\n"
                "R：复位\n"
                "F：适配视图\n"
                "Ctrl+O：加载地图\n"
                "Ctrl+B：切换全部车道边界\n"
                "Ctrl+L：切换车道中心线\n"
                "Ctrl+1：选择起点\n"
                "Ctrl+2：选择终点\n\n"
                "当前为精确输入模式。\n"
                "直接填写加速度和"
                "前轮转角。"
            )

        self.hint_label.setText(hint_text)

    def _update_keyboard_input_display(
        self,
    ) -> None:
        input_state = self.keyboard_control.input_state

        self.keyboard_steering_label.setText(f"{input_state.steering:+.3f}")

        self.keyboard_throttle_label.setText(f"{input_state.throttle:.3f}")

        self.keyboard_brake_label.setText(f"{input_state.brake:.3f}")

    def _set_driving_key_state(
        self,
        key: int,
        pressed: bool,
    ) -> bool:
        if key in {
            Qt.Key.Key_W,
            Qt.Key.Key_Up,
        }:
            self.keyboard_control.throttle_pressed = pressed

            return True

        if key in {
            Qt.Key.Key_S,
            Qt.Key.Key_Down,
        }:
            self.keyboard_control.brake_pressed = pressed

            return True

        if key in {
            Qt.Key.Key_A,
            Qt.Key.Key_Left,
        }:
            self.keyboard_control.left_pressed = pressed

            return True

        if key in {
            Qt.Key.Key_D,
            Qt.Key.Key_Right,
        }:
            self.keyboard_control.right_pressed = pressed

            return True

        return False

    def keyPressEvent(
        self,
        event: QKeyEvent,
    ) -> None:
        keyboard_driving_enabled = (
            self.current_control_mode() == "manual"
            and self.current_manual_input_mode() == "keyboard"
        )

        handled = False

        if keyboard_driving_enabled and not event.isAutoRepeat():
            handled = self._set_driving_key_state(
                key=event.key(),
                pressed=True,
            )

        if handled:
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(
        self,
        event: QKeyEvent,
    ) -> None:
        keyboard_driving_enabled = (
            self.current_control_mode() == "manual"
            and self.current_manual_input_mode() == "keyboard"
        )

        handled = False

        if keyboard_driving_enabled and not event.isAutoRepeat():
            handled = self._set_driving_key_state(
                key=event.key(),
                pressed=False,
            )

        if handled:
            event.accept()
            return

        super().keyReleaseEvent(event)

    def focusOutEvent(
        self,
        event: QFocusEvent,
    ) -> None:
        # 窗口失去焦点时释放所有驾驶键，
        # 防止油门、刹车或方向盘卡住。
        self.keyboard_control.reset()

        self._update_keyboard_input_display()

        super().focusOutEvent(event)

    def _configure_planner_reference_path(
        self,
    ) -> None:
        """
        根据当前地图和场景起终点配置规划器参考路径。

        成功：
            在 RoadNetwork lane graph 上生成有向路线；
            起点和终点 yaw 自动采用各自车道投影航向。

        失败：
            清除外部路径并回退到原 Bézier 路径。

        当前可达范围取决于 RoadNetwork 中已有的 successor_ids。
        现阶段已经支持同一 road 内跨 laneSection；
        后续补齐跨 road 和 junction 拓扑后，本函数无需改动。
        """
        self.current_lane_route = None

        if self.road_network is None:
            self.planner.clear_external_reference_path()

            self.append_log("MAP_ROUTE_FALLBACK " "reason=no_road_network")
            return

        try:
            route = build_lane_route(
                self.road_network,
                start_x=self.selected_initial_state.x,
                start_y=self.selected_initial_state.y,
                goal_x=self.selected_goal_state.x,
                goal_y=self.selected_goal_state.y,
                snap_distance=self.snap_distance_spin.value(),
            )

        except NoRouteError as error:
            self.planner.clear_external_reference_path()

            self.append_log(
                "MAP_ROUTE_FALLBACK " f"reason={error.reason} " f"error={error}"
            )
            return

        except Exception as error:
            self.planner.clear_external_reference_path()

            self.append_log(
                "MAP_ROUTE_FALLBACK "
                "reason=unexpected_error "
                f"error={type(error).__name__}: {error}"
            )
            return

        self.current_lane_route = route

        # 地图路线成功时，采用车道投影航向。
        # 位置仍保留用户输入值；参考路径本身从精确投影点开始。
        self.selected_initial_state = VehicleState(
            x=self.selected_initial_state.x,
            y=self.selected_initial_state.y,
            yaw=route.start_projection.yaw,
            speed=self.selected_initial_state.speed,
        )

        self.selected_goal_state = VehicleState(
            x=self.selected_goal_state.x,
            y=self.selected_goal_state.y,
            yaw=route.goal_projection.yaw,
            speed=self.selected_goal_state.speed,
        )

        self._sync_scenario_inputs_from_states()

        self.planner.set_reference_path(route.reference_path)

        self.append_log(
            "MAP_ROUTE_READY "
            f"lanes={route.lane_ids} "
            f"points={route.reference_path.shape[0]} "
            f"length={route.reference_path[-1, 3]:.3f} "
            f"start_s={route.start_projection.arc_length:.3f} "
            f"goal_s={route.goal_projection.arc_length:.3f} "
            f"start_yaw={route.start_projection.yaw:.3f} "
            f"goal_yaw={route.goal_projection.yaw:.3f}"
        )

    def reset_environment(
        self,
    ) -> None:
        self.pause_simulation()

        self.planner.reset()
        self.keyboard_control.reset()

        self.acceleration_spin.setValue(0.0)
        self.steering_spin.setValue(0.0)

        self._update_keyboard_input_display()
        self.cancel_point_selection()

        # planner.reset() 会清除外部路径，因此每次环境复位时
        # 都要重新构造地图参考路径。
        #
        # 路线成功时，该函数还会把起终点 yaw 同步为车道航向。
        self._configure_planner_reference_path()

        initial_state = self.selected_initial_state

        self.goal = GoalState(
            state=self.selected_goal_state,
            position_tolerance=self.goal.position_tolerance,
            yaw_tolerance=self.goal.yaw_tolerance,
            speed_tolerance=self.goal.speed_tolerance,
        )

        obstacles = (
            CircleObstacle(
                obstacle_id="circle_001",
                x=12.0,
                y=0.0,
                radius=1.0,
            ),
            BoxObstacle(
                obstacle_id="box_001",
                x=20.0,
                y=-2.4,
                yaw=0.2,
                length=2.5,
                width=1.3,
            ),
        )

        self.env.reset(
            initial_state=initial_state,
            goal=self.goal,
            obstacles=obstacles,
        )

        # 地图层交通灯映射为真实世界交通灯。
        # 当前默认无位姿偏差，后续可在这里配置地图误差。
        if self.road_network is not None:
            # self.env.initialize_world_traffic_signals(
            #     self.road_network.traffic_signals,
            # )
            self.env.initialize_world_traffic_signals(
                self.road_network.traffic_signals,
                position_offset_x=0.8,
                position_offset_y=0.4,
                yaw_offset=0.05,
                initial_state=TrafficLightState.RED,
            )

        # 地图路线无需等待第一帧 plan()，场景应用或复位后立即显示。
        if self.current_lane_route is not None:
            self.env.set_planner_debug(
                planned_trajectory=None,
                reference_path=(self.current_lane_route.reference_path),
                debug={
                    "planner": "BezierPlanner",
                    "status": "map_route_ready",
                    "reference_path_source": "external",
                    "route_lane_ids": (self.current_lane_route.lane_ids),
                    "reference_path_points": (
                        self.current_lane_route.reference_path.shape[0]
                    ),
                },
            )

        snapshot = self.env.get_snapshot()

        self.simulation_view.render_snapshot(
            snapshot=snapshot,
            goal=self.goal,
        )

        self._update_status(snapshot)

        route_source = "external" if self.current_lane_route is not None else "bezier"

        self.append_log(
            "RESET "
            f"mode={self.current_control_mode()} "
            f"reference={route_source} "
            f"dt={self.environment_config.dt:.3f}s "
            f"max_time={self.environment_config.max_time:.1f}s"
        )

    def current_control(
        self,
    ) -> VehicleControl:
        return VehicleControl(
            acceleration=(self.acceleration_spin.value()),
            steering=(self.steering_spin.value()),
        )

    def advance_one_step(
        self,
    ) -> None:
        if self.env.is_done:
            self.pause_simulation()

            self.append_log("Episode already finished")

            return

        control_mode = self.current_control_mode()

        mode_description: str

        if control_mode == "auto":
            observation = self.env.get_observation()

            plan_result = self.planner.plan(observation)

            control = plan_result.action

            self.env.set_planner_debug(
                planned_trajectory=(plan_result.trajectory),
                reference_path=(plan_result.reference_path),
                debug=(plan_result.debug),
            )

            mode_description = "auto"

        elif control_mode == "manual":
            manual_input_mode = self.current_manual_input_mode()

            mode_description = f"manual/{manual_input_mode}"

            if manual_input_mode == "keyboard":
                input_state = self.keyboard_control.update(
                    dt=(self.environment_config.dt)
                )

                control = self.manual_control_mapper.map(input_state)

                # 数值框保留，但在键盘驾驶模式下
                # 作为映射后的实际控制量显示。
                self.acceleration_spin.setValue(control.acceleration)

                self.steering_spin.setValue(control.steering)

                self._update_keyboard_input_display()

                debug = {
                    "control_mode": "manual",
                    "manual_input_mode": ("keyboard"),
                    "steering_input": (input_state.steering),
                    "throttle_input": (input_state.throttle),
                    "brake_input": (input_state.brake),
                    "mapped_acceleration": (control.acceleration),
                    "mapped_steering": (control.steering),
                }

            elif manual_input_mode == "numeric":
                control = self.current_control()

                debug = {
                    "control_mode": "manual",
                    "manual_input_mode": ("numeric"),
                    "mapped_acceleration": (control.acceleration),
                    "mapped_steering": (control.steering),
                }

            else:
                raise RuntimeError(
                    "Unsupported manual input mode: " f"{manual_input_mode!r}"
                )

            # 手动模式不应继续显示上一帧
            # 自动规划留下的预测轨迹和参考路径。
            self.env.set_planner_debug(
                planned_trajectory=None,
                reference_path=None,
                debug=debug,
            )

        else:
            raise RuntimeError("Unsupported control mode: " f"{control_mode!r}")

        result = self.env.step(control)

        snapshot = self.env.get_snapshot()

        self.simulation_view.render_snapshot(
            snapshot=snapshot,
            goal=self.goal,
        )

        self._update_status(snapshot)

        reference_source = "none"
        cross_track_error = math.nan
        nearest_index = -1
        target_index = -1

        if control_mode == "auto":
            reference_source = str(
                plan_result.debug.get(
                    "reference_path_source",
                    "unknown",
                )
            )

            cross_track_error = float(
                plan_result.debug.get(
                    "cross_track_error",
                    math.nan,
                )
            )

            nearest_index = int(
                plan_result.debug.get(
                    "nearest_index",
                    -1,
                )
            )

            target_index = int(
                plan_result.debug.get(
                    "target_index",
                    -1,
                )
            )

        self.append_log(
            f"step mode={mode_description} "
            f"reference={reference_source} "
            f"cte={cross_track_error:.3f} "
            f"nearest={nearest_index} "
            f"target={target_index} "
            f"control=("
            f"a={control.acceleration:.2f}, "
            f"delta={control.steering:.3f}) "
            f"state=("
            f"x={snapshot.ego.x:.3f}, "
            f"y={snapshot.ego.y:.3f}, "
            f"yaw={snapshot.ego.yaw:.3f}, "
            f"v={snapshot.ego.speed:.3f}) "
            f"reward={result.reward:.3f} "
            f"collision="
            f"{result.info['collision']} "
            f"goal="
            f"{result.info['goal_reached']} "
            f"timeout="
            f"{result.info['timeout']}"
        )

        if result.terminated or result.truncated:
            self.pause_simulation()

            # 回合结束后清理驾驶键状态。
            self.keyboard_control.reset()

            self._update_keyboard_input_display()

            if result.info["collision"]:
                reason = "collision"

            elif result.info["goal_reached"]:
                reason = "goal_reached"

            elif result.info["timeout"]:
                reason = "timeout"

            else:
                reason = "unknown"

            self.append_log("EPISODE_FINISHED " f"reason={reason}")

    def single_step(
        self,
    ) -> None:
        self.pause_simulation()
        self.advance_one_step()

    def start_simulation(
        self,
    ) -> None:
        if self.env.is_done:
            self.append_log("Episode finished. " "Press Reset first.")

            return

        if self.running:
            return

        self.running = True
        self.timer.start()

        self.status_label.setText("运行中")

        if self.current_control_mode() == "manual":
            mode_description = "manual/" f"{self.current_manual_input_mode()}"
        else:
            mode_description = "auto"

        self.append_log(f"RUN mode={mode_description}")

    def pause_simulation(
        self,
    ) -> None:
        if not self.running:
            return

        self.running = False
        self.timer.stop()

        self.status_label.setText("已暂停")

        self.append_log("PAUSE")

    def toggle_simulation(
        self,
    ) -> None:
        if self.running:
            self.pause_simulation()
        else:
            self.start_simulation()

    def _update_status(
        self,
        snapshot,
    ) -> None:
        ego = snapshot.ego

        self.frame_label.setText(str(snapshot.frame))

        self.time_label.setText(f"{snapshot.time:.2f} s")

        self.x_label.setText(f"{ego.x:.3f} m")

        self.y_label.setText(f"{ego.y:.3f} m")

        self.yaw_label.setText(f"{math.degrees(ego.yaw):.2f}°")

        self.speed_label.setText(f"{ego.speed:.3f} m/s")

        if snapshot.min_clearance is None:
            self.clearance_label.setText("--")
        else:
            self.clearance_label.setText(f"{snapshot.min_clearance:.3f} m")

        if snapshot.collision:
            self.status_label.setText("碰撞")

        elif self.env.terminated:
            self.status_label.setText("任务结束")

        elif self.env.truncated:
            self.status_label.setText("超时")

        elif self.running:
            self.status_label.setText("运行中")

        else:
            self.status_label.setText("已暂停")

    def append_log(
        self,
        message: str,
    ) -> None:
        if self.env.is_initialized:
            time_value = self.env.time
            frame_value = self.env.frame
        else:
            # GUI 启动和 reset() 前置配置阶段可能产生日志。
            time_value = 0.0
            frame_value = 0

        prefix = f"[t={time_value:7.3f}] " f"[frame={frame_value:05d}] "

        self.log_console.appendPlainText(prefix + message)

        scrollbar = self.log_console.verticalScrollBar()

        scrollbar.setValue(scrollbar.maximum())


def main() -> None:
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
