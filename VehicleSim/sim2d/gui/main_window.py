from __future__ import annotations

import math
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QFocusEvent,
    QFont,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
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
from sim2d.planning import BezierPlanner
from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    GoalState,
    VehicleConfig,
    VehicleControl,
    VehicleState,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("VehicleSim 2D")

        self.resize(
            1400,
            850,
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
            vehicle_config=self.vehicle_config,
            environment_config=(self.environment_config),
        )

        self.planner = BezierPlanner(
            vehicle_config=self.vehicle_config,
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

        self._build_ui()
        self._create_actions()
        self.reset_environment()

        QTimer.singleShot(
            0,
            self.simulation_view.fit_world,
        )

    def _build_ui(self) -> None:
        root_splitter = QSplitter(Qt.Orientation.Horizontal)

        control_panel = QWidget()
        control_panel.setMinimumWidth(280)
        control_panel.setMaximumWidth(390)

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

        right_splitter.setSizes([650, 180])

        root_splitter.addWidget(control_panel)

        root_splitter.addWidget(right_splitter)

        root_splitter.setStretchFactor(
            1,
            1,
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

        self._update_keyboard_input_display()
        self._update_control_mode_ui()

    def _create_actions(self) -> None:
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

    def current_control_mode(
        self,
    ) -> str:
        mode = self.control_mode_combo.currentData()

        if mode not in {
            "manual",
            "auto",
        }:
            raise RuntimeError(f"Unknown control mode: {mode!r}")

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
                "F：适配视图\n\n"
                "当前为自动规划模式。\n"
                "控制量由 SimplePlanner "
                "根据当前观测计算。"
            )

        elif manual_input_mode == "keyboard":
            hint_text = (
                "快捷键：\n"
                "Space：运行/暂停\n"
                "N：单步执行\n"
                "R：复位\n"
                "F：适配视图\n\n"
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
                "F：适配视图\n\n"
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

    def reset_environment(self) -> None:
        self.pause_simulation()

        self.planner.reset()
        self.keyboard_control.reset()

        self.acceleration_spin.setValue(0.0)

        self.steering_spin.setValue(0.0)

        self._update_keyboard_input_display()

        initial_state = VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=2.0,
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

        snapshot = self.env.get_snapshot()

        self.simulation_view.render_snapshot(
            snapshot=snapshot,
            goal=self.goal,
        )

        self._update_status(snapshot)

        self.append_log(
            "RESET "
            f"mode={self.current_control_mode()} "
            f"dt={self.environment_config.dt:.3f}s "
            f"max_time="
            f"{self.environment_config.max_time:.1f}s"
        )

    def current_control(
        self,
    ) -> VehicleControl:
        return VehicleControl(
            acceleration=(self.acceleration_spin.value()),
            steering=(self.steering_spin.value()),
        )

    def advance_one_step(self) -> None:
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
                debug=plan_result.debug,
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

            self.env.set_planner_debug(
                planned_trajectory=None,
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

        self.append_log(
            f"step mode={mode_description} "
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

    def single_step(self) -> None:
        self.pause_simulation()
        self.advance_one_step()

    def start_simulation(self) -> None:
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

    def pause_simulation(self) -> None:
        if not self.running:
            return

        self.running = False
        self.timer.stop()

        self.status_label.setText("已暂停")

        self.append_log("PAUSE")

    def toggle_simulation(self) -> None:
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
        prefix = f"[t={self.env.time:7.3f}] " f"[frame={self.env.frame:05d}] "

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
