from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from sim2d.gui.app_icon import apply_application_icon
from sim2d.gui.perception_extension import install as install_perception
from sim2d.gui.perception_object_debug_extension import (
    install as install_perception_object_debug,
)
from sim2d.gui.perception_viewer import install as install_perception_viewer
from sim2d.gui.perception_viewer_bootstrap import (
    install as install_perception_viewer_bootstrap,
)
from sim2d.gui.perception_viewer_interaction import (
    install as install_perception_viewer_interaction,
)
from sim2d.gui.road_compare_extension import install as install_road_compare
from sim2d.gui.road_layer_extension import install as install_road_layers
from sim2d.gui.road_topology_log_extension import (
    install as install_road_topology_log,
)
from sim2d.gui.traffic_actor_editor import install as install_traffic_actor_editor
from sim2d.perception.traffic_actor_extension import (
    install as install_traffic_actor_perception,
)
from sim2d.performance_debug import install as install_performance_debug
from sim2d.planning import (
    SpatiotemporalPlanner,
    SpatiotemporalPlannerConfig,
)

install_road_layers()
install_road_compare()
install_road_topology_log()
install_perception()
install_traffic_actor_perception()

# MainWindow.__init__ 内部会主动调用 reset_environment()。必须先安装
# 占位 controller，再安装感知窗口扩展，避免构造阶段访问尚未创建的属性。
install_perception_viewer_bootstrap()
install_perception_viewer()
install_perception_object_debug()
install_perception_viewer_interaction()

# 最后安装场景编辑器，使其能够覆盖现有 reset 链并使用已完成的主窗口扩展。
install_traffic_actor_editor()

# 性能调试必须在 MainWindow 实例化之前安装，才能覆盖每周期入口，
# 同时采集感知、规划、优化、rollout、cost 和 GUI 总周期耗时。
install_performance_debug()

from sim2d.gui.main_window import MainWindow  # noqa: E402


def main() -> None:
    QGuiApplication.setApplicationDisplayName("VehicleSim 2D")
    QGuiApplication.setDesktopFileName("vehiclesim")

    app = QApplication(sys.argv)
    app.setApplicationName("VehicleSim")
    app.setOrganizationName("VehicleSim")
    app.setStyle("Fusion")

    apply_application_icon(app)

    window = MainWindow()

    # MainWindow 为兼容旧入口仍会先构造 BezierPlanner。GUI 完成构造后，
    # 在这里切换为时空联合规划器，并重新执行 reset_environment()，使地图
    # 路线、感知输入和场景编辑扩展全部直接使用新规划器。
    window.planner = SpatiotemporalPlanner(
        vehicle_config=window.vehicle_config,
        config=SpatiotemporalPlannerConfig(
            dt=0.1,
            horizon_steps=20,
            target_speed=4.0,
            max_iterations=3,
            gradient_epsilon=1e-3,
            initial_step_size=0.02,
        ),
    )
    window.setWindowTitle("VehicleSim 2D - 时空联合规划器")
    window.reset_environment()

    apply_application_icon(app, window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
