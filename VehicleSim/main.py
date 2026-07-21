from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from sim2d.gui.app_icon import apply_application_icon
from sim2d.gui.demo_obstacle_extension import install as install_demo_obstacles
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

install_road_layers()
install_road_compare()
install_road_topology_log()
install_perception()

# GUI 示例障碍物使用车辆坐标定义，再根据每次起点转换到世界坐标。
install_demo_obstacles()

# MainWindow.__init__ 内部会主动调用 reset_environment()。必须先安装
# 占位 controller，再安装感知窗口扩展，避免构造阶段访问尚未创建的属性。
install_perception_viewer_bootstrap()
install_perception_viewer()

# perception_viewer.py 已直接使用车辆坐标系，并按 +x 向上、+y 向左绘制。
# 不再安装旧的 perception_view_rotation 扩展；旧扩展会把车辆坐标误当成
# 世界坐标，再减一次 ego 位姿，导致障碍物图元与数值标签位置不一致。

# 必须在感知图元类加载后安装，显示每个目标的实时车辆坐标。
install_perception_object_debug()

# 必须最后安装，覆盖逐帧 fitInView 行为并保留当前车辆坐标绘制逻辑。
install_perception_viewer_interaction()

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
    apply_application_icon(app, window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
