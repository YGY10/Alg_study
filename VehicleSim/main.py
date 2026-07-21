from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from sim2d.gui.app_icon import apply_application_icon
from sim2d.gui.perception_extension import install as install_perception
from sim2d.gui.perception_viewer import install as install_perception_viewer
from sim2d.gui.perception_viewer_bootstrap import (
    install as install_perception_viewer_bootstrap,
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

# MainWindow.__init__ 内部会主动调用 reset_environment()。必须先安装
# 占位 controller，再安装感知窗口扩展，避免构造阶段访问尚未创建的属性。
install_perception_viewer_bootstrap()
install_perception_viewer()

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
