from __future__ import annotations

from sim2d.gui.main_window import MainWindow

_INSTALLED = False


class _PendingPerceptionViewerController:
    """主窗口构造期间的空控制器。

    MainWindow.__init__ 会主动调用 reset_environment()。感知窗口扩展在原始
    __init__ 返回后才创建真实 controller，因此构造阶段需要一个可安全调用
    update() 的占位对象。
    """

    def update(self) -> None:
        return



def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_init = MainWindow.__init__

    def init(self: MainWindow, *args, **kwargs) -> None:
        self.perception_viewer_controller = (
            _PendingPerceptionViewerController()
        )
        original_init(self, *args, **kwargs)

    MainWindow.__init__ = init
    _INSTALLED = True
