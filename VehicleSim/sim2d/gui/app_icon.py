from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QWidget


def application_icon_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "vehiclesim.svg"


def apply_application_icon(
    app: QApplication,
    window: QWidget | None = None,
) -> QIcon:
    """设置应用和主窗口图标，供标题栏、任务栏和 Dock 使用。"""
    icon_path = application_icon_path()

    if not icon_path.is_file():
        return QIcon()

    icon = QIcon(str(icon_path))
    app.setWindowIcon(icon)

    if window is not None:
        window.setWindowIcon(icon)

    return icon
