from __future__ import annotations

import math
from typing import Literal

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem

from sim2d.gui.graphics_items import PIXELS_PER_METER
from sim2d.map.types import TrafficSignal
from sim2d.types import TrafficSignalSnapshot

TrafficSignalVisual = TrafficSignal | TrafficSignalSnapshot
TrafficSignalLayer = Literal["map", "world"]


class TrafficSignalGraphicsItem(QGraphicsItem):
    """
    固定屏幕尺寸的交通灯图元。

    layer:
        "map":
            地图先验层。使用灰色半透明外观。

        "world":
            真实世界层。使用 WorldTrafficSignal 的实时灯色。

    display_mode:
        "compact":
            远景模式，只绘制小圆点和短朝向线。

        "detailed":
            近景模式，绘制三灯灯箱和朝向箭头。
    """

    VALID_STATES = {
        "unknown",
        "red",
        "yellow",
        "green",
        "off",
    }

    VALID_DISPLAY_MODES = {
        "compact",
        "detailed",
    }

    VALID_LAYERS = {
        "map",
        "world",
    }

    def __init__(
        self,
        signal: TrafficSignalVisual,
        *,
        state: str = "unknown",
        show_id: bool = False,
        display_mode: str = "detailed",
        layer: TrafficSignalLayer = "map",
    ) -> None:
        super().__init__()

        self.signal = signal

        self.state = "unknown"
        self.show_id = bool(show_id)
        self.display_mode = "detailed"
        self.layer: TrafficSignalLayer = "map"

        self.housing_width = 44.0
        self.housing_height = 18.0
        self.corner_radius = 3.0

        self.lamp_radius = 5.0
        self.lamp_spacing = 14.0

        self.heading_line_length = 10.0
        self.arrow_size = 3.5

        self.compact_radius = 3.5
        self.compact_heading_length = 8.0

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )

        self.set_layer(layer)
        self.set_state(state)
        self.set_display_mode(display_mode)
        self.sync_from_signal()

    def update_signal(
        self,
        signal: TrafficSignalVisual,
        *,
        state: str | None = None,
    ) -> None:
        """更新图元绑定的交通灯数据。"""
        self.signal = signal

        if state is not None:
            self.set_state(state)

        self.sync_from_signal()
        self.update()

    def set_layer(
        self,
        layer: TrafficSignalLayer,
    ) -> None:
        normalized = str(layer).strip().lower()

        if normalized not in self.VALID_LAYERS:
            raise ValueError(f"Unsupported traffic signal layer: {layer!r}")

        self.layer = normalized  # type: ignore[assignment]

        if self.layer == "map":
            self.setZValue(34.0)
            self.setOpacity(0.42)
        else:
            self.setZValue(45.0)
            self.setOpacity(1.0)

        self.update()

    def set_state(
        self,
        state: str,
    ) -> None:
        normalized = str(state).strip().lower()

        if normalized not in self.VALID_STATES:
            raise ValueError(f"Unsupported traffic signal state: {state!r}")

        if self.state == normalized:
            return

        self.state = normalized
        self.update()

    def set_show_id(
        self,
        visible: bool,
    ) -> None:
        visible = bool(visible)

        if self.show_id == visible:
            return

        self.prepareGeometryChange()
        self.show_id = visible
        self.update()

    def set_display_mode(
        self,
        mode: str,
    ) -> None:
        normalized = str(mode).strip().lower()

        if normalized not in self.VALID_DISPLAY_MODES:
            raise ValueError(f"Unsupported traffic signal display mode: {mode!r}")

        if self.display_mode == normalized:
            return

        self.prepareGeometryChange()
        self.display_mode = normalized
        self.update()

    def boundingRect(self) -> QRectF:
        if self.display_mode == "compact":
            radius = self.compact_radius
            right = self.compact_heading_length + radius + 2.0

            return QRectF(
                -radius - 2.0,
                -radius - 2.0,
                right + radius + 4.0,
                2.0 * radius + 4.0,
            )

        half_width = 0.5 * self.housing_width
        half_height = 0.5 * self.housing_height

        left = -half_width - 2.0
        right = half_width + self.heading_line_length + self.arrow_size + 3.0
        top = -half_height - 3.0
        bottom = half_height + 3.0

        if self.show_id:
            bottom += 14.0

        return QRectF(
            left,
            top,
            right - left,
            bottom - top,
        )

    def paint(
        self,
        painter: QPainter,
        option,
        widget=None,
    ) -> None:
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        if self.display_mode == "compact":
            self._paint_compact(painter)
            return

        self._paint_detailed(painter)

    def _paint_compact(
        self,
        painter: QPainter,
    ) -> None:
        outline_color = QColor("#5f6872") if self.layer == "map" else QColor("#20242a")

        outline = QPen(
            outline_color,
            1.4,
        )
        outline.setCosmetic(True)
        painter.setPen(outline)

        if self.layer == "map":
            fill = QColor("#b9c0c7")
        elif self.state == "red":
            fill = QColor("#d65b5b")
        elif self.state == "yellow":
            fill = QColor("#e2bf5b")
        elif self.state == "green":
            fill = QColor("#54a36b")
        elif self.state == "off":
            fill = QColor("#66707a")
        else:
            fill = QColor("#f6f7f8")

        painter.setBrush(QBrush(fill))

        painter.drawEllipse(
            QPointF(0.0, 0.0),
            self.compact_radius,
            self.compact_radius,
        )

        heading_pen = QPen(
            outline_color,
            1.5,
        )
        heading_pen.setCosmetic(True)
        painter.setPen(heading_pen)

        painter.drawLine(
            QPointF(self.compact_radius, 0.0),
            QPointF(
                self.compact_radius + self.compact_heading_length,
                0.0,
            ),
        )

    def _paint_detailed(
        self,
        painter: QPainter,
    ) -> None:
        half_width = 0.5 * self.housing_width
        half_height = 0.5 * self.housing_height

        housing_rect = QRectF(
            -half_width,
            -half_height,
            self.housing_width,
            self.housing_height,
        )

        outline_color = QColor("#5f6872") if self.layer == "map" else QColor("#20242a")

        housing_pen = QPen(
            outline_color,
            1.6,
        )
        housing_pen.setCosmetic(True)

        painter.setPen(housing_pen)
        painter.setBrush(
            QBrush(QColor("#c5cbd1") if self.layer == "map" else QColor("#e7ebf1"))
        )

        painter.drawRoundedRect(
            housing_rect,
            self.corner_radius,
            self.corner_radius,
        )

        heading_pen = QPen(
            outline_color,
            1.6,
        )
        heading_pen.setCosmetic(True)
        painter.setPen(heading_pen)

        arrow_x = half_width + self.heading_line_length

        painter.drawLine(
            QPointF(half_width, 0.0),
            QPointF(arrow_x, 0.0),
        )

        painter.drawLine(
            QPointF(arrow_x, 0.0),
            QPointF(
                arrow_x - self.arrow_size,
                -self.arrow_size,
            ),
        )

        painter.drawLine(
            QPointF(arrow_x, 0.0),
            QPointF(
                arrow_x - self.arrow_size,
                self.arrow_size,
            ),
        )

        lamp_centers = (
            QPointF(-self.lamp_spacing, 0.0),
            QPointF(0.0, 0.0),
            QPointF(self.lamp_spacing, 0.0),
        )

        lamp_names = (
            "red",
            "yellow",
            "green",
        )

        lamp_colors = {
            "red": QColor("#d65b5b"),
            "yellow": QColor("#e2bf5b"),
            "green": QColor("#54a36b"),
        }

        inactive_fill = QColor("#f6f7f8")
        off_fill = QColor("#66707a")
        map_fill = QColor("#aeb6be")

        lamp_pen = QPen(
            outline_color,
            1.25,
        )
        lamp_pen.setCosmetic(True)
        painter.setPen(lamp_pen)

        for lamp_name, center in zip(
            lamp_names,
            lamp_centers,
        ):
            if self.layer == "map":
                fill = map_fill
            elif self.state == lamp_name:
                fill = lamp_colors[lamp_name]
            elif self.state == "off":
                fill = off_fill
            else:
                fill = inactive_fill

            painter.setBrush(QBrush(fill))

            painter.drawEllipse(
                center,
                self.lamp_radius,
                self.lamp_radius,
            )

        if self.show_id:
            painter.setPen(QPen(outline_color))

            painter.drawText(
                QPointF(
                    -half_width,
                    half_height + 12.0,
                ),
                self._display_id(),
            )

    def sync_from_signal(self) -> None:
        """
        根据交通灯快照更新位置和实体朝向。

        Qt scene 的 y 轴向下，旋转角正方向与世界坐标相反。
        """
        self.setPos(
            float(self.signal.x) * PIXELS_PER_METER,
            -float(self.signal.y) * PIXELS_PER_METER,
        )

        self.setRotation(-math.degrees(float(self.signal.yaw)))

    def _display_id(self) -> str:
        if isinstance(
            self.signal,
            TrafficSignalSnapshot,
        ):
            return self.signal.entity_id

        return str(self.signal.signal_id)
