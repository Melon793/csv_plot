import math
import time

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QElapsedTimer, QEventLoop, Signal
from PySide6.QtGui import (
    QPainter,
    QPixmap,
    QColor,
    QLinearGradient,
    QFont,
    QPen,
    QBrush,
    QPainterPath,
)
from PySide6.QtWidgets import QWidget

from src.core.startup_metrics import get_metrics

ICON_SIZE = 115

#: 启动看门狗：主窗口构造超过这个时长仍未发出 ``finished``，就认定启动失败。
SPLASH_WAIT_TIMEOUT_MS = 60_000


class SplashScreen(QWidget):
    """轻量启动图：静态层预渲染，仅呼吸环/进度条逐帧重绘。"""

    finished = Signal()

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window,
        )
        self.width = 500
        self.height = 280
        self._is_shown = False
        self._event_loop = None
        self._static_layer: QPixmap | None = None
        self._static_layer_dpr: float | None = None

        self.setFixedSize(self.width, self.height)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.elapsed_timer = QElapsedTimer()
        self.elapsed = 0

        # Qt 会自动回退缺失字体，无需为 splash 每帧/首帧枚举系统字体表。
        self._title_font = QFont("Space Grotesk", 34, QFont.Weight.Bold)
        self._title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        self._subtitle_font = QFont("Space Grotesk", 10)
        self._subtitle_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        self._subtitle_font.setCapitalization(QFont.Capitalization.AllUppercase)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._on_timer)

    def show(self):
        super().show()
        self._is_shown = True
        self.elapsed_timer.start()
        self.timer.start(16)

    def showEvent(self, event):
        super().showEvent(event)
        self._rebuild_static_layer()

    def _rebuild_static_layer(self) -> None:
        dpr = max(1.0, self.devicePixelRatioF())
        self._static_layer_dpr = dpr
        layer = QPixmap(int(self.width * dpr), int(self.height * dpr))
        layer.setDevicePixelRatio(dpr)
        layer.fill(Qt.GlobalColor.transparent)
        painter = QPainter(layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.draw_background(painter)
        self.draw_grid(painter)
        self.draw_icon_base(painter)
        self.draw_text(painter)
        painter.end()
        self._static_layer = layer

    def _on_timer(self):
        if not self._is_shown:
            return
        self.elapsed = self.elapsed_timer.elapsed()
        self.update()

    def finish(self, widget):
        self._is_shown = False
        self.timer.stop()
        self.close()

    def close(self):
        self._is_shown = False
        self.timer.stop()
        super().close()

    def wait_for_completion(self, timeout_ms: int = SPLASH_WAIT_TIMEOUT_MS) -> bool:
        """启动嵌套事件循环，直到 finished 信号触发或看门狗超时。"""
        if self._event_loop is not None:
            return True
        loop = QEventLoop()
        self._event_loop = loop
        timed_out: list[bool] = []

        watchdog = QTimer(self)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: timed_out.append(True))
        watchdog.timeout.connect(loop.quit)
        self.finished.connect(loop.quit)
        if timeout_ms > 0:
            watchdog.start(timeout_ms)
        try:
            loop.exec()
        finally:
            watchdog.stop()
            try:
                self.finished.disconnect(loop.quit)
            except (RuntimeError, TypeError):
                pass
            self._event_loop = None
        return not timed_out

    def signal_finish(self):
        self.finished.emit()

    def paintEvent(self, event):
        started = time.perf_counter()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        dpr = max(1.0, self.devicePixelRatioF())
        if self._static_layer is None or not math.isclose(
            self._static_layer_dpr or 0.0, dpr, rel_tol=0.0, abs_tol=1e-6
        ):
            self._rebuild_static_layer()
        painter.drawPixmap(0, 0, self._static_layer)
        self.draw_breathe_ring(painter)
        self.draw_progress(painter)
        painter.end()
        get_metrics().add_frame(started, time.perf_counter())

    def draw_background(self, painter: QPainter):
        gradient = QLinearGradient(0, 0, self.width, self.height)
        gradient.setColorAt(0, QColor(250, 250, 250))
        gradient.setColorAt(1, QColor(245, 245, 245))

        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width, self.height), 12, 12)
        painter.fillPath(path, QBrush(gradient))
        painter.setPen(QPen(QColor(224, 224, 224), 1))
        painter.drawPath(path)

    def draw_grid(self, painter: QPainter):
        grid_size = 40
        painter.setPen(QPen(QColor(100, 116, 139, 12), 1))
        for y in range(0, self.height, grid_size):
            painter.drawLine(0, y, self.width, y)
        for x in range(0, self.width, grid_size):
            painter.drawLine(x, 0, x, self.height)

    def draw_icon_base(self, painter: QPainter):
        center_x = self.width // 2
        center_y = 100
        icon_radius = ICON_SIZE // 2
        painter.save()
        painter.translate(center_x, center_y)
        painter.setBrush(QColor(14, 165, 233))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), icon_radius, icon_radius)

        arc_width = int(ICON_SIZE * 0.08)
        arc_center_offset_x = int(ICON_SIZE * 0.175)
        arc_center_offset_y = int(ICON_SIZE * 0.175)
        arc_radius = int(ICON_SIZE * 0.7 // 2)
        pen = QPen(QColor(255, 255, 255), arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            QRectF(
                arc_center_offset_x - arc_radius,
                arc_center_offset_y - arc_radius,
                arc_radius * 2,
                arc_radius * 2,
            ),
            80 * 16,
            100 * 16,
        )
        painter.restore()

    def draw_breathe_ring(self, painter: QPainter):
        phase = (self.elapsed % 2000) / 2000 * 2 * math.pi
        factor = (math.sin(phase) + 1) / 2
        opacity = 0.2 + 0.4 * factor
        scale = 1 + 0.12 * factor
        painter.save()
        painter.translate(self.width // 2, 100)
        painter.scale(scale, scale)
        painter.setPen(QPen(QColor(14, 165, 233, int(opacity * 255)), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), ICON_SIZE // 2 + 8, ICON_SIZE // 2 + 8)
        painter.restore()

    def draw_icon(self, painter: QPainter):
        """Compatibility helper for tests/callers that draw the complete icon."""
        self.draw_icon_base(painter)
        self.draw_breathe_ring(painter)

    def draw_text(self, painter: QPainter):
        painter.setFont(self._title_font)
        painter.setPen(QColor(26, 26, 26))
        painter.drawText(
            QRectF(0, 180, self.width, 40),
            Qt.AlignmentFlag.AlignCenter,
            "CSV PLOT",
        )

        painter.setFont(self._subtitle_font)
        painter.setPen(QColor(102, 102, 102))
        painter.drawText(
            QRectF(0, 230, self.width, 15),
            Qt.AlignmentFlag.AlignCenter,
            "Data Analysis Tool",
        )

    def draw_progress(self, painter: QPainter):
        bar_width = 100
        bar_height = 4
        bar_x = (self.width - bar_width) // 2
        bar_y = 255

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(14, 165, 233, 38))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_width, bar_height), 2, 2)

        progress = (self.elapsed % 1500) / 1500
        eased = progress * progress * (3 - 2 * progress)
        bar_segment_width = bar_width // 3
        bar_segment_x = bar_x - bar_segment_width + (bar_width + bar_segment_width) * eased

        gradient = QLinearGradient(bar_segment_x, 0, bar_segment_x + bar_segment_width, 0)
        gradient.setColorAt(0, QColor(14, 165, 233, 0))
        gradient.setColorAt(0.5, QColor(14, 165, 233))
        gradient.setColorAt(1, QColor(14, 165, 233, 0))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(
            QRectF(bar_segment_x, bar_y, bar_segment_width, bar_height), 2, 2
        )
