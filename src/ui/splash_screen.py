 
import math
import sys
from pathlib import Path
 
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QElapsedTimer
from PySide6.QtGui import (
    QPainter,
    QPixmap,
    QColor,
    QLinearGradient,
    QFont,
    QFontDatabase,
    QPen,
    QBrush,
    QPainterPath,
)
from PySide6.QtWidgets import QWidget
 
ICON_SIZE = 115
 
 
def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    elif getattr(sys, "frozen", False):
        return Path(sys.executable).parent / relative_path
    else:
        return Path(__file__).parent / relative_path
 
 
class SplashScreen(QWidget):

    def __init__(self):
        super().__init__()
        self.width = 500
        self.height = 280
        self.use_custom_icon = True
        self._is_shown = False

        self.setFixedSize(self.width, self.height)

        self.icon_path = resource_path("assets/icon.png")
        self.icon_pixmap = QPixmap(str(self.icon_path))

        self.elapsed_timer = QElapsedTimer()
        self.elapsed = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
 
    def show(self):
        super().show()
        self._is_shown = True
        self.elapsed_timer.start()
        self.timer.start(16)
 
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
 
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
 
        self.draw_background(painter)
        self.draw_grid(painter)
        self.draw_icon(painter)
        self.draw_text(painter)
        self.draw_progress(painter)
 
    def draw_background(self, painter: QPainter):
        gradient = QLinearGradient(0, 0, self.width, self.height)
        gradient.setColorAt(0, QColor(250, 250, 250))
        gradient.setColorAt(1, QColor(245, 245, 245))
 
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width, self.height), 12, 12)
        painter.fillPath(path, QBrush(gradient))
 
        pen = QPen(QColor(224, 224, 224), 1)
        painter.setPen(pen)
        painter.drawPath(path)
 
    def draw_grid(self, painter: QPainter):
        grid_size = 40
        color = QColor(100, 116, 139, 12)
 
        pen = QPen(color, 1)
        painter.setPen(pen)
 
        for y in range(0, self.height, grid_size):
            painter.drawLine(0, y, self.width, y)
 
        for x in range(0, self.width, grid_size):
            painter.drawLine(x, 0, x, self.height)
 
    def draw_icon(self, painter: QPainter):
        # ==================== 图标参数配置 ====================
        # 图标中心位置
        center_x = self.width // 2
        center_y = 100
        
        # 图标大小（直径）
        icon_size = ICON_SIZE
        icon_radius = icon_size // 2
        
        # ==================== 呼吸动画计算 ====================
        # 2秒一个周期的正弦波动画
        phase = (self.elapsed % 2000) / 2000 * 2 * math.pi
        factor = (math.sin(phase) + 1) / 2  # 0→1 平滑过渡
        
        # 呼吸环参数
        breathe_opacity = 0.2 + 0.4 * factor  # 透明度: 0.2→0.6
        breathe_scale = 1 + 0.12 * factor     # 缩放: 1→1.12
 
        # ==================== 绘制图标 ====================
        painter.save()
        painter.translate(center_x, center_y)
 
        # 1. 蓝色背景圆
        painter.setBrush(QColor(14, 165, 233))  # #0ea5e9
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), icon_radius, icon_radius)
 
        # 2. 白色弧线图标（圆心在主圆右下角）
        # 参数说明:
        # - pen width: 弧线粗细 (18px)
        # - arc_center_offset_x/y: 弧线圆心相对于主圆的偏移量 (5, 5)
        # - arc_radius: 弧线半径 (35)
        # - arc_start_angle: 起始角度 (225度)
        # - arc_span: 扫过角度 (100度)
        arc_width = ICON_SIZE * 0.08 // 1 
        arc_center_offset_x = ICON_SIZE * 0.175 // 1   # 圆心向右偏移5px
        arc_center_offset_y = ICON_SIZE * 0.175 // 1     # 圆心向下偏移5px
        arc_radius = ICON_SIZE * 0.7 // 2           # 弧线半径
        pen = QPen(QColor(255, 255, 255), arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)  # 圆角端点
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # QRectF: 左上角x = 圆心偏移 - 半径, 左上角y = 圆心偏移 - 半径, 宽, 高
        painter.drawArc(QRectF(arc_center_offset_x - arc_radius, arc_center_offset_y - arc_radius, arc_radius * 2, arc_radius * 2), 80 * 16, 100 * 16)
 
        painter.restore()
 
        # ==================== 绘制呼吸环 ====================
        painter.save()
        painter.translate(center_x, center_y)
        painter.scale(breathe_scale, breathe_scale)  # 缩放动画
 
        breathe_pen = QPen(QColor(14, 165, 233, int(breathe_opacity * 255)), 4)
        painter.setPen(breathe_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), icon_radius + 8, icon_radius + 8)
 
        painter.restore()
 
    def draw_text(self, painter: QPainter):
        font_family = "Space Grotesk"
        if font_family not in QFontDatabase.families():
            font_family = "Arial"
 
        title_font = QFont(font_family, 34, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(title_font)
        painter.setPen(QColor(26, 26, 26))
 
        painter.drawText(
            # 180 是标题的y坐标
            QRectF(0, 180, self.width, 40),
            Qt.AlignmentFlag.AlignCenter,
            "CSV PLOT",
        )
 
        subtitle_font = QFont(font_family, 10)
        subtitle_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        subtitle_font.setCapitalization(QFont.Capitalization.AllUppercase)
        painter.setFont(subtitle_font)
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
 
 
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
 
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    QTimer.singleShot(3000, splash.close)
    sys.exit(app.exec())