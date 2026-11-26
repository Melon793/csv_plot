"""Lightweight drag-and-drop overlay shared by the main window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget


class DropOverlay(QWidget):
    """显示文件拖入应用的半透明覆盖层，不依赖 pandas/数据表模块。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setStyleSheet("""
            background:rgba(255,255,255,200);
            border:none;
        """)

        self.label = QLabel("请丢入数据", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            background-color: rgba(168, 168, 168, 255);
            color:#333;
            font-size:36px;
            border-radius:12px;
            padding:20px 40px;
            color: rgba(128, 128, 128, 200);
        """)
        self.hide()

    def adjust_text(self, file_type_supported=True):
        self.label.setText("请丢入数据" if file_type_supported else "数据格式不支持")

    def adjust_font(self):
        side = min(self.label.width(), self.label.height())
        font_size = max(12, min(int(side * 0.3), 128))
        font = self.label.font()
        font.setPixelSize(font_size)
        font.setBold(True)
        self.label.setFont(font)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w_half = self.width()
        h_half = self.height()
        self.label.setFixedSize(w_half, h_half)
        self.adjust_font()
        self.label.move(
            (self.width() - self.label.width()) // 2,
            (self.height() - self.label.height()) // 2,
        )
