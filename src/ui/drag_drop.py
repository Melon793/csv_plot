"""
DragDropHandler —— 变量拖放统一工具类

封装多变量名解析（;; 分隔）、MIME 数据构建、拖拽预览图生成等可复用逻辑，
消除 MyTableWidget / DataTableDialog / MainWindow / DraggableGraphicsLayoutWidget
中 5+ 处重复的 split(';;') 模式。
"""

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import (
    QFontMetrics,
    QPainter,
    QPen,
    QColor,
    QPixmap,
    QFont,
)
from PyQt6.QtCore import QMimeData

VAR_SEPARATOR = ";;"


def parse_var_names_from_mimedata(mime_data) -> list[str]:
    """从 QMimeData 解析变量名列表（去重去空）。

    返回去重去空的变量名列表，保持原始顺序。
    """
    if not mime_data.hasText():
        return []
    text = mime_data.text()
    seen: set[str] = set()
    result: list[str] = []
    for name in text.split(VAR_SEPARATOR):
        name = name.strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def build_var_mimedata(var_names: list[str]) -> QMimeData:
    """构建包含多个变量名的 QMimeData，用 VAR_SEPARATOR 连接。"""
    mime = QMimeData()
    mime.setText(VAR_SEPARATOR.join(var_names))
    return mime


def create_drag_pixmap(
    var_names: list[str],
    font: QFont | None = None,
) -> QPixmap | None:
    """创建拖拽时显示的变量名缩略图。

    Args:
        var_names: 变量名列表
        font: 绘制字体，为 None 时使用默认系统字体

    Returns:
        QPixmap 缩略图，var_names 为空时返回 None
    """
    if not var_names:
        return None

    if font is None:
        font = QFont()

    metrics = QFontMetrics(font)
    bullet_names = [f"• {name}" for name in var_names]
    max_visible = 8
    display_lines = bullet_names[:max_visible]
    if len(bullet_names) > max_visible:
        display_lines.append(f"... 共{len(var_names)}项")

    text_width = max(
        (metrics.horizontalAdvance(line) for line in display_lines),
        default=80,
    )
    margin = 12
    line_height = metrics.lineSpacing()
    width = max(140, text_width + margin * 2)
    height = line_height * len(display_lines) + margin * 2

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#2b6def"), 2))
    painter.setBrush(QColor(255, 255, 255, 240))
    painter.drawRoundedRect(1, 1, width - 2, height - 2, 10, 10)
    painter.setPen(QColor("#1f1f1f"))

    y = margin + metrics.ascent()
    for line in display_lines:
        painter.drawText(QPoint(margin, y), line)
        y += line_height

    painter.end()
    return pixmap
