"""
DragDropHandler —— 变量拖放统一工具类

封装多变量名解析（;; 分隔）、MIME 数据构建、拖拽预览图生成等可复用逻辑，
消除 MyTableWidget / DataTableDialog / MainWindow / DraggableGraphicsLayoutWidget
中 5+ 处重复的 split(';;') 模式。

另提供 legend 拖拽专用扩展（legend_drag_to_plot_design.md §3.2）：
- LEGEND_MIME_FORMAT：自定义 MIME 格式，标记拖拽来源为 plot legend
- build_legend_var_mimedata：legend 来源 MIME（text/plain 兼容层 + 源 plot id）
- parse_anchor_var_name：curve:///xxx 锚点解析（legend 点击/拖拽共用）
- 拖拽上下文注册表：drop 端反查活的源 plot 对象
"""

from typing import Any
from urllib.parse import unquote

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import (
    QFontMetrics,
    QPainter,
    QPen,
    QColor,
    QPixmap,
    QFont,
)
from PySide6.QtCore import QMimeData

VAR_SEPARATOR = ";;"

# legend 锚点 URL 协议（curve:///URL编码变量名），与 multi_curve_manager 共用
ANCHOR_SCHEME = "curve"

# legend 来源拖拽的自定义 MIME 格式（text/plain 保持兼容层不变）
LEGEND_MIME_FORMAT = "application/x-csvplot-legend"

# 拖拽上下文注册表：发起 legend 拖拽前写入，drag.exec 返回后清空。
# drop 端通过 MIME 中的源 plot id 与注册表比对，拿到活的源 plot 对象。
_active_legend_drag: dict[str, Any] = {}


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


def build_legend_var_mimedata(var_names: list[str], source_plot: Any) -> QMimeData:
    """legend 来源的 MIME：text/plain 兼容层 + 自定义格式携带源 plot id。

    text/plain 保持与变量列表拖拽一致，兼容所有现有 drop 目标
    （数据表弹窗/变量编辑器等）；自定义格式仅供 plot drop 端识别来源。
    """
    mime = build_var_mimedata(var_names)
    mime.setData(LEGEND_MIME_FORMAT, str(id(source_plot)).encode())
    return mime


def parse_anchor_var_name(href: str) -> str | None:
    """解析 curve:///xxx 锚点为变量名（curve 前缀 + unquote）。

    空串/非法 scheme/空变量名返回 None。供 LegendTextBrowser（press 瞬间
    解析拖拽变量）与 MultiCurveManager._on_legend_anchor_clicked 共用。
    """
    if not href:
        return None
    prefix = f"{ANCHOR_SCHEME}:///"
    if not href.startswith(prefix):
        return None
    name = unquote(href[len(prefix):])
    return name or None


def set_active_legend_drag(source_plot: Any, var_names: list[str]) -> None:
    """登记当前 legend 拖拽上下文（drag.exec 前调用）"""
    _active_legend_drag.clear()
    _active_legend_drag["source"] = source_plot
    _active_legend_drag["vars"] = list(var_names)


def clear_active_legend_drag() -> None:
    """清理 legend 拖拽上下文（drag.exec 返回后调用）"""
    _active_legend_drag.clear()


def is_legend_drag_active() -> bool:
    """当前是否存在进行中的 legend 拖拽（供指示器轮询区分 Alt/Shift）"""
    return bool(_active_legend_drag)


def get_active_legend_drag_source(mime_data) -> Any | None:
    """drop 端反查活的源 plot 对象。

    要求：自定义格式存在 + MIME 内 id 与注册表一致（防伪造冗余校验）+ 对象存活
    （源 plot 在拖拽期间被关闭时静默返回 None，调用方降级为复制）。
    """
    if not _active_legend_drag or mime_data is None:
        return None
    if not mime_data.hasFormat(LEGEND_MIME_FORMAT):
        return None
    try:
        payload_id = int(bytes(mime_data.data(LEGEND_MIME_FORMAT)).decode())
    except (ValueError, UnicodeDecodeError):
        return None
    source = _active_legend_drag.get("source")
    if source is None or id(source) != payload_id:
        return None
    try:
        # 访问已销毁的 C++ 对象会抛 RuntimeError（popup 关闭等场景）
        source.window()
    except RuntimeError:
        return None
    return source


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
