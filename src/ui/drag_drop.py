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

import json
from dataclasses import dataclass
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

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    SYMBOL_POLICY_AUTO,
)

VAR_SEPARATOR = ";;"

# legend 锚点 URL 协议（curve:///URL编码变量名），与 multi_curve_manager 共用
ANCHOR_SCHEME = "curve"

# legend 色块锚点协议（color:///URL编码变量名）：左键弹调色板菜单（图例快速改色）
COLOR_SCHEME = "color"

# 锚点 scheme → 动作映射：curve:/// 切显隐（失效占位条目同协议，由分发端
# 按变量归属区分），color:/// 打开调色板
_LEGEND_ANCHOR_ACTIONS: dict[str, str] = {
    ANCHOR_SCHEME: "toggle",
    COLOR_SCHEME: "color",
}

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
    （变量数值表/变量编辑器等）；自定义格式仅供 plot drop 端识别来源。
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


def parse_legend_anchor(href: str) -> tuple[str, str] | None:
    """解析 legend 锚点 → (action, var_name)；action ∈ {"toggle", "color"}。

    toggle 即原 curve:/// 显隐锚点（失效占位条目同协议，由分发端按变量
    归属区分），color 为色块改色锚点。非法/空/空变量名返回 None。
    供 LegendTextBrowser（press 宽松解析：两种 scheme 都可发起拖拽/
    右键菜单）与 MultiCurveManager._on_legend_anchor_clicked（按 scheme
    分发）共用；parse_anchor_var_name 原样保留给既有调用方。
    """
    if not href:
        return None
    for scheme, action in _LEGEND_ANCHOR_ACTIONS.items():
        prefix = f"{scheme}:///"
        if href.startswith(prefix):
            name = unquote(href[len(prefix):])
            return (action, name) if name else None
    return None


def set_active_legend_drag(
    source_plot: Any, var_names: list[str], *, batch: bool = False
) -> None:
    """登记当前 legend 拖拽上下文（drag.exec 前调用）。

    ``batch=True`` 表示 P1 多选批量拖拽：drop 端按 skip + quiet 处理，
    与单变量 legend 拖拽（ask/notify 确认窗）区分。不能用变量个数推断
    —— 批量选择集可能只有 1 条。
    """
    _active_legend_drag.clear()
    _active_legend_drag["source"] = source_plot
    _active_legend_drag["vars"] = list(var_names)
    _active_legend_drag["batch"] = bool(batch)


def clear_active_legend_drag() -> None:
    """清理 legend 拖拽上下文（drag.exec 返回后调用）"""
    _active_legend_drag.clear()


def is_legend_drag_active() -> bool:
    """当前是否存在进行中的 legend 拖拽（供指示器轮询区分 Alt/Shift）"""
    return bool(_active_legend_drag)


def is_active_legend_batch_drag() -> bool:
    """当前 legend 拖拽是否为 P1 多选批量（决定 drop 端判重策略）。"""
    return bool(_active_legend_drag.get("batch"))


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


# ---------------------------------------------------------------------------
# 曲线暂存区（CurveShelf）拖拽协议
#
# 托盘拖出走独立 MIME：text/plain 兼容层仍写变量名（变量数值表/编辑器等
# 既有纯文本 drop 目标不受影响），自定义格式携带托盘 id + 条目 key 列表。
# 条目快照自持于托盘，drop 端经 id 比对 + 存活校验反查活托盘。
# ---------------------------------------------------------------------------
SHELF_MIME_FORMAT = "application/x-csvplot-shelf"

_active_shelf_drag: dict[str, Any] = {}


@dataclass(frozen=True)
class CurveStyleSnapshot:
    """曲线样式快照：搬运入区/出区全程自持，出区零源依赖。

    ``draw_mode_source`` 必须原样保留：出区落地后要回写（apply 入口会把
    来源强制钉成 'user'，直接落 'auto' 快照会改变 reload 后的行为）。
    """

    color: str = ""
    visible: bool = True
    draw_mode: str = DRAW_MODE_AUTO
    symbol_policy: str = SYMBOL_POLICY_AUTO
    draw_mode_source: str = DRAW_MODE_SOURCE_AUTO

    @classmethod
    def from_curve_info(cls, ci: Any) -> "CurveStyleSnapshot":
        """从 CurveInfo 拷贝全部样式值（须在源存活期调用）。"""
        return cls(
            color=getattr(ci, "color", "") or "",
            visible=bool(getattr(ci, "visible", True)),
            draw_mode=getattr(ci, "draw_mode", DRAW_MODE_AUTO),
            symbol_policy=getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO),
            draw_mode_source=getattr(
                ci, "draw_mode_source", DRAW_MODE_SOURCE_AUTO
            ),
        )


def build_shelf_mimedata(
    entry_keys: list[str], var_names: list[str], shelf: Any
) -> QMimeData:
    """托盘来源 MIME：text/plain 兼容层 + 自定义格式（托盘 id + 条目 key）。"""
    mime = build_var_mimedata(list(var_names))
    payload = json.dumps(
        {"id": id(shelf), "keys": [str(k) for k in entry_keys]},
        ensure_ascii=False,
    )
    mime.setData(SHELF_MIME_FORMAT, payload.encode("utf-8"))
    return mime


def parse_shelf_mime_payload(mime_data) -> dict | None:
    """解析托盘 MIME 自定义格式 → {"id": int|None, "keys": [str, ...]}。

    无效 JSON / 结构不符 / 空 key 列表返回 None（调用方静默降级）。
    """
    if mime_data is None or not mime_data.hasFormat(SHELF_MIME_FORMAT):
        return None
    try:
        data = json.loads(bytes(mime_data.data(SHELF_MIME_FORMAT)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    keys = data.get("keys")
    if not isinstance(keys, list):
        return None
    cleaned = [str(k) for k in keys if isinstance(k, str) and k]
    if not cleaned:
        return None
    return {"id": data.get("id"), "keys": cleaned}


def set_active_shelf_drag(shelf: Any, entry_keys: list[str]) -> None:
    """登记当前托盘拖拽上下文（drag.exec 前调用）。"""
    _active_shelf_drag.clear()
    _active_shelf_drag["source"] = shelf
    _active_shelf_drag["keys"] = [str(k) for k in entry_keys]


def clear_active_shelf_drag() -> None:
    """清理托盘拖拽上下文（drag.exec 返回后调用）。"""
    _active_shelf_drag.clear()


def is_shelf_drag_active() -> bool:
    """当前是否存在进行中的托盘拖拽（供 hover 指示器文案分支）。"""
    return bool(_active_shelf_drag)


def get_active_shelf_drag_source(mime_data) -> Any | None:
    """drop 端反查活的托盘对象。

    要求：自定义格式存在 + MIME 内 id 与注册表一致（防伪造冗余校验）+
    对象存活（拖拽期间托盘被关闭时静默返回 None，调用方降级忽略）。
    """
    payload = parse_shelf_mime_payload(mime_data)
    if payload is None or not _active_shelf_drag:
        return None
    source = _active_shelf_drag.get("source")
    if source is None or id(source) != payload.get("id"):
        return None
    try:
        # 访问已销毁的 C++ 对象会抛 RuntimeError
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
