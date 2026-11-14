"""拖拽视觉统一层：缩略图（含长按浮卡）、拖拽热点、子图落点覆盖层。

四个拖拽入口（变量列表 / 图例 / 曲线长按 / 暂存区）与子图落点共用这里的
一份材质：色板全部取自 ``theme.py``（不新增品牌色），几何与字号也只在这里
定义一次，避免同屏出现四套蓝、四档圆角。

本模块只依赖 PySide6 与 ``theme``，不引业务模块 —— ``drag_drop`` 与各入口
都可以安全地顶层 import。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from src.ui import theme

# === 色板（全部复用 theme 既有值） ===
DRAG_ACCENT = theme.CHIP_SEL_BD          # #2F6FB4：落点边框与提示文字
DRAG_ACCENT_SOFT = theme.CHIP_CUR_BG     # #E8F1FD：有效落点底色
DRAG_SURFACE = QColor(255, 255, 255, 247)
DRAG_BD = "#CBD9E8"
DRAG_TEXT = theme.TEXT_PRIMARY
DRAG_MUTED = theme.TEXT_MUTED
DRAG_SEP = theme.SEP                     # #EAECEF：头部分隔线
DRAG_HOLLOW = theme.SEP_ON_BAR           # #C9CDD4：无颜色曲线的空心点
DRAG_WARN_TXT = "#8A5300"                # 跳过 / 已存在
DRAG_WARN_BG = "#FFF4E1"

# === 几何 ===
RADIUS = 8
PAD_X = 12
PAD_Y = 8
ROW_H = 20
HEADER_H = 20
DOT_SIZE = 8
DOT_GAP = 8
#: 头部行里「提示」与「条数」之间的最小间距。
HEADER_GAP = 10
#: 头部分隔线占用的垂直空间（线自身 + 上下留白）。
HEADER_SEP_H = 4
#: 头部提示的左缩进 = 色点 + 间距：与正文变量名共用一条文字左边缘，同时把
#: 光标热点所在的左上角让出来。
HEADER_TEXT_X = DOT_SIZE + DOT_GAP
MIN_WIDTH = 140
MAX_WIDTH = 320
MAX_VISIBLE_ROWS = 5

#: 缩略图外扩留白：既给自绘阴影腾位置，也让热点能落在 pixmap 内。
PIXMAP_PAD = 18

#: 光标在缩略图**可见表面**内的落点：箭头尖正好落在外框左上角，卡片向右下展开。
CURSOR_INSET = QPoint(0, 0)

#: 落点框：虚线边框距容器边缘的留白（尽量贴近边缘）与文字字号。
INDICATOR_INSET = 2
INDICATOR_FONT_PX = 16


@dataclass(frozen=True)
class DragItem:
    """缩略图中的一行：变量名 + 曲线颜色（None → 空心灰点）。"""

    name: str
    color: str | None = None


def copy_modifier_hint() -> str:
    """复制修饰键的展示文案，与 ``drag_drop.is_copy_modifier`` 语义一致。"""
    return "⌥" if sys.platform == "darwin" else "Alt"


def _sized_font(font: QFont, delta: int) -> QFont:
    """按控件字体派生字号，保留系统缩放（pointSize 与 pixelSize 两态都覆盖）。"""
    small = QFont(font)
    if small.pointSizeF() > 0:
        small.setPointSizeF(max(6.0, small.pointSizeF() + delta))
    elif small.pixelSize() > 0:
        small.setPixelSize(max(9, small.pixelSize() + delta))
    return small


def _pixel_font(font: QFont, px: int) -> QFont:
    fixed = QFont(font)
    fixed.setPixelSize(px)
    fixed.setBold(False)
    return fixed


def _count_text(total: int) -> str:
    return "1 条" if total <= 1 else f"共 {total} 条"


def _draw_shadow(painter: QPainter, surface: QRectF) -> None:
    """透明底自绘柔和阴影（不用 QGraphicsDropShadowEffect，避免控件裁剪）。"""
    painter.setPen(Qt.PenStyle.NoPen)
    for spread, alpha in ((7, 8), (5, 10), (3, 14), (1, 18)):
        painter.setBrush(QColor(15, 23, 42, alpha))
        painter.drawRoundedRect(
            surface.adjusted(-spread, -spread + 1, spread, spread + 1),
            RADIUS + spread,
            RADIUS + spread,
        )


def build_drag_pixmap(
    items: Sequence[DragItem],
    font: QFont | None = None,
    *,
    hint: str | None = None,
    max_rows: int = MAX_VISIBLE_ROWS,
) -> QPixmap | None:
    """构建拖拽缩略图：头部提示 + 条数 / 色点 + 名称。

    长按浮卡与四个拖拽入口共用本函数，文案/材质只有这一处来源。

    不画动作徽标：拖影位图在 ``QDrag.exec()`` 后由系统接管、无法中途替换，
    任何「移动 / 复制」字样都会在拖动途中过期。动作语义交给实时更新的落点
    提示，卡片只负责「搬什么」。

    Args:
        items: 每行的变量名与颜色（``color=None`` 画空心灰点）
        font: 控件字体；为 None 时用默认系统字体
        hint: 头部左侧的提示（修饰键说明），与右侧条数同行，None 则不画
        max_rows: 最多可见行数，超出折叠成 ``+N 条…``

    Returns:
        带 DPR 的 QPixmap；items 为空时返回 None。
    """
    if not items:
        return None
    if font is None:
        font = QFont()

    metrics = QFontMetrics(font)
    meta_font = _sized_font(font, -1)
    meta_metrics = QFontMetrics(meta_font)

    visible = list(items[:max_rows])
    overflow = len(items) - len(visible)
    name_avail = MAX_WIDTH - PAD_X * 2 - DOT_SIZE - DOT_GAP
    rows = [
        metrics.elidedText(item.name, Qt.TextElideMode.ElideRight, name_avail)
        for item in visible
    ]

    count_text = _count_text(len(items))
    count_w = meta_metrics.horizontalAdvance(count_text)
    hint_text = hint or ""
    hint_w = meta_metrics.horizontalAdvance(hint_text) if hint_text else 0
    header_w = count_w + (hint_w + HEADER_GAP + HEADER_TEXT_X if hint_text else 0)
    body_w = max(
        (metrics.horizontalAdvance(text) for text in rows),
        default=0,
    ) + DOT_SIZE + DOT_GAP
    if overflow:
        body_w = max(body_w, meta_metrics.horizontalAdvance(f"+{overflow} 条…"))

    # +1：elidedText 在「可用宽度恰好等于文本宽度」时仍会截掉最后一个字符
    # （内部逐字累加的舍入），留 1px 余量让提示完整显示。
    width = max(
        MIN_WIDTH,
        min(MAX_WIDTH, max(header_w, body_w) + PAD_X * 2 + 1),
    )
    body_h = len(rows) * ROW_H + (ROW_H if overflow else 0)
    height = PAD_Y * 2 + HEADER_H + HEADER_SEP_H + body_h

    dpr = 1.0
    if QGuiApplication.instance() is not None:
        # 按光标（= 拖拽起点）所在屏取 DPR：混合缩放下起拖屏与主屏比例不同时，
        # 用主屏比例会把卡片画糊或画大。
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            dpr = float(screen.devicePixelRatio() or 1.0)

    total_w = width + PIXMAP_PAD * 2
    total_h = height + PIXMAP_PAD * 2
    pixmap = QPixmap(int(round(total_w * dpr)), int(round(total_h * dpr)))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    # 先按逻辑坐标绘制，最后再设 DPR：不依赖 QPainter 是否自动按 DPR 缩放。
    painter.scale(dpr, dpr)

    surface = QRectF(PIXMAP_PAD, PIXMAP_PAD, width, height)
    _draw_shadow(painter, surface)
    painter.setBrush(DRAG_SURFACE)
    painter.setPen(QPen(QColor(DRAG_BD), 1))
    painter.drawRoundedRect(
        surface.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS
    )

    # 头部：左侧提示 + 右侧条数
    #
    # 提示放在头部左侧而不是另起一行页脚：卡片不出现多余的空白带，高度少
    # 一节。提示起点与下方变量名的文字列对齐（跳过色点列）——光标热点在
    # 卡片左上角，箭头本体约 16px 宽，正好落在这条色点列里，既避开「首字
    # 被箭头压住」，也让头部与正文共用一条文字左边缘。
    header_y = surface.top() + PAD_Y
    left = surface.left() + PAD_X
    text_x = left + HEADER_TEXT_X
    painter.setFont(meta_font)
    painter.setPen(QColor(DRAG_MUTED))
    if hint_text:
        hint_avail = max(
            0, int(width - PAD_X * 2 - HEADER_TEXT_X - count_w - HEADER_GAP)
        )
        painter.drawText(
            QRectF(text_x, header_y, hint_avail, HEADER_H),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            meta_metrics.elidedText(
                hint_text, Qt.TextElideMode.ElideRight, hint_avail
            ),
        )
    painter.drawText(
        QRectF(
            surface.right() - PAD_X - count_w,
            header_y,
            count_w,
            HEADER_H,
        ),
        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
        count_text,
    )

    # 头部分隔线：把「提示 + 条数」这类元信息与下方变量名分成两级
    sep_y = header_y + HEADER_H + HEADER_SEP_H // 2
    painter.setPen(QPen(QColor(DRAG_SEP), 1))
    painter.drawLine(
        QPoint(int(left), sep_y), QPoint(int(surface.right() - PAD_X), sep_y)
    )

    # 主体：色点 + 名称
    row_y = header_y + HEADER_H + HEADER_SEP_H
    painter.setFont(font)
    for item, text in zip(visible, rows):
        painter.setPen(Qt.PenStyle.NoPen)
        if item.color:
            painter.setBrush(QColor(item.color))
            painter.drawEllipse(
                QPoint(int(left + DOT_SIZE / 2), int(row_y + ROW_H / 2)), 4, 4
            )
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(DRAG_HOLLOW), 1.4))
            painter.drawEllipse(
                QPoint(int(left + DOT_SIZE / 2), int(row_y + ROW_H / 2)), 3, 3
            )
        painter.setPen(QColor(DRAG_TEXT))
        painter.drawText(
            QRectF(left + DOT_SIZE + DOT_GAP, row_y, name_avail, ROW_H),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        row_y += ROW_H

    if overflow:
        painter.setFont(meta_font)
        painter.setPen(QColor(DRAG_MUTED))
        painter.drawText(
            QRectF(left + DOT_SIZE + DOT_GAP, row_y, name_avail, ROW_H),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"+{overflow} 条…",
        )

    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def drag_hotspot() -> QPoint:
    """缩略图热点：箭头尖落在外框左上角，卡片向右下展开。

    Qt 的 ``QDrag.setHotSpot()`` 按逻辑像素解释 pixmap 内位置（内部会自行
    除以 pixmap 的 DPR），所以这里只能给逻辑坐标 —— 再乘 DPR 会让光标多
    偏移一个 ``PIXMAP_PAD``，箭头就会扎到卡片正文上。
    """
    return QPoint(PIXMAP_PAD + CURSOR_INSET.x(), PIXMAP_PAD + CURSOR_INSET.y())


def transfer_hint() -> str:
    """搬运类拖拽（图例 / 曲线 / 暂存区 / 长按浮卡）的头部提示。"""
    return f"拖到其他子图释放 · {copy_modifier_hint()} 复制"


class DragIndicatorOverlay(QWidget):
    """子图落点：贴近边缘的虚线框 + 淡底 + 居中大字号提示，全部自绘。

    只做一个透明覆盖层，不额外挂子控件；``text`` 含「跳过/已存在」时整块
    转琥珀色（文案本身仍由 ``plot_widget`` 生成，这里只负责呈现）。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._text = ""

    def set_content(self, text: str) -> None:
        self._text = text or ""
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        frame = QRectF(self.rect()).adjusted(
            INDICATOR_INSET,
            INDICATOR_INSET,
            -INDICATOR_INSET,
            -INDICATOR_INSET,
        )
        if frame.width() <= 8 or frame.height() <= 8:
            return
        warn = "跳过" in self._text or "已存在" in self._text
        accent = QColor(DRAG_WARN_TXT if warn else DRAG_ACCENT)
        fill = QColor(DRAG_WARN_BG if warn else DRAG_ACCENT_SOFT)
        fill.setAlpha(48 if warn else 36)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dash = QPen(accent, 2, Qt.PenStyle.DashLine)
        dash.setDashPattern([5, 3])
        painter.setPen(dash)
        painter.setBrush(fill)
        painter.drawRoundedRect(frame, RADIUS, RADIUS)

        # 居中提示：大字号、单行省略，框太窄也不会溢出。
        font = _pixel_font(self.font(), INDICATOR_FONT_PX)
        font.setBold(True)
        metrics = QFontMetrics(font)
        avail = max(0.0, frame.width() - 16)
        label = metrics.elidedText(self._text, Qt.TextElideMode.ElideRight, int(avail))
        painter.setFont(font)
        painter.setPen(accent)
        painter.drawText(frame, int(Qt.AlignmentFlag.AlignCenter), label)
        painter.end()
