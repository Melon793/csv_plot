"""变量区折叠态细边栏（D1）＋左缘可点 affordance（D-b A）。

左栏折叠后剩下的 28px 竖条：» 按钮＋竖排「变量」二字常驻，**整条**都是
展开热区（点击任意位置都发 ``expand_requested``）。恢复入口是界面的一部分
而不是纪律问题 —— 这正是细边栏方案对「完全隐藏」的本质优势，折叠态才敢
持久化（D12）。

静置 affordance（拍板 D-b A）：rail 浅底＋右弱边框成"条"、» 为原生壳＋
ACCENT 矢量字形小按钮（C10a D-e：chrome 归平台、品牌色只作内容）、
splitter 手柄中部三点 grip —— 静置即答"能点"。hover 反馈按交互
类型分工（1E）：rail 是单击面，给浅底加深；手柄是拖拽面，**hover 保持
main 原样**（光标↔＋tooltip），不变色不改宽 —— 改宽会重排 splitter 扣
pane 宽，绘图区真实 resize 跳动（真机报障）。色值全复用 theme 既有 token
（ACCENT / TINT_SOFT / BORDER_SOFT，全平台固定品牌蓝），不新增颜色。

本体只发信号，折叠/展开的状态迁移统一走
``LayoutManager.set_var_pane_collapsed``，播报也收在那里。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QSplitterHandle,
    QVBoxLayout,
    QWidget,
)

from src.core.config import VAR_PANE_RAIL_WIDTH
from src.ui import theme
from src.ui.widgets.chrome_button import ChromeButton

#: grip 三点间距/半径：5px 手柄内唯一放得下的 glyph 尺度
GRIP_DOT_STEP = 5.0
GRIP_DOT_RADIUS = 1.0


class _RailExpandChip(ChromeButton):
    """rail » 芯片（D-e）：原生壳＋ACCENT 矢量双 chevron。

    取代 QSS 实色壳（C8b 旧形态）：实色背景/圆角是 chrome，归平台；
    品牌蓝只留在 » 字形上作内容。hover/pressed 反馈由壳的 state 位免费。
    """

    def _glyph_color(self) -> QColor:
        # 内容色固定 ACCENT（状态交给壳）；禁用退次级灰
        return QColor(theme.ACCENT if self.isEnabled() else theme.TEXT_MUTED)

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        center = QPointF(self.rect().center())
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        for dx in (-2.5, 2.5):  # 双 chevron：宽 3 高 8，整组水平居中
            painter.drawLine(
                center + QPointF(dx - 1.5, -4.0), center + QPointF(dx + 1.5, 0.0)
            )
            painter.drawLine(
                center + QPointF(dx + 1.5, 0.0), center + QPointF(dx - 1.5, 4.0)
            )


class VarPaneRail(QWidget):
    """折叠态细边栏：固定 28px 宽，整条可点。"""

    expand_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(VAR_PANE_RAIL_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("单击展开变量区 (Ctrl+B)")
        # _plain QWidget 子类的样式表背景必须显式置 WA_StyledBackground：
        # 不置时自 grab 有底、经父级渲染路径（grab/render）背景被跳过（实测）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 静置"条"感：10% 浅底＋右弱边框；hover 加深一档作交互确认
        self.setStyleSheet(
            f"VarPaneRail {{ background: {theme.TINT_SOFT};"
            f" border-right: 1px solid {theme.BORDER_SOFT}; }}"
            f"VarPaneRail:hover {{ background:"
            f" {theme.mix_hex(theme.ACCENT, '#FFFFFF', 0.8)}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self.expand_btn = _RailExpandChip(self)
        self.expand_btn.setToolTip("单击展开变量区 (Ctrl+B)")
        self.expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # F1：24×24＝mac Mini 档默认高＋WA_MacMiniSize → 圆角 aqua 与顶栏
        # 同族（qmacstyle_mac.mm:356/:1826）；mac-only 属性，Win/Fusion 忽略
        self.expand_btn.setAttribute(Qt.WidgetAttribute.WA_MacMiniSize, True)
        self.expand_btn.setFixedSize(24, 24)
        self.expand_btn.clicked.connect(self.expand_requested)
        layout.addWidget(self.expand_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        # 竖排「变量」：一字一行。QLabel 默认不吃鼠标点击，事件会冒到
        # 本体的 mousePressEvent —— 文字区也就是热区的一部分
        self._title_label = QLabel("变\n量", self)
        self._title_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._title_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.expand_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class _GripHandle(QSplitterHandle):
    """splitter 手柄：中部三点 grip（静置 affordance）。

    静置时灰线没有任何"我是控件"的信号（D-b 投诉根因）；grip 是
    JetBrains 分界线同口径的静置 affordance。**hover 不做任何视觉/布局
    变化**（1E，main 原样）：拖拽面的 hover 反馈由光标↔与 tooltip 承担，
    变色/改宽都是冗余信号，改宽还会重排 splitter 引起绘图区跳动。
    """

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.BORDER_SOFT))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        for dy in (-GRIP_DOT_STEP, 0.0, GRIP_DOT_STEP):
            painter.drawEllipse(QPointF(cx, cy + dy), GRIP_DOT_RADIUS, GRIP_DOT_RADIUS)
        painter.end()


class RailSplitter(QSplitter):
    """主 splitter：手柄换 _GripHandle（createHandle 虚函数覆写）。"""

    def createHandle(self):
        return _GripHandle(self.orientation(), self)