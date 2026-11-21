"""绘图区标签容器（PlotTabsWidget）。

设计文档 §4.4：选 ``QTabWidget`` 子类，可拖动 / 省略号；标签尾部放「＋」
新建页（平时跟随最后一个标签，页签放不下时停到暂存区胶囊左侧的预留槽）。
**不启用** ``setTabBarAutoHide(True)``：它会把「＋」一并隐藏，导致只有一页时
入口消失。

未加载数据时整个标签栏隐藏（同 VSCode：无文件即无标签栏），数据闸门
``set_new_page_enabled(True)`` 开启后显示且从此常显——空标签没有信息量，
加载前隐藏能让绘图区上移、入口心智更简单。

标签为 VSCode 扁平风：无圆角、激活页顶部 2px 强调线、标签相互贴合。关闭
按钮与「＋」均为 QPainter 矢量自绘：QSS 无法控制原生关闭图标的尺寸与对齐
（Windows 下贴右缘、竖直偏下），文本字形则随平台基线漂移；自绘后三平台
观感一致且任意 DPI 清晰。

「＋」的落位由 ``_TabsBar.tabLayoutChange`` 官方布局钩子驱动：插 / 删 /
拖动 / **重命名**等一切布局变化都会重算位置（PySide6 未暴露信号版
``tabLayoutChanged``，故经 ``setTabBar`` 挂入子类重写虚方法）。

原生滚动箭头不再显示（方向 B）：溢出导航由页签区滚轮（T4）与角位「▾」
页目录（T6）承担，右缘让位给页签。按钮本体保留为隐藏态——Qt 无公开滚动
API，滚轮经程序化 ``click()`` 驱动其内部偏移；Qt 每次布局会重新显示它们，
``tabLayoutChange`` 钩子里再藏回去。

本类只负责标签交互与信号转发，页对象的增删与切页时序由 MainWindow 掌握。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QTabBar,
    QTabWidget,
    QToolTip,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.drag_drop import SHELF_MIME_FORMAT
from src.ui import theme
from src.ui.drag_visuals import DRAG_ACCENT, DRAG_ACCENT_SOFT

#: 绘图页数量上限（设计文档 §4.4；2026-11-20 由 8 提升到 16）
MAX_PAGES = 16

#: 「＋」预留槽宽度：页签放不下时停到暂存区胶囊左侧，避免压住最后一个页签的关闭按钮
_PLUS_SLOT_WIDTH = 28

#: 页签宽度上限（V-A fit 语义）：VSCode 默认 fit 不压缩、按完整标签宽布局，
#: 放不下就溢出滚动。经典 fit 无上限，但本页名可为很长 CJK，故封顶 200px，
#: 超出截断＋T1 全名 tooltip 兜底。短签保持自然宽（main 口径，不人为撑宽）。
#: Windows 字体度量不同，常量合并前需复测。
_TAB_MAX_WIDTH = 200

#: 角位容器右缘留白（与胶囊保持视觉间距）
_CORNER_RIGHT_MARGIN = 2

#: 曲线暂存区入口回滚开关：False 时只保留 top_bar 旧按钮（MainWindow 同步读它）
_SHELF_CAPSULE_ENTRY = True

#: 角位胶囊尺寸：宽度写死 36（G1/G4——角位只按 sizeHint 留位，且窄窗只剩 2px 余量）
_SHELF_CAPSULE_SIZE = QSize(36, 20)

#: 页目录按钮尺寸（T6）：角位变为 [＋槽 28][▾ 20][胶囊 36]；G1 口径 sizeHint=fixedSize
_DIR_BUTTON_SIZE = QSize(20, 20)

#: 标签栏底色：角位容器与页签栏共用，保持右缘视觉连续
_TAB_BAR_BG = "#f3f3f3"

#: 激活页顶部强调线（与拖拽视觉同一份品牌蓝）
_ACCENT_COLOR = DRAG_ACCENT

#: 标签栏样式：VSCode 扁平风——选中页白底 + 顶部强调线，未选中透明，
#: 标签相互贴合，pane 顶边 1px 分隔线衔接绘图区。
#: 未选中页也画 2px 透明顶边占位，切换选中时高度不跳动。
#: 只作用于本控件的 QTabBar 内部，不影响页内容控件。
#: 注意：macOS 原生样式会在 QSS 之上再画一层 #bebebe 的 tab bar base，
#: 必须配合 ``tabBar().setDrawBase(False)`` 才看得到这里的底色。
_TABS_STYLE = f"""
QTabWidget::pane {{ border: 0; border-top: 1px solid #e5e5e5; }}
QTabBar {{ background: {_TAB_BAR_BG}; }}
QTabBar::tab {{
    background: transparent;
    color: #616161;
    padding: 4px 8px 4px 10px;
    margin: 0;
    border-top: 2px solid transparent;
}}
QTabBar::tab:hover:!selected {{ background: #eaeaea; }}
QTabBar::tab:selected {{
    background: #ffffff;
    color: #1f1f1f;
    border-top: 2px solid {_ACCENT_COLOR};
}}
"""


class _FlatIconButton(QToolButton):
    """扁平图标按钮基类：hover 圆角灰底 + 自绘字形。

    不用 QIcon/QSS image：位图在 150% DPI 下模糊、qproperty-icon 无法随
    状态变色；直接在 paintEvent 里矢量绘制最稳。子类实现 ``_draw_glyph``。
    """

    _COLOR_NORMAL = "#8a8a8a"
    _COLOR_HOVER = "#1f1f1f"
    _COLOR_DISABLED = "#b5b5b5"
    _BG_HOVER = QColor("#e0e0e0")
    _BG_PRESSED = QColor("#dcdcdc")
    _BG_RADIUS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        # 不抢焦点：点完后焦点应留在原交互入口（如搜索框）
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _base_color(self) -> str:
        """常态字形颜色，悬停/禁用时在 ``_glyph_color`` 里覆盖。"""
        return self._COLOR_NORMAL

    def _glyph_color(self) -> QColor:
        if not self.isEnabled():
            return QColor(self._COLOR_DISABLED)
        return QColor(self._COLOR_HOVER if self.underMouse() else self._base_color())

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        raise NotImplementedError

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isEnabled() and self.underMouse():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._BG_PRESSED if self.isDown() else self._BG_HOVER)
            painter.drawRoundedRect(
                self.rect().adjusted(1, 1, -1, -1), self._BG_RADIUS, self._BG_RADIUS
            )
        self._draw_glyph(painter, self._glyph_color())
        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().leaveEvent(event)
        self.update()


class _CloseTabButton(_FlatIconButton):
    """自绘关闭按钮：18×18 内画 10px 细「×」。

    字形颜色随所在页是否激活微调（VSCode 行为）。点击经所属
    PlotTabsWidget 直发 ``close_page_requested``。
    """

    _PEN_WIDTH = 1.4

    def _base_color(self) -> str:
        bar = self.parent()
        if isinstance(bar, QTabBar):
            for index in range(bar.count()):
                if bar.tabButton(index, QTabBar.ButtonPosition.RightSide) is self:
                    return "#5a5a5a" if index == bar.currentIndex() else self._COLOR_NORMAL
        return self._COLOR_NORMAL

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        center = QPointF(self.rect().center())
        half = 4.3  # 10px 字形：10/2 - 笔宽的一半留白
        pen = QPen(color, self._PEN_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(center + QPointF(-half, -half), center + QPointF(half, half))
        painter.drawLine(center + QPointF(half, -half), center + QPointF(-half, half))


class _NewPageButton(_FlatIconButton):
    """自绘「＋」：替代 16px 文本字形（Windows 基线漂移导致不居中）。"""

    _COLOR_NORMAL = "#555555"
    _COLOR_HOVER = "#111111"
    _BG_RADIUS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("newPageButton")

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        center = QPointF(self.rect().center())
        arm = 6.5  # 13px 十字
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(center + QPointF(0, -arm), center + QPointF(0, arm))
        painter.drawLine(center + QPointF(-arm, 0), center + QPointF(arm, 0))


class _ShelfCapsuleButton(_FlatIconButton):
    """页签栏右端 36×20 的曲线暂存区入口：托盘字形 + 常驻计数徽章。

    它同时是 **drop 落点**：浮窗关着时也必须能收拖拽（否则退回"先开窗才能
    拖"的鸡生蛋）。只发信号、不持有 shelf 引用（分层见模块 docstring），
    由 MainWindow 同步消费 ``drop_requested`` 里那个原始 ``QDropEvent``。

    Signals:
        toggle_requested(bool): 点击切换浮窗，参数是点击后的勾选态。
        drop_requested(object): 可接收的拖拽落下，转发同一个 QDropEvent。
        drag_hovered(bool): 拖拽进入/离开（供悬停探出去重）。
    """

    toggle_requested = Signal(bool)
    drop_requested = Signal(object)
    drag_hovered = Signal(bool)

    _BG_RADIUS = 4
    _BADGE_MAX = 99

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("shelfCapsule")
        self.setCheckable(True)
        self.setAcceptDrops(True)
        # G1：角位布局只认 sizeHint，setFixedSize 与 sizeHint 必须一致；
        # 不一致会让胶囊溢出右缘被裁（setFixedSize 单独用实测裁 7px）。
        self.setFixedSize(_SHELF_CAPSULE_SIZE)
        self._count = 0
        self._drag_hot = False
        self.clicked.connect(self._on_clicked)
        self._sync_tooltip()
        # G3：数据闸门开启前不得露面（未加载数据时页签栏整体隐藏，角位不跟随）
        self.setVisible(False)

    # -- 几何 ----------------------------------------------------------
    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 命名
        return QSize(_SHELF_CAPSULE_SIZE)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt 命名
        return QSize(_SHELF_CAPSULE_SIZE)

    # -- 对外状态 ------------------------------------------------------
    def set_count(self, count: int) -> None:
        self._count = max(0, int(count))
        self._sync_tooltip()
        self.update()

    def count(self) -> int:
        return self._count

    def _sync_tooltip(self) -> None:
        self.setToolTip(
            f"暂存区（{self._count} 条）：把图例/变量列表拖到这里，"
            "Ctrl+Shift+S 开关"
        )

    def _on_clicked(self, checked: bool) -> None:
        self.toggle_requested.emit(bool(checked))

    # -- 拖拽 ----------------------------------------------------------
    @staticmethod
    def _can_accept(mime) -> bool:
        """只收变量名拖拽；托盘自身 MIME 拖回胶囊必须拒绝（否则自己吃自己）。"""
        return (
            mime is not None
            and mime.hasText()
            and not mime.hasFormat(SHELF_MIME_FORMAT)
        )

    def _set_drag_hot(self, hot: bool) -> None:
        if hot == self._drag_hot:
            return
        self._drag_hot = hot
        self.update()
        self.drag_hovered.emit(hot)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self._can_accept(event.mimeData()):
            self._set_drag_hot(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self._can_accept(event.mimeData()):
            self._set_drag_hot(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 依赖 layout_manager 的 DragLeave 定向放行才收得到（应用级过滤器默认吞掉）
        self._set_drag_hot(False)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        accepted = self._can_accept(event.mimeData())
        # 兜底复位：不能只指望 DragLeave——应用级过滤器会吞掉它（G6）
        self._set_drag_hot(False)
        if not accepted:
            event.ignore()
            return
        # accept 由接收方决定：MainWindow 把同一个 event 转给托盘 dropEvent
        self.drop_requested.emit(event)

    # -- 绘制 ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(DRAG_ACCENT_SOFT))
            painter.drawRoundedRect(rect, self._BG_RADIUS, self._BG_RADIUS)
        elif self.isEnabled() and self.underMouse():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._BG_PRESSED if self.isDown() else self._BG_HOVER)
            painter.drawRoundedRect(rect, self._BG_RADIUS, self._BG_RADIUS)
        self._draw_glyph(painter, self._glyph_color())
        if self._drag_hot:
            pen = QPen(QColor(_ACCENT_COLOR), 1.4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, self._BG_RADIUS, self._BG_RADIUS)
        self._draw_badge(painter)
        painter.end()

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        """托盘字形：下开口方框（左侧 + 顶边 + 右侧）+ 向下箭头。"""
        center = QPointF(self.rect().center())
        pen = QPen(color, 1.3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half_w = 6.0
        top = center.y() - 5.5
        bottom = center.y() + 5.0
        left = center.x() - half_w
        right = center.x() + half_w
        # 方框：顶边 + 两侧竖线，底部开口
        painter.drawLine(QPointF(left, top), QPointF(right, top))
        painter.drawLine(QPointF(left, top), QPointF(left, bottom))
        painter.drawLine(QPointF(right, top), QPointF(right, bottom))
        # 向下箭头：从方框内指向开口
        painter.drawLine(
            QPointF(center.x(), top + 2.0), QPointF(center.x(), bottom + 2.0)
        )
        painter.drawLine(
            QPointF(center.x() - 2.6, bottom - 0.4),
            QPointF(center.x(), bottom + 2.0),
        )
        painter.drawLine(
            QPointF(center.x() + 2.6, bottom - 0.4),
            QPointF(center.x(), bottom + 2.0),
        )

    def _draw_badge(self, painter: QPainter) -> None:
        """右上角实心圆计数徽章；count==0 不画，>99 画 99+。"""
        if self._count <= 0:
            return
        text = f"{self._BADGE_MAX}+" if self._count > self._BADGE_MAX else str(self._count)
        # 字号/半径随位数收缩：8px 下 "99+" 实测 18px，13px 圆必裁
        if len(text) > 2:
            radius, font_px = 7.5, 6
        elif len(text) > 1:
            radius, font_px = 6.5, 7
        else:
            radius, font_px = 5.5, 8
        center = QPointF(self.width() - 1 - radius, 1 + radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_ACCENT_COLOR))
        painter.drawEllipse(center, radius, radius)
        font = painter.font()
        font.setPixelSize(font_px)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(theme.ON_ACCENT))
        painter.drawText(
            QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )


class _PageDirectoryButton(_FlatIconButton):
    """角位「▾」页目录入口（T6）：全部页一屏全见的补充入口，横条照常工作。"""

    _COLOR_NORMAL = "#5a5a5a"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageDirectoryButton")
        self.setFixedSize(_DIR_BUTTON_SIZE)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 命名
        return QSize(_DIR_BUTTON_SIZE)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt 命名
        return QSize(_DIR_BUTTON_SIZE)

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        center = QPointF(self.rect().center())
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(center.x() - 3.5, center.y() - 1.5),
            QPointF(center.x(), center.y() + 2.5),
        )
        painter.drawLine(
            QPointF(center.x(), center.y() + 2.5),
            QPointF(center.x() + 3.5, center.y() - 1.5),
        )


class _DirectoryRow(QWidget):
    """页目录单行：激活●＋页名＋布局角标＋hover 才现的 ×。"""

    def __init__(self, popup: "_PageDirectoryPopup", index: int,
                 title: str, badge: str, current: bool):
        super().__init__(popup._rows_host)
        self._popup = popup
        self._index = index
        self.setObjectName("dirRow")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#dirRow:hover {{ background: {DRAG_ACCENT_SOFT}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        dot = QLabel(self)
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background: {_ACCENT_COLOR if current else 'transparent'};"
            " border-radius: 4px;"
        )
        lay.addWidget(dot)
        name = QLabel(self)
        if current:
            font = name.font()
            font.setBold(True)
            name.setFont(font)
        # 省略必须按最终字体（含加粗）度量：先算省略后加粗会让实际渲染
        # 宽于省略宽，长名行溢出 popup 右缘、挤掉角标与 ×
        fm = QFontMetrics(name.font())
        name.setText(
            fm.elidedText(title, Qt.TextElideMode.ElideRight, popup._NAME_ELIDE_WIDTH)
        )
        name.setToolTip(title)
        # 平台字体更宽时允许收缩裁剪，绝不把行撑出 popup
        name.setMinimumWidth(0)
        policy = name.sizePolicy()
        policy.setHorizontalPolicy(policy.Policy.Ignored)
        name.setSizePolicy(policy)
        lay.addWidget(name, 1)
        if badge:
            badge_label = QLabel(badge, self)
            badge_label.setStyleSheet(
                "color: #8a8a8a; background: #f0f2f5; border-radius: 4px;"
                " padding: 0 6px; font-size: 10px;"
            )
            lay.addWidget(badge_label)
        self._close = _CloseTabButton(self)
        self._close.setFixedSize(16, 16)
        self._close.setToolTip("关闭绘图页")
        self._close.setVisible(False)
        self._close.clicked.connect(
            lambda _checked=False: popup._on_close(self._index)
        )
        lay.addWidget(self._close)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._close.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._close.setVisible(False)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton:
            self._popup._on_select(self._index)
        super().mouseReleaseEvent(event)


class _PageDirectoryPopup(QWidget):
    """锚定在「▾」正下方右对齐的页目录（T6）。

    数据经 owner 的 ``page_list_provider`` 每次打开时实时求值（固定顺序，
    保空间记忆）；点行切页、行尾 × 关页、底部＋新建，全部转发 owner 既有
    信号。``Qt.Popup`` 自带点外关闭，Esc 在 keyPressEvent 补；超高 clamp＋
    滚动（小屏兜底）。不做过滤框与拖拽重排——横条拖拽仍在，16 项可扫读。
    """

    _MAX_WIDTH = 320
    _NAME_ELIDE_WIDTH = 200

    def __init__(self, owner: "PlotTabsWidget"):
        super().__init__(
            owner, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self._owner = owner
        self.setObjectName("pageDirectoryPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "#pageDirectoryPopup { background: #ffffff;"
            " border: 1px solid #d5dbe3; border-radius: 6px; }"
            "#pageDirectoryPopup QScrollArea { background: transparent; border: none; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(0)
        self._scroll = QScrollArea(self)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # viewport/rows 容器默认灰底会盖住白色 popup 底（观感发灰），统一透明；
        # setWidget 会重置 autoFillBackground，故 rows_host 的透明在其后设置
        self._scroll.viewport().setAutoFillBackground(False)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._scroll.setWidget(self._rows_host)
        self._rows_host.setAutoFillBackground(False)
        outer.addWidget(self._scroll)
        self._new_btn = QPushButton("＋  新建绘图页", self)
        self._new_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._new_btn.setStyleSheet(
            "QPushButton { border: none; border-top: 1px solid #eceff3;"
            f" background: #ffffff; color: {_ACCENT_COLOR}; font-weight: 600;"
            " text-align: left; padding: 7px 10px; }"
            f"QPushButton:hover {{ background: {DRAG_ACCENT_SOFT}; }}"
            "QPushButton:disabled { color: #b5b5b5; }"
        )
        self._new_btn.clicked.connect(self._on_new)
        outer.addWidget(self._new_btn)

    def open_for(self, anchor: QWidget) -> None:
        provider = self._owner._page_list_provider
        if provider is not None:
            entries = list(provider())
        else:
            entries = [
                (self._owner.tabText(i), "", i == self._owner.currentIndex())
                for i in range(self._owner.count())
            ]
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, (title, badge, current) in enumerate(entries):
            self._rows_layout.addWidget(
                _DirectoryRow(self, index, title, badge, current)
            )
        self._new_btn.setEnabled(self._owner.new_page_enabled())
        self._rows_host.adjustSize()
        rows_height = self._rows_host.sizeHint().height()
        chrome = self._new_btn.sizeHint().height() + 10
        screen = QApplication.screenAt(
            anchor.mapToGlobal(QPoint(0, 0))
        ) or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        anchor_bottom = anchor.mapToGlobal(QPoint(0, anchor.height())).y()
        max_rows_height = max(120, avail.bottom() - anchor_bottom - 10 - chrome)
        self._scroll.setFixedHeight(min(rows_height, max_rows_height))
        self.adjustSize()
        if self.width() > self._MAX_WIDTH:
            self.setFixedWidth(self._MAX_WIDTH)
        pos = anchor.mapToGlobal(
            QPoint(anchor.width() - self.width(), anchor.height() + 4)
        )
        pos.setX(max(avail.left() + 4, min(pos.x(), avail.right() - self.width() - 4)))
        self.move(pos)
        self.show()
        self.setFocus()

    def _on_select(self, index: int) -> None:
        self.hide()
        self._owner.page_selected.emit(index)

    def _on_close(self, index: int) -> None:
        self.hide()
        self._owner.close_page_requested.emit(index)

    def _on_new(self) -> None:
        self.hide()
        self._owner.new_page_requested.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)


class _TabsBar(QTabBar):
    """挂 ``tabLayoutChange`` 官方布局钩子的标签栏。

    PySide6 里 ``tabLayoutChanged`` 信号未暴露（同名属性是 protected 虚
    方法），故经 ``QTabWidget.setTabBar`` 挂入子类重写。重命名 ``setTabText``
    只改变标签宽度，不走插 / 删 / Resize 任何路径——「＋」位置会过期；
    此钩子在插 / 删 / 拖动 / 重命名等一切布局变化后都会触发，一处覆盖全部。
    「＋」不再挂在 tabBar 上，钩子只负责重算「跟随最后一个页签 / 停入预留槽」。
    """

    def __init__(self, owner: "PlotTabsWidget"):
        super().__init__()
        self._owner = owner

    def tabLayoutChange(self):  # noqa: N802 - Qt 命名
        super().tabLayoutChange()
        # Qt 每次布局都会重新显示原生滚动按钮；方向 B 不显示箭头，
        # 在官方布局钩子里藏回去（按钮保留隐藏态供滚轮程序化驱动）。
        self._owner._hide_scroll_buttons()
        self._owner._place_new_page_button()

    def tabSizeHint(self, index):  # noqa: N802 - Qt 命名
        """V-A fit：自然宽封顶 ``_TAB_MAX_WIDTH``，超长名截断靠 tooltip 兜底。"""
        base = super().tabSizeHint(index)
        return QSize(min(base.width(), _TAB_MAX_WIDTH), base.height())

    def minimumTabSizeHint(self, index):  # noqa: N802 - Qt 命名
        """V-A fit：最小宽＝自然宽 → Qt 压不动 → 溢出滚动而非压缩页签。

        与曾回退的 T8 下限不同：这里不抬高低签，短签仍是 main 的自然宽。
        """
        return self.tabSizeHint(index)

    def wheelEvent(self, event):  # noqa: N802 - Qt 命名
        """T4：页签区滚轮＝横向滚动页签，不切页（QTabBar 默认会切页）。

        滚动仍由原生滚动按钮承担（一次一格＝一个页签）；按钮在 B 阶段
        隐藏后程序化 ``click()`` 依然驱动内部偏移。放得下时吞掉事件，
        既不切页也不漏给下层绘图区当缩放。
        """
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta != 0:
            direction = "left" if delta > 0 else "right"
            for button in self._owner._scroll_buttons():
                if direction in (button.objectName() or "").lower():
                    button.click()
                    break
        event.accept()


class PlotTabsWidget(QTabWidget):
    """多绘图页标签容器。

    Signals:
        new_page_requested: 点击标签尾部的「＋」。
        close_page_requested(int): 点标签关闭按钮或中键点击标签。
        rename_page_requested(int): 双击标签。
        close_other_pages_requested(int): 右键菜单「关闭其他」。
        close_right_pages_requested(int): 右键菜单「关闭右侧」。
        page_selected(int): 页目录下拉里点选某页。
        shelf_toggle_requested(bool): 点击右端暂存区胶囊。
        shelf_drop_requested(object): 拖拽落在胶囊上，参数为原始 QDropEvent。
        shelf_drag_hovered(bool): 拖拽进入/离开胶囊。
    """

    new_page_requested = Signal()
    close_page_requested = Signal(int)
    rename_page_requested = Signal(int)
    close_other_pages_requested = Signal(int)
    close_right_pages_requested = Signal(int)
    page_selected = Signal(int)
    shelf_toggle_requested = Signal(bool)
    shelf_drop_requested = Signal(object)
    shelf_drag_hovered = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 页签全名 tooltip 的数据源（MainWindow 注入，hover 时实时求值，
        # 改名/换布局/增删曲线后无需任何同步点，不会过期）
        self._tab_info_provider = None
        # 页目录数据源（MainWindow 注入）：() -> [(title, badge, current), ...]
        self._page_list_provider = None
        # 必须在一切 tabBar() 使用之前挂入：布局钩子依赖自定义子类
        self.setTabBar(_TabsBar(self))
        self.setDocumentMode(True)
        self.setMovable(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setUsesScrollButtons(True)
        # 不用 autoHide（它是"单页隐藏"语义，会把「＋」一起藏掉，加载后
        # 单页时新建入口消失）；改为"数据闸门驱动可见性"：
        # 未加载数据时整个标签栏隐藏，加载成功后显示且从此常显。
        self.setTabBarAutoHide(False)
        self.setStyleSheet(_TABS_STYLE)
        # macOS 原生样式会画 #bebebe 的 tab bar base，盖住 QSS 底色；
        # 关掉后由 QSS 的 #f3f3f3 接管（Windows/Fusion 本来没有这层）。
        self.tabBar().setDrawBase(False)

        self._new_page_allowed = False
        # 未加载数据：初始隐藏（set_new_page_enabled(True) 时显示）
        self.tabBar().setVisible(False)
        # 角位容器：[「＋」预留槽][▾ 页目录][暂存区胶囊]。
        # 「＋」不再挂在 tabBar 上：页签栏右侧必须永久留出槽位，页签再满也
        # 不会压住最后一个页签的关闭按钮。少量页签时「＋」仍紧跟
        # 最后一个页签，只是父控件换成这里，便于停入槽位。
        self._shelf_capsule = None
        self._corner_container = QWidget(self)
        corner_layout = QHBoxLayout(self._corner_container)
        corner_layout.setContentsMargins(0, 0, _CORNER_RIGHT_MARGIN, 0)
        corner_layout.setSpacing(0)
        self._plus_slot = QWidget(self._corner_container)
        self._plus_slot.setFixedWidth(_PLUS_SLOT_WIDTH)
        corner_layout.addWidget(self._plus_slot)
        # 页目录「▾」（T6）：停预留槽右侧、胶囊左侧，与＋/×同一套扁平语言
        self._dir_btn = _PageDirectoryButton(self._corner_container)
        self._dir_btn.setVisible(False)
        self._dir_btn.installEventFilter(self)
        corner_layout.addWidget(self._dir_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._dir_popup = _PageDirectoryPopup(self)
        self._dir_btn.clicked.connect(
            lambda: self._dir_popup.open_for(self._dir_btn)
        )
        if _SHELF_CAPSULE_ENTRY:
            self._shelf_capsule = _ShelfCapsuleButton(self._corner_container)
            self._shelf_capsule.toggle_requested.connect(
                self.shelf_toggle_requested
            )
            self._shelf_capsule.drop_requested.connect(self.shelf_drop_requested)
            self._shelf_capsule.drag_hovered.connect(self.shelf_drag_hovered)
            corner_layout.addWidget(
                self._shelf_capsule, 0, Qt.AlignmentFlag.AlignVCenter
            )
        self.setCornerWidget(
            self._corner_container, Qt.Corner.TopRightCorner
        )
        self._corner_container.setVisible(False)
        # 「＋」跟随最后一个标签；空间不足时停到胶囊左侧的预留槽
        self._new_page_btn = _NewPageButton(self)
        self._new_page_btn.setFixedSize(26, 24)
        self._new_page_btn.clicked.connect(self._on_new_page_clicked)
        self._new_page_btn.setVisible(False)

        self.tabCloseRequested.connect(self.close_page_requested)
        self.tabBarDoubleClicked.connect(self.rename_page_requested)
        self.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabBar().customContextMenuRequested.connect(self._on_tab_context_menu)
        self.tabBar().installEventFilter(self)
        # 角位容器位置由 QTabWidget 在 tabBar resize 之后才更新；必须监听它
        # 自己的 Move/Resize，否则「＋」会沿用旧槽位（变宽浮在页签上、变窄被裁）。
        self._corner_container.installEventFilter(self)
        if self._shelf_capsule is not None:
            self._shelf_capsule.installEventFilter(self)
        self._sync_new_page_button()
        self._place_new_page_button()

    @property
    def new_page_button(self) -> QToolButton:
        """新建页按钮（测试与外部提示共用）。"""
        return self._new_page_btn

    @property
    def shelf_capsule(self) -> _ShelfCapsuleButton | None:
        """暂存区胶囊（测试、拖拽放行判据与托盘锚点共用）；回滚开关关闭时为 None。"""
        return self._shelf_capsule

    def set_tab_info_provider(self, provider) -> None:
        """注入 ``provider(index) -> str``：页签 hover tooltip 文案（可含换行）。

        返回空串则该页签不弹 tooltip。provider 在 hover 瞬间求值，页名/布局/
        曲线数永远取当前值，调用方无需在状态变化处补同步。
        """
        self._tab_info_provider = provider

    def set_page_list_provider(self, provider) -> None:
        """注入 ``provider() -> [(title, badge, current), ...]``：页目录数据源。

        每次打开下拉时实时求值；缺省回退为 tabText（无角标）。
        """
        self._page_list_provider = provider

    def _hide_scroll_buttons(self) -> None:
        """方向 B：原生滚动箭头永不显示，右缘让位给页签。

        按钮本体保留隐藏态——Qt 无公开滚动 API，滚轮（T4）经程序化
        ``click()`` 驱动其内部偏移。Qt 每次布局会重新显示它们，故挂在
        ``tabLayoutChange`` 里于布局之后藏回。
        """
        for button in self._scroll_buttons():
            if button.isVisible():
                button.hide()

    def _hidden_tab_counts(self) -> tuple[int, int]:
        """视野外左侧/右侧的页数（▾ tooltip 的溢出计数，T5 的 B 形态）。"""
        bar = self.tabBar()
        width = bar.width()
        left = right = 0
        for index in range(bar.count()):
            rect = bar.tabRect(index)
            if rect.right() < 0:
                left += 1
            elif rect.left() > width:
                right += 1
        return left, right

    # ------------------------------------------------------------------
    # 暂存区胶囊：计数 / 勾选态 / 加载闸门（本类只转发，不认识 MainWindow）
    # ------------------------------------------------------------------
    def set_shelf_count(self, count: int) -> None:
        if self._shelf_capsule is not None:
            self._shelf_capsule.set_count(count)

    def set_shelf_active(self, active: bool) -> None:
        """同步浮窗开合态；blockSignals 防止回发 toggle 造成闭环。"""
        capsule = self._shelf_capsule
        if capsule is None:
            return
        capsule.blockSignals(True)
        try:
            capsule.setChecked(bool(active))
        finally:
            capsule.blockSignals(False)

    def set_shelf_enabled(self, enabled: bool) -> None:
        """加载闸门：加载中 / 加载失败无数据时置灰（与既有按钮同口径）。"""
        if self._shelf_capsule is not None:
            self._shelf_capsule.setEnabled(bool(enabled))

    def close_button(self, index: int) -> QToolButton | None:
        """指定页的自绘关闭按钮（测试共用）。"""
        return self.tabBar().tabButton(index, QTabBar.ButtonPosition.RightSide)

    def _sync_corner_width(self) -> None:
        """按胶囊显隐同步角位容器宽度，保证隐藏时 36px 全额归还给页签栏。

        QTabWidget 只在自身布局周期读取角位控件宽度；胶囊单独 hide 不会让
        外层重新计算。这里显式定宽并 updateGeometry，随后页签栏会重新铺排。
        """
        container = getattr(self, "_corner_container", None)
        slot = getattr(self, "_plus_slot", None)
        if container is None or slot is None:
            return
        width = slot.width() + _CORNER_RIGHT_MARGIN
        dir_btn = getattr(self, "_dir_btn", None)
        if dir_btn is not None and dir_btn.isVisible():
            width += dir_btn.sizeHint().width()
        capsule = getattr(self, "_shelf_capsule", None)
        if capsule is not None and capsule.isVisible():
            width += capsule.sizeHint().width()
        if container.width() != width:
            container.setFixedWidth(width)
            container.updateGeometry()
            self.updateGeometry()

    def _place_new_page_button(self) -> None:
        """把「＋」放到最后一个标签右侧；放不下时停到胶囊左侧的预留槽。

        「＋」是 ``PlotTabsWidget`` 的子控件而非 tabBar 子控件：
        - 少量页签：紧跟最后一个页签右侧（保持既有交互）；
        - 页签满 / 末标签在屏外 / 右侧放不下：停到 ``_plus_slot``，
          槽位在 tabBar 之外，页签和关闭按钮永远不会被压住。

        角位容器由 [预留槽][▾][胶囊] 组成，tabBar 宽度天然扣掉这几段，
        因此不需要再按右缘做临时钳制。
        """
        # setTabBar 挂钩子后 Qt 会在构造期同步触发布局回调，此时「＋」
        # 尚未创建——直接跳过（__init__ 尾部会显式放置一次）。
        button = getattr(self, "_new_page_btn", None)
        slot = getattr(self, "_plus_slot", None)
        if button is None or slot is None:
            return
        bar = self.tabBar()
        if not bar.isVisible() or bar.count() == 0 or not button.isVisible():
            return
        container = getattr(self, "_corner_container", None)
        if (
            container is not None
            and bar.height() > 0
            and container.height() != bar.height()
        ):
            # 角位容器按页签栏高度定高，胶囊由布局垂直居中；只管高度，
            # 宽度仍由 [槽][胶囊] 的 sizeHint 决定（隐藏胶囊时归还 36px）。
            container.setFixedHeight(bar.height())
        bar_pos = bar.mapTo(self, QPoint(0, 0))
        last_rect = bar.tabRect(bar.count() - 1)
        slot_pos = slot.mapTo(self, QPoint(0, 0))
        slot_center = max(
            2, slot_pos.x() + (slot.width() - button.width()) // 2
        )
        # 方向 B 无可见箭头：紧跟末标签，放不下（含末标签在屏外）停预留槽
        following = (
            bar_pos.x() + last_rect.right() + 5 if last_rect.isValid() else 5
        )
        x = min(following, slot_center)
        y = bar_pos.y() + max(0, (bar.height() - button.height()) // 2)
        button.move(x, y)
        button.raise_()

    def _scroll_buttons(self) -> list[QToolButton]:
        """全部原生滚动按钮（含隐藏）：滚轮程序化驱动不依赖可见性。"""
        out = []
        for child in self.tabBar().children():
            if not child.isWidgetType():
                continue
            name = (child.objectName() or "").lower()
            if "scroll" in name:
                out.append(child)
        return out

    # ------------------------------------------------------------------
    # 自绘关闭按钮：逐页挂载，替换 Qt 原生关闭图标
    # ------------------------------------------------------------------
    def _attach_close_button(self, index: int) -> None:
        bar = self.tabBar()
        button = _CloseTabButton(bar)
        button.setFixedSize(18, 18)
        button.setToolTip("关闭绘图页")
        button.clicked.connect(
            lambda _checked=False, b=button: self._on_close_button_clicked(b)
        )
        bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, button)

    def _on_close_button_clicked(self, button: _CloseTabButton) -> None:
        """点击时实时反查页索引：拖动重排后仍关闭正确的页。"""
        bar = self.tabBar()
        for index in range(bar.count()):
            if bar.tabButton(index, QTabBar.ButtonPosition.RightSide) is button:
                self.close_page_requested.emit(index)
                return

    # ------------------------------------------------------------------
    # 右键菜单（T2）：重命名 / 关闭 / 关闭其他 / 关闭右侧
    # ------------------------------------------------------------------
    def _build_tab_menu(self, index: int) -> QMenu:
        menu = QMenu(self)
        action = menu.addAction("重命名")
        action.triggered.connect(lambda: self.rename_page_requested.emit(index))
        menu.addSeparator()
        action = menu.addAction("关闭")
        action.triggered.connect(lambda: self.close_page_requested.emit(index))
        action = menu.addAction("关闭其他")
        action.setEnabled(self.count() > 1)
        action.triggered.connect(lambda: self.close_other_pages_requested.emit(index))
        action = menu.addAction("关闭右侧")
        action.setEnabled(index < self.count() - 1)
        action.triggered.connect(lambda: self.close_right_pages_requested.emit(index))
        return menu

    def _on_tab_context_menu(self, pos: QPoint) -> None:
        index = self.tabBar().tabAt(pos)
        if index < 0:
            return
        self._build_tab_menu(index).exec(self.tabBar().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # 「＋」可用性：未加载数据 / 已达上限时禁用（决策 D-D / §4.4）
    # ------------------------------------------------------------------
    def set_new_page_enabled(self, enabled: bool) -> None:
        """外部闸门（是否已加载数据）；与页数上限共同决定按钮可用性。

        未加载数据时整个标签栏一并隐藏；开启后显示且从此常显（含单页）。
        """
        self._new_page_allowed = bool(enabled)
        self.tabBar().setVisible(enabled)
        # 角位容器 / 胶囊 / 「＋」不跟随页签栏隐藏（G3），必须显式同步，
        # 否则未加载数据时会露出一个可见可点的死按钮
        if self._corner_container is not None:
            self._corner_container.setVisible(bool(enabled))
        if self._shelf_capsule is not None:
            self._shelf_capsule.setVisible(bool(enabled))
        self._dir_btn.setVisible(bool(enabled))
        if not enabled:
            self._dir_popup.hide()
        self._new_page_btn.setVisible(bool(enabled))
        self._sync_corner_width()
        self._sync_new_page_button()
        self._place_new_page_button()

    def new_page_enabled(self) -> bool:
        return self._can_add_page()

    def _can_add_page(self) -> bool:
        return self._new_page_allowed and self.count() < MAX_PAGES

    def _sync_new_page_button(self) -> None:
        enabled = self._can_add_page()
        self._new_page_btn.setEnabled(enabled)
        if not self._new_page_allowed:
            tip = "导入数据后可新建绘图页"
        elif self.count() >= MAX_PAGES:
            tip = "已达上限（%d 页）" % MAX_PAGES
        else:
            tip = "新建绘图页（最多 %d 页）" % MAX_PAGES
        self._new_page_btn.setToolTip(tip)

    def _on_new_page_clicked(self) -> None:
        if self._can_add_page():
            self.new_page_requested.emit()

    # 页增删后刷新按钮状态（不依赖 Qt 版本相关的 countChanged 信号）
    def tabInserted(self, index: int) -> None:  # noqa: N802 - Qt 命名
        super().tabInserted(index)
        self._attach_close_button(index)
        self._sync_new_page_button()
        self._place_new_page_button()

    def tabRemoved(self, index: int) -> None:  # noqa: N802 - Qt 命名
        super().tabRemoved(index)
        self._sync_new_page_button()
        self._place_new_page_button()

    # ------------------------------------------------------------------
    # 中键关闭 / 「＋」跟随标签重排
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is getattr(self, "_dir_btn", None):
            if event.type() == QEvent.Type.ToolTip:
                provider = self._page_list_provider
                count = len(provider()) if provider is not None else self.count()
                left, right = self._hidden_tab_counts()
                overflow = f" · 左隐 {left} / 右隐 {right}" if (left or right) else ""
                QToolTip.showText(
                    event.globalPos(), f"页目录（{count} 页）{overflow}", self._dir_btn
                )
                return True
        if obj is getattr(self, "_corner_container", None):
            if event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
                self._place_new_page_button()
        if obj is getattr(self, "_shelf_capsule", None):
            if event.type() in (QEvent.Type.Hide, QEvent.Type.Show):
                self._sync_corner_width()
                self._place_new_page_button()
        if obj is self.tabBar():
            event_type = event.type()
            if event_type == QEvent.Type.ToolTip:
                provider = self._tab_info_provider
                if provider is not None:
                    index = self.tabBar().tabAt(event.pos())
                    text = provider(index) if index >= 0 else ""
                    if text:
                        QToolTip.showText(event.globalPos(), text, self.tabBar())
                    return True
            if (
                event_type == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.MiddleButton
            ):
                index = self.tabBar().tabAt(event.position().toPoint())
                if index >= 0:
                    self.close_page_requested.emit(index)
                    return True
            if event_type in (
                QEvent.Type.Resize,
                QEvent.Type.LayoutRequest,
                QEvent.Type.Show,
                QEvent.Type.StyleChange,
            ):
                self._place_new_page_button()
        return super().eventFilter(obj, event)
