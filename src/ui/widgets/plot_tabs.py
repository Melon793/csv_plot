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

原生滚动箭头替换为与「＋」/ 关闭按钮同源的矢量字形。

本类只负责标签交互与信号转发，页对象的增删与切页时序由 MainWindow 掌握。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QTabBar, QTabWidget, QToolButton, QWidget

from src.ui.drag_drop import SHELF_MIME_FORMAT
from src.ui import theme
from src.ui.drag_visuals import DRAG_ACCENT, DRAG_ACCENT_SOFT

#: 绘图页数量上限（设计文档 §4.4；2026-11-20 由 8 提升到 16）
MAX_PAGES = 16

#: 「＋」预留槽宽度：页签放不下时停到暂存区胶囊左侧，避免压住最后一个页签的关闭按钮
_PLUS_SLOT_WIDTH = 28

#: 角位容器右缘留白（与胶囊保持视觉间距）
_CORNER_RIGHT_MARGIN = 2

#: 曲线暂存区入口回滚开关：False 时只保留 top_bar 旧按钮（MainWindow 同步读它）
_SHELF_CAPSULE_ENTRY = True

#: 角位胶囊尺寸：宽度写死 36（G1/G4——角位只按 sizeHint 留位，且窄窗只剩 2px 余量）
_SHELF_CAPSULE_SIZE = QSize(36, 20)

#: 标签栏底色：标签与滚动按钮共用，滚动按钮靠它遮住边界下被压住的部分标签
_TAB_BAR_BG = "#f3f3f3"

#: 滚动箭头：与 ＋/× 同一套扁平图标语言（16px 矢量、无边框、hover 圆角灰底）
_SCROLL_ARROW_ICON_SIZE = 16
_SCROLL_ARROW_COLOR = "#5a5a5a"
_SCROLL_ARROW_ACTIVE = "#1f1f1f"
_SCROLL_ARROW_DISABLED = "#b5b5b5"

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

#: 原生滚动箭头换成与自绘按钮一致的扁平视觉；底色遮住滚动边界下
#: 被压住的部分标签。点击 / autoRepeat 仍由 QTabBar 原生按钮承担。
_SCROLL_ARROW_STYLE = f"""
QToolButton {{
    background: {_TAB_BAR_BG};
    border: none;
    margin: 0;
    padding: 0;
}}
QToolButton:hover {{
    background: #e0e0e0;
    border-radius: 4px;
}}
QToolButton:pressed {{
    background: #dcdcdc;
    border-radius: 4px;
}}
QToolButton:disabled {{
    background: {_TAB_BAR_BG};
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


def _scroll_arrow_pixmap(direction: str, color: str, dpr: int) -> QPixmap:
    """画 16px chevron；使用逻辑坐标，DPR 由 QPixmap 自带。"""
    logical = _SCROLL_ARROW_ICON_SIZE
    pixmap = QPixmap(logical * dpr, logical * dpr)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    center = logical / 2.0
    arm = 4.2
    if direction == "left":
        painter.drawLine(
            QPointF(center + arm * 0.45, center - arm),
            QPointF(center - arm * 0.55, center),
        )
        painter.drawLine(
            QPointF(center - arm * 0.55, center),
            QPointF(center + arm * 0.45, center + arm),
        )
    else:
        painter.drawLine(
            QPointF(center - arm * 0.45, center - arm),
            QPointF(center + arm * 0.55, center),
        )
        painter.drawLine(
            QPointF(center + arm * 0.55, center),
            QPointF(center - arm * 0.45, center + arm),
        )
    painter.end()
    return pixmap


def _scroll_arrow_icon(direction: str) -> QIcon:
    """返回 normal / active / disabled 三态的矢量 chevron。"""
    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, _SCROLL_ARROW_COLOR),
        (QIcon.Mode.Active, _SCROLL_ARROW_ACTIVE),
        (QIcon.Mode.Selected, _SCROLL_ARROW_ACTIVE),
        (QIcon.Mode.Disabled, _SCROLL_ARROW_DISABLED),
    ):
        for dpr in (1, 2, 3):
            icon.addPixmap(
                _scroll_arrow_pixmap(direction, color, dpr),
                mode,
                QIcon.State.Off,
            )
    return icon


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
        # QTabBar 可能在本钩子触发的布局周期里才创建/重排滚动按钮；
        # 这里补一次，保证原生箭头不会在显示后退回平台样式。
        self._owner._style_scroll_buttons()
        self._owner._place_new_page_button()


class PlotTabsWidget(QTabWidget):
    """多绘图页标签容器。

    Signals:
        new_page_requested: 点击标签尾部的「＋」。
        close_page_requested(int): 点标签关闭按钮或中键点击标签。
        rename_page_requested(int): 双击标签。
        shelf_toggle_requested(bool): 点击右端暂存区胶囊。
        shelf_drop_requested(object): 拖拽落在胶囊上，参数为原始 QDropEvent。
        shelf_drag_hovered(bool): 拖拽进入/离开胶囊。
    """

    new_page_requested = Signal()
    close_page_requested = Signal(int)
    rename_page_requested = Signal(int)
    shelf_toggle_requested = Signal(bool)
    shelf_drop_requested = Signal(object)
    shelf_drag_hovered = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 箭头图标只在首次样式化时生成，tabLayoutChange 会反复补样式
        self._scroll_arrow_icons: dict[str, QIcon] = {}
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
        self._style_scroll_buttons()

        self._new_page_allowed = False
        # 未加载数据：初始隐藏（set_new_page_enabled(True) 时显示）
        self.tabBar().setVisible(False)
        # 角位容器：[「＋」预留槽][暂存区胶囊]。
        # 「＋」不再挂在 tabBar 上：页签栏右侧必须永久留出槽位，页签再满也
        # 不会压住最后一个页签的关闭按钮或滚动箭头。少量页签时「＋」仍紧跟
        # 最后一个页签，只是父控件换成这里，便于停入槽位。
        self._shelf_capsule = None
        self._corner_container = QWidget(self)
        corner_layout = QHBoxLayout(self._corner_container)
        corner_layout.setContentsMargins(0, 0, _CORNER_RIGHT_MARGIN, 0)
        corner_layout.setSpacing(0)
        self._plus_slot = QWidget(self._corner_container)
        self._plus_slot.setFixedWidth(_PLUS_SLOT_WIDTH)
        corner_layout.addWidget(self._plus_slot)
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

    def _style_scroll_buttons(self) -> None:
        """把 QTabBar 原生滚动箭头换成与 ＋/× 同一套的扁平矢量字形。

        按钮本身、点击与 autoRepeat 都保留；只替换绘制。底色调成标签栏
        同色，滚动边界下被压住的部分标签不会透出来。
        """
        for child in self.tabBar().children():
            if not isinstance(child, QToolButton):
                continue
            name = (child.objectName() or "").lower()
            if "scroll" not in name:
                continue
            if child.arrowType() == Qt.ArrowType.NoArrow and not child.icon().isNull():
                continue  # 已经换成自绘箭头；tabLayoutChange 会频繁回调
            if "left" in name:
                direction = "left"
            elif "right" in name:
                direction = "right"
            else:
                direction = (
                    "left"
                    if child.arrowType() == Qt.ArrowType.LeftArrow
                    else "right"
                )
            icon = self._scroll_arrow_icons.get(direction)
            if icon is None:
                icon = _scroll_arrow_icon(direction)
                self._scroll_arrow_icons[direction] = icon
            child.setArrowType(Qt.ArrowType.NoArrow)
            child.setIcon(icon)
            child.setIconSize(
                QSize(_SCROLL_ARROW_ICON_SIZE, _SCROLL_ARROW_ICON_SIZE)
            )
            child.setStyleSheet(_SCROLL_ARROW_STYLE)
            child.setCursor(Qt.CursorShape.ArrowCursor)
            child.setToolTip(
                "向左滚动标签" if direction == "left" else "向右滚动标签"
            )

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
        - 页签满 / 出现滚动箭头 / 右侧放不下：停到 ``_plus_slot``，
          槽位在 tabBar 之外，页签和关闭按钮永远不会被压住。

        角位容器由 [预留槽][胶囊] 组成，tabBar 宽度天然扣掉这两段，
        因此不需要再按滚动箭头或右缘做临时钳制。
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
        if self._visible_scroll_buttons():
            # 滚动箭头占住 tabBar 右缘：直接停槽，避免「＋」压箭头或页签
            x = slot_center
        else:
            following = (
                bar_pos.x() + last_rect.right() + 5 if last_rect.isValid() else 5
            )
            x = min(following, slot_center)
        y = bar_pos.y() + max(0, (bar.height() - button.height()) // 2)
        button.move(x, y)
        button.raise_()

    def _visible_scroll_buttons(self) -> list[QToolButton]:
        """可见的翻页滚动箭头（Qt 未暴露公开 API，按 objectName 反查）。

        不同版本/平台命名可能是 ``ScrollLeftButton`` /
        ``qt_tabbar_scrollbutton_left``，故按 "scroll" 关键字匹配。
        """
        out = []
        for child in self.tabBar().children():
            if not child.isWidgetType():
                continue
            name = (child.objectName() or "").lower()
            if "scroll" in name and child.isVisible():
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
        if obj is getattr(self, "_corner_container", None):
            if event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
                self._place_new_page_button()
        if obj is getattr(self, "_shelf_capsule", None):
            if event.type() in (QEvent.Type.Hide, QEvent.Type.Show):
                self._sync_corner_width()
                self._place_new_page_button()
        if obj is self.tabBar():
            event_type = event.type()
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
