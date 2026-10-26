"""绘图区标签容器（PlotTabsWidget）。

设计文档 §4.4：选 ``QTabWidget`` 子类，可拖动 / 省略号；标签尾部放「＋」
新建页（跟随最后一个标签）。**不启用** ``setTabBarAutoHide(True)``：它会把
「＋」一并隐藏，导致只有一页时入口消失。

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

本类只负责标签交互与信号转发，页对象的增删与切页时序由 MainWindow 掌握。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QTabBar, QTabWidget, QToolButton

#: 绘图页数量上限（设计文档 §4.4）
MAX_PAGES = 8

#: 激活页顶部强调线（VSCode 默认浅色主题蓝）
_ACCENT_COLOR = "#0078d4"

#: 标签栏样式：VSCode 扁平风——选中页白底 + 顶部强调线，未选中透明，
#: 标签相互贴合，pane 顶边 1px 分隔线衔接绘图区。
#: 未选中页也画 2px 透明顶边占位，切换选中时高度不跳动。
#: 只作用于本控件的 QTabBar 内部，不影响页内容控件。
#: 注意：macOS 原生样式会在 QSS 之上再画一层 #bebebe 的 tab bar base，
#: 必须配合 ``tabBar().setDrawBase(False)`` 才看得到这里的底色。
_TABS_STYLE = f"""
QTabWidget::pane {{ border: 0; border-top: 1px solid #e5e5e5; }}
QTabBar {{ background: #f3f3f3; }}
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


class _TabsBar(QTabBar):
    """挂 ``tabLayoutChange`` 官方布局钩子的标签栏。

    PySide6 里 ``tabLayoutChanged`` 信号未暴露（同名属性是 protected 虚
    方法），故经 ``QTabWidget.setTabBar`` 挂入子类重写。重命名 ``setTabText``
    只改变标签宽度，不走插 / 删 / Resize 任何路径——「＋」位置会过期并被
    拉宽的标签盖住；此钩子在插 / 删 / 拖动 / 重命名等一切布局变化后都会
    触发，一处覆盖全部。
    """

    def __init__(self, owner: "PlotTabsWidget"):
        super().__init__()
        self._owner = owner

    def tabLayoutChange(self):  # noqa: N802 - Qt 命名
        super().tabLayoutChange()
        self._owner._place_new_page_button()


class PlotTabsWidget(QTabWidget):
    """多绘图页标签容器。

    Signals:
        new_page_requested: 点击标签尾部的「＋」。
        close_page_requested(int): 点标签关闭按钮或中键点击标签。
        rename_page_requested(int): 双击标签。
    """

    new_page_requested = Signal()
    close_page_requested = Signal(int)
    rename_page_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
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
        # 「＋」跟随最后一个标签，而不是钉在标签栏最右端
        self._new_page_btn = _NewPageButton(self.tabBar())
        self._new_page_btn.setFixedSize(26, 24)
        self._new_page_btn.clicked.connect(self._on_new_page_clicked)

        self.tabCloseRequested.connect(self.close_page_requested)
        self.tabBarDoubleClicked.connect(self.rename_page_requested)
        self.tabBar().installEventFilter(self)
        self._sync_new_page_button()
        self._place_new_page_button()

    @property
    def new_page_button(self) -> QToolButton:
        """新建页按钮（测试与外部提示共用）。"""
        return self._new_page_btn

    def close_button(self, index: int) -> QToolButton | None:
        """指定页的自绘关闭按钮（测试共用）。"""
        return self.tabBar().tabButton(index, QTabBar.ButtonPosition.RightSide)

    def _place_new_page_button(self) -> None:
        """把「＋」放到最后一个标签右侧；空间不足时贴右缘。"""
        # setTabBar 挂钩子后 Qt 会在构造期同步触发布局回调，此时「＋」
        # 尚未创建——直接跳过（__init__ 尾部会显式放置一次）。
        button = getattr(self, "_new_page_btn", None)
        if button is None:
            return
        bar = self.tabBar()
        if bar.count() > 0:
            last_rect = bar.tabRect(bar.count() - 1)
            x = last_rect.right() + 5 if last_rect.isValid() else 5
        else:
            x = 5
        x = min(x, max(2, bar.width() - button.width() - 2))
        y = max(0, (bar.height() - button.height()) // 2)
        button.move(x, y)
        button.raise_()

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
        self._sync_new_page_button()

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
