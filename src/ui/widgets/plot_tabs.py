"""绘图区标签容器（PlotTabsWidget）。

设计文档 §4.4：选 ``QTabWidget`` 子类，自带关闭按钮 / 可拖动 / 省略号；
标签尾部放「＋」新建页（跟随最后一个标签）。**不启用**
``setTabBarAutoHide(True)``：它会把「＋」一并隐藏，导致只有一页时入口消失。

本类只负责标签交互与信号转发，页对象的增删与切页时序由 MainWindow 掌握。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QTabWidget, QToolButton

#: 绘图页数量上限（设计文档 §4.4）
MAX_PAGES = 8

#: 标签栏样式：选中项白底圆角，未选中透明，贴近顶部按钮的观感。
#: 只作用于本控件的 QTabBar 内部，不影响页内容控件。
#: 注意：macOS 原生样式会在 QSS 之上再画一层 #bebebe 的 tab bar base，
#: 必须配合 ``tabBar().setDrawBase(False)`` 才看得到这里的底色。
_TABS_STYLE = """
QTabWidget::pane { border: 0; }
QTabBar { background: #f3f3f3; }
QTabBar::tab {
    background: transparent;
    color: #333333;
    padding: 4px 8px;
    margin: 2px 3px 2px 0;
    border: 1px solid transparent;
    border-radius: 7px;
}
QTabBar::tab:hover:!selected { background: #e8e8e8; }
QTabBar::tab:selected {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    color: #111111;
}
QToolButton#newPageButton {
    background: transparent;
    border: 0;
    border-radius: 6px;
    color: #555555;
    font-size: 16px;
}
QToolButton#newPageButton:hover { background: #e6e6e6; color: #111111; }
QToolButton#newPageButton:pressed { background: #dcdcdc; }
QToolButton#newPageButton:disabled { color: #b5b5b5; background: transparent; }
"""


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
        self.setDocumentMode(True)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setUsesScrollButtons(True)
        # 常显 tab bar：autoHide 会把「＋」一起藏掉
        self.setTabBarAutoHide(False)
        self.setStyleSheet(_TABS_STYLE)
        # macOS 原生样式会画 #bebebe 的 tab bar base，盖住 QSS 底色；
        # 关掉后由 QSS 的 #f3f3f3 接管（Windows/Fusion 本来没有这层）。
        self.tabBar().setDrawBase(False)

        self._new_page_allowed = False
        # 「＋」跟随最后一个标签，而不是钉在标签栏最右端
        self._new_page_btn = QToolButton(self.tabBar())
        self._new_page_btn.setObjectName("newPageButton")
        self._new_page_btn.setText("+")
        self._new_page_btn.setAutoRaise(True)
        self._new_page_btn.setFixedSize(26, 24)
        self._new_page_btn.setToolTip("新建绘图页（最多 %d 页）" % MAX_PAGES)
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

    def _place_new_page_button(self) -> None:
        """把「＋」放到最后一个标签右侧；空间不足时贴右缘。"""
        bar = self.tabBar()
        button = self._new_page_btn
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
    # 「＋」可用性：未加载数据 / 已达上限时禁用（决策 D-D / §4.4）
    # ------------------------------------------------------------------
    def set_new_page_enabled(self, enabled: bool) -> None:
        """外部闸门（是否已加载数据）；与页数上限共同决定按钮可用性。"""
        self._new_page_allowed = bool(enabled)
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
