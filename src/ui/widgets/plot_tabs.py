"""绘图区标签容器（PlotTabsWidget）。

设计文档 §4.4：选 ``QTabWidget`` 子类，自带关闭按钮 / 可拖动 / 省略号；
角部放「＋」新建页。**不启用** ``setTabBarAutoHide(True)``：它会把角部
控件一并隐藏，导致只有一页时「＋」消失（实测）。

本类只负责标签交互与信号转发，页对象的增删与切页时序由 MainWindow 掌握。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QTabWidget, QToolButton

#: 绘图页数量上限（设计文档 §4.4）
MAX_PAGES = 8


class PlotTabsWidget(QTabWidget):
    """多绘图页标签容器。

    Signals:
        new_page_requested: 点击角部「＋」。
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
        # 常显 tab bar：autoHide 会连 corner 的「＋」一起藏掉
        self.setTabBarAutoHide(False)

        self._new_page_allowed = False
        self._new_page_btn = QToolButton(self)
        self._new_page_btn.setText("+")
        self._new_page_btn.setAutoRaise(True)
        self._new_page_btn.setToolTip("新建绘图页（最多 %d 页）" % MAX_PAGES)
        self._new_page_btn.clicked.connect(self._on_new_page_clicked)
        self.setCornerWidget(self._new_page_btn, Qt.Corner.TopRightCorner)

        self.tabCloseRequested.connect(self.close_page_requested)
        self.tabBarDoubleClicked.connect(self.rename_page_requested)
        self.tabBar().installEventFilter(self)
        self._sync_new_page_button()

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

    def tabRemoved(self, index: int) -> None:  # noqa: N802 - Qt 命名
        super().tabRemoved(index)
        self._sync_new_page_button()

    # ------------------------------------------------------------------
    # 中键关闭
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.tabBar() and event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.MiddleButton:
                index = self.tabBar().tabAt(event.position().toPoint())
                if index >= 0:
                    self.close_page_requested.emit(index)
                    return True
        return super().eventFilter(obj, event)
