"""PlotTabsWidget：尾部「＋」闸门、上限、信号转发与数据驱动的标签栏可见性。

对应设计文档 §4.4 与决策 D-D（未加载数据时禁用「＋」并隐藏整个标签栏，
加载成功后显示且从此常显；上限 8 页）。「＋」位置由 tabLayoutChange
钩子驱动，重命名变宽也不得重叠。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QToolButton

from src.ui.widgets.plot_tabs import MAX_PAGES, PlotTabsWidget


@pytest.fixture()
def tabs(qapp, qtbot):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.resize(640, 360)
    widget.show()
    # fixture 模拟"已加载数据"的使用态：标签栏可见；依赖"未加载态"的
    # 测试在开头自行 set_new_page_enabled(False) 复位。
    widget.set_new_page_enabled(True)
    qapp.processEvents()
    yield widget
    widget.close()


def _plus_button(tabs: PlotTabsWidget) -> QToolButton:
    button = tabs.new_page_button
    assert isinstance(button, QToolButton)
    return button


def _add_pages(tabs: PlotTabsWidget, count: int) -> None:
    for index in range(count):
        tabs.addTab(QLabel(f"page {index}"), f"page {index + 1}")


def test_corner_plus_disabled_until_data_loaded(tabs):
    tabs.set_new_page_enabled(False)
    assert not _plus_button(tabs).isEnabled()
    tabs.set_new_page_enabled(True)
    assert _plus_button(tabs).isEnabled()
    tabs.set_new_page_enabled(False)
    assert not _plus_button(tabs).isEnabled()


def test_plus_emits_only_when_allowed(tabs, qtbot):
    tabs.set_new_page_enabled(False)
    received = []
    tabs.new_page_requested.connect(lambda: received.append(True))

    qtbot.mouseClick(_plus_button(tabs), Qt.MouseButton.LeftButton)
    assert received == []          # 未加载数据：禁用，不发出请求

    tabs.set_new_page_enabled(True)
    qtbot.mouseClick(_plus_button(tabs), Qt.MouseButton.LeftButton)
    assert received == [True]


def test_plus_disables_at_page_limit(tabs):
    tabs.set_new_page_enabled(True)
    _add_pages(tabs, MAX_PAGES)
    assert tabs.count() == MAX_PAGES
    assert not _plus_button(tabs).isEnabled()

    tabs.removeTab(MAX_PAGES - 1)
    assert _plus_button(tabs).isEnabled()


def test_close_request_forwarded(tabs):
    _add_pages(tabs, 3)
    received = []
    tabs.close_page_requested.connect(received.append)
    tabs.tabCloseRequested.emit(2)
    assert received == [2]


def test_custom_close_button_attached_and_centered(tabs, qapp):
    """自绘关闭按钮替换原生图标：挂在每页右侧且在标签矩形内垂直居中。"""
    _add_pages(tabs, 3)
    qapp.processEvents()
    bar = tabs.tabBar()
    for index in range(3):
        button = tabs.close_button(index)
        assert isinstance(button, QToolButton)
        rect = bar.tabRect(index)
        button_rect = button.geometry()
        assert rect.isValid()
        assert rect.contains(button_rect)
        assert (rect.top() + rect.bottom()) == (button_rect.top() + button_rect.bottom())


def test_custom_close_button_click_emits_request(tabs, qtbot):
    _add_pages(tabs, 3)
    received = []
    tabs.close_page_requested.connect(received.append)
    qtbot.mouseClick(tabs.close_button(1), Qt.MouseButton.LeftButton)
    assert received == [1]


def test_custom_close_button_follows_tab_move(tabs, qtbot):
    """拖动重排后按钮随页移动，点击仍关闭正确的页（索引实时反查）。"""
    _add_pages(tabs, 3)
    received = []
    tabs.close_page_requested.connect(received.append)
    moved_button = tabs.close_button(0)
    tabs.tabBar().moveTab(0, 2)
    assert tabs.close_button(2) is moved_button
    qtbot.mouseClick(tabs.close_button(2), Qt.MouseButton.LeftButton)
    assert received == [2]


def test_middle_click_closes_tab(tabs, qtbot):
    _add_pages(tabs, 2)
    received = []
    tabs.close_page_requested.connect(received.append)
    center = tabs.tabBar().tabRect(1).center()
    qtbot.mouseClick(
        tabs.tabBar(), Qt.MouseButton.MiddleButton, pos=center
    )
    assert received == [1]


def test_double_click_requests_rename(tabs, qtbot):
    _add_pages(tabs, 2)
    received = []
    tabs.rename_page_requested.connect(received.append)
    center = tabs.tabBar().tabRect(0).center()
    qtbot.mouseDClick(tabs.tabBar(), Qt.MouseButton.LeftButton, pos=center)
    assert received == [0]


def test_tab_bar_hidden_on_fresh_construction(qapp, qtbot):
    """刚构造、未调用闸门时标签栏就应隐藏（空状态不留灰带）。"""
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    qapp.processEvents()
    assert not widget.tabBar().isVisible()
    widget.close()


def test_tab_bar_hidden_until_data_gate_opens(tabs, qapp):
    """未加载数据时整个标签栏隐藏（VSCode 无文件即无标签栏）。"""
    tabs.set_new_page_enabled(False)
    _add_pages(tabs, 1)
    qapp.processEvents()
    assert not tabs.tabBar().isVisible()

    tabs.set_new_page_enabled(True)
    qapp.processEvents()
    assert tabs.tabBar().isVisible()


def test_tab_bar_stays_visible_with_single_page(tabs, qapp):
    """闸门开启后 tab bar 必须常显：autoHide 是"单页隐藏"语义，会把「＋」
    一起藏掉，加载后单页时新建入口消失。"""
    _add_pages(tabs, 1)
    qapp.processEvents()
    assert tabs.tabBar().isVisible()
    assert _plus_button(tabs).isVisible()


def test_plus_reposition_after_rename(tabs, qapp):
    """重命名变长只改标签宽度、不走插删路径，「＋」必须随布局钩子重放置，
    不得被拉宽的标签盖住；改短后位置也要跟回来。"""
    _add_pages(tabs, 2)
    qapp.processEvents()
    bar = tabs.tabBar()
    plus = _plus_button(tabs)

    tabs.setTabText(1, "超长绘图页名称超长绘图页名称超长")
    qapp.processEvents()
    last_right = bar.tabRect(bar.count() - 1).right()
    assert plus.x() > last_right, "重命名变长后「＋」被标签盖住"

    wide_x = plus.x()
    tabs.setTabText(1, "短名")
    qapp.processEvents()
    last_right = bar.tabRect(bar.count() - 1).right()
    assert plus.x() > last_right
    assert plus.x() < wide_x, "重命名改短后「＋」位置没有跟回来"


def test_native_tab_bar_base_is_disabled(tabs):
    """macOS 原生 base 会盖住 QSS 底色（#bebebe），必须显式关掉。"""
    assert tabs.tabBar().drawBase() is False


def test_plus_follows_last_tab_instead_of_far_right(tabs, qapp):
    _add_pages(tabs, 3)
    qapp.processEvents()
    bar = tabs.tabBar()
    button = _plus_button(tabs)
    last = bar.tabRect(2)
    assert last.isValid()
    # 紧挨最后一个标签右侧，而不是标签栏最右端
    assert 0 <= button.x() - last.right() <= 12
    assert button.x() + button.width() < bar.width() - 100
