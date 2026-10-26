"""PlotTabsWidget：尾部「＋」闸门、上限、信号转发与常显 tab bar。

对应设计文档 §4.4 与决策 D-D（未加载数据时禁用「＋」，上限 8 页）。
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
    assert not _plus_button(tabs).isEnabled()
    tabs.set_new_page_enabled(True)
    assert _plus_button(tabs).isEnabled()
    tabs.set_new_page_enabled(False)
    assert not _plus_button(tabs).isEnabled()


def test_plus_emits_only_when_allowed(tabs, qtbot):
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


def test_tab_bar_stays_visible_with_single_page(tabs, qapp):
    """autoHide 会把「＋」一起隐藏，故 tab bar 必须常显。"""
    _add_pages(tabs, 1)
    qapp.processEvents()
    assert tabs.tabBar().isVisible()
    assert _plus_button(tabs).isVisible()


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
