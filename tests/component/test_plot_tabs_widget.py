"""PlotTabsWidget：尾部「＋」闸门、上限、信号转发与数据驱动的标签栏可见性。

对应设计文档 §4.4 与决策 D-D（未加载数据时禁用「＋」并隐藏整个标签栏，
加载成功后显示且从此常显；上限由 `MAX_PAGES` 决定）。「＋」位置由 tabLayoutChange
钩子驱动，重命名变宽也不得重叠。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtTest import QTest
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
    bar_x = bar.mapTo(tabs, QPoint(0, 0)).x()
    last_right = bar_x + bar.tabRect(bar.count() - 1).right()
    assert plus.x() > last_right, "重命名变长后「＋」被标签盖住"

    wide_x = plus.x()
    tabs.setTabText(1, "短名")
    qapp.processEvents()
    last_right = bar_x + bar.tabRect(bar.count() - 1).right()
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
    bar_x = bar.mapTo(tabs, QPoint(0, 0)).x()
    last_right = bar_x + last.right()
    # 紧挨最后一个标签右侧，而不是标签栏最右端
    assert 0 <= button.x() - last_right <= 12
    assert button.x() + button.width() < bar_x + bar.width() - 100


def test_scroll_arrows_use_flat_custom_icons(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    qapp.processEvents()
    scroll_buttons = [
        child
        for child in tabs.tabBar().children()
        if isinstance(child, QToolButton)
        and "scroll" in (child.objectName() or "").lower()
    ]
    assert len(scroll_buttons) == 2
    for button in scroll_buttons:
        assert button.arrowType() == Qt.ArrowType.NoArrow
        assert not button.icon().isNull()
        assert button.iconSize() == QSize(12, 12)


def _scroll_buttons(tabs):
    out = {}
    for child in tabs.tabBar().children():
        if isinstance(child, QToolButton) and "scroll" in (child.objectName() or "").lower():
            out["left" if "left" in child.objectName().lower() else "right"] = child
    return out


def test_scroll_buttons_adjacent_within_reserved_zone(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    bar = tabs.tabBar()
    btns = _scroll_buttons(tabs)
    left, right = btns["left"], btns["right"]
    assert left.isVisible() and right.isVisible(), "16 页窄窗应进入滚动模式"
    assert left.geometry().right() < right.geometry().x(), "两按钮不得重叠"
    assert right.geometry().right() <= bar.width() - 1
    last = bar.tabRect(bar.count() - 1)
    assert last.right() < left.geometry().x(), "按钮不得压住最后一个标签"
    assert left.isEnabled() and not right.isEnabled(), "停在最右：左可用右禁用"


def test_scroll_geometry_self_heals_on_resize(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    for width in (500, 640, 500):
        tabs.resize(width, 360)
        qapp.processEvents()
        bar = tabs.tabBar()
        btns = _scroll_buttons(tabs)
        assert btns["left"].isVisible(), "窄窗应始终处于滚动模式"
        assert btns["left"].geometry().right() < btns["right"].geometry().x()
        assert btns["right"].geometry().right() <= bar.width() - 1


def test_scroll_geometry_survives_gate_hide_show(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    tabs.set_new_page_enabled(False)
    qapp.processEvents()
    tabs.resize(420, 360)
    qapp.processEvents()
    tabs.set_new_page_enabled(True)
    qapp.processEvents()
    bar = tabs.tabBar()
    btns = _scroll_buttons(tabs)
    assert btns["left"].isVisible() and btns["right"].isVisible()
    assert btns["left"].geometry().right() < btns["right"].geometry().x()
    assert btns["right"].geometry().right() <= bar.width() - 1


def test_right_scroll_click_advances_from_left_end(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(0)
    qapp.processEvents()
    bar = tabs.tabBar()
    btns = _scroll_buttons(tabs)
    assert bar.tabRect(MAX_PAGES - 1).right() > bar.width(), "停最左时末页应在视野外"
    assert not btns["left"].isEnabled() and btns["right"].isEnabled()
    for _ in range(60):
        if bar.tabRect(MAX_PAGES - 1).right() <= bar.width():
            break
        QTest.mouseClick(btns["right"], Qt.MouseButton.LeftButton)
        qapp.processEvents()
    assert bar.tabRect(MAX_PAGES - 1).right() <= bar.width(), "右箭头应能滚到末页"


def test_left_scroll_click_returns_to_first_page(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    bar = tabs.tabBar()
    left = _scroll_buttons(tabs)["left"]
    assert bar.tabRect(0).x() < 0, "停在最右时第一页应在视野外"
    for _ in range(60):
        if bar.tabRect(0).x() >= 0:
            break
        QTest.mouseClick(left, Qt.MouseButton.LeftButton)
        qapp.processEvents()
    assert bar.tabRect(0).x() >= 0, "真实点击左箭头应能滚回第一页"
    assert not left.isEnabled(), "滚到最左后左箭头应禁用"
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(0).center())
    qapp.processEvents()
    assert tabs.currentIndex() == 0, "第一页应可点击切换"


# ---------------------------------------------------------------------------
# 角位暂存区胶囊（方案 A）：存在性 / 初始隐藏 / 闸门同步 / 状态转发
# ---------------------------------------------------------------------------


def _make_tabs_with_gate(qtbot):
    from PySide6.QtWidgets import QWidget

    tabs = PlotTabsWidget()
    qtbot.addWidget(tabs)
    tabs.resize(640, 300)
    tabs.addTab(QWidget(), "页面页1")
    return tabs


def test_corner_shelf_capsule_exists_and_initially_hidden(qapp, qtbot):
    tabs = _make_tabs_with_gate(qtbot)
    tabs.show()
    qapp.processEvents()
    capsule = tabs.shelf_capsule
    assert capsule is not None
    assert capsule.isVisible() is False

    tabs.set_new_page_enabled(True)
    qapp.processEvents()
    assert capsule.isVisible() is True


def test_new_page_gate_syncs_shelf_capsule_visibility(qapp, qtbot):
    tabs = _make_tabs_with_gate(qtbot)
    tabs.set_new_page_enabled(True)
    tabs.show()
    qapp.processEvents()
    tabs.set_new_page_enabled(False)
    qapp.processEvents()
    assert tabs.tabBar().isVisible() is False
    assert tabs.shelf_capsule.isVisible() is False


def test_shelf_capsule_state_api_forwards(qapp, qtbot):
    tabs = _make_tabs_with_gate(qtbot)
    tabs.set_new_page_enabled(True)
    tabs.show()
    qapp.processEvents()
    capsule = tabs.shelf_capsule

    tabs.set_shelf_count(3)
    assert capsule.count() == 3
    assert "3 条" in capsule.toolTip()

    tabs.set_shelf_active(True)
    assert capsule.isChecked() is True
    tabs.set_shelf_active(False)
    assert capsule.isChecked() is False

    tabs.set_shelf_enabled(False)
    assert capsule.isEnabled() is False
    tabs.set_shelf_enabled(True)
    assert capsule.isEnabled() is True


# ----------------------------------------------------------------------
# T1：页签全名 tooltip（provider hover 时实时求值，无同步点）
# ----------------------------------------------------------------------
def _send_tab_tooltip(tabs: PlotTabsWidget, index: int, qapp) -> None:
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent

    bar = tabs.tabBar()
    rect = bar.tabRect(index)
    event = QHelpEvent(
        QEvent.Type.ToolTip, rect.center(), bar.mapToGlobal(rect.center())
    )
    qapp.sendEvent(bar, event)


def test_tab_tooltip_shows_provider_text(tabs, qapp):
    from PySide6.QtWidgets import QToolTip

    _add_pages(tabs, 3)
    tabs.set_tab_info_provider(lambda i: f"完整页名 {i + 1}\n第 {i + 1}/3 页")
    _send_tab_tooltip(tabs, 1, qapp)
    assert "完整页名 2" in QToolTip.text()
    assert "第 2/3 页" in QToolTip.text()


def test_tab_tooltip_reevaluated_on_each_hover(tabs, qapp):
    from PySide6.QtWidgets import QToolTip

    _add_pages(tabs, 2)
    state = {"name": "旧名"}
    tabs.set_tab_info_provider(lambda i: state["name"])
    _send_tab_tooltip(tabs, 0, qapp)
    assert QToolTip.text() == "旧名"
    state["name"] = "改名后"
    _send_tab_tooltip(tabs, 0, qapp)
    assert QToolTip.text() == "改名后"


def test_tab_tooltip_empty_text_suppressed(tabs, qapp):
    from PySide6.QtWidgets import QToolTip

    _add_pages(tabs, 2)
    QToolTip.hideText()
    tabs.set_tab_info_provider(lambda i: "")
    _send_tab_tooltip(tabs, 0, qapp)
    assert QToolTip.text() == ""


def test_tab_tooltip_without_provider_falls_through(tabs, qapp):
    _add_pages(tabs, 2)
    _send_tab_tooltip(tabs, 0, qapp)  # 无 provider：不拦截、不崩


# ----------------------------------------------------------------------
# T2：页签右键菜单（重命名/关闭/关闭其他/关闭右侧）
# ----------------------------------------------------------------------
def test_context_menu_actions_emit_signals(tabs):
    _add_pages(tabs, 4)
    menu = tabs._build_tab_menu(1)
    assert [a.text() for a in menu.actions() if a.text()] == [
        "重命名", "关闭", "关闭其他", "关闭右侧",
    ]
    got = []
    tabs.rename_page_requested.connect(lambda i: got.append(("rename", i)))
    tabs.close_page_requested.connect(lambda i: got.append(("close", i)))
    tabs.close_other_pages_requested.connect(lambda i: got.append(("others", i)))
    tabs.close_right_pages_requested.connect(lambda i: got.append(("right", i)))
    for action in menu.actions():
        if action.text():
            action.trigger()
    assert got == [("rename", 1), ("close", 1), ("others", 1), ("right", 1)]
    menu.deleteLater()


def test_context_menu_batch_items_disabled_at_edges(tabs):
    _add_pages(tabs, 1)
    menu = tabs._build_tab_menu(0)
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    assert not by_text["关闭其他"].isEnabled()
    assert not by_text["关闭右侧"].isEnabled()
    menu.deleteLater()

    _add_pages(tabs, 2)                      # 共 3 页
    menu = tabs._build_tab_menu(0)
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    assert by_text["关闭其他"].isEnabled()
    assert by_text["关闭右侧"].isEnabled()
    menu.deleteLater()

    menu = tabs._build_tab_menu(2)           # 末页：右侧无页
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    assert by_text["关闭其他"].isEnabled()
    assert not by_text["关闭右侧"].isEnabled()
    menu.deleteLater()
