"""PlotTabsWidget：尾部「＋」闸门、上限、信号转发与数据驱动的标签栏可见性。

对应设计文档 §4.4 与决策 D-D（未加载数据时禁用「＋」并隐藏整个标签栏，
加载成功后显示且从此常显；上限由 `MAX_PAGES` 决定）。「＋」位置由 tabLayoutChange
钩子驱动，重命名变宽也不得重叠。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QPoint, Qt
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


def _scroll_buttons(tabs):
    out = {}
    for child in tabs.tabBar().children():
        if isinstance(child, QToolButton) and "scroll" in (child.objectName() or "").lower():
            out["left" if "left" in child.objectName().lower() else "right"] = child
    return out


# ---------------------------------------------------------------------------
# 方向 B：‹ › 永不显示；溢出导航＝滚轮 + 隐藏按钮程序化驱动 + ▾ 目录
# ---------------------------------------------------------------------------
def test_scroll_arrows_never_visible(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    qapp.processEvents()
    btns = _scroll_buttons(tabs)
    assert len(btns) == 2
    assert not btns["left"].isVisible() and not btns["right"].isVisible()


def test_arrows_stay_hidden_on_resize(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    for width in (500, 640, 500):
        tabs.resize(width, 360)
        qapp.processEvents()
        btns = _scroll_buttons(tabs)
        assert not btns["left"].isVisible() and not btns["right"].isVisible()


def test_arrows_stay_hidden_after_gate_cycle(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    tabs.set_new_page_enabled(False)
    qapp.processEvents()
    tabs.resize(420, 360)
    qapp.processEvents()
    tabs.set_new_page_enabled(True)
    qapp.processEvents()
    btns = _scroll_buttons(tabs)
    assert not btns["left"].isVisible() and not btns["right"].isVisible()


def test_hidden_right_button_scrolls_to_last_page(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(0)
    qapp.processEvents()
    bar = tabs.tabBar()
    right = _scroll_buttons(tabs)["right"]
    assert not right.isVisible()
    assert bar.tabRect(MAX_PAGES - 1).right() > bar.width(), "停最左时末页应在视野外"
    for _ in range(60):
        if bar.tabRect(MAX_PAGES - 1).right() <= bar.width():
            break
        right.click()
        qapp.processEvents()
    assert bar.tabRect(MAX_PAGES - 1).right() <= bar.width(), "隐藏右按钮应能滚到末页"


def test_hidden_left_button_returns_to_first_page(tabs, qapp):
    _add_pages(tabs, MAX_PAGES)
    tabs.setCurrentIndex(MAX_PAGES - 1)
    qapp.processEvents()
    bar = tabs.tabBar()
    left = _scroll_buttons(tabs)["left"]
    assert bar.tabRect(0).x() < 0, "停在最右时第一页应在视野外"
    for _ in range(60):
        if bar.tabRect(0).x() >= 0:
            break
        left.click()
        qapp.processEvents()
    assert bar.tabRect(0).x() >= 0, "隐藏左按钮应能滚回第一页"
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(0).center())
    qapp.processEvents()
    assert tabs.currentIndex() == 0, "第一页应可点击切换"


def test_dir_tooltip_reports_hidden_counts(tabs, qapp):
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip

    tabs.resize(640, 360)
    _add_pages(tabs, MAX_PAGES)
    qapp.processEvents()
    btn = tabs._dir_btn
    event = QHelpEvent(
        QEvent.Type.ToolTip,
        btn.rect().center(),
        btn.mapToGlobal(btn.rect().center()),
    )
    qapp.sendEvent(btn, event)
    text = QToolTip.text()
    assert "页目录（16 页）" in text
    assert "右隐" in text and "左隐 0" in text


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
    # 第 6 轮 NB6：tooltip 转义 + <qt> 富文本口径（与 shelf/var_tooltip 一致）
    assert QToolTip.text() == "<qt>旧名</qt>"
    state["name"] = "改名后"
    _send_tab_tooltip(tabs, 0, qapp)
    assert QToolTip.text() == "<qt>改名后</qt>"


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


# ----------------------------------------------------------------------
# T4：滚轮横向滚动（不切页）＋ 页签宽度恢复 main 自然宽（T8 已回退）
# ----------------------------------------------------------------------
def _wheel_bar(tabs: PlotTabsWidget, delta_y: int, qapp) -> None:
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    bar = tabs.tabBar()
    pos = QPointF(10, bar.height() / 2)
    event = QWheelEvent(
        pos,
        bar.mapToGlobal(pos.toPoint()),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    qapp.sendEvent(bar, event)


def test_tab_width_keeps_main_natural_behavior(tabs, qapp):
    """T8 回退：页签恢复 main 自然宽（默认名 < 96px），不再人为加宽。

    16 页窄窗仍进入溢出，辨识靠 T1 全名 tooltip＋T6 页目录，滚动靠 T4 滚轮。
    """
    tabs.resize(640, 360)
    _add_pages(tabs, 16)
    qapp.processEvents()
    bar = tabs.tabBar()
    # 平台中立：宽＝自然 hint（无下限撑宽、无压缩），且未触 200 封顶。
    # 曾用 <96 作 Mac 字体代理值，Windows 自然宽 112/124 误报（2026-11-21 实测）
    assert all(
        bar.tabRect(i).width() == bar.tabSizeHint(i).width() < 200
        for i in range(16)
    )
    assert bar.tabRect(15).right() > bar.width()   # 压不下 → 进入溢出


def test_wheel_scrolls_without_switching_page(tabs, qapp):
    tabs.resize(640, 360)
    _add_pages(tabs, 16)
    qapp.processEvents()
    bar = tabs.tabBar()
    bar.setCurrentIndex(0)
    x0 = bar.tabRect(0).x()
    _wheel_bar(tabs, -120, qapp)               # 滚轮下滑 → 向右滚
    qapp.processEvents()
    assert bar.currentIndex() == 0             # 不切页
    x1 = bar.tabRect(0).x()
    assert x1 < x0
    _wheel_bar(tabs, 120, qapp)                # 上滑 → 滚回
    qapp.processEvents()
    assert bar.tabRect(0).x() > x1
    assert bar.currentIndex() == 0


def test_wheel_without_overflow_does_not_switch(tabs, qapp):
    tabs.resize(1200, 360)
    _add_pages(tabs, 2)
    qapp.processEvents()
    bar = tabs.tabBar()
    bar.setCurrentIndex(0)
    _wheel_bar(tabs, -120, qapp)
    qapp.processEvents()
    assert bar.currentIndex() == 0


# ----------------------------------------------------------------------
# T6：角位「▾」页目录 + 锚定下拉
# ----------------------------------------------------------------------
def _open_dir_popup(tabs, qtbot, qapp):
    qtbot.mouseClick(tabs._dir_btn, Qt.MouseButton.LeftButton)
    popup = tabs._dir_popup
    qtbot.addWidget(popup)
    qapp.processEvents()
    return popup


def _dir_rows(popup):
    return [
        popup._rows_layout.itemAt(i).widget()
        for i in range(popup._rows_layout.count())
    ]


def test_dir_button_gate_and_corner_width(tabs, qapp):
    tabs.set_new_page_enabled(False)
    qapp.processEvents()
    assert not tabs._dir_btn.isVisible()
    tabs.set_new_page_enabled(True)
    qapp.processEvents()
    assert tabs._dir_btn.isVisible()
    # [＋槽 28][▾ 20][胶囊 52][右缘 2]（方案 A）
    assert tabs._corner_container.width() == 102


def test_dir_popup_lists_pages_and_selects(tabs, qtbot, qapp):
    _add_pages(tabs, 5)
    tabs.set_page_list_provider(
        lambda: [(f"完整页名 {i + 1}", "2×2", i == 2) for i in range(5)]
    )
    got = []
    tabs.page_selected.connect(lambda i: got.append(i))
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert popup.isVisible()
    rows = _dir_rows(popup)
    assert len(rows) == 5
    labels = rows[2].findChildren(QLabel)
    assert any("完整页名 3" in lb.text() for lb in labels)
    qtbot.mouseClick(rows[3], Qt.MouseButton.LeftButton)
    assert got == [3]
    assert not popup.isVisible()


def test_dir_popup_fallback_without_provider(tabs, qtbot, qapp):
    _add_pages(tabs, 3)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    rows = _dir_rows(popup)
    assert len(rows) == 3
    labels = rows[0].findChildren(QLabel)
    assert any("page 1" in lb.text() for lb in labels)
    popup.hide()


def test_dir_popup_row_close_and_footer_new(tabs, qtbot, qapp):
    _add_pages(tabs, 3)
    closed, created = [], []
    tabs.close_page_requested.connect(lambda i: closed.append(i))
    tabs.new_page_requested.connect(lambda: created.append(True))
    popup = _open_dir_popup(tabs, qtbot, qapp)
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent

    row = _dir_rows(popup)[1]
    row.enterEvent(QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)))
    assert row._close.isVisible()
    qtbot.mouseClick(row._close, Qt.MouseButton.LeftButton)
    assert closed == [1]
    assert not popup.isVisible()

    popup = _open_dir_popup(tabs, qtbot, qapp)
    qtbot.mouseClick(popup._new_row, Qt.MouseButton.LeftButton)
    assert created == [True]
    assert not popup.isVisible()


def test_dir_popup_esc_closes(tabs, qtbot, qapp):
    _add_pages(tabs, 2)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert popup.isVisible()
    QTest.keyClick(popup, Qt.Key.Key_Escape)
    assert not popup.isVisible()


def test_dir_popup_long_current_name_stays_inside(tabs, qtbot, qapp):
    """长名＋加粗当前行：省略按最终字体度量，行/角标/× 不得溢出 popup。"""
    _add_pages(tabs, 3)
    long_title = "a" * 30 + "史蒂夫" + "s" * 20
    tabs.set_page_list_provider(
        lambda: [("page 1", "1×1", False), ("page 2", "1×1", False),
                 (long_title, "2×2", True)]
    )
    popup = _open_dir_popup(tabs, qtbot, qapp)
    row = _dir_rows(popup)[2]
    assert row.width() <= popup.width()
    for child in row.findChildren(QLabel) + [row._close]:
        right = child.mapTo(popup, QPoint(0, 0)).x() + child.width()
        assert right <= popup.width(), "长名行的角标/× 被挤出 popup"
    name = [lb for lb in row.findChildren(QLabel) if lb.toolTip() == long_title][0]
    assert name.text().endswith("…")
    assert name.font().bold()
    popup.hide()


def test_dir_popup_background_is_white_not_gray(tabs, qtbot, qapp):
    _add_pages(tabs, 2)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert "background: #ffffff" in popup.styleSheet()
    assert "QScrollArea { background: transparent" in popup.styleSheet()
    assert not popup._scroll.viewport().autoFillBackground()
    assert not popup._rows_host.autoFillBackground()
    popup.hide()


# ----------------------------------------------------------------------
# V-A fit：不压缩（最小宽＝自然宽）＋ 200px 封顶（VSCode 默认 fit 语义）
# ----------------------------------------------------------------------
def test_fit_no_compression_short_tabs_keep_natural_width(tabs, qapp):
    tabs.resize(1288, 360)
    _add_pages(tabs, 16)
    qapp.processEvents()
    bar = tabs.tabBar()
    for i in range(16):
        assert bar.tabRect(i).width() == bar.tabSizeHint(i).width(), "fit 不得压缩"
    assert bar.tabRect(0).width() == bar.tabSizeHint(0).width() < 200  # 自然宽
    assert bar.tabRect(15).right() > bar.width()   # 放不下 → 溢出滚动


def test_fit_long_name_capped_at_max(tabs, qapp):
    tabs.resize(1288, 360)
    tabs.addTab(QLabel("x"), "长" * 60)
    qapp.processEvents()
    bar = tabs.tabBar()
    assert bar.tabRect(0).width() == 200


# ----------------------------------------------------------------------
# V-A：3px 溢出指示条（仅指示、不交互、溢出才显示、随滚动同步）
# ----------------------------------------------------------------------
def test_overflow_indicator_hidden_when_no_overflow(tabs, qapp):
    tabs.resize(1200, 360)
    _add_pages(tabs, 2)
    qapp.processEvents()
    assert not tabs._overflow_bar.isVisible()


def test_overflow_indicator_visible_and_tracks_scroll(tabs, qapp, qtbot):
    tabs.resize(640, 360)
    _add_pages(tabs, 16)
    qapp.processEvents()
    ind = tabs._overflow_bar
    bar = tabs.tabBar()
    assert ind.isVisible()
    assert ind.height() == 3
    assert ind.width() == bar.width()
    assert ind.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert ind.y() == bar.mapTo(tabs, QPoint(0, bar.height() - 3)).y()
    assert ind._total > ind._visible
    offset0 = ind._offset
    _wheel_bar(tabs, -120, qapp)
    qapp.processEvents()
    assert ind._offset > offset0, "滚动后指示条偏移应同步"
    # auto-hide：停滚且鼠标不在栏上 500ms 后隐藏
    # （offscreen show() 会合成 enter，先模拟鼠标离开页签栏）
    bar.leaveEvent(QEvent(QEvent.Type.Leave))
    qtbot.wait(700)
    assert not ind.isVisible(), "空闲 500ms 后应 auto-hide"
    # 鼠标进入页签栏 → 常显（超过 500ms 也不隐）
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent

    bar.enterEvent(QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)))
    assert ind.isVisible()
    qtbot.wait(700)
    assert ind.isVisible(), "鼠标在栏上期间应常显"
    # 离开立即隐藏
    bar.leaveEvent(QEvent(QEvent.Type.Leave))
    assert not ind.isVisible()
    # 再滚一次 → 重新 reveal
    _wheel_bar(tabs, -120, qapp)
    qapp.processEvents()
    assert ind.isVisible()


# ----------------------------------------------------------------------
# 方案 A：角位双 chip 常驻容器；胶囊内联计数；＋ 保持扁平
# ----------------------------------------------------------------------
def test_corner_chips_rest_container_and_plus_stays_flat(tabs, qapp):
    _add_pages(tabs, 3)
    qapp.processEvents()
    assert tabs._dir_btn._CHIP_REST is True
    assert hasattr(tabs.shelf_capsule, "_draw_count")   # 胶囊自绘 chip＋内联计数
    assert tabs.new_page_button._CHIP_REST is False
    assert tabs.shelf_capsule.width() == 52
    # rest / 计数 0 / 99+ / 打开态 / 禁用态 绘制均不崩
    for count in (0, 3, 100):
        tabs.set_shelf_count(count)
        tabs.shelf_capsule.grab()
    tabs.set_shelf_active(True)
    tabs.shelf_capsule.grab()
    tabs.set_shelf_enabled(False)
    tabs.shelf_capsule.grab()
    tabs._dir_btn.grab()


def test_capsule_content_group_centered_for_all_counts(tabs, qapp):
    """字形＋4px＋计数实测宽整组居中：左右留白恒等（Win 真机错位回归）。"""
    _add_pages(tabs, 2)
    qapp.processEvents()
    cap = tabs.shelf_capsule
    for count in (0, 3, 12, 100):
        tabs.set_shelf_count(count)
        cap.grab()
        left = cap._glyph_cx - 7
        right = cap.width() - cap._count_rect.right()
        assert abs(left - right) < 0.6, f"count={count} 左右留白不等"
        assert abs(cap._count_rect.left() - (cap._glyph_cx + 7) - 4) < 0.01


def test_dir_rows_and_footer_share_fixed_height(tabs, qtbot, qapp):
    """A：目录行与 footer 新建行同高 26px，Win/mac 节奏恒等。"""
    _add_pages(tabs, 3)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    rows = _dir_rows(popup)
    assert all(r.height() == 26 for r in rows)
    assert popup._new_row.height() == 26
    popup.hide()


def test_dir_new_row_enter_triggers_and_disabled_at_limit(tabs, qtbot, qapp):
    _add_pages(tabs, 3)
    created = []
    tabs.new_page_requested.connect(lambda: created.append(True))
    popup = _open_dir_popup(tabs, qtbot, qapp)
    popup._new_row.setFocus()
    QTest.keyClick(popup._new_row, Qt.Key.Key_Return)
    assert created == [True]
    popup.hide()

    tabs.addTab(QLabel("x"), "page 4")
    for _ in range(MAX_PAGES - 4):
        tabs.addTab(QLabel("x"), "page x")
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert not popup._new_row.isEnabled()
    QTest.keyClick(popup._new_row, Qt.Key.Key_Return)
    qtbot.mouseClick(popup._new_row, Qt.MouseButton.LeftButton)
    assert created == [True], "上限禁用后不得再发射"
    popup.hide()


def test_corner_background_matches_tab_bar_across_styles(tabs, qapp):
    """角位容器与 tabBar 底色恒等（Win/Fusion palette 缝回归）。"""
    from PySide6.QtWidgets import QApplication

    _add_pages(tabs, 3)
    qapp.processEvents()
    original = QApplication.style().objectName()
    try:
        for style in ("Windows", "Fusion"):
            QApplication.setStyle(style)
            qapp.processEvents()
            img = tabs.grab().toImage()
            bar = tabs.tabBar()
            y = bar.height() // 2
            assert img.pixelColor(bar.width() - 6, y).name() == img.pixelColor(
                bar.width() + 10, y
            ).name(), f"{style} 下角位与标签栏存在色差"
    finally:
        QApplication.setStyle(original)


def test_dir_popup_height_exact_no_dead_zone(tabs, qtbot, qapp):
    """单页/多页 popup 高度＝行高×N＋分隔块9＋新建行26＋外margin10，无死区。"""
    _add_pages(tabs, 1)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert popup.height() == 26 + 9 + 26 + 10
    sep = popup.layout().itemAt(1).widget()
    assert sep.height() == 9
    assert popup.width() >= 200, "单页 popup 不得窄到截断页名"
    popup.hide()
    for _ in range(15):
        tabs.addTab(QLabel("x"), "page x")
    popup = _open_dir_popup(tabs, qtbot, qapp)
    assert popup.height() == 16 * 26 + 9 + 26 + 10
    popup.hide()


def test_corner_chips_vertically_centered_with_tabs_across_styles(tabs, qapp):
    """Win/Fusion 角位容器被风格 hint 区域放低 8px 的错位回归。"""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    _add_pages(tabs, 2)
    qapp.processEvents()
    original = QApplication.style().objectName()
    try:
        for style in ("Windows", "Fusion", "macOS"):
            QApplication.setStyle(style)
            qapp.processEvents()
            bar = tabs.tabBar()
            bar_y = bar.mapTo(tabs, QPoint(0, 0)).y()
            assert tabs._corner_container.mapTo(tabs, QPoint(0, 0)).y() == bar_y
            tab_center = bar.tabRect(0).center().y()
            for btn in (tabs._dir_btn, tabs.shelf_capsule):
                center = btn.mapTo(tabs, QPoint(0, btn.height() // 2)).y()
                assert abs(center - (bar_y + tab_center)) <= 1, f"{style} chip 垂直错位"
    finally:
        QApplication.setStyle(original)


def test_dir_popup_rebuild_leaves_no_ghost_rows(tabs, qtbot, qapp):
    from PySide6.QtCore import QEvent

    _add_pages(tabs, 6)
    popup = _open_dir_popup(tabs, qtbot, qapp)
    tabs.set_page_list_provider(lambda: [("page 1", "", True), ("page 2", "", False)])
    popup.open_for(tabs._dir_btn)
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    host = popup._rows_host
    assert len([c for c in host.children() if c.isWidgetType()]) == 2
    popup.hide()


def test_dir_popup_flips_above_when_below_screen_edge(tabs, qtbot, qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    _add_pages(tabs, MAX_PAGES)
    screen = QApplication.primaryScreen().availableGeometry()
    tabs.move(0, screen.bottom() - 40)     # 页签栏贴屏幕底
    qapp.processEvents()
    popup = _open_dir_popup(tabs, qtbot, qapp)
    anchor_top = tabs._dir_btn.mapToGlobal(QPoint(0, 0)).y()
    assert popup.y() + popup.height() <= anchor_top, "下方放不下应上翻"
    assert popup.y() >= screen.top()
    popup.hide()
    tabs.move(0, 0)


def test_dir_popup_flip_up_recomputes_height_from_above_space(
    tabs, qtbot, qapp, monkeypatch
):
    """上翻后行高预算＝锚点上方空间，不得再按下方剩余截成 120px。"""
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication

    class _FakeScreen:
        def availableGeometry(self):
            return QRect(0, 0, 800, 600)

    monkeypatch.setattr(QApplication, "screenAt", lambda *a: _FakeScreen())
    _add_pages(tabs, MAX_PAGES)
    tabs.move(0, 560)     # 锚点贴 800×600 假屏幕底缘
    qapp.processEvents()
    popup = _open_dir_popup(tabs, qtbot, qapp)
    anchor_top = tabs._dir_btn.mapToGlobal(QPoint(0, 0)).y()
    assert anchor_top >= 560
    expected_rows = min(MAX_PAGES * 26, max(120, anchor_top - 10 - 45))
    assert popup._scroll.height() == expected_rows, "上翻后应按上方空间重算行高"
    assert popup.height() == expected_rows + 9 + 26 + 10
    assert popup.y() + popup.height() <= anchor_top
    popup.hide()
    tabs.move(0, 0)


def test_hidden_tab_counts_single_pixel_boundary(tabs, qapp, monkeypatch):
    """right()==0 / left()==width-1 仍有一列像素可见不算隐藏；left()==width 零像素须计右隐。"""
    from PySide6.QtCore import QRect

    _add_pages(tabs, 4)
    qapp.processEvents()
    bar = tabs.tabBar()
    width = bar.width()
    rects = [
        QRect(-100, 0, 100, 30),        # right()=-1：完全在左视野外
        QRect(-99, 0, 100, 30),         # right()=0：仍露 1px
        QRect(width - 1, 0, 100, 30),   # left=width-1：仍露 1px
        QRect(width, 0, 100, 30),       # left=width：零像素
    ]
    monkeypatch.setattr(bar, "tabRect", lambda index: rects[index])
    assert tabs._hidden_tab_counts() == (1, 1)
