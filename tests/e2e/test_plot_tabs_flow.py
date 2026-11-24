"""多绘图页标签链路（真实 MainWindow + 已加载 CSV）。

覆盖：加载后「＋」可用、新建页继承当前可见行列并在激活时建矩阵、
切页互不串页、X 范围跟随、关闭页与最后一页保留、切页前校验「＋」闸门。
"""

from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QMessageBox, QToolButton

from src.ui.widgets.plot_tabs import LAST_PAGE_CLOSE_HINT, MAX_PAGES


def _plus_button(mw) -> QToolButton:
    button = mw.plot_tabs.new_page_button
    assert isinstance(button, QToolButton)
    return button


def _wait_new_page_matrix(qtbot, page):
    qtbot.waitUntil(lambda: len(page.plot_widgets) > 0, timeout=3000)


def _assert_range(qtbot, view_box, expected):
    qtbot.waitUntil(
        lambda: abs(view_box.viewRange()[0][0] - expected[0]) < 1e-6
        and abs(view_box.viewRange()[0][1] - expected[1]) < 1e-6,
        timeout=3000,
    )


def test_plus_enabled_after_load(loaded_window):
    mw = loaded_window
    assert mw.loader is not None
    assert _plus_button(mw).isEnabled()


def test_new_page_gets_global_x_limits(loaded_window, qtbot):
    """新建页在加载后创建，激活时必须补齐全局 X limits，不能无限缩放。"""
    mw = loaded_window
    first_master = mw.pages[0].master_viewbox()
    assert first_master is not None
    expected = tuple(first_master.state["limits"]["xLimits"])
    assert expected != (-1e307, 1e307)

    page = mw.new_page()
    _wait_new_page_matrix(qtbot, page)
    qtbot.waitUntil(
        lambda: tuple(
            page.master_viewbox().state["limits"]["xLimits"]
        ) == expected,
        timeout=3000,
    )

    pw = page.plot_widgets[0].plot_widget
    pw.view_box.setXRange(-1e6, 1e6, padding=0)
    x_min, x_max = pw.view_box.viewRange()[0]
    assert x_min >= expected[0] - 1e-6
    assert x_max <= expected[1] + 1e-6


def test_new_page_inherits_visible_grid_and_builds_on_activation(loaded_window, qtbot):
    mw = loaded_window
    old = mw.current_page
    assert len(old.plot_widgets) > 0

    page = mw.new_page()
    assert page is not None
    assert mw.pages == [old, page]
    assert mw.plot_tabs.count() == 2
    assert mw.current_page is page
    # 新建时先 hide 再入 tab，激活后必须真正可见
    assert page.widget.isVisible()
    # 新建瞬间不建矩阵（延迟到激活后），随后按继承的可见行列建
    _wait_new_page_matrix(qtbot, page)
    assert (page.rows_current, page.cols_current) == (
        old.rows_current, old.cols_current,
    )
    assert len(page.plot_widgets) == old.rows_current * old.cols_current
    assert page.plot_widgets[0] is not old.plot_widgets[0]
    assert mw.plot_widgets is page.plot_widgets


def test_switch_pages_keeps_widget_sets_isolated(loaded_window, qtbot):
    mw = loaded_window
    first_page = mw.current_page
    first_pw = first_page.plot_widgets[0].plot_widget

    second_page = mw.new_page()
    _wait_new_page_matrix(qtbot, second_page)
    second_pw = second_page.plot_widgets[0].plot_widget

    mw.activate_page(first_page)
    assert mw.current_page is first_page
    assert mw.plot_widgets is first_page.plot_widgets
    assert mw.plot_widgets[0].plot_widget is first_pw

    mw.activate_page(second_page)
    assert mw.current_page is second_page
    assert mw.plot_widgets is second_page.plot_widgets
    assert mw.plot_widgets[0].plot_widget is second_pw


def test_x_range_follows_the_previous_page(loaded_window, qtbot):
    mw = loaded_window
    first_page = mw.current_page
    first_page.master_viewbox().enableAutoRange(x=False)
    first_page.master_viewbox().setXRange(10.0, 20.0, padding=0)
    _assert_range(qtbot, first_page.master_viewbox(), (10.0, 20.0))

    second_page = mw.new_page()
    _wait_new_page_matrix(qtbot, second_page)
    _assert_range(qtbot, second_page.master_viewbox(), (10.0, 20.0))

    second_page.master_viewbox().enableAutoRange(x=False)
    second_page.master_viewbox().setXRange(30.0, 40.0, padding=0)
    _assert_range(qtbot, second_page.master_viewbox(), (30.0, 40.0))

    mw.activate_page(first_page)
    _assert_range(qtbot, first_page.master_viewbox(), (30.0, 40.0))


def test_close_current_page_switches_to_neighbor(loaded_window, qtbot):
    mw = loaded_window
    first_page = mw.current_page
    second_page = mw.new_page()
    _wait_new_page_matrix(qtbot, second_page)
    assert mw.current_page is second_page

    mw.close_page(second_page)
    assert mw.pages == [first_page]
    assert mw.plot_tabs.count() == 1
    assert mw.current_page is first_page
    assert mw.plot_tabs.currentWidget() is first_page.widget


def test_closing_last_page_clears_instead_of_removing(loaded_window):
    mw = loaded_window
    page = mw.current_page
    mw.close_page(page)
    assert mw.pages == [page]
    assert mw.plot_tabs.count() == 1
    # 容器保留，只是曲线被清空
    assert page.plot_widgets
    assert all(not c.plot_widget.curves for c in page.plot_widgets)


def test_single_page_middle_click_blocked_and_hinted(loaded_window, qtbot):
    """单页中键不关页也不清空：只播报原因（× 已置灰，中键是唯一活入口）。"""
    from PySide6.QtCore import Qt

    mw = loaded_window
    page = mw.current_page
    pw = page.master_container().plot_widget
    assert pw.plot_variable("speed")
    assert not mw.plot_tabs.close_button(0).isEnabled()

    rect = mw.plot_tabs.tabBar().tabRect(0)
    qtbot.mouseClick(
        mw.plot_tabs.tabBar(), Qt.MouseButton.MiddleButton, pos=rect.center()
    )

    assert mw._message_text == LAST_PAGE_CLOSE_HINT
    assert mw.pages == [page]
    assert mw.plot_tabs.count() == 1
    assert page.master_container().plot_widget.curves   # 未被清空
    # 页签 hover 是禁用 × 之外唯一在原地可触达的说明
    assert LAST_PAGE_CLOSE_HINT in mw._tab_tooltip_text(0)


def test_close_button_reenabled_after_second_page_closed(loaded_window, qtbot):
    """× 的置灰随页数同步：加页解禁，关回单页再次置灰。"""
    mw = loaded_window
    assert not mw.plot_tabs.close_button(0).isEnabled()

    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    assert mw.plot_tabs.close_button(0).isEnabled()
    assert mw.plot_tabs.close_button(1).isEnabled()
    assert LAST_PAGE_CLOSE_HINT not in mw._tab_tooltip_text(0)

    mw.close_page(second, confirm=False)
    assert mw.plot_tabs.count() == 1
    assert not mw.plot_tabs.close_button(0).isEnabled()
    assert LAST_PAGE_CLOSE_HINT in mw._tab_tooltip_text(0)


def test_single_page_clear_button_still_works(loaded_window, qtbot):
    """替代路径可用：× 置灰后，单页重来仍能靠顶部「清除绘图」完成。"""
    from PySide6.QtCore import Qt

    mw = loaded_window
    page = mw.current_page
    assert page.master_container().plot_widget.plot_variable("speed")
    assert not mw.plot_tabs.close_button(0).isEnabled()

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert mw.pages == [page]
    assert all(not c.plot_widget.curves for c in page.plot_widgets)
    assert "已清除全部绘图" in mw._message_text


def test_close_page_with_only_placeholder_still_confirms(
    loaded_window, qtbot, dialog_stubs
):
    """B2 §7：只有失效占位（curves 为空）的页关闭前也要确认，不得静默销毁资产。"""
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    pw = second.plot_widgets[0].plot_widget
    # 只降级为占位：curves 保持空，invalid_curves 有内容
    pw.demote_to_invalid("speed", "gray")
    assert not pw.curves
    assert "speed" in pw.invalid_curves

    # 替身返回 No → 页必须保留，且确认弹窗被触发
    mw.close_page(second)
    assert len(dialog_stubs["question"]) == 1
    assert second in mw.pages


def test_close_page_marks_destroyed_and_filters_global_iteration(
    loaded_window, qtbot, dialog_stubs
):
    """R5：关页先确认；确认后旧页面所有子图置销毁位，跨页遍历不再返回。"""
    mw = loaded_window
    first = mw.current_page
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    assert second.plot_widgets[0].plot_widget.plot_variable("speed")
    containers = list(second.plot_widgets)
    widgets = [container.plot_widget for container in containers]

    # 默认确认口径：替身返回 No，页必须保留
    mw.close_page(second)
    assert second in mw.pages
    assert len(dialog_stubs["question"]) == 1

    mw.close_page(second, confirm=False)
    assert second not in mw.pages
    assert mw.current_page is first
    assert all(widget._is_being_destroyed for widget in widgets)
    remaining = {container.plot_widget for container in mw.iter_all_plot_widgets()}
    assert remaining.isdisjoint(widgets)


def test_close_invalidates_pending_epoch(loaded_window, qtbot):
    mw = loaded_window
    page = mw.new_page()
    _wait_new_page_matrix(qtbot, page)
    epoch_before = page.epoch
    page.schedule_xlink_sync(10)
    mw.close_page(page)
    assert page.epoch > epoch_before
    qtbot.wait(30)   # 给迟到回调落地机会；epoch 已作废，不应报错


def test_new_page_is_gated_before_data(main_window):
    mw = main_window
    assert not _plus_button(mw).isEnabled()
    assert mw.new_page() is None
    assert len(mw.pages) == 1
    assert "导入数据" in mw._message_text


def test_new_page_title_skips_used_numbers(loaded_window, qtbot):
    mw = loaded_window
    p2 = mw.new_page()
    _wait_new_page_matrix(qtbot, p2)
    p3 = mw.new_page()
    _wait_new_page_matrix(qtbot, p3)
    assert (mw.pages[0].title, p2.title, p3.title) == (
        "page 1", "page 2", "page 3",
    )

    mw.close_page(p2)
    p4 = mw.new_page()
    _wait_new_page_matrix(qtbot, p4)
    assert p4.title == "page 2"


def test_rename_updates_page_title(loaded_window, monkeypatch):
    mw = loaded_window
    page = mw.pages[0]
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: ("分析 A", True)),
    )
    mw._on_tab_rename_requested(0)
    assert page.title == "分析 A"
    assert mw.plot_tabs.tabText(0) == "分析 A"


def test_page_limit_is_enforced(loaded_window, monkeypatch):
    mw = loaded_window
    monkeypatch.setattr(mw, "MAX_PLOT_PAGES", 2)
    assert mw.new_page() is not None
    assert len(mw.pages) == 2
    assert mw.new_page() is None
    assert "最多支持 2 个绘图页" in mw._message_text
    assert MAX_PAGES >= 2


def test_rename_rejects_duplicate_title(loaded_window, qtbot, monkeypatch):
    """重命名不得产生重名页。

    回归点：页标题是统计窗口分组键与 ``StatRow.key`` 的组成部分，两页重名会让
    它们的统计行共用同一个 key，``replace_entries`` 原位刷新时把两条数据写进
    同一行——表格静默显示错误/过期数字，页筛选也会同时放行两页。
    """
    mw = loaded_window
    p2 = mw.new_page()
    _wait_new_page_matrix(qtbot, p2)
    first_title = mw.pages[0].title
    assert p2.title != first_title

    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(
            lambda parent, title, text, *args, **kwargs: warnings.append(text)
        ),
    )
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: (first_title, True)),
    )

    before = p2.title
    mw._on_tab_rename_requested(mw.plot_tabs.indexOf(p2.widget))
    assert p2.title == before, "重名标题不得生效"
    assert mw.plot_tabs.tabText(mw.plot_tabs.indexOf(p2.widget)) == before
    assert warnings and first_title in warnings[0]

    # 换成不重名的标题仍应正常生效（查重不得误伤正常路径）
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: ("分析 B", True)),
    )
    mw._on_tab_rename_requested(mw.plot_tabs.indexOf(p2.widget))
    assert p2.title == "分析 B"


# ----------------------------------------------------------------------
# T2：批量关页（右键菜单「关闭其他/关闭右侧」）——一次汇总确认
# ----------------------------------------------------------------------
def test_close_other_pages_batch_confirm_yes(loaded_window, qtbot, monkeypatch):
    mw = loaded_window
    first = mw.pages[0]
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    third = mw.new_page()
    _wait_new_page_matrix(qtbot, third)
    mw.activate_page(first)

    calls = []

    def yes(parent, title, text, *args, **kwargs):
        calls.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(yes))
    mw._close_other_pages(0)
    assert len(calls) == 1
    assert "将关闭 2 个绘图页" in calls[0]
    assert mw.pages == [first]
    assert mw.plot_tabs.count() == 1


def test_close_right_pages_cancel_keeps_pages(loaded_window, qtbot, dialog_stubs):
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    third = mw.new_page()
    _wait_new_page_matrix(qtbot, third)

    mw._close_right_pages(0)                 # 替身默认 No → 全保留
    assert len(mw.pages) == 3
    assert any("将关闭 2 个绘图页" in text for _, text in dialog_stubs["question"])


def test_batch_close_reports_curve_pages(loaded_window, qtbot, dialog_stubs):
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    second.plot_widgets[0].plot_widget.demote_to_invalid("speed", "gray")
    third = mw.new_page()
    _wait_new_page_matrix(qtbot, third)

    mw._close_right_pages(0)
    assert any(
        "将关闭 2 个绘图页（其中 1 页含曲线）" in text
        for _, text in dialog_stubs["question"]
    )
    assert len(mw.pages) == 3                # No → 未动


# ----------------------------------------------------------------------
# T3：Ctrl+PgUp/Dn 顺序切页 + Ctrl+Tab MRU 对比翻切
# ----------------------------------------------------------------------
def test_neighbor_page_switch_clamps_at_bounds(loaded_window, qtbot):
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    third = mw.new_page()
    _wait_new_page_matrix(qtbot, third)
    mw.activate_page(mw.pages[0])

    mw._activate_neighbor_page(1)
    assert mw.current_page is mw.pages[1]
    mw._activate_neighbor_page(1)
    assert mw.current_page is mw.pages[2]
    mw._activate_neighbor_page(1)          # 末页边界：停住不循环
    assert mw.current_page is mw.pages[2]
    mw._activate_neighbor_page(-1)
    assert mw.current_page is mw.pages[1]
    mw._activate_neighbor_page(-1)
    mw._activate_neighbor_page(-1)         # 首页边界
    assert mw.current_page is mw.pages[0]


def test_ctrl_tab_mru_toggles_between_two_pages(loaded_window, qtbot):
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    first = mw.pages[0]
    mw.activate_page(first)                # MRU：second → first

    mw._toggle_mru_page()
    assert mw.current_page is second
    mw._toggle_mru_page()
    assert mw.current_page is first


def test_mru_cleared_when_prev_page_closed(loaded_window, qtbot):
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    first = mw.pages[0]
    mw.activate_page(first)
    mw.close_page(second, confirm=False)
    mw._toggle_mru_page()                  # 上一激活页已销毁：不动
    assert mw.current_page is first


def test_page_switch_shortcuts_registered(loaded_window):
    import sys

    from PySide6.QtGui import QKeySequence

    mw = loaded_window
    assert mw._page_prev_shortcut.key().toString() == "Ctrl+PgUp"
    assert mw._page_next_shortcut.key().toString() == "Ctrl+PgDown"
    # native 口径：mac 的 Ctrl+Tab 会被映射成 ⌘⇥（系统截走），必须绑物理 ⌃⇥
    native = QKeySequence.SequenceFormat.NativeText
    if sys.platform == "darwin":
        assert mw._page_mru_shortcut.key().toString(native) == "⌃⇥"
        assert mw._page_mru_back_shortcut.key().toString(native) == "⌃⇧⇥"
    else:
        assert mw._page_mru_shortcut.key().toString(native) == "Ctrl+Tab"
        assert mw._page_mru_back_shortcut.key().toString(native) == "Ctrl+Shift+Tab"


def test_mru_real_key_toggles(loaded_window, qtbot):
    import sys

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    first = mw.pages[0]
    mw.activate_page(first)
    modifier = (
        Qt.KeyboardModifier.MetaModifier
        if sys.platform == "darwin"
        else Qt.KeyboardModifier.ControlModifier
    )
    QTest.keyClick(mw, Qt.Key.Key_Tab, modifier)
    qtbot.waitUntil(lambda: mw.current_page is second, timeout=2000)
    QTest.keyClick(mw, Qt.Key.Key_Tab, modifier)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=2000)


def test_close_other_pages_from_non_current_index(loaded_window, qtbot, monkeypatch):
    """批量中途关到当前页：走既有激活链，索引页保留。"""
    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    third = mw.new_page()
    _wait_new_page_matrix(qtbot, third)
    mw.activate_page(third)                # 当前页在批量目标里
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    mw._close_other_pages(0)
    assert mw.pages == [mw.pages[0]] or len(mw.pages) == 1
    assert mw.current_page is mw.pages[0]
    assert mw.plot_tabs.count() == 1


def test_ctrl_pgdown_key_switches_page(loaded_window, qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    mw.activate_page(mw.pages[0])
    QTest.keyClick(mw, Qt.Key.Key_PageDown, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: mw.current_page is mw.pages[1], timeout=2000)


# ----------------------------------------------------------------------
# T6：页目录 ▾ —— 全部页一屏全见，点行切页
# ----------------------------------------------------------------------
def test_page_directory_selects_page(loaded_window, qtbot):
    from PySide6.QtCore import Qt

    mw = loaded_window
    second = mw.new_page()
    _wait_new_page_matrix(qtbot, second)
    mw.activate_page(mw.pages[0])

    qtbot.mouseClick(mw.plot_tabs._dir_btn, Qt.MouseButton.LeftButton)
    popup = mw.plot_tabs._dir_popup
    qtbot.waitUntil(popup.isVisible, timeout=2000)
    rows = [
        popup._rows_layout.itemAt(i).widget()
        for i in range(popup._rows_layout.count())
    ]
    assert len(rows) == 2
    qtbot.mouseClick(rows[1], Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: mw.current_page is second, timeout=2000)
    assert not popup.isVisible()
