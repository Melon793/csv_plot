"""多绘图页标签链路（真实 MainWindow + 已加载 CSV）。

覆盖：加载后「＋」可用、新建页继承当前可见行列并在激活时建矩阵、
切页互不串页、X 范围跟随、关闭页与最后一页保留、切页前校验「＋」闸门。
"""

from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QToolButton

from src.ui.widgets.plot_tabs import MAX_PAGES


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
