"""多绘图页 P2：游标 / 标记区 / 统计的跨页语义（真实 MainWindow）。

核心不变量（设计文档 §6.3 / §6.4）：
- 游标状态全局共享，但场景写入只发生在当前页；隐藏页置 ``_scene_dirty``，
  切过去时由 ``restore_scene()`` 按全局 pin 值补建 vline。
- 标记区间全局共享，item 每页各挂；隐藏页只记区间，激活时补建。
- 关闭标记区必须全量拆除，否则隐藏页的 region 仍连着全局槽（幽灵 region）。
- 统计窗口汇总所有页并按页分组；隐藏页没有 item 时按全局区间直接算。
"""

from __future__ import annotations

import pytest

from src.ui.mark_stats import MarkStatsWindow


@pytest.fixture()
def mark_stats_singleton_cleanup(qapp):
    """清理 MarkStatsWindow 类级单例，避免污染同进程后续用例。"""
    yield
    window = MarkStatsWindow._instance
    MarkStatsWindow._instance = None
    if window is None:
        return
    try:
        window.hide()
        window.deleteLater()
    except RuntimeError:
        # 主窗口已先销毁，子窗口的 C++ 对象随之消失
        pass
    qapp.processEvents()


def _two_pages(mw, qtbot, *, with_curve: bool = False):
    """建第二页并在激活时建好矩阵，随后切回第一页。"""
    first = mw.current_page
    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    if with_curve:
        assert second.plot_widgets[0].plot_widget.plot_variable("speed")
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
    if with_curve:
        assert first.plot_widgets[0].plot_widget.plot_variable("speed")
    return first, second


def _vline_positions(plot_widget) -> list[float]:
    positions = [plot_widget.vline.value()]
    vline2 = getattr(plot_widget, "vline2", None)
    if vline2 is not None:
        positions.append(vline2.value())
    return positions


def test_cursor_state_is_shared_and_rebuilt_on_activation(loaded_window, qtbot):
    mw = loaded_window
    first, second = _two_pages(mw, qtbot)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    hidden_before = _vline_positions(second_pw)
    mw.cursor_sync_manager.set_cursor_mode(
        "2 anchored cursor", source_plot=first_pw, context_x=15.0
    )
    pinned = list(mw.pinned_x_values)
    assert len(pinned) == 2
    assert first_pw.is_cursor_pinned is True

    # 隐藏页只记脏，场景未被跨页改写
    assert second._scene_dirty is True
    assert _vline_positions(second_pw) == pytest.approx(hidden_before)

    mw.activate_page(second)
    qtbot.waitUntil(lambda: not second._scene_dirty, timeout=3000)
    qtbot.waitUntil(lambda: second_pw.vline.isVisible(), timeout=3000)
    assert second_pw.is_cursor_pinned is True
    assert _vline_positions(second_pw) == pytest.approx(pinned)

    # 切回第一页：pin 位置不变
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
    assert first_pw.vline.isVisible() is True
    assert _vline_positions(first_pw) == pytest.approx(pinned)


def test_mark_region_is_shared_and_stats_grouped_by_page(
    loaded_window, qtbot, mark_stats_singleton_cleanup
):
    mw = loaded_window
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_mark = first.plot_widgets[0].plot_widget
    second_pw = second.plot_widgets[0].plot_widget

    mw.layout_manager.toggle_mark_region(True)
    try:
        assert mw.mark_region_btn.isChecked()
        assert first_mark.mark_region is not None
        saved = tuple(mw.saved_mark_range)
        # 隐藏页暂不建 item，但区间已记下
        assert second._scene_dirty is True
        assert second_pw.mark_region is None

        mw.layout_manager.update_mark_stats()
        tree = mw.mark_stats_window.tree
        labels = [
            tree.topLevelItem(i).text(0)
            for i in range(tree.topLevelItemCount())
        ]
        assert "图 1" in labels and "图 2" in labels
        for page_name in ("图 1", "图 2"):
            page_item = tree.topLevelItem(labels.index(page_name))
            assert page_item.childCount() >= 1
            assert "speed" in page_item.child(0).text(0)

        mw.activate_page(second)
        qtbot.waitUntil(
            lambda: second_pw.mark_region is not None, timeout=3000
        )
        assert tuple(second_pw.mark_region.getRegion()) == pytest.approx(saved)
    finally:
        mw.layout_manager.toggle_mark_region(False)


def test_closing_mark_region_removes_items_on_every_page(
    loaded_window, qtbot, mark_stats_singleton_cleanup
):
    mw = loaded_window
    first, second = _two_pages(mw, qtbot, with_curve=True)

    mw.layout_manager.toggle_mark_region(True)
    first_region = first.plot_widgets[0].plot_widget.mark_region
    assert first_region is not None

    # 让隐藏页也真正补建过 item，再回第一页关闭
    mw.activate_page(second)
    qtbot.waitUntil(
        lambda: second.plot_widgets[0].plot_widget.mark_region is not None,
        timeout=3000,
    )
    mw.activate_page(first)

    mw.layout_manager.toggle_mark_region(False)
    for page in mw.pages:
        for container in page.plot_widgets:
            assert container.plot_widget.mark_region is None

    # 旧 region 仍被测试引用，但信号已断开：拖拽不得再改写全局区间
    saved_before = mw.saved_mark_range
    first_region.setRegion([saved_before[0] - 5.0, saved_before[1] + 5.0])
    assert mw.saved_mark_range == saved_before
    assert mw.mark_region_btn.isChecked() is False
