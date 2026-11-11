"""第 4 轮全仓审查批次 2 的回归测试：跨页口径收口（P1-E / P1-F / P1-G）。

- P1-E: 「清除绘图」只清当前页却清全局状态并播报「已清除全部绘图」；
  CAN 数据库重选后隐藏页旧曲线被 _apply_loader(is_reload=True) 按同名
  静默重绑到新库变量。
- P1-F: 关闭非当前页没有激活链兜底，非模态统计窗口滞留已销毁页的行。
- P1-G: 模板套用路径 _apply_page_title 无重名守卫，重名页让统计行共 key。
"""

from __future__ import annotations

import pytest

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.ui.mark_stats import MarkStatsWindow


@pytest.fixture()
def mark_stats_singleton_cleanup(qapp):
    yield
    window = MarkStatsWindow._instance
    MarkStatsWindow._instance = None
    if window is None:
        return
    try:
        window.hide()
        window.deleteLater()
    except RuntimeError:
        pass
    qapp.processEvents()


def _two_pages_with_curve(mw, qtbot):
    """建第二页并各画一条曲线，最后停回第一页（同 test_plot_tabs_review_fixes）。"""
    first = mw.current_page
    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    assert second.plot_widgets[0].plot_widget.plot_variable("speed")
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
    assert first.plot_widgets[0].plot_widget.plot_variable("speed")
    return first, second


def _wait_activation(mw, page, qtbot):
    qtbot.waitUntil(
        lambda: mw.current_page is page and mw._pending_activation is None,
        timeout=3000,
    )


# ----------------------------------------------------------------------
# P1-E: 清除绘图必须跨页
# ----------------------------------------------------------------------
def test_clear_all_plots_clears_hidden_pages(loaded_window, qtbot):
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages_with_curve(mw, qtbot)
    assert mw.current_page is first

    mw.cursor_sync_manager.clear_all_plots()

    assert first.plot_widgets[0].plot_widget.curves == {}
    hidden_pw = second.plot_widgets[0].plot_widget
    assert hidden_pw.curves == {}, (
        "隐藏页曲线必须一并清除，否则 CAN 重选后按同名静默重绑到新数据"
    )
    assert hidden_pw.invalid_curves == {}
    assert not mw.value_cache
    assert mw.saved_mark_range is None


def test_clear_all_plots_then_activate_hidden_page_stays_empty(
    loaded_window, qtbot
):
    """被清的隐藏页置脏后，激活补建必须是空页而不是旧曲线复活。"""
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages_with_curve(mw, qtbot)

    mw.cursor_sync_manager.clear_all_plots()
    mw.activate_page(second)
    _wait_activation(mw, second, qtbot)

    assert second.plot_widgets[0].plot_widget.curves == {}


# ----------------------------------------------------------------------
# P1-F: 关非当前页后统计窗口刷新
# ----------------------------------------------------------------------
def test_close_hidden_page_refreshes_mark_stats(
    loaded_window, qtbot, mark_stats_singleton_cleanup
):
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages_with_curve(mw, qtbot)

    mw.layout_manager.toggle_mark_region(True)
    assert mw.mark_region_btn.isChecked()
    # 让页 2 的 region 补建，统计窗收进两页的行
    mw.activate_page(second)
    _wait_activation(mw, second, qtbot)
    second_pw = second.master_container().plot_widget
    qtbot.waitUntil(lambda: second_pw.mark_region is not None, timeout=3000)
    mw.activate_page(first)
    _wait_activation(mw, first, qtbot)

    window = mw.mark_stats_window
    assert window is not None
    closed_title = second.title
    assert closed_title in window._pages, "前置条件：统计窗已收录页 2"

    mw.close_page(second, confirm=False)

    assert second not in mw.pages
    assert closed_title not in window._pages, (
        "关闭非当前页后统计窗口必须刷新，不得滞留已销毁页"
    )
    assert all(row.page != closed_title for row in window.model.rows())


# ----------------------------------------------------------------------
# P1-G: 模板重名页自动加后缀
# ----------------------------------------------------------------------
def test_apply_config_dedups_page_titles(loaded_window, qtbot):
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)

    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="同名", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["speed"])],
            ),
            PlotPageConfig(
                title="同名", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["speed"])],
            ),
        ],
        active_page=0,
    )
    assert mw.plot_config_manager.apply_config(mw, config)

    titles = [page.title for page in mw.pages]
    assert len(titles) == 2
    assert len(set(titles)) == 2, f"页标题必须唯一，实际: {titles}"
    assert titles[0] == "同名"
    assert titles[1] == "同名 (2)"
    # 标签文本与页标题同步
    tab_texts = [
        mw.plot_tabs.tabText(i) for i in range(mw.plot_tabs.count())
    ]
    assert "同名 (2)" in tab_texts
