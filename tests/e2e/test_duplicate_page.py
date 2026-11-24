"""绘图页复制（tab 右键「复制绘图页」）回归测试（真实 MainWindow）。

覆盖：
- 布局/曲线/样式/可见性/X 视图整体克隆；源页零变化
- 缩列后的隐藏格曲线复制（按已建网格对位，不丢不错位）
- 无效占位复制与纯占位格图例
- 页数上限双闸门（菜单置灰 + 服务层拒绝）
- 复制中途异常回滚，不留幽灵页
"""

from __future__ import annotations

import math

import pytest

from src.ui.widgets import plot_tabs as plot_tabs_module


def _wait_activation(mw, page, qtbot):
    qtbot.waitUntil(
        lambda: mw.current_page is page and mw._pending_activation is None,
        timeout=3000,
    )


def _view_x(plot_widget) -> tuple[float, float]:
    view_min, view_max = plot_widget.view_box.viewRange()[0]
    return float(view_min), float(view_max)


def test_duplicate_page_copies_curves_styles_and_x_view(loaded_window, qtbot):
    mw = loaded_window
    src = mw.pages[0]
    src_pw = src.plot_widgets[0].plot_widget

    assert src_pw.plot_variable("speed", preferred_color="#1658DB")
    assert src_pw.plot_variable("rpm", preferred_color="#D9480F")
    mcm = src_pw._multi_curve_manager
    assert mcm.apply_curve_line_preset("speed", "dash")
    assert mcm.apply_curve_display_name("speed", "车速")
    assert mcm.apply_curve_transform("rpm", 2.0, -1.0, allow_y_rescue=False)
    mcm.toggle_curve_visibility_by_name("rpm")
    assert src_pw.curves["rpm"].visible is False

    src_names = list(src_pw.curves)
    src_colors = {name: ci.color for name, ci in src_pw.curves.items()}
    src_range = _view_x(src_pw)

    new_page = mw.duplicate_page(src)
    assert new_page is not None
    # visibility_now：隐藏态在 copy_curves_from 返回时已同步落地，不依赖
    # 0ms 定时器（否则收尾的 Y/图例会先计入隐藏曲线）
    dst_immediate = new_page.plot_widgets[0].plot_widget
    dst_rpm = dst_immediate.curves["rpm"]
    assert dst_rpm.visible is False
    assert dst_rpm.curve is not None
    assert dst_rpm.curve.isVisible() is False
    _wait_activation(mw, new_page, qtbot)

    assert mw.pages.index(new_page) == 1
    assert new_page.title == "page 1 副本"
    assert len(new_page.plot_widgets) == len(src.plot_widgets)

    dst_pw = new_page.plot_widgets[0].plot_widget
    assert list(dst_pw.curves) == src_names
    for name, ci in dst_pw.curves.items():
        src_ci = src_pw.curves[name]
        assert ci.color == src_ci.color
        assert ci.line_preset == src_ci.line_preset
        assert ci.display_name == src_ci.display_name
        assert (ci.factor, ci.offset) == (src_ci.factor, src_ci.offset)

    qtbot.waitUntil(lambda: dst_pw.curves["rpm"].visible is False, timeout=3000)

    # X 视图与全局 limits
    qtbot.waitUntil(
        lambda: all(
            abs(actual - expected) < 1e-6
            for actual, expected in zip(_view_x(dst_pw), src_range)
        ),
        timeout=3000,
    )
    x_limits = dst_pw.view_box.state["limits"]["xLimits"]
    assert math.isfinite(x_limits[0]) and math.isfinite(x_limits[1])

    # 源页零变化
    assert {name: ci.color for name, ci in src_pw.curves.items()} == src_colors
    assert list(src_pw.curves) == src_names

    # 关闭复制页不影响源页
    mw.close_page(new_page, confirm=False)
    assert src in mw.pages
    assert list(src_pw.curves) == src_names


@pytest.mark.full_matrix
def test_duplicate_keeps_curves_in_hidden_cells(loaded_window, qtbot):
    """缩列只隐藏不销毁容器：复制必须按已建网格对位，否则错位/丢曲线。"""
    mw = loaded_window
    src = mw.pages[0]
    grid_rows, grid_cols = src.built_grid_size
    assert (grid_rows, grid_cols) == (5, 3)  # full_matrix：真实默认容量
    assert len(src.plot_widgets) == grid_rows * grid_cols

    src.apply_visibility(3, 2)
    qtbot.waitUntil(
        lambda: (src.rows_current, src.cols_current) == (3, 2), timeout=3000
    )

    # 只在前两行铺曲线：索引 1/2/4/5 落在后续要被隐藏的列上
    for index in range(min(6, len(src.plot_widgets))):
        var_name = "speed" if index % 2 == 0 else "rpm"
        assert src.plot_widgets[index].plot_widget.plot_variable(var_name)
    names_by_index = {
        index: list(container.plot_widget.curves)
        for index, container in enumerate(src.plot_widgets)
    }

    src.apply_visibility(3, 1)  # 第 2/3 列隐藏（容器保留）
    qtbot.waitUntil(
        lambda: (src.rows_current, src.cols_current) == (3, 1), timeout=3000
    )
    assert src.built_grid_size == (grid_rows, grid_cols)
    assert len(src.plot_widgets) == grid_rows * grid_cols

    new_page = mw.duplicate_page(src)
    assert new_page is not None
    _wait_activation(mw, new_page, qtbot)

    assert new_page.built_grid_size == (grid_rows, grid_cols)
    assert (new_page.rows_current, new_page.cols_current) == (3, 1)
    assert len(new_page.plot_widgets) == grid_rows * grid_cols
    for index in range(len(src.plot_widgets)):
        assert list(new_page.plot_widgets[index].plot_widget.curves) == (
            names_by_index[index]
        )
        row, col = divmod(index, grid_cols)
        assert new_page.plot_widgets[index].isHidden() is (col >= 1 or row >= 3)


def test_duplicate_copies_invalid_placeholder_and_refreshes_legend(
    loaded_window, qtbot
):
    mw = loaded_window
    src = mw.pages[0]
    src_pw = src.plot_widgets[0].plot_widget
    src_pw.demote_to_invalid(
        "ghost_var", "#123456", False,
        display_name="幽灵", factor=3.0, offset=2.0,
    )

    new_page = mw.duplicate_page(src)
    assert new_page is not None
    _wait_activation(mw, new_page, qtbot)

    dst_pw = new_page.plot_widgets[0].plot_widget
    ref = dst_pw.invalid_curves["ghost_var"]
    assert ref.color == "#123456"
    assert ref.visible_preference is False
    assert ref.display_name == "幽灵"
    assert (ref.factor, ref.offset) == (3.0, 2.0)
    # 纯占位格：transfer 不 finalize，copy_curves_from 必须补刷图例
    legend_text = dst_pw.legend_label.toPlainText()
    assert "幽灵" in legend_text or "ghost_var" in legend_text


def test_duplicate_gate_menu_and_service(loaded_window, qtbot, monkeypatch):
    mw = loaded_window

    monkeypatch.setattr(plot_tabs_module, "MAX_PAGES", 1)
    menu = mw.plot_tabs._build_tab_menu(0)
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    assert not by_text["复制绘图页"].isEnabled()
    menu.deleteLater()

    monkeypatch.setattr(type(mw), "MAX_PLOT_PAGES", 1)
    before = len(mw.pages)
    assert mw.duplicate_page(mw.pages[0]) is None
    assert len(mw.pages) == before


def test_duplicate_failure_rolls_back_without_ghost_tab(
    loaded_window, qtbot, monkeypatch
):
    mw = loaded_window
    src = mw.pages[0]
    pw_cls = type(src.plot_widgets[0].plot_widget)

    def _boom(self, source, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pw_cls, "copy_curves_from", _boom)
    before_pages = len(mw.pages)
    before_tabs = mw.plot_tabs.count()

    assert mw.duplicate_page(src) is None
    assert len(mw.pages) == before_pages
    assert mw.plot_tabs.count() == before_tabs
    assert mw.current_page in mw.pages
    # 回滚销毁了新页 → 回到单页，× 必须重新置灰（tabRemoved 同步）
    assert not mw.plot_tabs.close_button(0).isEnabled()


def test_duplicate_from_background_page_uses_tab_index_entry(
    loaded_window, qtbot
):
    """右键入口按 tab 索引取页：源页是后台页也要复制它，而不是当前页。"""
    mw = loaded_window
    first = mw.pages[0]
    first_pw = first.plot_widgets[0].plot_widget
    assert first_pw.plot_variable("speed")

    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    _wait_activation(mw, second, qtbot)
    assert mw.current_page is second

    mw._on_duplicate_page_requested(0)  # tab 0（后台页）
    qtbot.waitUntil(lambda: len(mw.pages) == 3, timeout=3000)
    new_page = mw.pages[1]
    assert new_page.title == "page 1 副本"
    assert mw.pages[2] is second
    _wait_activation(mw, new_page, qtbot)
    assert list(new_page.plot_widgets[0].plot_widget.curves) == ["speed"]
    assert list(first_pw.curves) == ["speed"]