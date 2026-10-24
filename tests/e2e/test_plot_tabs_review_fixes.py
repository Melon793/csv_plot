"""dev/plot-multitab code review 修复的回归测试（真实 MainWindow）。

每个用例对应一个评审缺陷：
- P1-A: PageServices 缺 saved_mark_range 写转发 → restore_mark_regions 的
  回读写成影子状态，隐藏页补建用过时区间。
- P1-B: 模板应用（隐藏页 scene=False）单曲线走 plot_variable 首绘收尾，
  绕过场景分级；setBounds 钳制回写污染全局 pinned_x_values。
- P1-C: X 轴双击确认后 findChildren 跨窗广播，把隐藏页 X 范围一并改写。
- P1-D: event_handler 兄弟交互源仍读 window().plot_widgets（= 当前页），
  隐藏页拿当前页的交互标志做误判（设计 §4.2 P0 验收项）。
- P1-E: 切页跟随用过的 saved_x_range 不清除，同页重激活时把用户缩放
  拉回上次的旧范围。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSignalBlocker, Qt, QRectF
from PySide6.QtTest import QTest

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


def _two_pages(mw, qtbot, *, with_curve: bool = False):
    """建第二页并等矩阵就绪，随后切回第一页（同 test_plot_tabs_view_state）。"""
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


def _wait_activation(mw, page, qtbot):
    qtbot.waitUntil(
        lambda: mw.current_page is page and mw._pending_activation is None,
        timeout=3000,
    )


def _view_x(plot_widget) -> tuple[float, float]:
    view_min, view_max = plot_widget.view_box.viewRange()[0]
    return float(view_min), float(view_max)


# ----------------------------------------------------------------------
# P1-A: saved_mark_range 影子状态
# ----------------------------------------------------------------------
def test_mark_range_readback_writes_global_not_page_shadow(
    loaded_window, qtbot, mark_stats_singleton_cleanup
):
    """restore_mark_regions 的区间回读必须写回全局 mw.saved_mark_range。

    链条：页 2 布局变更时回读写了 PageServices 影子 → 关/开标记区把页 2 的
    item 拆掉 → 页 1 拖新区间（全局已更新，页 2 无 item 被广播跳过）→
    切页 2 补建时读到影子旧值，两页区间不一致。
    """
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages(mw, qtbot, with_curve=True)

    mw.layout_manager.toggle_mark_region(True)
    assert mw.mark_region_btn.isChecked()
    try:
        # 页 2 补建 region item
        mw.activate_page(second)
        _wait_activation(mw, second, qtbot)
        second_pw = second.master_container().plot_widget
        qtbot.waitUntil(lambda: second_pw.mark_region is not None, timeout=3000)

        # 页 2 布局变更 → restore_mark_regions 回读 master region（写影子的入口）
        mw.layout_manager.set_plots_visible(1, 1)
        shadow = vars(second.services).get("saved_mark_range", None)
        assert shadow is None, (
            f"PageServices 出现影子属性 saved_mark_range={shadow}："
            "回读没有写回全局"
        )

        # 回页 1，把区间拖到 (1.0, 2.0)
        mw.activate_page(first)
        _wait_activation(mw, first, qtbot)
        first_pw = first.master_container().plot_widget
        qtbot.waitUntil(lambda: first_pw.mark_region is not None, timeout=3000)
        first_pw.mark_region.setRegion([1.0, 2.0])
        assert tuple(mw.saved_mark_range) == pytest.approx((1.0, 2.0))

        # 关/开标记区：页 2 的 item 被全量拆除，重新变回“待补建”
        mw.layout_manager.toggle_mark_region(False)
        mw.layout_manager.toggle_mark_region(True)
        assert second.master_container().plot_widget.mark_region is None

        # 页 1 再拖到 (2.0, 3.0)：页 2 无 item，广播跳过，只有全局值更新
        first_pw.mark_region.setRegion([2.0, 3.0])
        assert tuple(mw.saved_mark_range) == pytest.approx((2.0, 3.0))

        # 切页 2：补建必须用全局 (2.0, 3.0)，而不是影子旧值
        mw.activate_page(second)
        _wait_activation(mw, second, qtbot)
        qtbot.waitUntil(
            lambda: second.master_container().plot_widget.mark_region is not None,
            timeout=3000,
        )
        region = tuple(second.master_container().plot_widget.mark_region.getRegion())
        assert region == pytest.approx((2.0, 3.0))
    finally:
        mw.layout_manager.toggle_mark_region(False)


# ----------------------------------------------------------------------
# P1-B: 模板应用（隐藏页）不得污染全局 pinned
# ----------------------------------------------------------------------
def test_apply_template_hidden_page_does_not_pollute_global_pin(
    loaded_window, qtbot
):
    """隐藏页重建曲线时，vline bounds 钳制不得回写全局 pinned_x_values。

    模板应用的单曲线 plot 走 plot_variable 首绘收尾（无 _batch_adding 保护、
    无 scene 分级）：_set_vline_bounds → setBounds 钳制 → sigPositionChanged
    → on_vline_position_changed → pinned 状态被隐藏页的钳制值改写。
    """
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    mw.cursor_sync_manager.set_cursor_mode(
        "1 anchored cursor", source_plot=first_pw, context_x=4.0
    )
    assert mw.pinned_x_values == pytest.approx([4.0])

    # 让页 2 完成一次游标恢复（vline=4.0、is_cursor_pinned=True）
    mw.activate_page(second)
    qtbot.waitUntil(lambda: second_pw.vline.value() == pytest.approx(4.0), timeout=3000)
    mw.activate_page(first)
    _wait_activation(mw, first, qtbot)

    # 构造“vline 值在曲线域外”的状态（时间修正/数据切换后同样会出现）：
    # 先放宽 bounds 再把 vline 挪到 0.5（低于数据域下界 1.0），全程静默
    with QSignalBlocker(second_pw.vline):
        second_pw.vline.setBounds([0.0, 200.0])
        second_pw.vline.setPos(0.5)
    assert second_pw.is_cursor_pinned is True

    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="page 1", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["speed"])],
            ),
            PlotPageConfig(
                title="page 2", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["rpm"])],
            ),
        ],
        active_page=0,
    )
    assert mw.plot_config_manager.apply_config(mw, config) is True

    assert mw.pinned_x_values == pytest.approx([4.0]), (
        "模板应用的隐藏页把钳制后的 vline 值写回了全局 pinned_x_values"
    )


# ----------------------------------------------------------------------
# P1-C: X 轴双击广播只限本页
# ----------------------------------------------------------------------
def test_x_axis_double_click_broadcast_stays_in_page(
    loaded_window, qtbot, monkeypatch
):
    """页 1 双击 X 轴确认范围，不得改写隐藏页 2 的 X 范围。"""
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    import src.ui.dialogs.axis as axis_mod

    monkeypatch.setattr(axis_mod.AxisDialog, "exec", lambda self: True)

    # 页 2 停在一个独特的窄范围（宽度须大于 minXRange≈2.94，否则被钳制）
    second_pw.view_box.setXRange(0.5, 4.0, padding=0)
    qtbot.waitUntil(lambda: not second_pw._is_interacting, timeout=3000)
    before = _view_x(second_pw)
    assert before == pytest.approx((0.5, 4.0), abs=1e-6)

    # 在页 1 的 X 轴标签区域双击（AxisDialog.exec 已被替身放行）
    axis_scene = first_pw.axis_x.mapToScene(
        first_pw.axis_x.boundingRect()
    ).boundingRect()
    label_rect = QRectF(
        axis_scene.left(), axis_scene.bottom() - 30, axis_scene.width(), 30
    )
    viewport_pos = first_pw.mapFromScene(label_rect.center())
    QTest.mouseDClick(
        first_pw.viewport(), Qt.MouseButton.LeftButton, pos=viewport_pos
    )
    qapp = QApplication_instance(first_pw)
    qapp.processEvents()

    assert _view_x(second_pw) == pytest.approx(before, abs=1e-6), (
        "X 轴双击确认把范围广播到了其它页"
    )


def QApplication_instance(widget):
    from PySide6.QtWidgets import QApplication

    return QApplication.instance()


# ----------------------------------------------------------------------
# P1-D: 兄弟交互源必须按“我所在页”判定
# ----------------------------------------------------------------------
def test_sibling_interaction_source_is_page_scoped(loaded_window, qtbot):
    """隐藏页的 pw 判定交互源时不得读当前页的兄弟标志（设计 §4.2 P0 项）。"""
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    first_pw._is_interacting = True
    try:
        handler = second_pw._event_handler
        assert handler._sibling_is_interaction_source() is False, (
            "页 2 的交互源判定读到了页 1 的 _is_interacting"
        )
    finally:
        first_pw._is_interacting = False


# ----------------------------------------------------------------------
# P1-E: 切页跟随的 saved_x_range 一次性消费
# ----------------------------------------------------------------------
def test_saved_x_range_consumed_once_on_activation(loaded_window, qtbot):
    """同页重激活（模板应用/关页回退）不得把视图拉回上次切页的旧范围。"""
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    second_pw = second.master_container().plot_widget

    # A→B 切页：B.saved_x_range 记录页 1 的范围并在激活时应用
    mw.activate_page(second)
    _wait_activation(mw, second, qtbot)

    # 用户在页 2 缩放到独特范围（宽度须大于 minXRange≈2.94，否则被钳制）
    second_pw.view_box.setXRange(0.5, 4.0, padding=0)
    qtbot.waitUntil(lambda: not second_pw._is_interacting, timeout=3000)
    assert _view_x(second_pw) == pytest.approx((0.5, 4.0), abs=1e-6)

    # 重激活同一页（close_page 最后一页分支 / apply_config 目标页=当前页）
    mw.activate_page(second)
    _wait_activation(mw, second, qtbot)
    qapp = QApplication_instance(second_pw)
    qapp.processEvents()

    assert _view_x(second_pw) == pytest.approx((0.5, 4.0), abs=1e-6), (
        "重激活把视图拉回了上次切页跟随的旧范围（saved_x_range 未消费清除）"
    )
