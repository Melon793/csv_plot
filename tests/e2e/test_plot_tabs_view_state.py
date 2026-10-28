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
from PySide6.QtCore import Qt

from src.ui.mark_stats import MarkStatsWindow


@pytest.fixture()
def dense_loaded_window(main_window, qapp, qtbot, dialog_stubs, tmp_path):
    """200 行合成 CSV：全览约 200 个可见点，窄窗约 10 个。

    ``loaded_window`` 固定 50 行，全览也不足 ``XRANGE_THRESHOLD_FOR_SYMBOLS``
    （100），跨不过粗细线判据；本用例需要能双向切换样式才能复现切页过期。
    """
    from tests.fixtures.data_factory import make_simple_rows, write_csv

    csv = write_csv(
        tmp_path / "e2e_dense.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(200),
    )
    dialog_stubs["open_file"] = (str(csv), "CSV/TXT Files (*.csv)")
    qtbot.mouseClick(main_window.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: main_window.loader is not None, timeout=5000)
    qtbot.waitUntil(lambda: len(main_window.plot_widgets) > 0, timeout=5000)
    qtbot.waitUntil(lambda: not main_window._is_loading_new_data, timeout=5000)
    qapp.processEvents()
    assert not dialog_stubs["critical"], f"加载出现错误弹窗: {dialog_stubs['critical']}"
    return main_window


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


def _view_x(plot_widget) -> tuple[float, float]:
    view_min, view_max = plot_widget.view_box.viewRange()[0]
    return float(view_min), float(view_max)


def _has_symbols(plot_widget, var_name: str = "speed") -> bool:
    curve = plot_widget.curves[var_name].curve
    return bool(getattr(curve, "_has_symbols", False))


def test_page_switch_syncs_curve_style_with_followed_x_range(
    dense_loaded_window, qtbot
):
    """切页跟随 X 范围后，曲线粗细/symbol 必须同步刷新。

    回归：``apply_x_range`` 用 ``_is_syncing_range`` 屏蔽了 ``sigRangeChanged``，
    切回页 2 时范围已跟随页 1，但样式仍停在页 2 原来的细线+symbol。
    """
    mw = dense_loaded_window
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    def wait_activation(page) -> None:
        qtbot.waitUntil(
            lambda: mw.current_page is page and mw._pending_activation is None,
            timeout=3000,
        )

    def wait_style(plot_widget, symbols: bool) -> None:
        curve = plot_widget.curves["speed"].curve
        qtbot.waitUntil(
            lambda: bool(getattr(curve, "_has_symbols", False)) is symbols,
            timeout=3000,
        )

    # 页 2 窄窗：约 10 个可见点 → 细线 + symbol
    mw.activate_page(second)
    wait_activation(second)
    second_pw.view_box.setXRange(10.0, 20.0, padding=0)
    wait_style(second_pw, True)

    # 页 1 放宽：约 200 个可见点 → 粗线无 symbol
    mw.activate_page(first)
    wait_activation(first)
    first_pw.view_box.setXRange(1.0, 200.0, padding=0)
    wait_style(first_pw, False)

    # 切回页 2：跟随页 1 的宽范围，样式必须一起变粗
    mw.activate_page(second)
    wait_activation(second)
    assert _view_x(second_pw) == pytest.approx((1.0, 200.0), abs=1e-6)
    assert _has_symbols(second_pw) is False
    assert second_pw.curves["speed"].curve.opts["symbol"] is None

    # 反向：页 2 缩窄 → 切页 1 跟随窄范围 → 样式回到细线+symbol
    second_pw.view_box.setXRange(10.0, 20.0, padding=0)
    wait_style(second_pw, True)

    mw.activate_page(first)
    wait_activation(first)
    assert _view_x(first_pw) == pytest.approx((10.0, 20.0), abs=1e-6)
    assert _has_symbols(first_pw) is True
    assert first_pw.curves["speed"].curve.opts["symbol"] == "s"


def test_pinned_drag_updates_global_and_clean_page_syncs_on_activation(
    loaded_window, qtbot
):
    """拖动 pin 写全局；切到“干净页”也要按全局 pin 对齐 vline。

    回归：代理没有 setter 时拖动只写影子属性，隐藏页/非脏页切回时仍拿旧
    pin，叠加“切页跟随”的视图范围后游标会落到 viewRange 外。
    """
    mw = loaded_window
    # loaded_window 只等到 loader/控件就绪；UI 加载锁要到下一轮事件循环才放行。
    # 拖动固定游标走的是带锁回调，必须先等锁清空，否则 setPos 会被静默忽略。
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=3000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    mw.cursor_sync_manager.set_cursor_mode(
        "1 anchored cursor", source_plot=first_pw, context_x=10.0
    )
    assert mw.pinned_x_values == pytest.approx([10.0])

    # 先让第二页完成一次恢复，之后它就是没有任何待补场景的“干净页”
    mw.activate_page(second)
    qtbot.waitUntil(lambda: not second._scene_dirty, timeout=3000)
    assert second_pw.vline.value() == pytest.approx(10.0)
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)

    # 仿真拖动固定游标：全局状态必须立即更新，代理不得留影子属性
    first_pw.vline.setPos(35.0)
    assert mw.pinned_x_values == pytest.approx([35.0])
    services = first_pw.plot_context._services
    assert set(vars(services)) == {"mw", "page"}

    # 拖拽只同步同页兄弟子图：第二页仍是干净的，vline 暂未改写
    assert second._scene_dirty is False
    assert second_pw.vline.value() == pytest.approx(10.0)

    # 激活干净页：按全局 pin 落位，并且落在跟随后的 viewRange 内
    mw.activate_page(second)
    qtbot.waitUntil(lambda: mw.current_page is second, timeout=3000)
    qtbot.waitUntil(
        lambda: second_pw.vline.value() == pytest.approx(35.0), timeout=3000
    )
    view_min, view_max = _view_x(second_pw)
    assert view_min <= second_pw.vline.value() <= view_max
    # 单游标模式不得被 vline2 的 bounds 钳制信号补成两条
    assert len(mw.pinned_x_values) == 1
    assert second_pw.is_cursor_pinned is True
    # 程序化 setXRange 也会进入交互态；等防抖收尾后标签才会重绘
    qtbot.waitUntil(lambda: not second_pw._is_interacting, timeout=3000)
    qtbot.waitUntil(
        lambda: len(second_pw.multi_cursor_items) > 0, timeout=3000
    )


def test_time_correction_and_page_rounds_keep_single_pin(loaded_window, qtbot):
    """×10 修正后两轮切页：vline 始终等于全局 pin，不贴边界、不漂移。

    回归：恢复场景时 setBounds 的钳制回写会把旧 vline 写进全局 pin，
    下一轮再按旧值反算，factor 越大越容易落在边界上。
    """
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=3000)
    first, second = _two_pages(mw, qtbot, with_curve=True)
    first_pw = first.master_container().plot_widget
    second_pw = second.master_container().plot_widget

    mw.cursor_sync_manager.set_cursor_mode(
        "1 anchored cursor", source_plot=first_pw, context_x=40.0
    )
    assert mw.pinned_x_values == pytest.approx([40.0])

    # 先让第二页恢复一次；时间修正会再次把它置脏
    mw.activate_page(second)
    qtbot.waitUntil(lambda: not second._scene_dirty, timeout=3000)
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)

    assert mw.layout_manager.apply_time_correction(10.0, 0.0) is True
    assert mw.pinned_x_values == pytest.approx([400.0])
    assert second._scene_dirty is True

    for _ in range(2):
        mw.activate_page(second)
        qtbot.waitUntil(lambda: mw.current_page is second, timeout=3000)
        qtbot.waitUntil(
            lambda: second_pw.vline.value() == pytest.approx(400.0), timeout=3000
        )
        view_min, view_max = _view_x(second_pw)
        assert view_min <= 400.0 <= view_max
        assert len(mw.pinned_x_values) == 1
        qtbot.waitUntil(lambda: not second_pw._is_interacting, timeout=3000)
        qtbot.waitUntil(
            lambda: len(second_pw.multi_cursor_items) > 0, timeout=3000
        )

        mw.activate_page(first)
        qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
        assert first_pw.vline.value() == pytest.approx(400.0)
        assert len(mw.pinned_x_values) == 1
        view_min, view_max = _view_x(first_pw)
        assert view_min <= 400.0 <= view_max


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
        # 统计窗口重构后改为扁平表格：按 model 行断言页分组（不再访问 .tree）
        rows = mw.mark_stats_window.model.rows()
        assert {r.page for r in rows} == {"page 1", "page 2"}
        for page_name in ("page 1", "page 2"):
            page_rows = [r for r in rows if r.page == page_name]
            assert page_rows, f"{page_name} 无统计行"
            assert any(
                "speed" in (r.entry.var_name or r.entry.label) for r in page_rows
            )

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
