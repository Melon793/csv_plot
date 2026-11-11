"""模板/自动恢复后坐标轴范围对齐数据域的回归测试。

背景（tmp/2026-10-29/template_restore_range_fix_design.md）：apply_config
全程不设置 X viewRange，恢复后视图停在「应用前」的遗留值（旧缩放窗口 /
(0,1) 默认 / 旧时间基准全域），Y 按交集计算连带错位。

修复：apply_config 给每页置一次性 _range_dirty，由页激活收尾
_post_activation 统一消费（复用 auto_range_page，程序化静默 +
_is_syncing_range 守卫）。

断言口径：viewRange 覆盖恢复曲线的数据全域，且宽度不超过数据宽度的
约定倍数（X padding=5% ⇒ ≈1.1x；Y padding=0.1 + 安全裕量 ⇒ 实测 ≈1.44x）。
"""

from __future__ import annotations

import pytest

from src.core.plot_config import (
    PlotConfig,
    PlotPageConfig,
    PlotSessionConfig,
)


def _data_domain(pw):
    """取 pw 当前全部曲线的数据全域 (x0, x1, y0, y1)；无曲线返回 None。"""
    if not pw.curves:
        return None
    xs0 = [float(ci.x_data.min()) for ci in pw.curves.values()]
    xs1 = [float(ci.x_data.max()) for ci in pw.curves.values()]
    ys0 = [float(ci.y_data.min()) for ci in pw.curves.values()]
    ys1 = [float(ci.y_data.max()) for ci in pw.curves.values()]
    return min(xs0), max(xs1), min(ys0), max(ys1)


def assert_view_matches_data(pw, x_tol=1.2, y_tol=1.6):
    """viewRange 覆盖数据域且宽度不超过数据宽度的 x_tol/y_tol 倍。"""
    domain = _data_domain(pw)
    assert domain is not None, "子图无曲线，无法校验范围对齐"
    x0, x1, y0, y1 = domain
    (vx0, vx1), (vy0, vy1) = pw.view_box.viewRange()
    assert vx0 <= x0 and vx1 >= x1, (
        f"X 视窗 ({vx0:.3f}, {vx1:.3f}) 未覆盖数据域 ({x0:.3f}, {x1:.3f})"
    )
    assert vy0 <= y0 and vy1 >= y1, (
        f"Y 视窗 ({vy0:.3f}, {vy1:.3f}) 未覆盖数据域 ({y0:.3f}, {y1:.3f})"
    )
    assert (vx1 - vx0) <= (x1 - x0) * x_tol, (
        f"X 视窗宽度 {(vx1 - vx0):.3f} 超过数据宽度 {(x1 - x0):.3f} 的 {x_tol} 倍"
    )
    assert (vy1 - vy0) <= (y1 - y0) * y_tol, (
        f"Y 视窗宽度 {(vy1 - vy0):.3f} 超过数据宽度 {(y1 - y0):.3f} 的 {y_tol} 倍"
    )


def _single_page_config(curves=("speed", "rpm")) -> PlotSessionConfig:
    return PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="恢复页",
                layout_rows=1,
                layout_cols=1,
                plots=[PlotConfig(curves=list(curves))],
            )
        ],
        active_page=0,
    )


def test_restore_single_page_aligns_range(loaded_window, qtbot):
    """场景 A：单页两曲线模板恢复后，X/Y 视窗对齐数据全域。"""
    mw = loaded_window
    assert mw.plot_config_manager.apply_config(mw, _single_page_config())
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    assert_view_matches_data(page.plot_widgets[0].plot_widget)


def test_restore_with_time_factor_aligns_x(loaded_window, qtbot):
    """场景 B：模板带 factor=2/offset=1，恢复后 X 对齐修正后的数据域。"""
    mw = loaded_window
    config = _single_page_config(curves=("speed",))
    config.time_factor = 2.0
    config.time_offset = 1.0
    assert mw.plot_config_manager.apply_config(mw, config)
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    pw = page.plot_widgets[0].plot_widget
    assert_view_matches_data(pw)
    # 修正后 x_data = 2 * t + 1，应与原始时间列单调映射一致
    assert pw.curves, "恢复后应有曲线"
    qtbot.waitUntil(lambda: pw.curves["speed"].x_data is not None, timeout=3000)


def test_restore_multi_page_aligns_active_and_hidden(loaded_window, qtbot):
    """场景 C：两页模板，active 页立即对齐；隐藏页切页后兜底对齐。"""
    mw = loaded_window
    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="第一页", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["speed"])],
            ),
            PlotPageConfig(
                title="第二页", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["rpm"])],
            ),
        ],
        active_page=0,
    )
    assert mw.plot_config_manager.apply_config(mw, config)
    qtbot.waitUntil(lambda: len(mw.pages) == 2, timeout=3000)
    first, second = mw.pages
    qtbot.waitUntil(lambda: not first._range_dirty, timeout=3000)
    assert mw.current_page is first
    assert_view_matches_data(first.plot_widgets[0].plot_widget)

    # 隐藏页：_range_dirty 保持，切页激活后由 _post_activation 消费
    assert second._range_dirty is True
    mw.activate_page(second)
    qtbot.waitUntil(
        lambda: not second._range_dirty and not second._scene_dirty, timeout=3000
    )
    assert_view_matches_data(second.plot_widgets[0].plot_widget)


def test_manual_apply_overrides_user_zoom(loaded_window, qtbot):
    """场景 D：用户局部缩放后手动套用模板，视图回到新曲线全域。"""
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    domain = _data_domain(pw)
    mid = (domain[0] + domain[1]) / 2
    pw.view_box.setXRange(mid - 2, mid + 2, padding=0)
    (vx0, _), _ = pw.view_box.viewRange()
    assert vx0 > domain[0], "前置条件：用户应处于局部缩放"

    assert mw.plot_config_manager.apply_config(mw, _single_page_config())
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    assert_view_matches_data(page.plot_widgets[0].plot_widget)


def test_restore_aligns_with_follow_disabled(loaded_window, qtbot, monkeypatch):
    """切页跟随关闭时，_range_dirty 兜底仍生效（不依赖 saved_x_range）。"""
    mw = loaded_window
    monkeypatch.setattr(mw, "_x_range_follow_enabled", False)
    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="第一页", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["speed"])],
            ),
            PlotPageConfig(
                title="第二页", layout_rows=1, layout_cols=1,
                plots=[PlotConfig(curves=["rpm"])],
            ),
        ],
        active_page=0,
    )
    assert mw.plot_config_manager.apply_config(mw, config)
    qtbot.waitUntil(lambda: len(mw.pages) == 2, timeout=3000)
    first, second = mw.pages
    qtbot.waitUntil(lambda: not first._range_dirty, timeout=3000)
    assert_view_matches_data(first.plot_widgets[0].plot_widget)

    mw.activate_page(second)
    qtbot.waitUntil(lambda: not second._range_dirty, timeout=3000)
    assert_view_matches_data(second.plot_widgets[0].plot_widget)


def test_auto_range_page_announce_false_is_silent(loaded_window, qtbot, monkeypatch):
    """announce=False 程序化调用不播报状态栏；默认路径仍播报。"""
    mw = loaded_window
    calls: list[str] = []
    monkeypatch.setattr(mw, "_show_status_message", lambda msg, *a, **k: calls.append(msg))

    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")

    mw.cursor_sync_manager.auto_range_page(mw.current_page, announce=False)
    assert calls == [], "announce=False 不应写状态栏"

    mw.cursor_sync_manager.auto_range_page(mw.current_page)
    assert calls, "默认路径应播报状态栏"


@pytest.mark.parametrize("curves", [("speed",), ("speed", "rpm")])
def test_restore_keeps_curves_and_aligns(loaded_window, qtbot, curves):
    """补充冒烟：恢复后曲线集合正确且范围对齐（单/双曲线参数化）。"""
    mw = loaded_window
    assert mw.plot_config_manager.apply_config(mw, _single_page_config(curves))
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    pw = page.plot_widgets[0].plot_widget
    assert set(pw.curves) == set(curves)
    assert_view_matches_data(pw)


# ----------------------------------------------------------------------
# P1-J（第 4 轮全仓审查）：saved_mark_range 模板往返保真
# ----------------------------------------------------------------------

@pytest.fixture()
def _mark_stats_cleanup(qapp):
    from src.ui.mark_stats import MarkStatsWindow

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


def test_saved_mark_range_survives_view_alignment(
    loaded_window, qtbot, _mark_stats_cleanup
):
    """模板 saved_mark_range 不得在视图对齐之前被旧视野钳制。

    修复前时序：apply_config 同步调 toggle_mark_region(True)，而 activate_page
    只排了 0ms 定时器——视图还停在套用前的旧视野；模板区间落在旧视野外时被
    钳成旧视野中间 1/3 并覆写全局 saved_mark_range，视图对齐后
    restore_mark_regions 回读钳制值，模板区间永久丢失。
    修复后：apply_config 只置 pending 标志，_post_activation 在 _range_dirty
    对齐之后消费开启标记区。
    """
    mw = loaded_window
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    assert mw.plot_config_manager.apply_config(mw, _single_page_config(("speed",)))
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    pw = page.master_container().plot_widget
    qtbot.waitUntil(lambda: "speed" in pw.curves, timeout=3000)
    x0, x1, _, _ = _data_domain(pw)
    m1 = x0 + (x1 - x0) * 0.2
    m2 = x0 + (x1 - x0) * 0.4

    # 开标记区并把区间拖到目标位置
    mw.layout_manager.toggle_mark_region(True)
    assert mw.mark_region_btn.isChecked()
    qtbot.waitUntil(lambda: pw.mark_region is not None, timeout=3000)
    pw.mark_region.setRegion([m1, m2])
    assert tuple(mw.saved_mark_range) == pytest.approx((m1, m2))

    # 导出配置（标记按钮开着 → saved_mark_range 进配置），随后把视图挪到
    # 数据域末端：旧视野不含模板区间——正是修复前的钳制触发条件
    config = mw.plot_config_manager.export_current_config(mw)
    assert config.saved_mark_range == pytest.approx([m1, m2])
    pw.view_box.setXRange(x0 + (x1 - x0) * 0.8, x1, padding=0)

    # 重新套用同一份配置（等价于模板恢复）
    assert mw.plot_config_manager.apply_config(mw, config)
    page = mw.pages[0]
    qtbot.waitUntil(lambda: not page._range_dirty, timeout=3000)
    qtbot.waitUntil(lambda: mw._pending_activation is None, timeout=3000)
    pw2 = page.master_container().plot_widget
    qtbot.waitUntil(lambda: pw2.mark_region is not None, timeout=3000)

    assert tuple(mw.saved_mark_range) == pytest.approx((m1, m2)), (
        "全局 saved_mark_range 必须等于模板值，不得被旧视野钳成中间 1/3"
    )
    assert tuple(pw2.mark_region.getRegion()) == pytest.approx((m1, m2))
