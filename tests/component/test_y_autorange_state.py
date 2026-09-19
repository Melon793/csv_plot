"""P1-22：Y 轴 autoRange 状态收进 `set_y_autorange` / `y_autorange_preserved` 两个入口。

Ctrl+Y 的语义是「Y 跟随当前 X 可见段」，但它原先只做 `enableAutoRange(y=True)` 一步：
配对（autoVisibleOnly）靠 `plot_ui_manager._setup_plot_area` 在别处设过、重算靠
`enableAutoRange` 内部顺带触发（标记已为真时它整段跳过）。两处都不是本函数的契约。
`zoom_x` 里手写的「读标记 → 改范围 → 回写标记」也收进上下文管理器，避免每个程序化
改范围点各抄一份。

注意：`tests/component/test_y_autorange.py` 已固定 Ctrl+Y 的可见段语义，本文件只补
状态归属，不与其竞争断言。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ui.widgets.axis_manager import set_y_autorange, y_autorange_preserved

N = 10000


@pytest.fixture()
def plotted(plot_factory, qapp):
    """主体 40~60 的正弦曲线 + 窗口外极值，宿主窗口 show（auto-range 需要有效视图）"""
    y = np.sin(np.linspace(0, 100, N) / 5.0) * 10 + 50
    y[100] = 500.0
    y[9500] = -300.0
    pw = plot_factory(pd.DataFrame({"y": y}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("y")
    pw.view_box.enableAutoRange(x=False)
    pw.view_box.setXRange(4000, 6000, padding=0)
    qapp.processEvents()
    return pw


class TestSetYAutoRangeOwnsTheState:
    def test_pairing_is_reasserted_by_the_command_itself(self, plotted, qapp):
        """配对被别处改掉后，Ctrl+Y 必须自己补回来，而不是依赖建图时那次设置。"""
        vb = plotted.view_box
        vb.setAutoVisible(x=True, y=False)
        assert vb.state["autoVisibleOnly"] == [True, False]

        plotted.auto_y_in_x_range()
        qapp.processEvents()

        assert vb.state["autoVisibleOnly"] == [False, True]
        assert bool(vb.state["autoRange"][1]) is True

    def test_visible_segment_fit_is_not_polluted_by_out_of_view_extremes(
        self, plotted, qapp
    ):
        """正向对照：补配对之后 Y 仍只覆盖可见段（40~60），没被 500/-300 带跑。"""
        vb = plotted.view_box
        vb.setAutoVisible(x=True, y=False)

        plotted.auto_y_in_x_range()
        qapp.processEvents()
        qapp.processEvents()

        y_min, y_max = (float(v) for v in vb.viewRange()[1])
        assert 30 < y_min < 45, f"Y 下界异常: {y_min}"
        assert 55 < y_max < 70, f"Y 上界异常: {y_max}"

    def test_rearming_recomputes_even_when_the_flag_is_already_on(self, plotted, qapp):
        """标记为真但范围陈旧时，再按一次 Ctrl+Y 也要重算（enableAutoRange 会整段跳过）。"""
        vb = plotted.view_box
        plotted.auto_y_in_x_range()
        qapp.processEvents()

        vb.setYRange(-300.0, 500.0, padding=0)
        vb.state["autoRange"][1] = True  # 只把标记拨回真，不触发重算

        plotted.auto_y_in_x_range()
        qapp.processEvents()
        qapp.processEvents()

        y_min, y_max = (float(v) for v in vb.viewRange()[1])
        assert -300 < y_min < 45 and 55 < y_max < 70, f"未被重算: {y_min}~{y_max}"

    def test_helper_is_idempotent(self, plotted):
        """连按两次不改变状态语义（值可重复设，不叠加副作用）。"""
        vb = plotted.view_box
        set_y_autorange(vb)
        once = (vb.state["autoRange"][1], vb.state["autoVisibleOnly"][:])
        set_y_autorange(vb)
        assert (vb.state["autoRange"][1], vb.state["autoVisibleOnly"][:]) == once


class TestPreservedContext:
    def test_restores_a_mode_the_user_had_on(self, plotted):
        vb = plotted.view_box
        set_y_autorange(vb)

        with y_autorange_preserved(vb):
            vb.disableAutoRange()  # 模拟 scaleBy 那类连带关掉两轴的程序化改动
            assert bool(vb.state["autoRange"][1]) is False

        assert bool(vb.state["autoRange"][1]) is True

    def test_does_not_clobber_a_mode_turned_on_inside_the_block(self, plotted):
        """守卫只补回「进入前就开着」的模式，不做回滚。

        写成无条件复原的话，块内合法开启 Y autoRange 的调用方（例如嵌套了
        ``set_y_autorange``）会被它默默关掉。
        """
        vb = plotted.view_box
        assert bool(vb.state["autoRange"][1]) is False

        with y_autorange_preserved(vb):
            vb.enableAutoRange(axis=vb.YAxis, enable=True)

        assert bool(vb.state["autoRange"][1]) is True

    def test_leaves_a_never_enabled_mode_alone(self, plotted):
        """反向对照：进入前没开，退出后也不许被守卫打开。"""
        vb = plotted.view_box
        assert bool(vb.state["autoRange"][1]) is False

        with y_autorange_preserved(vb):
            vb.disableAutoRange()

        assert bool(vb.state["autoRange"][1]) is False

    def test_restores_even_when_the_body_raises(self, plotted):
        vb = plotted.view_box
        set_y_autorange(vb)

        with pytest.raises(RuntimeError):
            with y_autorange_preserved(vb):
                vb.disableAutoRange()
                raise RuntimeError("改范围中途炸了")

        assert bool(vb.state["autoRange"][1]) is True

    def test_zoom_x_routes_through_the_guard(self, plotted, monkeypatch):
        """zoom_x 必须走守卫：改范围把 Y autoRange 关掉时，缩放收尾要把它带回来。"""
        vb = plotted.view_box
        set_y_autorange(vb)
        killed = []

        original = plotted._axis_manager.set_xrange_with_link_handling

        def killer(*args, **kwargs):
            original(*args, **kwargs)
            vb.disableAutoRange()  # 模拟「这一步顺手废掉了 Y autoRange」
            killed.append(bool(vb.state["autoRange"][1]))

        monkeypatch.setattr(
            plotted._axis_manager, "set_xrange_with_link_handling", killer
        )
        plotted._axis_manager.zoom_x(0.8, 5000.0)

        assert killed == [False]
        assert bool(vb.state["autoRange"][1]) is True
