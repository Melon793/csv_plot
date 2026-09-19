"""P1-20：游标热路径改读 CurveInfo 缓存的 x_min/x_max。

`_update_multi_curve_cursor_label` 每帧对每条曲线做一次 `x_data.min()` +
`x_data.max()`（百万点实测 0.106 ms/次，12 曲线 × 2 游标 ≈ 2.5 ms/帧），而
`CurveInfo` 早已缓存这两个端点（`x_data` 唯一写入点 `plot_data_manager.py:587-588`
必跟着调 `update_x_range()`）。这里钉两件事：热路径不再做全量扫描、缓存语义与
原来的 `min()/max()` 等价。

判据用「这一帧真正画出来的圆点坐标」：`_update_multi_curve_cursor_label` 整体包在
`except Exception` 里，任何异常都会伪装成「什么都没画」，所以每个反向用例都配一个
正向对照；pool 里的 ScatterPlotItem 默认 `isVisible()` 为真，拿可见性当判据会假绿。
"""

from __future__ import annotations

import numpy as np
import pytest

range_calls: list[str] = []


class _CountingX(np.ndarray):
    """记录 .min()/.max() 调用次数的 ndarray 视图（全量扫描的探针）。"""

    def min(self, *args, **kwargs):
        range_calls.append("min")
        return super().min(*args, **kwargs)

    def max(self, *args, **kwargs):
        range_calls.append("max")
        return super().max(*args, **kwargs)


def _arm_cursor(plot_factory, x_data=(0.0, 10.0), n=51, y_value=3.0):
    """造一个「曲线 x∈[0,10]、cursor 位置可控」的最小现场。"""
    pw = plot_factory()
    assert pw.plot_variable("a") is True
    ci = pw.curves["a"]

    lo, hi = x_data
    # 探针挂在 x_data 本体上：热路径只要还调 min()/max() 就必被记录
    _set_curve_data(ci, np.linspace(lo, hi, n).view(_CountingX), np.full(n, y_value))

    vb = pw.view_box
    vb.enableAutoRange(x=False, y=False)
    # 只需 Y 覆盖 y_value；X 方向 anchored 模式本就不按视图范围过滤
    vb.setYRange(0.0, 10.0, padding=0)

    pw.vline.setValue(0.0)
    pw.vline.setVisible(True)
    # 走「用户固定」的 cursor：非 anchored 模式会先按视图范围过滤，测不到数据范围
    pw.plot_context.cursor_mode = "1 anchored cursor"
    cm = pw._cursor_manager
    # 这两个开关在现场里默认可能为真，都会让被测函数直接早退（假绿）
    pw._is_interacting = False
    cm.show_values_only = False
    cm.pinned_x_value = None
    return pw


def _set_curve_data(ci, x_arr, y_arr):
    ci.x_data = x_arr
    ci.y_data = y_arr
    ci.update_x_range()


def _run_with_x(pw, x: float) -> list[float]:
    """跑一帧游标刷新，返回这一帧真正画出来的圆点 x 坐标。"""
    cm = pw._cursor_manager
    cm.pinned_x_value = x
    pw._last_cursor_update_time = 0.0
    range_calls.clear()
    cm._update_multi_curve_cursor_label()

    drawn: list[float] = []
    for item in pw.multi_cursor_items:
        data = getattr(item, "data", None)
        if isinstance(data, np.ndarray) and data.dtype.names and "x" in data.dtype.names:
            drawn.extend(float(v) for v in data["x"])
    return drawn


class TestHotPathUsesCachedEndpoints:
    def test_no_full_scan_and_dot_still_drawn(self, plot_factory):
        pw = _arm_cursor(plot_factory)

        drawn = _run_with_x(pw, 5.0)

        assert drawn == [5.0], "范围内的 cursor 必须画出圆点（否则下面的空断言是空跑）"
        assert range_calls == [], f"热路径不得再全量扫描: {range_calls}"

    def test_cursor_outside_cached_data_range_draws_nothing(self, plot_factory):
        pw = _arm_cursor(plot_factory)

        assert _run_with_x(pw, 10.0) == [10.0], "端点上算范围内（与原 min()/max() 一致）"
        assert _run_with_x(pw, 10.5) == []
        assert _run_with_x(pw, -0.5) == []

    def test_probe_records_a_full_scan(self, plot_factory):
        """前提：``range_calls == []`` 是真断言，不是探针空转。

        旧代码写 ``x < x_data.min() or x > x_data.max()``，探针就记两次；探针一旦
        失效，上面那两条「不再全量扫描」的用例全部退化成恒真。
        """
        pw = _arm_cursor(plot_factory)
        x_data = pw.curves["a"].x_data

        range_calls.clear()
        assert float(x_data.min()) == pytest.approx(0.0)
        assert float(x_data.max()) == pytest.approx(10.0)
        assert range_calls == ["min", "max"]

    def test_rejection_boundary_follows_the_cache_not_the_array(self, plot_factory):
        """判别式用例：故意让缓存与数组不一致，放行/拒绝只能由其中一方决定。

        数组仍是 [0,10]，缓存改成 [100,200]：读缓存才会放行 x=150（再由
        ``searchsorted`` 夹到数组末端 10.0 画点），读数组则直接拒绝、什么都不画。
        因此本用例在 ``x_data.min()/max()`` 版本上必红。
        """
        pw = _arm_cursor(plot_factory)
        ci = pw.curves["a"]

        assert _run_with_x(pw, 150.0) == [], "现场自检：默认缓存下 150 该被拒绝"

        ci.x_min, ci.x_max = 100.0, 200.0

        assert _run_with_x(pw, 150.0) == [10.0]
        assert _run_with_x(pw, 250.0) == [], "缓存上界之外仍要拒绝"


class TestCacheTracksDataChange:
    def test_new_x_data_moves_the_rejection_boundary(self, plot_factory):
        pw = _arm_cursor(plot_factory)
        ci = pw.curves["a"]

        assert _run_with_x(pw, 5.0) == [5.0]

        _set_curve_data(ci, np.linspace(100.0, 200.0, 51).view(_CountingX), np.full(51, 3.0))

        assert _run_with_x(pw, 5.0) == []
        assert _run_with_x(pw, 150.0) == [150.0]
        assert range_calls == [], f"换新数据后热路径仍不得全量扫描: {range_calls}"

    def test_empty_data_never_reaches_the_index_lookup(self, plot_factory):
        pw = _arm_cursor(plot_factory)
        ci = pw.curves["a"]
        # 空数组不刷新缓存（_refresh_x_cache 的 len>1 / len==1 两支都不进），
        # 于是缓存仍是 (0,10)：判空必须排在范围检查之前，否则会拿 stale 端点放行
        _set_curve_data(ci, np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        assert (ci.x_min, ci.x_max) == (0.0, 10.0)

        assert _run_with_x(pw, 5.0) == []
