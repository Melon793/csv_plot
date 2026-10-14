"""symbol 切换 component 测试（offscreen）：真实 plot widget 上的端到端行为。

对应修复：update_plot_style 判据改为「估算可见点数 = x宽度 × 采样密度 < 100」
（方案 B1，见 tmp/2026-10-15/blf_symbol_switch_fix_plan.md §4.1/§8.2）。

覆盖：
- BLF 风格数据（x=秒、20 Hz、57 s 全览 ≈1140 点）→ 粗线无 symbol（本 bug 回归）
- 缩放到 4 s 窗口（80 点）→ 细线+symbol；再缩回 → 粗线（双向切换覆盖缓存）
- 真实链路：setXRange → sigRangeChanged → 防抖 → _end_interaction → 样式刷新
- 隐藏高密度曲线后，低密度曲线不再被"连坐"（per-plot + visible 过滤）
- 重载/交互守卫不刷新

注意（tests/README.md 陷阱）：#1 必须走 plot_factory（FakePlotContext 防
QMessageBox 挂起）；#7 范围断言用 bool()，涉及真实布局需先 show()。
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.waits import pump, settle


HZ = 20.0          # BLF 样本的主采样率
SPAN_S = 57.0      # 样本 x 轴总跨度（秒）
N_POINTS = int(HZ * SPAN_S)  # ≈1140 点


def _blf_x() -> np.ndarray:
    return np.linspace(0.0, SPAN_S, N_POINTS)


def _blf_y() -> np.ndarray:
    return np.sin(np.linspace(0.0, 30.0, N_POINTS))


@pytest.fixture()
def blf_plot(plot_factory, qapp):
    """注入 BLF 风格曲线的独立 plot widget。

    必须显式传 x_values：CSV loader 的 x 恒为 arange（行号），默认 3 行
    df 构造不出 20 Hz 秒轴数据（见 base_loader.py 的 x 数据来源）。
    """
    pw = plot_factory(pd.DataFrame({"v": _blf_y()}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=_blf_x(), y_values=_blf_y())
    settle()
    return pw


def _curve(pw):
    return pw.curves["v"].curve


def _apply_style(pw, var_name: str = "v") -> bool:
    """直调样式刷新（同步、确定性），返回指定曲线当前是否带 symbol。

    setXRange 会经 sigRangeChanged 置位 _is_interacting（50ms 防抖后才
    由 _end_interaction 统一刷新），直调前必须复位交互态并停掉防抖
    定时器，否则刷新被守卫跳过、断言读到旧状态。
    """
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw.update_plot_style(pw.view_box, pw.view_box.viewRange(), None)
    curve = pw.curves[var_name].curve
    return bool(getattr(curve, "_has_symbols", False))


def test_blf_full_window_is_thick_line(blf_plot):
    """核心回归：20 Hz × 57 s 全览（≈1140 点）→ 粗线无 symbol。

    旧实现按"57 < 100 秒"误判为显示点。
    """
    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    assert not _apply_style(blf_plot)
    assert _curve(blf_plot).opts["symbol"] is None


def test_zoom_in_shows_symbols_and_back(blf_plot):
    """放大到 4 s 窗口（80 点 < 100）→ 细线+symbol；缩回全览 → 粗线。

    双向切换同时覆盖 thin/thick pen 缓存与 _has_symbols 状态翻转。
    """
    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    assert not _apply_style(blf_plot)

    blf_plot.view_box.setXRange(0.0, 4.0, padding=0)
    assert _apply_style(blf_plot)
    assert _curve(blf_plot).opts["symbol"] == "s"

    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    assert not _apply_style(blf_plot)
    assert _curve(blf_plot).opts["symbol"] is None


def test_real_chain_range_changed_triggers_refresh(blf_plot):
    """端到端：setXRange → sigRangeChanged → 防抖定时器 → _end_interaction
    → _queue_ui_refresh → 样式刷新（不直调 update_plot_style）。"""
    curve = _curve(blf_plot)
    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    _apply_style(blf_plot)  # 建立基线：全览 → 粗线
    assert curve.opts["symbol"] is None

    # 缩小窗口：经真实信号链刷新（防抖 UI_DEBOUNCE_DELAY_MS=50ms）
    blf_plot.view_box.setXRange(0.0, 4.0, padding=0)
    pump(150)
    assert curve.opts["symbol"] == "s"

    # 再放大回全览：仍经真实链刷新
    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    pump(150)
    assert curve.opts["symbol"] is None


def test_hidden_high_density_curve_not_counted(plot_factory):
    """per-plot + visible 过滤：隐藏 200 Hz 曲线后，1 Hz 曲线不再被连坐。

    窗口 3 s：两曲线都可见时按最大密度 200 判（600 点 → 粗线）；
    隐藏高密度曲线后按 1 Hz 判（3 点 → 细线+symbol）。
    """
    pw = plot_factory(pd.DataFrame({"a": np.zeros(600)}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    x_fast = np.linspace(0.0, 3.0, 600)   # 200 Hz
    y_fast = np.zeros(600)
    x_slow = np.linspace(0.0, 57.0, 57)   # 1 Hz
    y_slow = np.zeros(57)
    assert pw.add_variable_to_plot("fast", x_values=x_fast, y_values=y_fast)
    assert pw.add_variable_to_plot("slow", x_values=x_slow, y_values=y_slow)
    settle()

    pw.view_box.setXRange(0.0, 3.0, padding=0)
    assert not _apply_style(pw, "fast")  # max(200, 1) × 3s = 600 点 → 粗线

    pw.toggle_curve_visibility_by_name("fast")
    settle()
    assert not pw.curves["fast"].visible
    assert _apply_style(pw, "slow")  # 仅剩 1 Hz：3 点 < 100 → 细线+symbol


def test_guards_skip_style_refresh(blf_plot):
    """交互 / 重载期间不刷新（沿用既有守卫，样式保持进入前状态）。"""
    blf_plot.view_box.setXRange(0.0, SPAN_S, padding=0)
    assert not _apply_style(blf_plot)
    curve = _curve(blf_plot)

    # 交互中：样式刷新被跳过
    blf_plot._is_interacting = True
    blf_plot.view_box.setXRange(0.0, 4.0, padding=0)
    blf_plot.update_plot_style(blf_plot.view_box, blf_plot.view_box.viewRange(), None)
    assert curve.opts["symbol"] is None
    blf_plot._is_interacting = False

    # 重载中：样式刷新被跳过
    host = blf_plot.window()
    host._is_loading_new_data = True
    try:
        blf_plot.view_box.setXRange(0.0, 4.0, padding=0)
        blf_plot.update_plot_style(blf_plot.view_box, blf_plot.view_box.viewRange(), None)
        assert curve.opts["symbol"] is None
    finally:
        host._is_loading_new_data = False
