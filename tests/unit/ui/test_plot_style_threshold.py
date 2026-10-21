"""symbol 切换阈值（可见点数判据）单元测试。

对应修复：update_plot_style 判据从「index 空间宽度 vs max(N, min(N/d, 2N))」
改为「估算可见点数 = x宽度 × 采样密度 < N」（方案 B1，见
tmp/2026-10-15/blf_symbol_switch_fix_plan.md）。

核心回归：BLF/MDF 的 x 轴是秒，20 Hz × 57 s 窗口 ≈ 1140 个可见点，
必须判为"粗线无 symbol"；旧实现把它误判成"57 < 100 秒 → 显示点"。

判据表（N = XRANGE_THRESHOLD_FOR_SYMBOLS = 100，"显示点" ⟺ 可见点数 < 100）
覆盖 CSV / CSV+时间修正 / CSV 稀疏 / MDF / BLF 各场景（方案 §5 十行）。
"""

import numpy as np
import pytest

from src.core.config import XRANGE_THRESHOLD_FOR_SYMBOLS
from src.core.data_types import CurveInfo
from src.ui.widgets.plot_ui_manager import PlotUIManager

N = XRANGE_THRESHOLD_FOR_SYMBOLS


class StubMainWindow:
    def __init__(self):
        self._is_loading_new_data = False


class StubPlotWidget:
    """最小 pw 替身：只实现 update_plot_style 触及的表面。"""

    def __init__(self, curves=None):
        self.curves = curves or {}
        self.plot_item = object()  # 仅作 hasattr 探测
        self._is_updating_data = False
        self._is_being_destroyed = False
        self._is_interacting = False
        self._style_calls: list[bool] = []
        self._main_window = StubMainWindow()

    def window(self):
        return self._main_window

    def _apply_plot_style(self, show_symbols: bool):
        self._style_calls.append(show_symbols)


def _make_curve(density: float, span: float, visible: bool = True) -> CurveInfo:
    """构造指定采样密度的 CurveInfo（density = 点数 / x跨度，两者须整除）。"""
    n = int(density * span)
    if n <= 1:
        x_data = np.array([0.0])  # 单点：density 恒为 0
    else:
        x_data = np.linspace(0.0, span, n)
    return CurveInfo(
        var_name="v",
        curve=None,
        x_data=x_data,
        y_data=np.zeros_like(x_data),
        visible=visible,
    )


def _run_style(pw: StubPlotWidget, x_width: float) -> bool:
    """以指定可见 x 宽度触发一次样式刷新，返回 show_symbols 判定。"""
    manager = PlotUIManager(pw)
    rng = [[0.0, x_width], [0.0, 1.0]]
    manager.update_plot_style(pw, None, rng, None)
    assert pw._style_calls, "样式刷新被守卫拦截，未到达判据"
    return pw._style_calls[-1]


# ---------------------------------------------------------------------------
# 判据表（方案 §5 / §8.1）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "x_width, density, expected",
    [
        # CSV 默认（x=行号，d≈1）
        (50.0, 1.0, True),     # 50 点 < 100 → 显示点
        (200.0, 1.0, False),   # 200 点 → 粗线
        (100.0, 1.0, False),   # 边界：严格小于
        # CSV + 时间修正 20 Hz（x=秒，可见 2.5s = 50 行）
        (2.5, 20.0, True),     # 50 点
        # CSV factor=10 稀疏（d=0.1，1500 x 单位 = 150 行）
        (1500.0, 0.1, False),  # 150 点 → 粗线（旧行为 200 行才切，过松）
        # MDF 20 Hz
        (2.0, 20.0, True),     # 40 点
        # BLF 20 Hz（本 bug 场景：57s 全览 ≈ 1140 点）
        (57.0, 20.0, False),   # 核心回归：旧实现误判为显示点
        (4.0, 20.0, True),     # 放大到 4s = 80 点 → 显示点
        # BLF 1 Hz
        (57.0, 1.0, True),     # 57 点
        # BLF 200 Hz
        (1.0, 200.0, False),   # 200 点
        # BLF 0.2 Hz 稀疏
        (400.0, 0.2, True),    # 80 点
        # 单点曲线（density=0）恒显示点，否则不可见
        (57.0, 0.0, True),
    ],
)
def test_symbol_threshold_table(x_width, density, expected):
    """判据表：可见点数 = x宽度 × 密度 < 100 → 显示 symbol。"""
    pw = StubPlotWidget({"v": _make_curve(density, span=max(x_width, 1.0))})
    assert _run_style(pw, x_width) is expected


def test_threshold_constant_value():
    """钉住阈值语义：N=100 指"可见点数"而非秒/行。"""
    assert N == 100.0


# ---------------------------------------------------------------------------
# 密度来源：per-plot + 只统计可见曲线（方案 §4.3）
# ---------------------------------------------------------------------------

def test_density_ignores_hidden_curves():
    """隐藏的高密度曲线不参与判定：200Hz 隐藏 + 1Hz 可见 → 密度取 1。"""
    pw = StubPlotWidget({
        "dense": _make_curve(200.0, span=1.0, visible=False),
        "sparse": _make_curve(1.0, span=57.0, visible=True),
    })
    manager = PlotUIManager(pw)
    assert manager._resolve_visible_density(pw) == pytest.approx(1.0)


def test_density_all_hidden_or_empty_is_zero():
    """全部隐藏 / 无曲线 → 密度 0（判据 fallback：恒显示点，无害空刷）。"""
    empty = StubPlotWidget({})
    manager = PlotUIManager(empty)
    assert manager._resolve_visible_density(empty) == 0.0

    hidden = StubPlotWidget({"v": _make_curve(20.0, span=57.0, visible=False)})
    assert manager._resolve_visible_density(hidden) == 0.0


def test_density_per_plot_not_global():
    """per-plot 口径：低采样子图不被全仓高密度连坐。

    同仓另有一个 200Hz 子图时，本子图（1Hz）仍按自身密度判定。
    """
    pw = StubPlotWidget({"v": _make_curve(1.0, span=57.0)})
    # 模拟全局聚合值存在但远高于本子图密度（旧实现会读它）
    pw._main_window._global_max_density = 200.0
    manager = PlotUIManager(pw)
    assert manager._resolve_visible_density(pw) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 守卫：交互 / 数据更新 / 重载期间不刷新（沿用既有行为）
# ---------------------------------------------------------------------------

def test_guard_is_interacting_skips_refresh():
    pw = StubPlotWidget({"v": _make_curve(20.0, span=57.0)})
    pw._is_interacting = True
    PlotUIManager(pw).update_plot_style(pw, None, [[0.0, 57.0], [0.0, 1.0]], None)
    assert pw._style_calls == []


def test_guard_is_updating_data_skips_refresh():
    pw = StubPlotWidget({"v": _make_curve(20.0, span=57.0)})
    pw._is_updating_data = True
    PlotUIManager(pw).update_plot_style(pw, None, [[0.0, 57.0], [0.0, 1.0]], None)
    assert pw._style_calls == []


def test_guard_loading_new_data_skips_refresh():
    pw = StubPlotWidget({"v": _make_curve(20.0, span=57.0)})
    pw._main_window._is_loading_new_data = True
    PlotUIManager(pw).update_plot_style(pw, None, [[0.0, 57.0], [0.0, 1.0]], None)
    assert pw._style_calls == []


def test_guard_missing_plot_item_skips_refresh():
    pw = StubPlotWidget({"v": _make_curve(20.0, span=57.0)})
    del pw.plot_item
    PlotUIManager(pw).update_plot_style(pw, None, [[0.0, 57.0], [0.0, 1.0]], None)
    assert pw._style_calls == []
