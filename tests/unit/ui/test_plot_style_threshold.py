"""symbol 切换阈值（可见点数判据）单元测试。

判据（v0.5.6 前已是）：估算可见点数 = x宽度 × 采样密度 < N → 显示 symbol。
v0.5.7 起为逐曲线判定（方案 A）：每条曲线各自按自身密度判定，
稀疏曲线不再被同图密集曲线"连坐"；``symbol_policy`` 非 'auto'
时以用户显式钉住值覆盖判定。

判据表（N = XRANGE_THRESHOLD_FOR_SYMBOLS = 100，"显示点" ⟺ 可见点数 < 100）
覆盖 CSV / CSV+时间修正 / CSV 稀疏 / MDF / BLF 各场景。
"""

import numpy as np
import pytest

from src.core.config import XRANGE_THRESHOLD_FOR_SYMBOLS
from src.core.data_types import (
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_OFF,
    SYMBOL_POLICY_ON,
    CurveInfo,
)
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
        # (var_name, show_symbols) 逐曲线调用记录
        self._style_calls: list[tuple[str, bool]] = []
        self._main_window = StubMainWindow()

    def window(self):
        return self._main_window

    def _apply_curve_style(self, var_name: str, show_symbols: bool):
        self._style_calls.append((var_name, show_symbols))


def _make_curve(
    density: float,
    span: float,
    visible: bool = True,
    symbol_policy: str = SYMBOL_POLICY_AUTO,
) -> CurveInfo:
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
        symbol_policy=symbol_policy,
    )


def _run_style(pw: StubPlotWidget, x_width: float) -> bool:
    """以指定可见 x 宽度触发一次样式刷新，返回末次单曲线判定的 show_symbols。"""
    manager = PlotUIManager(pw)
    rng = [[0.0, x_width], [0.0, 1.0]]
    manager.update_plot_style(pw, None, rng, None)
    assert pw._style_calls, "样式刷新被守卫拦截，未到达判据"
    return pw._style_calls[-1][1]


# ---------------------------------------------------------------------------
# 判据表（policy=auto 时每条曲线各自按可见点数判定）
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
        (1500.0, 0.1, False),  # 150 点 → 粗线
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
# 逐曲线判定：稀疏曲线不被同图密集曲线连坐（方案 A 核心）
# ---------------------------------------------------------------------------

def test_per_curve_decision_no_guilt_by_association():
    """同一子图内 1Hz 稀疏 + 200Hz 密集：各自判定，互不影响。

    窗口 3s：slow（1Hz）3 点 < 100 → symbol；fast（200Hz）600 点 → 粗线。
    旧实现取全图最大密度统一判定，slow 被 fast 连坐成粗线。
    """
    pw = StubPlotWidget({
        "fast": _make_curve(200.0, span=1.0),   # 跨度 1s / 200 点
        "slow": _make_curve(1.0, span=57.0),    # 跨度 57s / 57 点
    })
    manager = PlotUIManager(pw)
    manager.update_plot_style(pw, None, [[0.0, 3.0], [0.0, 1.0]], None)
    got = dict(pw._style_calls)
    assert got["fast"] is False
    assert got["slow"] is True


def test_hidden_dense_curve_irrelevant():
    """隐藏的密集曲线存在与否不改变其他曲线判定（per-curve 天然成立）。"""
    pw = StubPlotWidget({
        "fast": _make_curve(200.0, span=1.0, visible=False),
        "slow": _make_curve(1.0, span=57.0),
    })
    manager = PlotUIManager(pw)
    manager.update_plot_style(pw, None, [[0.0, 3.0], [0.0, 1.0]], None)
    got = dict(pw._style_calls)
    assert got["slow"] is True


def test_empty_plot_is_noop():
    """无曲线：守卫通过但循环为空，无任何样式调用。"""
    pw = StubPlotWidget({})
    PlotUIManager(pw).update_plot_style(pw, None, [[0.0, 57.0], [0.0, 1.0]], None)
    assert pw._style_calls == []


# ---------------------------------------------------------------------------
# symbol_policy 显式钉住覆盖判定
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "policy, x_width, density, expected",
    [
        # 'on'：无论可见点数多少都显示
        (SYMBOL_POLICY_ON, 57.0, 20.0, True),    # 1140 点也显示
        # 'off'：无论可见点数多少都隐藏
        (SYMBOL_POLICY_OFF, 4.0, 20.0, False),   # 80 点也隐藏
        # 'auto'：回落到阈值判定
        (SYMBOL_POLICY_AUTO, 4.0, 20.0, True),
        (SYMBOL_POLICY_AUTO, 57.0, 20.0, False),
    ],
)
def test_symbol_policy_override(policy, x_width, density, expected):
    """symbol_policy 非 auto 时以用户钉住值覆盖阈值判定。"""
    pw = StubPlotWidget({"v": _make_curve(density, span=max(x_width, 1.0), symbol_policy=policy)})
    assert _run_style(pw, x_width) is expected


def test_symbol_policy_per_curve_mixed():
    """同图混合策略：on / off / auto 三条曲线同窗各自生效。"""
    pw = StubPlotWidget({
        "pinned_on": _make_curve(20.0, span=57.0, symbol_policy=SYMBOL_POLICY_ON),
        "pinned_off": _make_curve(1.0, span=57.0, symbol_policy=SYMBOL_POLICY_OFF),
        "auto_v": _make_curve(1.0, span=57.0),
    })
    manager = PlotUIManager(pw)
    manager.update_plot_style(pw, None, [[0.0, 3.0], [0.0, 1.0]], None)
    got = dict(pw._style_calls)
    assert got["pinned_on"] is True     # 钉住显示
    assert got["pinned_off"] is False   # 钉住隐藏
    assert got["auto_v"] is True        # 3 点 < 100 自动显示


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
