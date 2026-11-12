"""曲线绘制方式（draw_mode）组件测试（offscreen）：真实 plot widget 端到端。

方案 A：per-curve 曲线绘制方式 + symbol 粒度控制。覆盖：
- apply_curve_draw_mode 切换直线/step-post/step-pre → curve.opts['stepMode']
  在 None/'left'/'right' 间正确翻转（含回到 auto 的往返）
- 非法 draw_mode / 未知变量 → False 且不污染 ci.draw_mode
- 时间修正重算 x 后 stepMode 透传保持（阶梯语义不因 factor 变化丢失）
- symbol 与 stepMode 正交：阶梯模式下 symbol 仍画在原始数据点上
- NaN 回归：ragged CSV 尾部补 NaN 的数据在 step 模式下渲染不崩溃、
  非有限值仍走 connect='finite' 断线语义（无横线伪影路径）

注意（tests/README.md 陷阱）：直调样式刷新前需复位 _is_interacting。
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.waits import settle

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    DRAW_MODE_STEP_PRE,
    SYMBOL_POLICY_ON,
    step_mode_for,
)


def _plain_y(n: int) -> np.ndarray:
    return np.sin(np.linspace(0.0, 6.0, n))


@pytest.fixture()
def step_plot(plot_factory, qapp):
    """注入 1 Hz 秒轴数据的独立 plot widget（30 点）。"""
    n = 30
    pw = plot_factory(pd.DataFrame({"v": _plain_y(n)}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=np.arange(n, dtype=float), y_values=_plain_y(n))
    settle()
    return pw


# ---------------------------------------------------------------------------
# step_mode_for 映射（单元层语义钉住）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "draw_mode, expected",
    [
        (DRAW_MODE_AUTO, None),
        (DRAW_MODE_STEP_POST, "left"),
        (DRAW_MODE_STEP_PRE, "right"),
        ("unknown-garbage", None),  # 防御：未知值按直线处理
    ],
)
def test_step_mode_for_mapping(draw_mode, expected):
    assert step_mode_for(draw_mode) == expected


# ---------------------------------------------------------------------------
# apply_curve_draw_mode：opts 翻转与往返
# ---------------------------------------------------------------------------

def test_draw_mode_switch_flips_step_mode(step_plot):
    """切换 step-post/step-pre → stepMode 'left'/'right'；回 auto → None。"""
    pw = step_plot
    mcm = pw._multi_curve_manager
    curve = pw.curves["v"].curve

    assert mcm.apply_curve_draw_mode("v", DRAW_MODE_STEP_POST)
    assert curve.opts["stepMode"] == "left"

    assert mcm.apply_curve_draw_mode("v", DRAW_MODE_STEP_PRE)
    assert curve.opts["stepMode"] == "right"

    assert mcm.apply_curve_draw_mode("v", DRAW_MODE_AUTO)
    assert curve.opts["stepMode"] is None
    assert pw.curves["v"].draw_mode == DRAW_MODE_AUTO


def test_draw_mode_invalid_inputs(step_plot):
    """非法 draw_mode / 未知变量 → False，字段不被污染。"""
    mcm = step_plot._multi_curve_manager
    assert mcm.apply_curve_draw_mode("v", "diagonal") is False
    assert step_plot.curves["v"].draw_mode == DRAW_MODE_AUTO
    assert mcm.apply_curve_draw_mode("ghost", DRAW_MODE_STEP_POST) is False


def test_draw_mode_keeps_pen_and_symbol(step_plot):
    """切换绘制方式不影响既有 pen / symbol 状态（正交性）。"""
    pw = step_plot
    curve = pw.curves["v"].curve
    # 建立带 symbol 的基线（30 点 < 100 → 显示点）
    pw.view_box.setXRange(0.0, 30.0, padding=0)
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw.update_plot_style(pw.view_box, pw.view_box.viewRange(), None)
    assert getattr(curve, "_has_symbols", False)
    pen_before = curve.opts["pen"]

    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_STEP_POST)
    assert getattr(curve, "_has_symbols", False)          # symbol 保留
    assert curve.opts["pen"] is pen_before                 # pen 保留
    assert curve.opts["symbol"] == "s"                     # symbol 仍在原始数据点上


# ---------------------------------------------------------------------------
# 时间修正：setData 透传 stepMode
# ---------------------------------------------------------------------------

def test_time_correction_preserves_step_mode(step_plot):
    """时间修正重算 x 后，阶梯语义保持（stepMode 不丢）。"""
    pw = step_plot
    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_STEP_POST)
    curve = pw.curves["v"].curve
    assert curve.opts["stepMode"] == "left"

    pw.update_time_correction(2.0, 1.0)
    assert curve.opts["stepMode"] == "left"
    # x 确实被重算了（factor 2 → 跨度翻倍）
    assert pw.curves["v"].x_data[1] == pytest.approx(3.0)  # 1 + 2*1
    # y 不受影响
    assert pw.curves["v"].y_data[1] == pytest.approx(_plain_y(30)[1])


# ---------------------------------------------------------------------------
# NaN 回归：step 模式下含 NaN 数据渲染不崩溃
# ---------------------------------------------------------------------------

def test_step_mode_with_nan_renders(plot_factory, qapp):
    """ragged CSV 尾部 NaN：step 模式下渲染（grab 栅格化）不崩溃且产出位图。

    阶梯路径 + 非有限值由 pyqtgraph connect='auto' 自动切 'finite' 断线，
    避免 NaN 坐标进入 Qt 光栅化产生横线伪影（与直线模式同语义）。
    """
    n = 30
    y = _plain_y(n).copy()
    y[-3:] = np.nan  # 尾部 3 个 NaN
    pw = plot_factory(pd.DataFrame({"v": y}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=np.arange(n, dtype=float), y_values=y)
    settle()

    assert pw._multi_curve_manager.apply_curve_draw_mode("v", DRAW_MODE_STEP_POST)
    ci = pw.curves["v"]
    assert ci.curve.opts["stepMode"] == "left"

    # 真实渲染一帧：NaN 坐标若进入光栅化会在这里崩溃/产出垃圾
    pm = pw.grab()
    assert not pm.isNull()
    # 数据集确实含非有限值（触发 dataRect 的有限性检查后自证）
    assert ci.curve._dataset.dataRect() is not None
    assert ci.curve._dataset.containsNonfinite is True


def test_symbol_policy_pins_symbols_on_dense_curve(plot_factory):
    """symbol_policy='on'：密集曲线（1140 点）也被钉住显示 symbol。"""
    n = 1140
    x = np.linspace(0.0, 57.0, n)   # 20 Hz × 57 s
    y = np.sin(np.linspace(0.0, 30.0, n))
    pw = plot_factory(pd.DataFrame({"v": y}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=x, y_values=y)
    settle()

    pw.view_box.setXRange(0.0, 57.0, padding=0)
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw.update_plot_style(pw.view_box, pw.view_box.viewRange(), None)
    assert not getattr(pw.curves["v"].curve, "_has_symbols", False)  # 基线：粗线无点

    assert pw._multi_curve_manager.apply_curve_symbol_policy("v", SYMBOL_POLICY_ON)
    assert pw.curves["v"].symbol_policy == SYMBOL_POLICY_ON
    assert getattr(pw.curves["v"].curve, "_has_symbols", False)     # 钉住后立即显示
