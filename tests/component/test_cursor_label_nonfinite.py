"""P1-1（第 5 轮）：inf 样本不得使整帧游标标签全灭（offscreen）。

边缘吸附（CURSOR_EDGE_PIN）把 Y 出界样本保留入列后，inf 落到格式化分支：
enum 的 int(inf) 抛 OverflowError、时间格式 inf%86400→nan 抛 ValueError，
被整帧 except 吞掉 → 所有曲线标签消失（probe E1 实测）。修复：inf 与 NaN
同口径在入列前跳过；有限出界样本仍正常吸附。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _setup_inf(plot_factory, qapp, fmt):
    df = pd.DataFrame({
        "plain": [1.0, 2.0, 3.0, 4.0, 5.0],
        "bad": [1.0, 2.0, np.inf, 4.0, 5.0],
    })
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("plain")
    assert pw.plot_variable("bad")
    pw.curves["bad"].y_format = fmt
    pw.show_values_only = False
    ci = pw.curves["bad"]
    assert np.isinf(ci.y_data).any(), "inf 必须保留到 CurveInfo（未被上游洗掉）"
    x_inf = float(np.asarray(ci.x_data).ravel()[np.isinf(ci.y_data).ravel()][0])
    pw.view_box.setXRange(float(ci.x_data[0]), float(ci.x_data[-1]), padding=0)
    pw.view_box.setYRange(0, 10, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(x_inf)  # 精确停在 inf 样本
    pw._is_interacting = False
    pw._last_cursor_update_time = 0.0
    pw._cursor_manager._update_multi_curve_cursor_label()
    return pw


@pytest.mark.parametrize("fmt", ["enum", "s", "date", "float"])
def test_inf_sample_does_not_kill_frame_labels(plot_factory, qapp, fmt):
    pw = _setup_inf(plot_factory, qapp, fmt)
    labels = [lab for lab in pw._cursor_item_pool["labels"] if lab.isVisible()]
    assert len(labels) >= 1, f"{fmt} 格式 inf 样本不得使整帧标签全灭"
    assert not any("inf" in lab.toPlainText().lower() for lab in labels), (
        "inf 样本应被跳过，不得渲染 inf 标签"
    )


def test_finite_out_of_range_still_pins(plot_factory, qapp):
    """正面护栏：有限出界样本的边缘吸附不受 isfinite 守卫影响。"""
    y = np.zeros(100)
    y[50] = 100.0
    pw = plot_factory(pd.DataFrame({"y": y}))
    pw.resize(500, 160)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("y")
    pw.show_values_only = False
    pw.view_box.setXRange(0, 99, padding=0.05)
    pw.view_box.setYRange(0, 10, padding=0)  # y=100 出界（有限）
    qapp.processEvents()
    ci = pw.curves["y"]
    x_peak = float(ci.x_data[int(np.where(ci.y_data == 100.0)[0][0])])
    pw.vline.setVisible(True)
    pw.vline.setPos(x_peak)
    pw._is_interacting = False
    pw._last_cursor_update_time = 0.0
    pw._cursor_manager._update_multi_curve_cursor_label()
    labels = [lab for lab in pw._cursor_item_pool["labels"] if lab.isVisible()]
    assert any(lab.toPlainText().startswith("▲") for lab in labels), (
        "有限出界交点必须吸附显示且带 ▲ 前缀（CURSOR_EDGE_PIN，与 edge 测试同口径）"
    )

def test_nan_sample_still_skipped(plot_factory, qapp):
    """isnan→isfinite 的 NaN 等价性回归护栏（评审 NB2）。"""
    df = pd.DataFrame({
        "plain": [1.0, 2.0, 3.0, 4.0, 5.0],
        "bad": [1.0, 2.0, np.nan, 4.0, 5.0],
    })
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("plain")
    assert pw.plot_variable("bad")
    pw.show_values_only = False
    pw.view_box.setXRange(0, 4, padding=0)
    pw.view_box.setYRange(0, 10, padding=0)
    qapp.processEvents()
    ci = pw.curves["bad"]
    x_nan = float(np.asarray(ci.x_data).ravel()[np.isnan(ci.y_data).ravel()][0])
    pw.vline.setVisible(True)
    pw.vline.setPos(x_nan)
    pw._is_interacting = False
    pw._last_cursor_update_time = 0.0
    pw._cursor_manager._update_multi_curve_cursor_label()
    labels = [lab for lab in pw._cursor_item_pool["labels"] if lab.isVisible()]
    assert len(labels) == 1, "NaN 样本跳过，plain 标签必须在"


def test_x_actual_inf_does_not_kill_frame(plot_factory, qapp):
    """守卫 x 分支护栏：最近样本 x=inf 时整帧不得全灭（评审 NB2）。"""
    df = pd.DataFrame({
        "plain": [1.0, 2.0, 3.0, 4.0, 5.0],
        "bad": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("plain")
    assert pw.plot_variable("bad")
    pw.show_values_only = False
    # 全 inf 退化 x_data：searchsorted 落 idx0 → x_actual=inf。
    # （部分 inf 时最近样本比较 |inf-x|=inf 永不获胜，不可达）
    pw.curves["bad"].x_data = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    pw.view_box.setXRange(0, 4, padding=0)
    pw.view_box.setYRange(0, 10, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(2.0)  # plain 数据域内；bad 最近样本 x=inf
    pw._is_interacting = False
    pw._last_cursor_update_time = 0.0
    pw._cursor_manager._update_multi_curve_cursor_label()
    labels = [lab for lab in pw._cursor_item_pool["labels"] if lab.isVisible()]
    assert len(labels) >= 1, "x=inf 样本不得使整帧标签全灭"
