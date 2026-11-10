"""拖入新变量时 Y 视野保持 component 测试（offscreen）。

对应修复（multi_curve_manager.add_variable_to_plot has_other_curves 分支）：
1. 视野污染：全量 _set_safe_y_range(set_limits=True) 先覆盖 Y viewRange，
   之后读 viewRange 当"用户当前视野"做并集 → 并集恒等于全量数据范围，
   v5.x 问题 B 的窗口交集保护失效（视野跳到全量量级）；
2. padding 二次膨胀：把已含 padding 的 viewRange 传回 _set_safe_y_range
   再次 setYRange(padding=0.1) → 视野超过全量数据范围；
3. 空交集回退：新曲线 X 范围与当前窗口无交集时，_get_y_range_in_x_window
   回退到新曲线全量范围并入视野（窗口内根本看不到这些数据）。

修复后语义：
- 新曲线窗口内范围 ⊆ 用户视野 → 精确还原用户视野（padding=0）
- 超出 → 与用户视野取并集后 padding 一次
- 窗口内无数据 → 保持用户视野
- Y limits 仍为全部可见曲线全量数据 ±50%（最大可调范围不受影响）

场景构造（合成数据，非真实采集文件）：全量范围悬殊但当前 X 窗口内
数值接近的两条曲线——拖入第二条后视野应保持在用户设定的窄窗口。
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.waits import settle


# ---------- 合成数据 ----------

X_MIN, X_MAX = 0.0, 200.0
N = 201


def _x() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, N)


def _y_a() -> np.ndarray:
    """已有曲线：全量 35~36.2（缓变），窗口内 ~35.5。"""
    x = _x()
    return 35.0 + 1.2 * (x - X_MIN) / (X_MAX - X_MIN)


def _y_b() -> np.ndarray:
    """新曲线：全量 0~130（两端极端值在窗口外），窗口内 36.2~36.3。"""
    x = _x()
    y = 36.0 + 0.5 * (x - 50.0) / 100.0
    y = np.where(x < 50.0, 0.0, y)
    y = np.where(x > 150.0, 130.0, y)
    return y


VIEW_X = (90.0, 110.0)     # 用户当前 X 窗口
# 用户当前 Y 视野（含两曲线窗口内范围；须落在曲线 a 的 Y limits
# [35, 36.2]±50% = [34.4, 36.8] 内，否则 setYRange 会被 pyqtgraph 钳制）
USER_Y = (35.0, 36.5)


@pytest.fixture()
def plot_with_a(plot_factory, qapp):
    """已有曲线 a + 用户设定的 X/Y 窗口。"""
    pw = plot_factory(pd.DataFrame({"a": _y_a(), "b": _y_b()}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("a", x_values=_x(), y_values=_y_a())
    pw.view_box.setXRange(*VIEW_X, padding=0)
    pw.view_box.setYRange(*USER_Y, padding=0)
    settle()
    # 前置 sanity：添加 b 之前用户视野必须是设定值
    assert pw.view_box.viewRange()[1] == pytest.approx(list(USER_Y), abs=1e-6)
    return pw


# ---------- 修复回归 ----------

def test_y_view_preserved_when_new_curve_within_view(plot_with_a):
    """回归 1+2：新曲线窗口内范围在用户视野内 → 视野精确保持。

    修复前：全量调用污染"当前视野"→ 并集退化成全量 [0,130] 再二次
    padding，视野跳到约 [-16, 151]；修复后应保持 USER_Y。
    """
    pw = plot_with_a
    assert pw.add_variable_to_plot("b", x_values=_x(), y_values=_y_b())

    view_y = pw.view_box.viewRange()[1]
    assert view_y == pytest.approx(list(USER_Y), abs=1e-6)


def test_y_view_expands_by_union_when_new_curve_exceeds(plot_factory, qapp):
    """扩张语义：新曲线窗口内范围超出视野 → 并集 + 单次 padding。

    窗口内 b 范围 [30, 40]，用户视野 [35.0, 36.5] → 并集 [30, 40]，
    padding 0.1 → 最终视野约 [28, 42]（绝不能到全量量级）。
    """
    pw = plot_factory(pd.DataFrame({"a": _y_a(), "b": _y_b()}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("a", x_values=_x(), y_values=_y_a())

    x = _x()
    # b 在窗口 [90, 110] 内从 30 线性升到 40（超出用户视野两端）
    y_b = 30.0 + 0.5 * (x - 90.0)
    pw.view_box.setXRange(*VIEW_X, padding=0)
    pw.view_box.setYRange(*USER_Y, padding=0)
    settle()

    assert pw.add_variable_to_plot("b", x_values=x, y_values=y_b)

    view_y = pw.view_box.viewRange()[1]
    # 并集 [30, 40]（span=10）经 padding 0.1 → 每侧扩 1.0 → [29, 41]
    assert view_y == pytest.approx([29.0, 41.0], abs=1e-6)


def test_y_view_preserved_when_x_window_disjoint(plot_with_a):
    """回归 3：新曲线 X 范围与当前窗口无交集 → 视野保持不变。"""
    pw = plot_with_a
    x_far = np.linspace(300.0, 500.0, N)
    y_far = np.linspace(0.0, 130.0, N)  # 全量范围悬殊，但都在窗口外
    assert pw.add_variable_to_plot("b", x_values=x_far, y_values=y_far)

    view_y = pw.view_box.viewRange()[1]
    assert view_y == pytest.approx(list(USER_Y), abs=1e-6)


def test_y_view_preserved_when_no_sample_in_window(plot_with_a):
    """回归 3 主干：X 范围覆盖窗口但窗口内无采样点 → 视野保持不变。

    与上一个用例不同，这里交集非空（x_min/x_max 落在数据范围内），
    只有 window_mask 为空，专门覆盖 np.any 判定的主干分支。
    """
    pw = plot_with_a
    x_sparse = np.array([0.0, 50.0, 150.0, 200.0])
    y_sparse = np.array([0.0, 130.0, 0.0, 130.0])  # 窗口 [90,110] 内无采样点
    assert pw.add_variable_to_plot("b", x_values=x_sparse, y_values=y_sparse)

    view_y = pw.view_box.viewRange()[1]
    assert view_y == pytest.approx(list(USER_Y), abs=1e-6)


def test_y_view_preserved_when_window_y_all_nan(plot_with_a):
    """回归 3 补充：窗口内有采样点但 y 全为 NaN → 视野保持不变。

    ragged CSV 尾部补 NaN 的真实场景：窗口内无有效 y 时不应回退到
    新曲线全量范围扩张视野。
    """
    pw = plot_with_a
    y_nan = _y_b().copy()
    y_nan[( _x() >= VIEW_X[0]) & (_x() <= VIEW_X[1])] = np.nan
    assert pw.add_variable_to_plot("b", x_values=_x(), y_values=y_nan)

    view_y = pw.view_box.viewRange()[1]
    assert view_y == pytest.approx(list(USER_Y), abs=1e-6)


def test_y_limits_still_cover_full_data(plot_with_a):
    """最大可调范围不受影响：limits 仍为全部可见曲线全量数据 ±50%。

    全量并集 [0, 130] → limits ≈ [-65, 195]。
    """
    pw = plot_with_a
    assert pw.add_variable_to_plot("b", x_values=_x(), y_values=_y_b())

    y_limits = pw.view_box.state["limits"]["yLimits"]
    assert y_limits[0] == pytest.approx(-65.0, abs=1.0)
    assert y_limits[1] == pytest.approx(195.0, abs=1.0)
