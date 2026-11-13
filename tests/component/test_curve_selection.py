"""曲线选中联动：选择集 → 曲线加粗投影 / plot 命中入口 / 手势播报（offscreen）。

对应施工计划 tmp/2026-11-13/curve-selection-plan.md：
- P0（§3.1/§3.2）：pen 缓存 _hl 维度、加粗 2→4 / 1→3、提 Z=10、隐藏曲线
  只 tint 不加粗、热路径重刷高亮存活、两投影收口、curves_changed 零触发；
- P1（§3.3）：命中纯函数容差/step 拐角/NaN、press-release 手势分派、
  跨区误触与陈旧记录防护；
- P2（§3.4）：文案函数与 announce_selection（component 层只断言文案与
  调用，真实状态栏通道断言在 tests/e2e/test_curve_selection_flow.py）。

纪律：component 层宿主为 FakeHost（无状态栏通道，conftest.py:59-69），
不得断言 _message_label/toolTip/timer；夹具一律 data_factory / plot_factory。
"""

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.core.config import (
    CURVE_HIT_TOLERANCE_PX,
    HIGHLIGHT_CURVE_Z,
    HIGHLIGHT_LINE_WIDTH,
    HIGHLIGHT_THIN_LINE_WIDTH,
    THICK_LINE_WIDTH,
    THIN_LINE_WIDTH,
)
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    DRAW_MODE_STEP_PRE,
    SYMBOL_POLICY_OFF,
    SYMBOL_POLICY_ON,
)
from src.ui.widgets.multi_curve_manager import hit_test_curve
from tests.fixtures.data_factory import make_timeseries
from tests.fixtures.waits import settle


@pytest.fixture()
def three_curves(plot_factory):
    pw = plot_factory()
    for name in ("a", "b", "c"):
        assert pw.plot_variable(name)
    return pw


def _pin_symbol_state(pw, policy):
    """钉住 symbol 策略，让加粗断言不受可见点数阈值影响。"""
    for ci in pw.curves.values():
        ci.symbol_policy = policy
    pw._refresh_selection_projections()


def _width(pw, name):
    return pw.curves[name].curve.opts["pen"].widthF()


def _z(pw, name):
    return pw.curves[name].curve.zValue()


# ---------------------------------------------------------------------------
# P0 · 加粗投影（验收 §6.1 / §6.2 / §6.3）
# ---------------------------------------------------------------------------


def test_select_thickens_pen_and_raises_z(three_curves):
    """粗线态：选中 2→4px + z=10，未选中曲线不动。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert _z(pw, "a") == 0

    pw.legend_label._toggle_legend_selection("a")

    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH
    assert _z(pw, "a") == HIGHLIGHT_CURVE_Z
    assert pw.curves["a"].curve._cached_pen_key.endswith("_hl")
    assert _width(pw, "b") == THICK_LINE_WIDTH
    assert _z(pw, "b") == 0


def test_deselect_restores_pen_and_z(three_curves):
    """取消选中恢复原线宽与 z=0（Ctrl 减选路径）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    label = pw.legend_label
    label._toggle_legend_selection("a")
    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH

    label._toggle_legend_selection("a")

    assert pw._legend_selection == set()
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert _z(pw, "a") == 0
    assert not pw.curves["a"].curve._cached_pen_key.endswith("_hl")


def test_symbol_state_thin_pen_highlight(three_curves):
    """symbol 态：细线 1→3px（验收 §6.1 的 symbol 分支）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_ON)
    assert _width(pw, "a") == THIN_LINE_WIDTH

    pw.legend_label._toggle_legend_selection("a")

    assert _width(pw, "a") == HIGHLIGHT_THIN_LINE_WIDTH
    assert pw.curves["a"].curve._has_symbols is True
    assert _z(pw, "a") == HIGHLIGHT_CURVE_Z


def test_hidden_curve_selected_keeps_pen(three_curves):
    """隐藏曲线被选中：pen/z 不变（显隐正交，R5），tint 仍在。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    pw.curves["a"].visible = False
    pw.curves["a"].curve.hide()

    pw.legend_label._toggle_legend_selection("a")

    assert "a" in pw._legend_selection
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert _z(pw, "a") == 0
    assert pw.legend_label.extraSelections(), "隐藏曲线仍应有 legend tint"


def test_multi_select_all_highlighted(three_curves):
    """Ctrl 多选：三条同时加粗；减选一条后剩两条（验收 §6.2）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    label = pw.legend_label
    for name in ("a", "b", "c"):
        label._toggle_legend_selection(name)

    assert len(pw._legend_selection) == 3
    assert all(_width(pw, n) == HIGHLIGHT_LINE_WIDTH for n in ("a", "b", "c"))

    label._toggle_legend_selection("b")

    assert len(pw._legend_selection) == 2
    assert _width(pw, "b") == THICK_LINE_WIDTH
    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "c") == HIGHLIGHT_LINE_WIDTH


def test_highlight_survives_hot_path_restyle(three_curves):
    """缩放热路径 update_plot_style 重刷后高亮存活（D3 / 风险 #2）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    pw.legend_label._toggle_legend_selection("a")
    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH

    x_min, x_max = pw.view_box.viewRange()[0]
    pw.update_plot_style(pw.view_box, [[x_min, x_max], [0.0, 10.0]])

    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH
    assert _z(pw, "a") == HIGHLIGHT_CURVE_Z
    assert _width(pw, "b") == THICK_LINE_WIDTH


def test_pen_cache_key_carries_hl_dimension(three_curves):
    """pen 缓存 key 含 _hl 维度：同色同 symbol 态下选中/未选中不互相吞刷新。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    # 两条曲线同色，排除颜色差异造成的 key 变化
    pw.curves["b"].color = pw.curves["a"].color
    pw._refresh_selection_projections()
    key_plain = pw.curves["a"].curve._cached_pen_key

    pw.legend_label._toggle_legend_selection("a")

    key_hl = pw.curves["a"].curve._cached_pen_key
    assert key_hl != key_plain
    assert key_hl == f"{key_plain}_hl"
    assert pw.curves["b"].curve._cached_pen_key == key_plain


def test_range_select_refreshes_emphasis(three_curves):
    """Shift 范围选经收口函数：范围内全部加粗（反向联动，验收 §6.3）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    label = pw.legend_label
    label._toggle_legend_selection("a")  # 立锚点
    label._range_select_legend("c")

    assert pw._legend_selection == {"a", "b", "c"}
    assert all(_width(pw, n) == HIGHLIGHT_LINE_WIDTH for n in ("a", "b", "c"))


def test_clear_selection_restores_all(three_curves):
    """clear_legend_selection 两投影同步恢复（批量菜单「清除选择」同路径）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    label = pw.legend_label
    for name in ("a", "b"):
        label._toggle_legend_selection(name)

    label.clear_legend_selection()

    assert pw._legend_selection == set()
    assert pw._legend_sel_anchor is None
    assert label.extraSelections() == []
    assert all(_width(pw, n) == THICK_LINE_WIDTH for n in ("a", "b", "c"))
    assert all(_z(pw, n) == 0 for n in ("a", "b", "c"))


def test_sethtml_replay_syncs_emphasis_and_prunes(three_curves):
    """图例重建重放：加粗同步，且已删除曲线被 prune 后不再残留高亮。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    label = pw.legend_label
    label._toggle_legend_selection("a")
    label._toggle_legend_selection("b")

    # 模拟删除 a 后图例重建（系统路径）
    del pw.curves["a"]
    pw.update_legend_label('<a href="curve:///b">b</a>')

    assert pw._legend_selection == {"b"}
    assert _width(pw, "b") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "c") == THICK_LINE_WIDTH


def test_placeholder_clear_restores_emphasis(three_curves):
    """图例回归占位态：选择集清空 + 加粗全部恢复（系统路径）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    pw.legend_label._toggle_legend_selection("a")

    pw.update_legend_label("绘图区为空")

    assert pw._legend_selection == set()
    assert all(_width(pw, n) == THICK_LINE_WIDTH for n in ("a", "b", "c"))


def test_selection_change_does_not_emit_curves_changed(three_curves):
    """选择变化不得 emit curves_changed（R8 / 风险 #10）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    calls = []
    pw.curves_changed.connect(lambda: calls.append(1))
    label = pw.legend_label

    label._toggle_legend_selection("a")
    label._range_select_legend("c")
    label.clear_legend_selection()
    pw._refresh_selection_projections()

    assert calls == []


def test_emphasis_skips_during_data_update(three_curves):
    """数据更新/销毁中间态跳过加粗刷新（守卫对齐热路径，风险 #11）。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    pw.legend_label._toggle_legend_selection("a")
    pw.curves["a"].curve.setPen(width=THICK_LINE_WIDTH)
    pw.curves["a"].curve._cached_pen_key = "stale"

    pw._is_updating_data = True
    try:
        pw._multi_curve_manager.refresh_selection_emphasis()
        assert pw.curves["a"].curve._cached_pen_key == "stale"
    finally:
        pw._is_updating_data = False

    pw._multi_curve_manager.refresh_selection_emphasis()
    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH


def test_emphasis_tolerates_missing_selection_attr(three_curves):
    """替身/早期初始化无 _legend_selection 时不抛异常，按未选中处理。"""
    pw = three_curves
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    saved = pw._legend_selection
    del pw._legend_selection
    try:
        pw._multi_curve_manager.refresh_selection_emphasis()
        assert all(_width(pw, n) == THICK_LINE_WIDTH for n in ("a", "b", "c"))
    finally:
        pw._legend_selection = saved


# ---------------------------------------------------------------------------
# P1 · 命中纯函数（计划 §3.3 / 风险 #6 #13）
# ---------------------------------------------------------------------------


def _shown_plot(plot_factory, df):
    """show 宿主后再量测：未 show 时 vb 仅 ~21px 高，像素容差断言会失真。

    先例 test_curve_draw_mode.py:40-41 / test_cursor_label_edge.py:36-37。
    """
    pw = plot_factory(df)
    pw.window().resize(900, 620)
    pw.resize(880, 600)
    pw.window().show()
    pw.show()
    settle()
    return pw


@pytest.fixture()
def hit_plot(plot_factory, qapp):
    """三条分离带的真实几何 plot：a/b 正弦（相差 20），c 水平线 y=30。

    数据源自 data_factory.make_timeseries；c 取常量，使"垂直像素偏移"
    等于"点到线段距离"，容差边界断言才成立。
    """
    _, y = make_timeseries(n=60)
    df = pd.DataFrame({"a": y, "b": y + 20.0, "c": np.full_like(y, 30.0)})
    pw = _shown_plot(plot_factory, df)
    for name in ("a", "b", "c"):
        assert pw.plot_variable(name)
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    settle()
    assert pw.view_box.sceneBoundingRect().height() > 100, "几何失真，容差断言不可信"
    return pw


@pytest.fixture()
def step_plot(plot_factory, qapp):
    """枚举阶梯小数据（合成 5 点）：y = 1,1,9,9,3，跳变在 idx1→idx2。

    x_data 为 1 基行号 [1..5]（diag_step_hit.py 实测）：step-post 的竖直
    riser 落在 x_data[2]，step-pre 提前到 x_data[1]。断言一律取 x_data
    实值，不硬编码行号基准。
    """
    df = pd.DataFrame({"s": [1.0, 1.0, 9.0, 9.0, 3.0]})
    pw = _shown_plot(plot_factory, df)
    assert pw.plot_variable("s")
    pw.curves["s"].symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    settle()
    return pw


def _scene_at(pw, name, frac=0.5):
    """曲线上某个数据点的 scene 坐标（按索引比例取，不假设 x 基准）。"""
    ci = pw.curves[name]
    i = min(len(ci.x_data) - 1, int(len(ci.x_data) * frac))
    return pw.view_box.mapViewToScene(
        QPointF(float(ci.x_data[i]), float(ci.y_data[i]))
    )


def _vp(pw, scene_pt):
    return pw.mapFromScene(scene_pt)


def _mouse(kind, pos, button, mods=Qt.KeyboardModifier.NoModifier):
    p = QPointF(pos)
    return QMouseEvent(kind, p, p, QPointF(0.0, 0.0), button, button, mods)


def _press(pw, pos, button=Qt.MouseButton.LeftButton,
           mods=Qt.KeyboardModifier.NoModifier):
    pw.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pos, button, mods))


def _release(pw, pos, button=Qt.MouseButton.LeftButton,
             mods=Qt.KeyboardModifier.NoModifier):
    pw.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pos, button, mods))


def _click(pw, pos, mods=Qt.KeyboardModifier.NoModifier):
    _press(pw, pos, mods=mods)
    _release(pw, pos, mods=mods)


def _blank_scene(pw):
    """a(40~60) 与 c(30) 之间的空带中央：离两侧都远超容差。"""
    ci = pw.curves["c"]
    x_mid = float(ci.x_data[len(ci.x_data) // 2])
    p = pw.view_box.mapViewToScene(QPointF(x_mid, 35.0))
    assert hit_test_curve(pw, p) is None, "空白点前提不成立"
    return p


def test_hit_tolerance_boundary(hit_plot):
    """8px 容差：7px 命中 / 9px 不中（水平线，垂直偏移即最短距离）。"""
    pw = hit_plot
    base = _scene_at(pw, "c")
    inside = QPointF(base.x(), base.y() + CURVE_HIT_TOLERANCE_PX - 1)
    outside = QPointF(base.x(), base.y() + CURVE_HIT_TOLERANCE_PX + 1)
    assert hit_test_curve(pw, inside) == "c"
    assert hit_test_curve(pw, outside) is None


def test_hit_prefers_nearest_over_insertion_order(plot_factory, qapp):
    """两条都在容差内时取像素距离更近者，而非插入序在前者。"""
    _, y = make_timeseries(n=20)
    df = pd.DataFrame({
        "far": np.full_like(y, 30.0),
        "near": np.full_like(y, 30.4),
        "wide": np.full_like(y, 100.0),  # 撑开 y 轴，使 0.4 数据单位 < 8px
    })
    pw = _shown_plot(plot_factory, df)
    for name in ("far", "near", "wide"):
        assert pw.plot_variable(name)
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    settle()
    sf = _scene_at(pw, "far")
    sn = _scene_at(pw, "near")
    gap = abs(sn.y() - sf.y())
    assert 0 < gap < CURVE_HIT_TOLERANCE_PX, f"两条线须同在容差内，实测 {gap:.2f}px"
    # 落在两线之间、偏向 near（3:1）→ 最近者胜
    p = QPointF(sf.x(), sf.y() + (sn.y() - sf.y()) * 0.75)
    assert hit_test_curve(pw, p) == "near"


def test_hit_tie_takes_insertion_order(plot_factory, qapp):
    """完全重合（距离相等）→ 取 curves 插入序在前者（严格小于比较）。"""
    _, y = make_timeseries(n=20)
    flat = np.full_like(y, 30.0)
    df = pd.DataFrame({"first": flat, "second": flat.copy()})
    pw = _shown_plot(plot_factory, df)
    for name in ("first", "second"):
        assert pw.plot_variable(name)
    _pin_symbol_state(pw, SYMBOL_POLICY_OFF)
    settle()
    assert hit_test_curve(pw, _scene_at(pw, "first")) == "first"


def test_hit_step_post_riser_and_platform(step_plot):
    """step-post：竖直 riser 中点/两端 + 平台段都命中；直线模式对照不中。"""
    pw = step_plot
    assert pw._multi_curve_manager.apply_curve_draw_mode("s", DRAW_MODE_STEP_POST)
    assert pw.curves["s"].curve.opts["stepMode"] == "right"
    settle()
    vb = pw.view_box
    x_post = float(pw.curves["s"].x_data[2])
    x_prev = float(pw.curves["s"].x_data[1])
    for y in (1.0, 5.0, 9.0):  # riser 下端 / 中点 / 上端
        assert hit_test_curve(pw, vb.mapViewToScene(QPointF(x_post, y))) == "s"
    # 平台段：旧值保持到跳变点
    assert hit_test_curve(
        pw, vb.mapViewToScene(QPointF((x_prev + x_post) / 2, 1.0))
    ) == "s"
    # 对照：同一点在直线模式下是空中点 → 证明 L 形拐角分支确实生效
    assert pw._multi_curve_manager.apply_curve_draw_mode("s", DRAW_MODE_AUTO)
    settle()
    assert hit_test_curve(pw, vb.mapViewToScene(QPointF(x_post, 5.0))) is None


def test_hit_step_pre_riser_shifts_left(step_plot):
    """step-pre：跳变提前一个采样间隔，riser 落在 x_data[1]。"""
    pw = step_plot
    assert pw._multi_curve_manager.apply_curve_draw_mode("s", DRAW_MODE_STEP_PRE)
    assert pw.curves["s"].curve.opts["stepMode"] == "left"
    settle()
    vb = pw.view_box
    x_pre = float(pw.curves["s"].x_data[1])
    x_post = float(pw.curves["s"].x_data[2])
    assert hit_test_curve(pw, vb.mapViewToScene(QPointF(x_pre, 5.0))) == "s"
    assert hit_test_curve(pw, vb.mapViewToScene(QPointF(x_post, 5.0))) is None


def test_hit_skips_nan_segments(plot_factory, qapp):
    """NaN 空洞：非有限端点跳过，不崩也不误命中；有限段照常命中。"""
    df = pd.DataFrame({"n": [0.0, 1.0, np.nan, np.nan, 4.0, 5.0]})
    pw = _shown_plot(plot_factory, df)
    assert pw.plot_variable("n")
    settle()
    ci = pw.curves["n"]
    vb = pw.view_box
    x_mid = (float(ci.x_data[2]) + float(ci.x_data[3])) / 2
    assert hit_test_curve(pw, vb.mapViewToScene(QPointF(x_mid, 2.5))) is None
    assert hit_test_curve(
        pw, vb.mapViewToScene(QPointF(float(ci.x_data[0]), float(ci.y_data[0])))
    ) == "n"


def test_hit_skips_hidden_curve(hit_plot):
    """隐藏曲线不参与命中（点不到的东西不该被选中）。"""
    pw = hit_plot
    p = _scene_at(pw, "c")
    assert hit_test_curve(pw, p) == "c"
    pw.curves["c"].visible = False
    assert hit_test_curve(pw, p) is None


def test_hit_skips_missing_curve_object(hit_plot):
    """失效占位无曲线对象（R9）：跳过而非抛异常。"""
    pw = hit_plot
    p = _scene_at(pw, "c")
    pw.curves["c"].curve = None
    assert hit_test_curve(pw, p) is None


def test_hit_skips_single_point_curve(plot_factory, qapp):
    """单点曲线无线段可测（len < 2 守卫）。"""
    df = pd.DataFrame({"one": [5.0]})
    pw = _shown_plot(plot_factory, df)
    assert pw.plot_variable("one")
    settle()
    assert hit_test_curve(pw, _scene_at(pw, "one")) is None


def test_hit_outside_data_x_range(hit_plot):
    """端点缓存区间排除：x 超出 [x_min, x_max] 直接不中。"""
    pw = hit_plot
    ci = pw.curves["c"]
    p = pw.view_box.mapViewToScene(
        QPointF(float(ci.x_max) + 1e3, float(ci.y_data[0]))
    )
    assert hit_test_curve(pw, p) is None


def test_hit_tolerates_missing_view_box():
    """纯函数防御：无 view_box 的替身返回 None，不抛异常。"""
    class Stub:
        curves = {}

    assert hit_test_curve(Stub(), QPointF(0.0, 0.0)) is None


# ---------------------------------------------------------------------------
# P1 · press/release 手势分派（计划 §3.3 / 评审 B3、N2）
# ---------------------------------------------------------------------------


def test_qtest_click_selects_thickens_and_tints(hit_plot, qapp):
    """真实点击链路（QTest 打 viewport）：选择集 + 加粗 + z + tint 四断言。"""
    pw = hit_plot
    pos = _vp(pw, _scene_at(pw, "b"))
    QTest.mouseClick(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, pos)
    qapp.processEvents()

    assert pw._legend_selection == {"b"}
    assert _width(pw, "b") == HIGHLIGHT_LINE_WIDTH
    assert _z(pw, "b") == HIGHLIGHT_CURVE_Z
    # tint 按 fragment 计（色块 + 名字各一段），故断言"被 tint 的文本"
    # 而非段数：b 的名字在 tint 内，未选中的 a/c 不在。
    tinted = [es.cursor.selectedText() for es in pw.legend_label.extraSelections()]
    assert "b" in tinted, tinted
    assert "a" not in tinted and "c" not in tinted, tinted
    assert pw._curve_click_press_pos is None, "记录须被消费，不留陈旧值"


def test_ctrl_click_adds_then_toggles(hit_plot, qapp):
    """Ctrl 加选第二条 → 两条都加粗；Ctrl 再击同一条 → 减选并恢复线宽。"""
    pw = hit_plot
    pa = _vp(pw, _scene_at(pw, "a"))
    pb = _vp(pw, _scene_at(pw, "b"))
    QTest.mouseClick(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, pb)
    QTest.mouseClick(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier, pa)
    qapp.processEvents()
    assert pw._legend_selection == {"a", "b"}
    assert _width(pw, "a") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "b") == HIGHLIGHT_LINE_WIDTH

    QTest.mouseClick(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier, pa)
    qapp.processEvents()
    assert pw._legend_selection == {"b"}
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert _z(pw, "a") == 0


def test_bare_click_replaces_selection_and_anchor(hit_plot):
    """裸单击 = 替换单选（标准文件管理器语义），anchor 同步重置。"""
    pw = hit_plot
    pw.legend_label._toggle_legend_selection("a")
    pw.legend_label._toggle_legend_selection("c")
    assert pw._legend_selection == {"a", "c"}

    _click(pw, _vp(pw, _scene_at(pw, "b")))

    assert pw._legend_selection == {"b"}
    assert pw._legend_sel_anchor == "b"
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert _width(pw, "c") == THICK_LINE_WIDTH
    assert _width(pw, "b") == HIGHLIGHT_LINE_WIDTH


def test_bare_click_blank_clears(hit_plot):
    """裸单击空白 → 清选 + 全部恢复原线宽。"""
    pw = hit_plot
    pw.legend_label._toggle_legend_selection("a")
    _click(pw, _vp(pw, _blank_scene(pw)))
    assert pw._legend_selection == set()
    assert _width(pw, "a") == THICK_LINE_WIDTH
    assert pw.legend_label.extraSelections() == []


def test_ctrl_click_blank_keeps_selection(hit_plot):
    """Ctrl+空白 → 选择保持（与 legend 口径一致）。"""
    pw = hit_plot
    pw.legend_label._toggle_legend_selection("c")
    _click(pw, _vp(pw, _blank_scene(pw)), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"c"}
    assert _width(pw, "c") == HIGHLIGHT_LINE_WIDTH


def test_drag_displacement_does_not_change_selection(hit_plot):
    """位移 ≥ startDragDistance 的 press/release 是拖拽不是点击（N1）。"""
    pw = hit_plot
    pw.legend_label._toggle_legend_selection("a")
    before = set(pw._legend_selection)
    pos = _vp(pw, _scene_at(pw, "b"))
    assert 25 >= QApplication.startDragDistance()

    _press(pw, pos)
    _release(pw, QPoint(pos.x() + 25, pos.y() + 25))

    assert pw._legend_selection == before
    assert pw._curve_click_press_pos is None


def test_small_jitter_still_counts_as_click(hit_plot):
    """位移 < startDragDistance 且 release 仍命中 → 照常选中（手抖容忍）。"""
    pw = hit_plot
    pos = _vp(pw, _scene_at(pw, "b"))
    jitter = QPoint(pos.x() + 4, pos.y())
    assert (jitter - pos).manhattanLength() < QApplication.startDragDistance()
    assert hit_test_curve(pw, pw.mapToScene(jitter)) == "b"

    _press(pw, pos)
    _release(pw, jitter)

    assert pw._legend_selection == {"b"}


def test_shift_press_not_recorded_and_rubber_band(hit_plot, qapp):
    """Shift 留给框选缩放（R4）：不记录起点、起 rubber band、选择集不动。"""
    pw = hit_plot
    pos = _vp(pw, _scene_at(pw, "b"))
    _press(pw, pos, mods=Qt.KeyboardModifier.ShiftModifier)
    assert pw._curve_click_press_pos is None
    assert pw.rubberBand.isVisible()

    _release(pw, pos, mods=Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()
    assert pw._legend_selection == set()
    assert not pw.rubberBand.isVisible()


def test_press_on_legend_release_in_vb_not_triggered(hit_plot):
    """B3：press 落图例、release 移 ≤10px 进绘图区 → 不触发选择。"""
    pw = hit_plot
    proxy = pw.legend_label.graphicsProxyWidget()
    assert proxy is not None
    lg = proxy.sceneBoundingRect()
    vb_rect = pw.view_box.sceneBoundingRect()
    press_scene = QPointF(vb_rect.center().x(), lg.bottom() - 2)
    release_scene = QPointF(vb_rect.center().x(), lg.bottom() + 4)
    p_vp, r_vp = _vp(pw, press_scene), _vp(pw, release_scene)
    assert (r_vp - p_vp).manhattanLength() < QApplication.startDragDistance()
    assert vb_rect.contains(release_scene), "release 须真的落在绘图区内"

    pw.legend_label._toggle_legend_selection("a")
    before = set(pw._legend_selection)
    _press(pw, p_vp)
    assert pw._curve_click_press_pos is None, "图例上的 press 不得记录"
    _release(pw, r_vp)
    assert pw._legend_selection == before


def test_release_outside_vb_not_triggered(hit_plot):
    """press 在绘图区内、release 移出 vb（位移 <10px）→ 区域校验拦下。"""
    pw = hit_plot
    vb_rect = pw.view_box.sceneBoundingRect()
    press_scene = QPointF(vb_rect.center().x(), vb_rect.top() + 3)
    release_scene = QPointF(vb_rect.center().x(), vb_rect.top() - 3)
    p_vp, r_vp = _vp(pw, press_scene), _vp(pw, release_scene)
    assert (r_vp - p_vp).manhattanLength() < QApplication.startDragDistance()
    assert not vb_rect.contains(release_scene)

    pw.legend_label._toggle_legend_selection("a")
    before = set(pw._legend_selection)
    _press(pw, p_vp)
    assert pw._curve_click_press_pos is not None
    _release(pw, r_vp)
    assert pw._legend_selection == before


def test_middle_press_clears_stale_record(hit_plot):
    """N2：中键 press 早退分支也必须清记录，中键 release 不触发选择。"""
    pw = hit_plot
    pos = _vp(pw, _scene_at(pw, "b"))
    _press(pw, pos)
    assert pw._curve_click_press_pos is not None

    _press(pw, pos, button=Qt.MouseButton.MiddleButton)
    assert pw._curve_click_press_pos is None
    _release(pw, pos, button=Qt.MouseButton.MiddleButton)
    assert pw._legend_selection == set()


def test_non_left_release_does_not_trigger(hit_plot):
    """N2：左键 press 后跟右键 release → 不触发，且记录被清空。"""
    pw = hit_plot
    pos = _vp(pw, _scene_at(pw, "b"))
    _press(pw, pos)
    _release(pw, pos, button=Qt.MouseButton.RightButton)
    assert pw._legend_selection == set()
    assert pw._curve_click_press_pos is None


def test_click_does_not_emit_curves_changed(hit_plot, qapp):
    """硬护栏：plot 点击选择不得刷新变量编辑器/搜索栏。"""
    pw = hit_plot
    calls = []
    pw.curves_changed.connect(lambda: calls.append(1))

    _click(pw, _vp(pw, _scene_at(pw, "b")))
    _click(pw, _vp(pw, _scene_at(pw, "a")), Qt.KeyboardModifier.ControlModifier)
    _click(pw, _vp(pw, _blank_scene(pw)))
    qapp.processEvents()

    assert calls == []


def test_click_adds_no_scene_items(hit_plot, qapp):
    """D2 rev2：无徽章/无覆盖物，选择前后 scene item 计数不变。"""
    pw = hit_plot
    n0 = len(pw.scene().items())
    _click(pw, _vp(pw, _scene_at(pw, "b")))
    _click(pw, _vp(pw, _scene_at(pw, "a")), Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert len(pw.scene().items()) == n0


def test_ctrl_click_tolerates_legend_stand_in(hit_plot, qapp, monkeypatch):
    """替身 label 缺 toggle 方法：不抛异常（getattr 防御分支）。"""
    pw = hit_plot
    monkeypatch.setattr(
        type(pw.legend_label), "_toggle_legend_selection", None, raising=False
    )
    _click(pw, _vp(pw, _scene_at(pw, "b")), Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert pw._legend_selection == set()


def test_bare_click_tolerates_missing_selection_attr(hit_plot):
    """早期初始化/替身无 _legend_selection：裸单击不抛异常。"""
    pw = hit_plot
    saved = pw._legend_selection
    del pw._legend_selection
    try:
        _click(pw, _vp(pw, _scene_at(pw, "b")))
    finally:
        pw._legend_selection = saved