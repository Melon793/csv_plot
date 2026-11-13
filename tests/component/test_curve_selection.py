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

import pytest

from src.core.config import (
    HIGHLIGHT_CURVE_Z,
    HIGHLIGHT_LINE_WIDTH,
    HIGHLIGHT_THIN_LINE_WIDTH,
    THICK_LINE_WIDTH,
    THIN_LINE_WIDTH,
)
from src.core.data_types import SYMBOL_POLICY_OFF, SYMBOL_POLICY_ON


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