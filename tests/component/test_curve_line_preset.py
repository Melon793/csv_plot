"""曲线样式预设（line_preset）组件测试（offscreen）：真实 plot widget。

覆盖：
- apply_curve_line_preset 切换 → pen 宽度/线型翻转（含回 auto 往返）
- 非法 preset / 未知变量 → False 且不污染 ci.line_preset
- 钉住态宽度不随 symbol 态切换；symbol 尺寸 = 生效宽度 +2
- 选中高亮 +2（dash 2→4 / bold 4→6），缓存 key 以 _hl 结尾
- 三层 Z：默认 0 < 钉住预设 5 < 选中 10
- 改色保线型与 hl 宽度（选中态改色不回落）
- 批量窗口抑制 curves_changed
- _recreate_curve 恢复钉住预设
- 搬运快照（CurveStyleSnapshot / ShelfEntry / 去重更新）携带预设

面板样式组（单选语义/播报）见 test_legend_line_preset_panel.py。
"""

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from tests.fixtures.waits import settle

from src.core.data_types import (
    LINE_PRESET_AUTO,
    LINE_PRESET_BOLD,
    LINE_PRESET_DASH,
    LINE_PRESET_DOT,
    SYMBOL_POLICY_ON,
)
from src.ui.drag_drop import CurveStyleSnapshot
from src.ui.widgets.curve_shelf import CurveShelfWidget, ShelfEntry


def _y(n: int) -> np.ndarray:
    return np.sin(np.linspace(0.0, 6.0, n))


@pytest.fixture()
def preset_plot(plot_factory, qapp):
    n = 30
    pw = plot_factory(pd.DataFrame({"v": _y(n), "w": _y(n)}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=np.arange(n, dtype=float), y_values=_y(n))
    assert pw.add_variable_to_plot("w", x_values=np.arange(n, dtype=float), y_values=_y(n))
    settle()
    return pw


def _pen(pw, name="v"):
    return pw.curves[name].curve.opts["pen"]


def _style_refresh(pw):
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw.update_plot_style(pw.view_box, pw.view_box.viewRange(), None)


# ---------------------------------------------------------------------------
# apply 入口：pen 翻转 / 守卫
# ---------------------------------------------------------------------------

def test_preset_switch_flips_pen(preset_plot):
    pw = preset_plot
    mcm = pw._multi_curve_manager
    # 全景基线：拉宽视野 → 无 symbol 粗线态
    pw.view_box.setXRange(0.0, 200.0, padding=0)
    _style_refresh(pw)

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    assert _pen(pw).widthF() == 2
    assert _pen(pw).style() == Qt.PenStyle.DashLine

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_BOLD)
    assert _pen(pw).widthF() == 4
    assert _pen(pw).style() == Qt.PenStyle.SolidLine

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DOT)
    assert _pen(pw).widthF() == 2
    assert _pen(pw).style() == Qt.PenStyle.DotLine

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_AUTO)
    assert _pen(pw).widthF() == 2  # 回到状态机粗线态
    assert _pen(pw).style() == Qt.PenStyle.SolidLine
    assert pw.curves["v"].line_preset == LINE_PRESET_AUTO


def test_preset_invalid_inputs(preset_plot):
    mcm = preset_plot._multi_curve_manager
    assert mcm.apply_curve_line_preset("v", "diagonal") is False
    assert preset_plot.curves["v"].line_preset == LINE_PRESET_AUTO
    assert mcm.apply_curve_line_preset("ghost", LINE_PRESET_DASH) is False


def test_pinned_width_constant_across_symbol_states(preset_plot):
    """钉住态宽度恒定：zoom-in 出 symbol 也不回落细 1；点尺寸随宽联动。"""
    pw = preset_plot
    mcm = pw._multi_curve_manager
    pw.view_box.setXRange(0.0, 200.0, padding=0)
    _style_refresh(pw)
    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    assert not getattr(pw.curves["v"].curve, "_has_symbols", False)

    # zoom-in + 钉住 symbol：宽度仍 2（不回落 1），symbolSize = 2+2
    pw.view_box.setXRange(0.0, 30.0, padding=0)
    assert mcm.apply_curve_symbol_policy("v", SYMBOL_POLICY_ON)
    curve = pw.curves["v"].curve
    assert getattr(curve, "_has_symbols", False)
    assert _pen(pw).widthF() == 2
    assert _pen(pw).style() == Qt.PenStyle.DashLine
    assert curve.opts["symbolSize"] == 4


def test_symbol_size_links_width(preset_plot):
    """auto 细态尺寸 3（现状恒等）；bold+symbol 尺寸 6（点不埋进粗线）。"""
    pw = preset_plot
    mcm = pw._multi_curve_manager
    pw.view_box.setXRange(0.0, 30.0, padding=0)
    _style_refresh(pw)
    curve = pw.curves["v"].curve
    assert getattr(curve, "_has_symbols", False)
    assert curve.opts["symbolSize"] == 3  # auto 细 1 → 1+2

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_BOLD)
    assert curve.opts["symbolSize"] == 6  # bold 4 → 4+2
    assert _pen(pw).widthF() == 4


# ---------------------------------------------------------------------------
# 选中高亮 +2 与改色保持
# ---------------------------------------------------------------------------

def test_highlight_plus_two_and_cache_key(preset_plot):
    pw = preset_plot
    mcm = pw._multi_curve_manager
    pw.view_box.setXRange(0.0, 200.0, padding=0)
    _style_refresh(pw)
    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)

    pw._legend_selection = {"v"}
    mcm.refresh_selection_emphasis()
    assert _pen(pw).widthF() == 4
    assert pw.curves["v"].curve._cached_pen_key.endswith("_hl")

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_BOLD)
    assert _pen(pw).widthF() == 6

    pw._legend_selection = set()
    mcm.refresh_selection_emphasis()
    assert _pen(pw).widthF() == 4


def test_preset_z_order(preset_plot):
    """三层 Z：默认 0 < 钉住预设 5 < 选中 10；回 auto 落回 0。"""
    pw = preset_plot
    mcm = pw._multi_curve_manager
    pw.view_box.setXRange(0.0, 200.0, padding=0)
    _style_refresh(pw)
    curve = pw.curves["v"].curve
    assert curve.zValue() == 0

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    assert curve.zValue() == 5

    pw._legend_selection = {"v"}
    mcm.refresh_selection_emphasis()
    assert curve.zValue() == 10

    pw._legend_selection = set()
    mcm.refresh_selection_emphasis()
    assert curve.zValue() == 5

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_AUTO)
    assert curve.zValue() == 0


def test_color_change_keeps_preset_and_hl_width(preset_plot):
    """选中态改色：线型保持、hl 宽度不回落（4→4 而非 4→2）。"""
    pw = preset_plot
    mcm = pw._multi_curve_manager
    pw.view_box.setXRange(0.0, 200.0, padding=0)
    _style_refresh(pw)
    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    pw._legend_selection = {"v"}
    mcm.refresh_selection_emphasis()
    assert _pen(pw).widthF() == 4

    assert mcm.apply_curve_color("v", "#D62728")
    assert _pen(pw).widthF() == 4
    assert _pen(pw).style() == Qt.PenStyle.DashLine
    assert _pen(pw).color().name().lower() == "#d62728"


# ---------------------------------------------------------------------------
# 批量抑制 / 重建恢复 / 搬运快照
# ---------------------------------------------------------------------------

def test_batch_window_suppresses_emit(preset_plot):
    pw = preset_plot
    mcm = pw._multi_curve_manager
    calls = []
    pw.curves_changed.connect(lambda: calls.append(1))

    pw._batch_adding = True
    try:
        assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    finally:
        pw._batch_adding = False
    assert calls == []

    assert mcm.apply_curve_line_preset("v", LINE_PRESET_BOLD)
    assert calls == [1]


def test_recreate_restores_pinned_preset(preset_plot):
    pw = preset_plot
    mcm = pw._multi_curve_manager
    assert mcm.apply_curve_line_preset("v", LINE_PRESET_DASH)
    mcm._recreate_curve("v")
    assert pw.curves["v"].line_preset == LINE_PRESET_DASH
    assert _pen(pw).widthF() == 2
    assert _pen(pw).style() == Qt.PenStyle.DashLine


def test_transfer_snapshots_carry_preset(preset_plot, qapp):
    pw = preset_plot
    assert pw._multi_curve_manager.apply_curve_line_preset("v", LINE_PRESET_DASH)
    ci = pw.curves["v"]

    snap = CurveStyleSnapshot.from_curve_info(ci)
    assert snap.line_preset == LINE_PRESET_DASH

    entry = ShelfEntry.from_curve_info("v", ci)
    assert entry.line_preset == LINE_PRESET_DASH

    # 暂存区按名去重更新：二次入区必须带上新预设值
    shelf = CurveShelfWidget()
    assert shelf.add_entry(entry) == "added"
    assert pw._multi_curve_manager.apply_curve_line_preset("v", LINE_PRESET_BOLD)
    entry2 = ShelfEntry.from_curve_info("v", pw.curves["v"])
    assert shelf.add_entry(entry2) == "updated"
    assert shelf.entry_for_name("v").line_preset == LINE_PRESET_BOLD
    shelf.deleteLater()