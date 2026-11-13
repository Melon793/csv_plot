"""绘图变量编辑器：绘制方式/数据点下拉（方案 A UI 入口）组件测试。

覆盖：
- 表格 5 列结构，真实曲线行带绘制方式/数据点下拉（itemData 绑常量）
- 下拉切换 → MultiCurveManager 应用（ci.draw_mode / stepMode / symbol_policy）
- 外部（图例菜单等）变更后 _refresh_rows_in_place 就地回填下拉状态
- 失效占位行不建下拉（无曲线可操作）
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_ON,
)
from src.ui.plot_variable_editor import PlotVariableEditorDialog

NAMES = ["a", "b"]


@pytest.fixture()
def editor_env(plot_factory, qapp):
    df = pd.DataFrame({name: [float(i), float(i + 1)] for i, name in enumerate(NAMES)})
    pw = plot_factory(df)
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=list(NAMES), units={})
    host.data_validity = {}
    for name in NAMES:
        assert pw.add_variable_to_plot(name)
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    return dlg, pw


def test_table_structure_with_mode_columns(editor_env):
    """5 列结构；真实曲线行第 3/4 列为下拉，初值 auto。"""
    dlg, _pw = editor_env
    table = dlg.var_table
    assert table.columnCount() == 5
    labels = [
        table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
    ]
    assert labels == ["显示", "变量名", "颜色", "绘制方式", "数据点"]
    for row in range(table.rowCount()):
        draw_combo = table.cellWidget(row, 3)
        assert draw_combo is not None
        assert draw_combo.currentData() == DRAW_MODE_AUTO
        symbol_combo = table.cellWidget(row, 4)
        assert symbol_combo is not None
        assert symbol_combo.currentData() == SYMBOL_POLICY_AUTO


def test_draw_mode_combo_switches_curve(editor_env):
    """绘制方式下拉切到 step-post：ci.draw_mode + curve.opts['stepMode'] 跟随。"""
    dlg, pw = editor_env
    draw_combo = dlg.var_table.cellWidget(0, 3)
    draw_combo.setCurrentIndex(draw_combo.findData(DRAW_MODE_STEP_POST))
    var_name = dlg.var_table.item(0, 1).data(Qt.ItemDataRole.UserRole)
    ci = pw.curves[var_name]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.curve.opts["stepMode"] == "right"


def test_symbol_policy_combo_pins_symbols(editor_env):
    """数据点下拉切到始终显示：symbol_policy=on 且立即应用。"""
    dlg, pw = editor_env
    symbol_combo = dlg.var_table.cellWidget(0, 4)
    symbol_combo.setCurrentIndex(symbol_combo.findData(SYMBOL_POLICY_ON))
    var_name = dlg.var_table.item(0, 1).data(Qt.ItemDataRole.UserRole)
    ci = pw.curves[var_name]
    assert ci.symbol_policy == SYMBOL_POLICY_ON
    assert getattr(ci.curve, "_has_symbols", False)


def test_refresh_rows_in_place_updates_combos(editor_env):
    """外部变更（图例菜单路径）经 curves_changed 自动回填下拉（真实信号链）。

    apply_curve_draw_mode / apply_curve_symbol_policy 成功后 emit
    curves_changed，编辑器增量刷新场景 1 就地回填下拉——不再手动调
    _refresh_rows_in_place（曾掩盖链路断裂）。
    """
    dlg, pw = editor_env
    var_name = dlg.var_table.item(0, 1).data(Qt.ItemDataRole.UserRole)
    # 模拟图例菜单入口改 draw_mode / symbol_policy
    assert pw._multi_curve_manager.apply_curve_draw_mode(var_name, DRAW_MODE_STEP_POST)
    pw.curves[var_name].symbol_policy = SYMBOL_POLICY_ON
    pw.curves_changed.emit()  # 后者直改字段，补一次信号（等价链路）

    draw_combo = dlg.var_table.cellWidget(0, 3)
    assert draw_combo.currentData() == DRAW_MODE_STEP_POST
    symbol_combo = dlg.var_table.cellWidget(0, 4)
    assert symbol_combo.currentData() == SYMBOL_POLICY_ON
    assert pw.curves[var_name].curve.opts["stepMode"] == "right"


def test_status_message_reports_chinese_label(editor_env):
    """状态栏回执：区分绘制方式/数据点且播报中文标签（与图例菜单同口径）。"""
    dlg, pw = editor_env
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)
    draw_combo = dlg.var_table.cellWidget(0, 3)
    draw_combo.setCurrentIndex(draw_combo.findData(DRAW_MODE_STEP_POST))
    assert any("绘制方式" in m and "阶梯(后)" in m for m in msgs)
    assert not any("auto" in m for m in msgs)  # 不播报内部常量

    symbol_combo = dlg.var_table.cellWidget(0, 4)
    symbol_combo.setCurrentIndex(symbol_combo.findData(SYMBOL_POLICY_ON))
    assert any("数据点" in m and "始终显示" in m for m in msgs)


def test_invalid_placeholder_rows_have_no_combos(plot_factory, qapp):
    """失效占位行第 3/4 列留空（无曲线可操作，行构造独立于真实曲线行）。"""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    pw = plot_factory(df)
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=["a"], units={})
    host.data_validity = {}
    assert pw.add_variable_to_plot("a")

    # 手工制造一条失效占位（reload 路径的降级形态）
    from src.core.data_types import InvalidCurveRef
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost")

    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    table = dlg.var_table
    ghost_rows = [
        r for r in range(table.rowCount())
        if table.item(r, 1).data(Qt.ItemDataRole.UserRole) == "ghost"
    ]
    assert ghost_rows, "占位行应在表格中"
    for row in ghost_rows:
        assert table.cellWidget(row, 3) is None
        assert table.cellWidget(row, 4) is None


def test_combo_lambda_uses_current_data_not_index(editor_env):
    """下拉信号经 currentData() 取常量值：选项文案与顺序变化不影响语义。"""
    dlg, pw = editor_env
    draw_combo = dlg.var_table.cellWidget(0, 3)
    # 逐个切换所有取值，断言 field 跟随（含往返回 auto）
    from src.core.data_types import DRAW_MODE_STEP_PRE
    for mode in (DRAW_MODE_STEP_POST, DRAW_MODE_STEP_PRE, DRAW_MODE_AUTO):
        draw_combo.setCurrentIndex(draw_combo.findData(mode))
        var_name = dlg.var_table.item(0, 1).data(Qt.ItemDataRole.UserRole)
        assert pw.curves[var_name].draw_mode == mode
