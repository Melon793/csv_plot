"""绘图变量编辑器 C4：6 列结构 / 样式摘要格 / Legend 名 / Factor / Offset /
右键重置 / 列宽引擎（D9=A2、D2/D3/D4/D7/D8、D1+D10）。

旧「绘制方式/数据点下拉」入口已撤（实现保留），样式类变更改走 legend
同款快捷面板共享入口 apply_curve_panel_picks（parity）。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QMenu

import src.ui.plot_variable_editor as ed_mod
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    SYMBOL_POLICY_ON,
    InvalidCurveRef,
)
from src.ui.plot_variable_editor import PlotVariableEditorDialog, _mode_combo
from src.ui.widgets import curve_style_panel as csp

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
    yield dlg, pw
    if dlg.isVisible():
        dlg.close()
    dlg.deleteLater()


def _var_at(dlg, row):
    return dlg.var_table.item(row, 1).data(Qt.ItemDataRole.UserRole)


class TestStructure:
    def test_six_columns_and_cell_types(self, editor_env):
        dlg, _pw = editor_env
        table = dlg.var_table
        assert table.columnCount() == 6
        labels = [
            table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
        ]
        assert labels == ["显示", "变量名", "Legend 名", "样式", "Factor", "Offset"]
        for row in range(table.rowCount()):
            assert isinstance(table.cellWidget(row, 2), QLineEdit)
            assert isinstance(table.cellWidget(row, 3), ed_mod._StyleCell)
            assert isinstance(table.cellWidget(row, 4), QDoubleSpinBox)
            assert isinstance(table.cellWidget(row, 5), QDoubleSpinBox)
            assert table.cellWidget(row, 4).isEnabled()

    def test_all_columns_interactive(self, editor_env):
        dlg, _pw = editor_env
        header = dlg.var_table.horizontalHeader()
        for col in range(6):
            assert header.sectionResizeMode(col) == header.ResizeMode.Interactive


class TestStylePanelParity:
    def test_panel_pick_draw_mode_applies_and_backfills(self, editor_env):
        """面板共享入口改绘制方式：ci 跟随 + 摘要格经 curves_changed 回填。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        csp.apply_curve_panel_picks(pw, var, {"draw_mode": DRAW_MODE_STEP_POST})
        ci = pw.curves[var]
        assert ci.draw_mode == DRAW_MODE_STEP_POST
        assert ci.curve.opts["stepMode"] == "right"
        assert "阶梯后" in dlg.var_table.cellWidget(0, 3).text.text()

    def test_panel_pick_symbol_applies(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        csp.apply_curve_panel_picks(pw, var, {"symbol_policy": SYMBOL_POLICY_ON})
        assert pw.curves[var].symbol_policy == SYMBOL_POLICY_ON
        assert "显示" in dlg.var_table.cellWidget(0, 3).text.text()

    def test_panel_pick_broadcasts_chinese(self, editor_env):
        dlg, pw = editor_env
        msgs = []
        pw.window()._show_status_message = lambda m: msgs.append(m)
        var = _var_at(dlg, 0)
        csp.apply_curve_panel_picks(pw, var, {"draw_mode": DRAW_MODE_STEP_POST})
        assert any("绘制方式" in m and "阶梯(后)" in m for m in msgs)

    def test_style_cell_click_opens_panel(self, editor_env, monkeypatch):
        dlg, pw = editor_env
        opened = []
        monkeypatch.setattr(
            ed_mod, "exec_curve_panel",
            lambda pw_, var, pos: opened.append(var),
        )
        dlg.var_table.cellWidget(0, 3).clicked.emit(_var_at(dlg, 0))
        # singleShot(0) 落地
        from tests.fixtures.waits import settle
        settle()
        assert opened == [_var_at(dlg, 0)]


class TestLegendNameAndTransform:
    def test_legend_name_edit_updates_ci_and_legend(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        edit = dlg.var_table.cellWidget(0, 2)
        edit.setText("扭矩A")
        edit.editingFinished.emit()
        assert pw.curves[var].display_name == "扭矩A"
        assert "扭矩A" in pw._multi_curve_manager._build_legend_html()
        assert "·名" in dlg.var_table.item(0, 1).text()

    def test_legend_name_empty_resets(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        edit = dlg.var_table.cellWidget(0, 2)
        edit.setText("扭矩A")
        edit.editingFinished.emit()
        edit.setText("")
        edit.editingFinished.emit()
        assert pw.curves[var].display_name == ""

    def test_factor_spin_projects_y(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_spin = dlg.var_table.cellWidget(0, 4)
        o_spin = dlg.var_table.cellWidget(0, 5)
        f_spin.setValue(2.0)
        o_spin.setValue(-10.0)
        ci = pw.curves[var]
        assert ci.factor == 2.0 and ci.offset == -10.0
        import numpy as np
        np.testing.assert_allclose(ci.y_data, 2.0 * ci.y_raw - 10.0)
        assert "·ƒ" in dlg.var_table.item(0, 1).text()

    def test_refresh_backfills_spins_without_loop(self, editor_env):
        """外部改换算 → curves_changed 就地回填 spinbox（ blocker 防回环）。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        assert pw._multi_curve_manager.apply_curve_transform(var, 3.0, 1.0)
        assert dlg.var_table.cellWidget(0, 4).value() == 3.0
        assert dlg.var_table.cellWidget(0, 5).value() == 1.0

    def test_enum_channel_spins_disabled(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        pw.curves[var].y_format = "enum"
        dlg.load_current_curves()
        row = dlg._row_of_var(var)
        assert not dlg.var_table.cellWidget(row, 4).isEnabled()
        assert not dlg.var_table.cellWidget(row, 5).isEnabled()


class TestInvalidRows:
    def test_invalid_row_cells(self, plot_factory, qapp):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        pw = plot_factory(df)
        host = pw.window()
        host.loader = types.SimpleNamespace(var_names=["a"], units={})
        host.data_validity = {}
        assert pw.add_variable_to_plot("a")
        pw.invalid_curves["ghost"] = InvalidCurveRef(
            var_name="ghost", display_name="旧名", factor=2.0, offset=1.0
        )
        dlg = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        row = dlg._row_of_var("ghost")
        assert row is not None
        assert dlg.var_table.cellWidget(row, 2).text() == "旧名"
        assert not dlg.var_table.cellWidget(row, 3).isEnabled()
        assert not dlg.var_table.cellWidget(row, 4).isEnabled()
        assert dlg.var_table.cellWidget(row, 4).value() == 2.0
        dlg.deleteLater()


class TestContextMenuReset:
    def test_reset_all_via_context_menu(self, editor_env, monkeypatch):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        pw._multi_curve_manager.apply_curve_display_name(var, "扭矩A")
        pw._multi_curve_manager.apply_curve_transform(var, 2.0, -10.0)

        captured = {}

        class _FakeMenu:
            def __init__(self, parent=None):
                self._acts = []

            def addAction(self, text):
                act = types.SimpleNamespace(text=text)
                self._acts.append(act)
                return act

            def exec(self, pos=None):
                captured["menu"] = self
                return self._acts[2]  # 重置该行全部

            def deleteLater(self):
                pass

        monkeypatch.setattr(ed_mod, "QMenu", _FakeMenu)
        dlg._on_row_context_menu(dlg.var_table.visualItemRect(
            dlg.var_table.item(0, 1)
        ).center())
        ci = pw.curves[var]
        assert ci.display_name == ""
        assert ci.factor == 1.0 and ci.offset == 0.0


class TestColumnWidthEngine:
    def test_drag_marks_user_fixed_and_excludes_elastic(self, editor_env):
        dlg, _pw = editor_env
        dlg._on_section_resized(1, 210, 300)
        assert 1 in dlg._user_fixed_cols
        dlg.var_table.setColumnWidth(1, 300)
        dlg.resize(1200, 600)
        dlg._distribute_column_widths()
        assert dlg.var_table.columnWidth(1) == 300  # 固定列不被弹性分配改写

    def test_double_click_restores_elastic(self, editor_env):
        dlg, _pw = editor_env
        dlg._on_section_resized(1, 210, 300)
        dlg._on_handle_double_clicked(1)
        assert 1 not in dlg._user_fixed_cols

    def test_min_width_clamp(self, editor_env):
        dlg, _pw = editor_env
        dlg._on_section_resized(0, 46, 5)
        assert dlg.var_table.columnWidth(0) >= dlg._col_min[0]

    def test_narrow_window_elastic_at_min_and_scroll(self, editor_env):
        dlg, _pw = editor_env
        dlg.resize(560, 400)
        # 定宽列加宽到超视口 → 弹性列落最小宽 + 横滚兜底
        dlg.var_table.setColumnWidth(3, 400)
        dlg._distribute_column_widths()
        for c in (1, 2):
            assert dlg.var_table.columnWidth(c) == dlg._col_min[c]
        total = sum(dlg.var_table.columnWidth(c) for c in range(6))
        assert total > dlg.var_table.viewport().width()

    def test_widths_persist_and_restore(self, editor_env, qapp):
        dlg, pw = editor_env
        dlg._on_section_resized(1, 210, 321)
        dlg.var_table.setColumnWidth(1, 321)
        dlg._save_column_widths()
        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        assert dlg2.var_table.columnWidth(1) == 321
        assert 1 in dlg2._user_fixed_cols
        dlg2.deleteLater()


class TestRetainedImpl:
    def test_mode_combo_still_bind_current_data(self):
        """撤入口不撤实现：_mode_combo 保留，itemData 绑常量语义不变。"""
        seen = []
        combo = _mode_combo(
            ed_mod._DRAW_MODE_OPTIONS, DRAW_MODE_AUTO,
            lambda v, label: seen.append(v),
        )
        combo.setCurrentIndex(combo.findData(DRAW_MODE_STEP_POST))
        assert seen == [DRAW_MODE_STEP_POST]
        combo.deleteLater()