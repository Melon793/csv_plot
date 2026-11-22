"""X2（F3+F5）：面板改色即时回填编辑器 + 显隐共享入口与 emit 口径。

真机缺陷：图例/快捷面板改色后编辑器样式格色块不刷新（唯一不 emit
curves_changed 的 style 入口）；编辑器复选框直写 visible，与 legend/轴脱节。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from src.core.data_types import InvalidCurveRef
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from src.ui.widgets import curve_style_panel as csp

NAMES = ["a", "b"]


@pytest.fixture()
def editor_env(plot_factory, qapp, app_settings):
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


def _var_at(dlg, row=0):
    return dlg.var_table.item(row, 1).data(Qt.ItemDataRole.UserRole)


def _y_limits(pw):
    return pw.view_box.state["limits"]["yLimits"]


class TestColorRefresh:
    def test_panel_color_pick_updates_editor_swatch(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg)
        csp.apply_curve_panel_picks(pw, var, {"color": "#123456"})
        cell = dlg.var_table.cellWidget(0, 3)
        assert "#123456" in cell.swatch.styleSheet()
        assert pw.curves[var].color == "#123456"

    def test_color_pick_emits_once(self, editor_env):
        dlg, pw = editor_env
        calls = []
        pw.curves_changed.connect(lambda: calls.append(1))
        csp.apply_curve_panel_picks(pw, _var_at(dlg), {"color": "red"})
        assert len(calls) == 1

    def test_reset_colors_emits_once(self, editor_env):
        """N 条逐色改色包在批量窗口内，收尾只 emit 一次（评审 P1-10）。"""
        dlg, pw = editor_env
        calls = []
        pw.curves_changed.connect(lambda: calls.append(1))
        dlg.reset_curve_colors()
        assert len(calls) == 1
        assert len({ci.color for ci in pw.curves.values()}) >= 1


class TestVisibilitySharedEntry:
    def test_checkbox_goes_through_shared_entry(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg)
        calls = []
        pw.curves_changed.connect(lambda: calls.append(1))

        dlg.var_table.cellWidget(0, 0).setChecked(False)

        assert pw.curves[var].visible is False
        assert not pw.curves[var].curve.isVisible()
        assert len(calls) == 1, "复选框不得直写 visible 绕过共享入口"
        assert dlg.var_table.cellWidget(0, 0).isChecked() is False

    def test_checkbox_recomputes_y_limits(self, editor_env):
        """隐藏一条曲线后 limits 必须按剩余可见数据收缩（评审 P1-8）。"""
        dlg, pw = editor_env
        assert pw.curves[_var_at(dlg, 1)].visible is True
        before = _y_limits(pw)
        assert before[1] >= 2.0

        dlg.var_table.cellWidget(1, 0).setChecked(False)

        after = _y_limits(pw)
        assert after[1] == pytest.approx(1.5, abs=1e-9)

    def test_toggle_and_checkbox_parity(self, editor_env):
        """图例切换与编辑器复选框同走 _apply_visibility：各 emit 一次。"""
        dlg, pw = editor_env
        var = _var_at(dlg)
        calls = []
        pw.curves_changed.connect(lambda: calls.append(1))

        pw._multi_curve_manager.toggle_curve_visibility_by_name(var)
        assert pw.curves[var].visible is False
        assert len(calls) == 1

        calls.clear()
        pw._multi_curve_manager.toggle_curve_visibility_by_name(var)
        assert pw.curves[var].visible is True
        assert len(calls) == 1

    def test_apply_curve_visibility_noop_returns_false(self, editor_env):
        _dlg, pw = editor_env
        var = NAMES[0]
        assert pw._multi_curve_manager.apply_curve_visibility(var, True) is False
        assert pw._multi_curve_manager.apply_curve_visibility(var, False) is True


class TestInvalidVisibility:
    def test_invalid_checkbox_writes_preference_only(self, plot_factory, qapp):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        pw = plot_factory(df)
        host = pw.window()
        host.loader = types.SimpleNamespace(var_names=["a"], units={})
        host.data_validity = {}
        assert pw.add_variable_to_plot("a")
        pw.invalid_curves["ghost"] = InvalidCurveRef(
            var_name="ghost", visible_preference=True
        )
        dlg = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            calls = []
            pw.curves_changed.connect(lambda: calls.append(1))
            row = dlg._row_of_var("ghost")
            assert row is not None
            dlg.var_table.cellWidget(row, 0).setChecked(False)
            assert pw.invalid_curves["ghost"].visible_preference is False
            assert calls == []
        finally:
            dlg.deleteLater()


class TestRawValueTooltips:
    def test_table_header_marks_raw_values(self):
        from src.ui.table_dialog import PandasTableModel

        model = PandasTableModel(pd.DataFrame({"a": [1.0]}), {"a": "km/h"})
        tip = model.headerData(
            0, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole
        )
        assert "原始值" in tip
        assert "km/h" in tip