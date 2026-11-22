"""编辑器列宽记忆与排版（X1：F1 存盘/跨会话 + F2 去箭头/定宽）。

真机缺陷：拖列宽关窗重开恢复默认（ini 列表跨进程退化 + 只 closeEvent 存盘）；
Factor/Offset 被极值文本 sizeHint 撑宽且带无意义箭头。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox

import src.ui.plot_variable_editor as ed_mod
from src.core.settings import AppSettings, ConfigKey
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from tests.fixtures.waits import wait_until

NAMES = ["a", "b"]


@pytest.fixture()
def editor_env(plot_factory, qapp, app_settings):
    # ini 残留跨用例污染（fixture 只重置单例不删 ini），先显式清键
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")
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
    # 评审 P1-1：teardown close() 会经 closeEvent 把 333/固定列写回；配置目录
    # 是会话级共享，不清键会污染同 worker 后续用例（顺序依赖）。显式回干净态。
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")
    app_settings.sync()


class TestTransformColumnLayout:
    def test_no_buttons_and_narrow(self, editor_env):
        dlg, _pw = editor_env
        for col in (4, 5):
            spin = dlg.var_table.cellWidget(0, col)
            assert spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
            assert spin.minimumWidth() == 60
            assert dlg.var_table.columnWidth(col) == ed_mod._TRANSFORM_COL_WIDTH

    def test_transform_col_drag_keeps_width_and_persists(self, editor_env, qapp):
        """D16（反转 v1 P2-2）：换算列可拖；拖过即固定 + 落盘，重开恢复。"""
        dlg, pw = editor_env
        dlg.resize(1100, 600)
        for col in (4, 5):
            dlg._on_section_resized(col, ed_mod._TRANSFORM_COL_WIDTH, 150)
            dlg.var_table.setColumnWidth(col, 150)
        assert dlg.var_table.columnWidth(4) == 150
        assert dlg.var_table.columnWidth(5) == 150
        assert {4, 5} <= dlg._user_fixed_cols
        dlg._save_column_widths()

        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            assert dlg2.var_table.columnWidth(4) == 150
            assert dlg2.var_table.columnWidth(5) == 150
            assert {4, 5} <= dlg2._user_fixed_cols
        finally:
            dlg2.deleteLater()

    def test_transform_col_drag_clamps_min_width(self, editor_env):
        """换算列拖到过窄时夹回列最小宽。"""
        dlg, _pw = editor_env
        dlg._on_section_resized(4, ed_mod._TRANSFORM_COL_WIDTH, 10)
        assert dlg.var_table.columnWidth(4) == dlg._col_min[4]

    def test_transform_col_double_click_restores_84_and_saves(
        self, editor_env, app_settings
    ):
        """双击换算列：回默认宽 84 + 退出固定列 + 落盘（评审 P1-5）。"""
        dlg, _pw = editor_env
        dlg._on_section_resized(4, ed_mod._TRANSFORM_COL_WIDTH, 150)
        dlg.var_table.setColumnWidth(4, 150)
        dlg._on_handle_double_clicked(4)
        assert dlg.var_table.columnWidth(4) == ed_mod._TRANSFORM_COL_WIDTH
        assert 4 not in dlg._user_fixed_cols
        assert dlg._width_save_timer.isActive()
        dlg._width_save_timer.stop()
        dlg._save_column_widths()
        assert AppSettings().get_plot_variable_editor_column_widths()[4] == (
            ed_mod._TRANSFORM_COL_WIDTH
        )
        assert 4 not in (AppSettings().get_plot_variable_editor_fixed_cols() or [])

    def test_huge_value_does_not_widen_column(self, editor_env):
        """极值文本（1234.500000）不得把列撑回 sizeHint 宽度。"""
        dlg, pw = editor_env
        var = dlg.var_table.item(0, 1).data(Qt.ItemDataRole.UserRole)
        assert pw._multi_curve_manager.apply_curve_transform(var, 1234.5, -999.25)
        dlg._adjust_fixed_column_widths()
        assert dlg.var_table.columnWidth(4) == ed_mod._TRANSFORM_COL_WIDTH
        assert dlg.var_table.columnWidth(5) == ed_mod._TRANSFORM_COL_WIDTH

    def test_elastic_min_independent_constants(self, editor_env):
        dlg, _pw = editor_env
        assert set(ed_mod._ELASTIC_MIN).isdisjoint(ed_mod._FIXED_MIN_WIDTHS)
        assert set(ed_mod._ELASTIC_MIN) == {1, 2}
        assert dlg._col_min[1] >= ed_mod._ELASTIC_MIN[1]
        assert dlg._col_min[2] >= ed_mod._ELASTIC_MIN[2]


class TestArchiveRestore:
    def test_archive_still_clamps_unfixed_transform_cols(
        self, editor_env, app_settings, qapp
    ):
        """D16：未固定换算列仍压回 84（旧档 sizeHint 膨胀值不恢复）。"""
        dlg, pw = editor_env
        dlg.close()
        app_settings.set_plot_variable_editor_column_widths(
            [46, 210, 170, 168, 200, 200]
        )
        app_settings.set_plot_variable_editor_fixed_cols([1, 3])
        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            assert dlg2.var_table.columnWidth(4) == ed_mod._TRANSFORM_COL_WIDTH
            assert dlg2.var_table.columnWidth(5) == ed_mod._TRANSFORM_COL_WIDTH
            assert dlg2.var_table.columnWidth(3) == 168
            assert dlg2._columns_sized is True
        finally:
            dlg2.deleteLater()

    def test_archive_restores_dragged_transform_col(
        self, editor_env, app_settings, qapp
    ):
        """D16：存档固定列含 4 时恢复存档宽；未固定的 5 仍回 84。"""
        dlg, pw = editor_env
        dlg.close()
        app_settings.set_plot_variable_editor_column_widths(
            [46, 210, 170, 168, 150, 200]
        )
        app_settings.set_plot_variable_editor_fixed_cols([4])
        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            assert dlg2.var_table.columnWidth(4) == 150
            assert dlg2.var_table.columnWidth(5) == ed_mod._TRANSFORM_COL_WIDTH
            assert 4 in dlg2._user_fixed_cols
            assert 5 not in dlg2._user_fixed_cols
        finally:
            dlg2.deleteLater()

    def test_empty_fixed_archive_compatible(self, editor_env, app_settings, qapp):
        """settings 返回空固定列列表：等同"未拖过"，换算列回 84。"""
        dlg, pw = editor_env
        dlg.close()
        app_settings.set_plot_variable_editor_column_widths(
            [46, 210, 170, 168, 200, 200]
        )
        app_settings.set_plot_variable_editor_fixed_cols([])
        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            assert dlg2._user_fixed_cols == set()
            assert dlg2.var_table.columnWidth(4) == ed_mod._TRANSFORM_COL_WIDTH
            assert dlg2.var_table.columnWidth(5) == ed_mod._TRANSFORM_COL_WIDTH
        finally:
            dlg2.deleteLater()


class TestSaveOnEveryClosePath:
    @pytest.mark.parametrize("shutdown", ["accept", "reject", "close"])
    def test_width_persists_on_done_and_close(self, editor_env, qapp, shutdown):
        dlg, pw = editor_env
        dlg._on_section_resized(1, 210, 333)
        dlg.var_table.setColumnWidth(1, 333)
        getattr(dlg, shutdown)()
        qapp.processEvents()

        dlg2 = PlotVariableEditorDialog(pw)
        dlg2.show()
        qapp.processEvents()
        try:
            assert dlg2.var_table.columnWidth(1) == 333
            assert 1 in dlg2._user_fixed_cols
        finally:
            dlg2.deleteLater()

    def test_debounced_save_without_closing(self, editor_env, app_settings):
        """主窗销毁不走 closeEvent：拖宽后 400ms 内必须已落盘。"""
        dlg, _pw = editor_env
        dlg._width_save_timer.setInterval(10)
        dlg._on_section_resized(1, 210, 321)
        dlg.var_table.setColumnWidth(1, 321)
        ok = wait_until(
            lambda: AppSettings().get_plot_variable_editor_column_widths() is not None
            and AppSettings().get_plot_variable_editor_column_widths()[1] == 321,
            timeout_s=3.0,
        )
        assert ok, "拖宽后 debounce 未落盘"
        assert AppSettings().get_plot_variable_editor_fixed_cols() == [1]


class TestDefaultWidthUnchanged:
    def test_initial_window_unchanged_by_layout(self, editor_env):
        """默认初宽仍按定宽列实际宽 + 余量计算（约 800px 级）。"""
        dlg, pw = editor_env
        host = pw.window()
        fixed = sum(dlg.var_table.columnWidth(c) for c in (0, 3, 4, 5))
        expected = min(fixed + 380 + 24 + 40, int(host.width() * 0.9))
        assert dlg.width() == expected