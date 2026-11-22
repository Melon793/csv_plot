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

    def test_transform_col_drag_rejected(self, editor_env):
        """评审 P2-2：换算列拖拽/双击被压回定宽，且不入固定列（防重开闪回）。"""
        dlg, _pw = editor_env
        for col in (4, 5):
            dlg._on_section_resized(col, 84, 300)
            assert dlg.var_table.columnWidth(col) == ed_mod._TRANSFORM_COL_WIDTH
            assert col not in dlg._user_fixed_cols
            dlg._on_handle_double_clicked(col)
            assert dlg.var_table.columnWidth(col) == ed_mod._TRANSFORM_COL_WIDTH

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
    def test_archive_still_clamps_transform_cols(self, editor_env, app_settings, qapp):
        """评审 P0-1：旧档里的 200px 膨胀值必须压回定宽，否则自适应永不生效。"""
        dlg, pw = editor_env
        dlg.close()
        app_settings.set_plot_variable_editor_column_widths(
            [46, 210, 170, 168, 200, 200]
        )
        app_settings.set_plot_variable_editor_fixed_cols([1, 3, 4, 5])
        dlg2 = PlotVariableEditorDialog(pw)
        qapp.processEvents()
        try:
            assert dlg2.var_table.columnWidth(4) == ed_mod._TRANSFORM_COL_WIDTH
            assert dlg2.var_table.columnWidth(5) == ed_mod._TRANSFORM_COL_WIDTH
            assert dlg2.var_table.columnWidth(3) == 168
            assert dlg2._columns_sized is True
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