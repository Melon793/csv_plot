"""编辑器 Factor/Offset 文本输入（D17 / R2）。

真机缺陷：QDoubleSpinBox 的 decimals 是控件级设置，恒定显示 1.000000；
且 ↑↓/滚轮属于误触面。改为 QLineEdit 子类：严格正则、失焦/Enter 主动
提交、非法回退、去尾零显示。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import src.ui.plot_variable_editor as ed_mod
from src.core.settings import ConfigKey
from src.ui.plot_variable_editor import PlotVariableEditorDialog, format_number

NAMES = ["a", "b"]


@pytest.fixture()
def editor_env(plot_factory, qapp, app_settings):
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
    dlg._width_save_timer.stop()
    if dlg.isVisible():
        dlg.close()
    dlg.deleteLater()
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS, "")
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")
    app_settings.sync()


def _edits(dlg, row=0):
    return dlg.var_table.cellWidget(row, 4), dlg.var_table.cellWidget(row, 5)


def _var_at(dlg, row):
    return dlg.var_table.item(row, 1).data(Qt.ItemDataRole.UserRole)


class TestFormatNumber:
    @pytest.mark.parametrize(
        "value,text",
        [
            (1.0, "1"),
            (0.0, "0"),
            (-0.0, "0"),
            (0.5, "0.5"),
            (2.50, "2.5"),
            (1234.5, "1234.5"),
            (1e12, "1000000000000"),
            (1e-13, "0"),  # 已知边界（评审 P2-11）
            (None, "0"),
        ],
    )
    def test_format(self, value, text):
        assert format_number(value) == text


class TestDefaultDisplay:
    def test_default_text_is_1_and_0(self, editor_env):
        dlg, _pw = editor_env
        f_edit, o_edit = _edits(dlg)
        assert f_edit.text() == "1"
        assert o_edit.text() == "0"

    def test_no_spin_stepping(self, editor_env):
        """↑↓ 与滚轮不改值（D17：只打字，无步进）。"""
        dlg, _pw = editor_env
        f_edit, _o = _edits(dlg)
        f_edit.setFocus()
        QTest.keyClick(f_edit, Qt.Key.Key_Up)
        QTest.keyClick(f_edit, Qt.Key.Key_Down)
        wheel = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(f_edit, wheel)
        assert f_edit.text() == "1"


class TestCommit:
    def test_typing_preserves_precision(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("0.333")
        f_edit.commit()
        assert pw.curves[var].factor == pytest.approx(0.333)
        assert f_edit.text() == "0.333"

    def test_enter_commits_and_normalizes(self, editor_env):
        """Enter 提交 "1." → factor=1.0，文本回写去尾零。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("1.")
        QTest.keyClick(f_edit, Qt.Key.Key_Return)
        assert pw.curves[var].factor == 1.0
        assert f_edit.text() == "1"

    def test_invalid_text_reverts_on_commit(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("abc")
        f_edit.commit()
        assert pw.curves[var].factor == 1.0
        assert f_edit.text() == "1"

    def test_empty_text_reverts_on_focus_out(self, editor_env):
        """空文本失焦回退上次合法值（editingFinished 在 Invalid 态不触发）。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("2.5")
        f_edit.commit()
        assert pw.curves[var].factor == 2.5
        f_edit.setText("")
        f_edit.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
        assert pw.curves[var].factor == 2.5
        assert f_edit.text() == "2.5"

    def test_comma_not_turned_into_15(self, editor_env):
        """评审 P0-2 回归：QDoubleValidator 的 locale fixup 会把 "1,5" 变 15。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("1,5")
        f_edit.commit()
        assert pw.curves[var].factor == 1.0, "逗号文本必须回退，不得解析成 15"
        assert f_edit.text() == "1"

    def test_value_clamped_to_limit(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setText("9999999999999")
        f_edit.commit()
        assert pw.curves[var].factor == 1e12
        assert f_edit.text() == "1000000000000"

    def test_negative_and_decimal(self, editor_env):
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        f_edit, o_edit = _edits(dlg)
        f_edit.setText("-.25")
        f_edit.commit()
        o_edit.setText("-10.50")
        o_edit.commit()
        ci = pw.curves[var]
        assert ci.factor == -0.25
        assert ci.offset == -10.5
        assert o_edit.text() == "-10.5"

    def test_enter_does_not_close_dialog(self, editor_env):
        """评审 P2-10：格内 Enter 必须被消费，不得触发默认「确定」关窗。"""
        dlg, _pw = editor_env
        dlg.show()
        QApplication.processEvents()
        f_edit, _o = _edits(dlg)
        f_edit.setFocus()
        QTest.keyClick(f_edit, Qt.Key.Key_Return)
        QApplication.processEvents()
        assert dlg.isVisible()
        # autoDefault 双保险：焦点不在输入框时 Enter 也不应直接接受对话框
        assert not dlg.ok_btn.autoDefault()
        assert not dlg.cancel_btn.autoDefault()


class TestBackfill:
    def test_external_transform_backfills_without_second_apply(
        self, editor_env, monkeypatch
    ):
        """外部改换算 → 回填两框，且回填不再触发第二次 apply（无回环）。"""
        dlg, pw = editor_env
        var = _var_at(dlg, 0)
        calls = []
        mcm = pw._multi_curve_manager
        orig = mcm.apply_curve_transform

        def spy(*args, **kwargs):
            calls.append(args)
            return orig(*args, **kwargs)

        monkeypatch.setattr(mcm, "apply_curve_transform", spy)
        assert mcm.apply_curve_transform(var, 3.0, 1.5)
        f_edit, o_edit = _edits(dlg)
        assert f_edit.text() == "3"
        assert o_edit.text() == "1.5"
        assert len(calls) == 1, "回填不得再走 apply（防回环）"

    def test_backfill_skips_focused_editor(self, editor_env):
        """评审 P1-8：正在输入的框不被程序化回填打断。"""
        dlg, pw = editor_env
        dlg.show()
        QApplication.processEvents()
        var = _var_at(dlg, 0)
        f_edit, _o = _edits(dlg)
        f_edit.setFocus()
        QApplication.processEvents()
        assert f_edit.hasFocus()
        f_edit.setText("1.2")
        pw._multi_curve_manager.apply_curve_transform(var, 7.0, 0.0)
        assert f_edit.text() == "1.2", "聚焦中的输入不得被回填覆盖"


class TestDisabledRows:
    def test_invalid_row_edits_disabled_showing_saved_value(self, editor_env):
        dlg, pw = editor_env
        from src.core.data_types import InvalidCurveRef

        pw.invalid_curves["ghost"] = InvalidCurveRef(
            var_name="ghost", factor=2.0, offset=1.0
        )
        dlg.load_current_curves()
        row = dlg._row_of_var("ghost")
        f_edit, o_edit = _edits(dlg, row)
        assert not f_edit.isEnabled()
        assert not o_edit.isEnabled()
        assert f_edit.text() == "2"
        assert o_edit.text() == "1"