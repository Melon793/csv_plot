"""编辑器布局：首拖不跳列（R3）与样式格垂直居中（R4）。

R3：列宽分布只在 resizeEvent 执行；构造/恢复几何后视口尺寸已变而未再
分布，用户第一次拖窗宽才触发比例分配 → 弹性列瞬间跳变。showEvent 后
补一次分布（singleShot(0) 等布局落定）。
R4：_StyleCell 曾 setFixedHeight(28)，溢出行高（约 30）使内容整体偏下；
改为填满单元格 + AlignVCenter。像素断言统一用 viewport 坐标系（P0-3）。
R5（D20）：行高钉在 _ROW_HEIGHT，行内输入框高度由行高决定（框高 = 行高 −
1 − 2×item padding），太矮会把 QLineEdit 压扁到字体行高上下只剩 1px。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import src.ui.plot_variable_editor as ed_mod
from src.core.settings import ConfigKey
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from tests.fixtures.waits import settle

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


def _elastic_total(dlg) -> int:
    return sum(dlg.var_table.columnWidth(c) for c in ed_mod._ELASTIC_COLS)


def _elastic_avail(dlg) -> int:
    table = dlg.var_table
    fixed_sum = sum(
        table.columnWidth(c)
        for c in range(ed_mod._COL_COUNT)
        if c not in ed_mod._ELASTIC_COLS
    )
    return table.viewport().width() - fixed_sum


class TestShowDistribution:
    def test_show_fills_elastic_columns(self, editor_env, qapp):
        """show 后弹性列已按视口铺满，不再停留默认 210/170。"""
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        avail = _elastic_avail(dlg)
        assert avail > 0
        assert abs(_elastic_total(dlg) - avail) <= 2, "弹性列应铺满视口余量"
        for col in ed_mod._ELASTIC_COLS:
            assert dlg.var_table.columnWidth(col) > ed_mod._COL_DEFAULT_WIDTHS[col]

    def test_first_resize_after_show_has_no_jump(self, editor_env, qapp):
        """首拖窗宽不再跳列：每 10px 步进下弹性列宽连续变化。"""
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        table = dlg.var_table
        prev = [table.columnWidth(c) for c in range(ed_mod._COL_COUNT)]
        for width in range(910, 1000, 10):
            dlg.resize(width, 600)
            settle()
            cur = [table.columnWidth(c) for c in range(ed_mod._COL_COUNT)]
            deltas = [abs(a - b) for a, b in zip(prev, cur)]
            assert max(deltas) <= 10, f"窗宽 {width} 出现列宽跳变: {deltas}"
            prev = cur

    def test_show_keeps_archived_fixed_columns(
        self, editor_env, app_settings, qapp
    ):
        """show 补分布不得改写用户固定列存档宽。"""
        dlg, pw = editor_env
        dlg.close()
        app_settings.set_plot_variable_editor_column_widths(
            [46, 321, 170, 168, 84, 84]
        )
        app_settings.set_plot_variable_editor_fixed_cols([1])
        dlg2 = PlotVariableEditorDialog(pw)
        dlg2.resize(900, 600)
        dlg2.show()
        settle()
        try:
            assert dlg2.var_table.columnWidth(1) == 321
        finally:
            dlg2.close()
            dlg2.deleteLater()


class TestRowHeightAndInputBox:
    def test_row_height_keeps_input_box_off_squashed(self, editor_env, qapp):
        """D20：行高钉在 _ROW_HEIGHT，行内输入框不再被压到字体行高上下 1px。

        框高由表格决定（单元格高 − 2×::item padding 6px），与 QLineEdit 自身的
        sizeHint 无关；旧 30px 行高把框压到 17px（字体行高 15px，上下各 1px）。
        这里只钉"高度杠杆生效 + 上下至少各留 2px"，不拍绝对像素（三平台字体不同）。
        """
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        table = dlg.var_table
        assert table.rowHeight(0) == ed_mod._ROW_HEIGHT
        for col in (2, 4, 5):
            edit = table.cellWidget(0, col)
            assert edit is not None
            assert edit.height() > 20, (
                f"列 {col} 输入框高 {edit.height()}，仍被压扁"
            )
            assert edit.height() >= edit.fontMetrics().height() + 4, (
                f"列 {col} 输入框 {edit.height()}px 装不下 {edit.fontMetrics().height()}px 文本"
            )


class TestStyleCellAlignment:
    def test_style_cell_centered_not_fixed_height(self, editor_env, qapp):
        """样式格不再定高 28；控件按 sizeHint 居中（格中心 == 行中心）。

        Qt 对 index widget 按 sizeHint 定尺寸并在单元格内居中；旧版固定高 28
        使内容中心比行中心低 5px（真机缺陷），此处只钉"无定高 + 居中"。
        """
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        table = dlg.var_table
        viewport = table.viewport()
        row_center = table.visualItemRect(table.item(0, 1)).center().y()
        cell = table.cellWidget(0, 3)
        assert cell.minimumHeight() < 28, "不得残留 setFixedHeight(28)"
        assert cell.height() <= table.rowHeight(0)
        cell_center = cell.mapTo(viewport, cell.rect().center()).y()
        assert abs(cell_center - row_center) <= 2, (
            f"样式格中心 {cell_center} 偏离行中心 {row_center}"
        )

    def test_swatch_and_text_centered_in_row(self, editor_env, qapp):
        """色块/文字与行中心、邻列 QLineEdit 中心同高（viewport 坐标 ≤2px）。"""
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        table = dlg.var_table
        viewport = table.viewport()
        row_center = table.visualItemRect(table.item(0, 1)).center().y()
        cell = table.cellWidget(0, 3)
        swatch_center = cell.swatch.mapTo(
            viewport, cell.swatch.rect().center()
        ).y()
        text_center = cell.text.mapTo(viewport, cell.text.rect().center()).y()
        edit = table.cellWidget(0, 2)
        edit_center = edit.mapTo(viewport, edit.rect().center()).y()
        assert abs(swatch_center - row_center) <= 2, (
            f"色块中心 {swatch_center} 偏离行中心 {row_center}"
        )
        assert abs(text_center - row_center) <= 2, (
            f"文字中心 {text_center} 偏离行中心 {row_center}"
        )
        assert abs(swatch_center - edit_center) <= 2, (
            f"色块中心 {swatch_center} 与邻列输入框中心 {edit_center} 不同高"
        )

    def test_style_cell_item_band_click_opens_panel(
        self, editor_env, qapp, monkeypatch
    ):
        """评审 P1-2：样式格控件只覆盖单元格内容区（约 25px），行内其余条带点击也要开面板。"""
        dlg, _pw = editor_env
        dlg.resize(900, 600)
        dlg.show()
        settle()
        opened = []
        monkeypatch.setattr(
            ed_mod, "exec_curve_panel",
            lambda pw_, var, pos: opened.append(var),
        )
        table = dlg.var_table
        viewport = table.viewport()
        cell = table.cellWidget(0, 3)
        geom = cell.geometry()
        click_y = geom.bottom() + 3
        assert click_y < table.rowHeight(0), "行内控件下方应还有条带"
        pos = QPoint(geom.center().x(), click_y)
        QTest.mouseClick(
            viewport, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, pos,
        )
        settle()
        var = table.item(0, 1).data(Qt.ItemDataRole.UserRole)
        assert opened == [var]