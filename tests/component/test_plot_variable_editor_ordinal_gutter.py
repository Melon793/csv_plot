"""绘图变量编辑器：行号槽钉宽 + viewport 重分布防回归（D1/D3）。

覆盖：
- 行号槽最少 2 位、空/满两态宽度与 viewport 一致（列位不跳）
- 空表 → 加 1 条 → 加 30 条（竖滚出现）不再挤出尘埃横滚条
- 行数到 3 位数（100 行）槽宽扩一档

夹具注意（评审 P0）：
- 必须清列宽/固定列存档：``test_plot_variable_editor_style_transform`` 会把
  300/333 宽写回，同进程后续用例会随机拿到脏存档（AppSettings 单例）。
- 必须把对话框拉到 900×600：FakeHost 默认 640 宽 → 视口 ~533 小于
  定宽列+弹性最小宽，此时横滚条是既有"最小宽兜底"，与本次缺陷无关。
- 每次改行/改尺寸后 ``settle(3)``：重分布要经过"横滚显隐 → viewport
  Resize → 再调度"的级联，单次 processEvents 不够。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import QPoint, Qt

from src.core.settings import ConfigKey
from src.ui.plot_variable_editor import (
    _ORDINAL_GUTTER_PAD,
    _ORDINAL_MIN_DIGITS,
    PlotVariableEditorDialog,
)
from tests.fixtures.waits import flush_deferred_deletes, settle

NAMES = [f"var_{i}" for i in range(30)]


@pytest.fixture()
def make_editor(plot_factory, qapp, app_settings):
    created = []
    for key in (
        ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY,
        ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS,
        ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS,
    ):
        app_settings.set_value(key, "")

    def _make():
        # 数据列必须覆盖 NAMES：缺列时 add_variable_to_plot 会弹 QMessageBox
        # 阻塞用例（tests/README.md 陷阱 #1）
        df = pd.DataFrame({name: [1.0, 2.0] for name in NAMES})
        pw = plot_factory(df)
        host = pw.window()
        host.loader = types.SimpleNamespace(var_names=list(NAMES), units={})
        host.data_validity = {}
        dlg = PlotVariableEditorDialog(pw)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dlg.resize(900, 600)
        dlg.show()
        settle()
        created.append(dlg)
        return dlg, pw

    yield _make
    for dlg in created:
        if dlg.isVisible():
            dlg.close()
        dlg.deleteLater()
    flush_deferred_deletes()


def _expected_gutter(header, digits: int) -> int:
    return header.fontMetrics().horizontalAdvance("9" * digits) + _ORDINAL_GUTTER_PAD


def _header_left_x(dlg) -> int:
    """「显示」列左边缘的窗口坐标：行号槽出现时它会右移。"""
    header = dlg.var_table.horizontalHeader()
    return header.mapTo(dlg, QPoint(0, 0)).x()


def test_gutter_keeps_min_two_digits_on_empty(make_editor):
    """空表：槽宽按最少 2 位钉死，不写死像素（按 header 字体度量算）。"""
    dlg, _pw = make_editor()
    header = dlg.var_table.verticalHeader()
    assert _ORDINAL_MIN_DIGITS == 2
    assert header.width() == _expected_gutter(header, _ORDINAL_MIN_DIGITS)


def test_gutter_width_and_viewport_constant_after_first_row(make_editor):
    """空表 → 加 1 条：槽宽、viewport 宽度、显示列左边缘都不动。"""
    dlg, pw = make_editor()
    table = dlg.var_table
    gutter = table.verticalHeader()
    width0 = gutter.width()
    viewport0 = table.viewport().width()
    left0 = _header_left_x(dlg)

    assert pw.add_variable_to_plot("var_0")
    settle()

    assert table.rowCount() == 1
    assert gutter.width() == width0
    assert table.viewport().width() == viewport0
    assert _header_left_x(dlg) == left0


def test_no_horizontal_overflow_after_rows_appear(make_editor):
    """空表 → 1 条 → 30 条（竖滚出现）：横滚条最大值保持 0。"""
    dlg, pw = make_editor()
    table = dlg.var_table
    assert table.horizontalScrollBar().maximum() == 0

    assert pw.add_variable_to_plot(NAMES[0])
    settle()
    assert table.horizontalScrollBar().maximum() == 0

    for name in NAMES[1:]:
        assert pw.add_variable_to_plot(name)
    settle()
    assert table.rowCount() == len(NAMES)
    assert table.horizontalScrollBar().maximum() == 0


def test_gutter_widens_at_three_digits(make_editor):
    """补到 100 行：槽宽扩一档到 3 位数口径，且仍无横滚。"""
    dlg, _pw = make_editor()
    table = dlg.var_table
    gutter = table.verticalHeader()
    two_digit = _expected_gutter(gutter, 2)

    while table.rowCount() < 100:
        table.insertRow(table.rowCount())
    settle()

    assert gutter.width() == _expected_gutter(gutter, 3)
    assert gutter.width() > two_digit
    assert table.horizontalScrollBar().maximum() == 0
