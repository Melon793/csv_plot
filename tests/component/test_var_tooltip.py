"""变量名悬停提示（长名截断时的全文出口）。

覆盖：公共文案 helper、数值表表头（含 tab 模式重建）、数值表 tab 定位下拉、
主窗口变量列表（全量填充 + 排序后重建两条路径）、编辑器左表（正常行 +
缺失占位行）、搜索候选列表（含增量标记不冲掉提示）。
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from PySide6.QtCore import QObject, Qt, Signal

from src.ui import theme
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from src.ui.table_dialog import PandasTableModel
from src.ui.tooltip_text import var_tooltip
from src.ui.variable_list import MyTableWidget
from src.ui.widgets.variable_search_bar import VariableSearchBar

LONG = "EGRVIv_uEIStc_mass_flow_correction_factor_bank2_sensor_primary"


# ---------------------------------------------------------------- 公共文案
def test_tooltip_with_unit():
    text = var_tooltip("AI05_2", "deg")
    assert text.startswith("<qt>")
    assert "<b>AI05_2</b>" in text  # 名称是主体
    assert "单位: deg" in text
    assert theme.TEXT_MUTED in text  # 单位退为次级灰
    assert "font-size: 11px" in text


def test_tooltip_style_is_compact_card():
    """QSS 只补留白与配色，外形（边框/圆角）交给平台：

    QToolTip 浮层是不透明方窗，自绘圆角会在方窗里露出直角（直角套圆角）。
    紧凑档实测：mac 6.5pt 四边，Windows 6~8px @96dpi。
    """
    qss = theme.TOOLTIP_QSS
    assert "padding: 1px 2px;" in qss
    assert f"background: {theme.BG};" in qss
    assert "border-radius" not in qss
    assert "border:" not in qss


def test_tooltip_qss_only_targets_tooltip():
    """护栏：应用级样式表只允许命中原生绘制之外的 QToolTip。

    theme.py 的既有决策是「不给原生控件挂 QSS，否则不再跟随平台绘制」；
    这条断言锁住"规则只有 QToolTip 一个选择器"。
    """
    rule = theme.TOOLTIP_QSS.split("{", 1)[0].strip()
    assert rule == "QToolTip", f"应用级 QSS 引入了别的选择器: {rule!r}"
    assert theme.TOOLTIP_QSS.count("{") == 1


@pytest.mark.parametrize("unit", ["", "-", None])
def test_tooltip_without_unit_is_single_line(unit):
    text = var_tooltip("AI05_2", unit)
    assert "AI05_2" in text
    assert "单位" not in text
    assert "<br>" not in text


def test_tooltip_escapes_markup():
    """变量名含尖括号/& 时按原文显示，不被当标签解析。"""
    text = var_tooltip("A<B>&C", "deg")
    assert "&lt;" in text and "&amp;" in text
    assert "<B>" not in text


# ---------------------------------------------------------------- 数值表表头
def test_header_tooltip_has_full_name_and_unit():
    df = pd.DataFrame({LONG: [1.0], "AI05_2": [2.0]})
    model = PandasTableModel(df, {LONG: "deg", "AI05_2": "deg"})

    tip = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
    assert LONG in tip and "单位: deg" in tip

    # 未改 DisplayRole：表头仍是「名\n(单位)」
    shown = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
    assert shown == f"{LONG}\n(deg)"

    # 行号列不弹
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.ToolTipRole) is None


def test_header_tooltip_no_unit_marker_is_single_line():
    """loader 缺单位写 "-"，此时提示只给名称一行（与 helper 约定一致）。"""
    df = pd.DataFrame({LONG: [1.0]})
    model = PandasTableModel(df, {LONG: "-"})
    tip = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
    assert LONG in tip
    assert "单位" not in tip


def test_header_tooltip_survives_model_rebuild(tab_dialog):
    """tab 模式每页各自 new 一个模型，提示必须跟着这些重建后的模型仍在。"""
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    state = tab_dialog._group_tabs[0]
    columns = [str(c) for c in state.model._df.columns]
    idx = columns.index("Press_G0")
    tip = state.model.headerData(
        idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole
    )
    assert "Press_G0" in tip


@pytest.mark.parametrize(
    "role",
    [Qt.ItemDataRole.ToolTipRole, Qt.ItemDataRole.DisplayRole],
)
@pytest.mark.parametrize("section", [5, -1])
def test_header_out_of_range_section_returns_none(role, section):
    """越界 section 不能抛 IndexError，也不能悄悄取到末列（section=-1）。

    两种 role 口径必须一致：DisplayRole 此前漏了判断，越界抛 IndexError、
    -1 返回最后一列的列名。
    """
    df = pd.DataFrame({LONG: [1.0]})
    model = PandasTableModel(df, {})
    assert model.headerData(section, Qt.Orientation.Horizontal, role) is None


def test_header_in_range_section_keeps_both_roles():
    """护栏不得误伤正常列：首列两个 role 仍各自返回应有内容。"""
    df = pd.DataFrame({LONG: [1.0]})
    model = PandasTableModel(df, {})
    assert model.headerData(
        0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
    ) == f"{LONG}\n()"
    assert LONG in model.headerData(
        0, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole
    )


# ---------------------------------------------------------------- 变量列表
@pytest.fixture()
def var_table(qapp):
    widget = MyTableWidget(None)
    widget.populate([LONG, "zulu"], {LONG: "deg", "zulu": "s"}, {LONG: 1, "zulu": 1})
    yield widget
    widget.deleteLater()


def _tooltip_of(widget, name):
    for row in range(widget.rowCount()):
        item = widget.item(row, 0)
        if item.data(Qt.ItemDataRole.UserRole + 1) == name:
            return item.toolTip()
    raise AssertionError(f"未找到变量 {name}")


def test_variable_list_tooltip_populate(var_table):
    assert LONG in _tooltip_of(var_table, LONG)
    assert "单位: deg" in _tooltip_of(var_table, LONG)


def test_variable_list_tooltip_survives_sort(var_table):
    """排序会重建 item，提示必须跟着重建路径一起补。"""
    var_table.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert LONG in _tooltip_of(var_table, LONG)


# ---------------------------------------------------------------- 数值表定位下拉
def test_tab_locator_items_have_tooltip(tab_dialog):
    tab_dialog._add_variable_to_tab("Press_G0", 0)
    combo = tab_dialog._var_locator
    tips = [
        combo.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(combo.count())
    ]
    assert tips and all("Press_G0" in tip for tip in tips)


# ---------------------------------------------------------------- 搜索候选列表
class FakePlotWidget(QObject):
    curves_changed = Signal()

    def __init__(self):
        super().__init__()
        self.curves: dict = {}
        self.units: dict = {}


@pytest.fixture()
def search_bar(qapp):
    pw = FakePlotWidget()
    bar = VariableSearchBar([LONG], {LONG: "deg"}, {LONG: 1}, pw)
    bar.search_edit.setText("EGR")
    bar._refresh_list()
    yield bar, pw
    bar.deleteLater()


def test_candidate_tooltip_has_name_without_added_suffix(search_bar):
    bar, _ = search_bar
    tip = bar.candidate_list.item(0).toolTip()
    assert LONG in tip and "单位: deg" in tip
    assert "已添加" not in tip


def test_candidate_tooltip_survives_incremental_mark(search_bar):
    """增量标记只改显示文本（setText），不得冲掉悬停提示。"""
    bar, pw = search_bar
    pw.curves[LONG] = object()
    bar.mark_added(LONG)

    item = bar.candidate_list.item(0)
    assert item.text().endswith("(已添加)")
    assert LONG in item.toolTip()


# ---------------------------------------------------------------- 编辑器左表
@pytest.fixture()
def editor(plot_factory, qapp):
    pw = plot_factory()
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=[LONG], units={LONG: "deg"})
    host.data_validity = {LONG: 1}
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    dlg.plot_widget.units = {LONG: "deg"}
    yield dlg
    dlg.deleteLater()


def test_editor_name_column_tooltip(editor):
    editor._add_variable_to_table(
        LONG, types.SimpleNamespace(visible=True, color="#ff0000")
    )
    tip = editor.var_table.item(0, 1).toolTip()
    assert LONG in tip and "单位: deg" in tip


def test_editor_missing_row_tooltip_uses_raw_name(editor):
    editor._add_invalid_to_table(
        LONG, types.SimpleNamespace(visible_preference=True, color="#ff0000")
    )
    item = editor.var_table.item(0, 1)
    assert "（缺失）" in item.text()
    assert LONG in item.toolTip()