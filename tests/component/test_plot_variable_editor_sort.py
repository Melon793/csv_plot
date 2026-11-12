"""绘图变量编辑器变量名排序防回归（自然排序/仅选中行/撤销快照）。

排序链路与"上移/下移"一致：改 curves 字典顺序 → load_current_curves 重建表格，
因此断言同时覆盖表格行序与 plot_widget.curves 顺序两侧。
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QMenu

from src.ui.plot_variable_editor import PlotVariableEditorDialog, natural_sort_key

# 初始刻意乱序：覆盖 10/2 字典序陷阱与乱插场景
NAMES = ["AI05_1", "AI05_4", "AI05_2", "AI05_10", "AI05_3"]
ASC = ["AI05_1", "AI05_2", "AI05_3", "AI05_4", "AI05_10"]


@pytest.fixture()
def sort_env(plot_factory, qapp):
    df = pd.DataFrame(
        {name: [float(i), float(i + 1)] for i, name in enumerate(NAMES)}
    )
    pw = plot_factory(df)
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=list(NAMES), units={})
    host.data_validity = {}
    for name in NAMES:
        assert pw.add_variable_to_plot(name)
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    return dlg, pw


def _select_rows(dlg, rows):
    """多行选中辅助：编辑器表格为 ExtendedSelection，用选择模型逐行累加"""
    table = dlg.var_table
    table.clearSelection()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    for row in rows:
        table.selectionModel().select(table.model().index(row, 0), flags)


def test_natural_sort_key_compares_digits_numerically():
    """核心陷阱：字典序会把 AI05_10 排在 AI05_2 前"""
    assert sorted(["AI05_10", "AI05_2", "AI05_1"], key=natural_sort_key) == [
        "AI05_1", "AI05_2", "AI05_10",
    ]


def test_natural_sort_key_case_insensitive():
    assert sorted(["b_2", "A_1", "a_10"], key=natural_sort_key) == [
        "A_1", "a_10", "b_2",
    ]


def test_sort_ascending_reorders_table_and_curves(sort_env):
    dlg, pw = sort_env
    dlg._sort_variables()
    assert dlg._table_order() == ASC
    assert list(pw.curves.keys()) == ASC


def test_sort_descending(sort_env):
    dlg, _ = sort_env
    dlg._sort_variables(descending=True)
    assert dlg._table_order() == ASC[::-1]


def test_sort_selected_only_keeps_unselected_positions(sort_env):
    """选中 AI05_4/AI05_2（行 1、2）：只在这两个槽位内互换，其余行不动"""
    dlg, _ = sort_env
    _select_rows(dlg, [1, 2])
    dlg._sort_variables(selected_only=True)
    assert dlg._table_order() == ["AI05_1", "AI05_2", "AI05_4", "AI05_10", "AI05_3"]
    # 选中跟随：排序后仍选中同一批变量
    assert dlg._selected_var_names() == {"AI05_2", "AI05_4"}


def test_undo_restores_order_before_sort(sort_env):
    dlg, _ = sort_env
    dlg._sort_variables()
    dlg._undo_last_sort()
    assert dlg._table_order() == NAMES
    # 单层撤销：用后即清
    assert dlg._last_sort_order is None


def test_resorting_sorted_list_keeps_original_snapshot(sort_env):
    """已有序时重复排序为无操作，不覆盖快照（撤销仍回初始序）"""
    dlg, _ = sort_env
    dlg._sort_variables()
    snapshot = dlg._last_sort_order
    assert snapshot == NAMES
    dlg._sort_variables()
    assert dlg._table_order() == ASC
    assert dlg._last_sort_order == snapshot


def test_manual_move_invalidates_undo_snapshot(sort_env):
    """上移/下移后快照已过期，必须作废，否则'撤销'会连带回滚手动移动"""
    dlg, _ = sort_env
    dlg._sort_variables()
    assert dlg._last_sort_order is not None
    dlg.var_table.selectRow(0)
    dlg._move_selected_row(1)
    assert dlg._last_sort_order is None


def test_sort_menu_action_states(sort_env):
    dlg, _ = sort_env
    dlg._update_sort_menu_actions(dlg.sort_menu)
    states = {
        a.data(): a.isEnabled()
        for a in dlg.sort_menu.actions()
        if a.data() is not None
    }
    assert states["asc"] and states["desc"]
    assert not states["sel"]  # 无选中
    assert not states["undo"]  # 未排序过
    _select_rows(dlg, [0, 1])
    dlg._update_sort_menu_actions(dlg.sort_menu)
    states = {
        a.data(): a.isEnabled()
        for a in dlg.sort_menu.actions()
        if a.data() is not None
    }
    assert states["sel"]
    dlg._sort_variables()
    dlg._update_sort_menu_actions(dlg.sort_menu)
    states = {
        a.data(): a.isEnabled()
        for a in dlg.sort_menu.actions()
        if a.data() is not None
    }
    assert states["undo"]


def test_context_menu_shares_sort_action_group(sort_env):
    """右键菜单与'排序'按钮共用同一动作组（顺序与标识一致）"""
    dlg, _ = sort_env
    menu = QMenu(dlg.var_table)
    dlg._add_sort_actions(menu)
    dlg._update_sort_menu_actions(menu)
    assert [a.data() for a in menu.actions() if a.data() is not None] == [
        "asc", "desc", "sel", "undo",
    ]


def test_sort_disabled_with_single_row(sort_env):
    """按钮可用态：行数 < 2 时排序按钮禁用"""
    dlg, _ = sort_env
    assert dlg.sort_btn.isEnabled()
    dlg.var_table.setRowCount(1)
    dlg.update_button_states()
    assert not dlg.sort_btn.isEnabled()
