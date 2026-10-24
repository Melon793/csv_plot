"""变量列表排序语义回归测试。

有效性始终是第一优先级：
- 分组顺序固定为 VALID(1) → CONST(0) → INVALID(-1) → UNKNOWN(-2)，
  不随升/降序变化；缺失有效性(-999)排在最后；
- 组内分别按变量名、单位文本、文件序号升/降序；
- 序号升序在首次点击时与初始加载顺序一致（设计使然，不是 bug）。

另覆盖排序状态复位、指示器隐藏、按变量名保持选中、过滤重放。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from src.ui.variable_list import MyTableWidget

NAMES = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"]
UNITS = {
    "zulu": "s",
    "alpha": "m",
    "mike": "s",
    "bravo": "m",
    "yankee": "s",
    "charlie": "m",
}
VALID = {
    "zulu": 1,
    "alpha": 0,
    "mike": -1,
    "bravo": 1,
    "yankee": 0,
    "charlie": -1,
}


@pytest.fixture()
def table(qapp):
    widget = MyTableWidget(None)
    widget.populate(NAMES, UNITS, VALID)
    yield widget
    widget.deleteLater()


def _names(table):
    return [
        table.item(row, 0).data(Qt.ItemDataRole.UserRole + 1)
        for row in range(table.rowCount())
    ]


def _visible_names(table):
    return [
        name for row, name in enumerate(_names(table)) if not table.isRowHidden(row)
    ]


def _hidden_names(table):
    return [
        name for row, name in enumerate(_names(table)) if table.isRowHidden(row)
    ]


def test_populate_orders_by_validity_then_file_order(table):
    """初始加载：有效性优先，同组保持文件顺序。"""
    assert _names(table) == [
        "zulu",
        "bravo",
        "alpha",
        "yankee",
        "mike",
        "charlie",
    ]


def test_name_asc_sorts_within_validity_groups(table):
    """变量名升序：每组内部 A→Z，组顺序不变。"""
    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "bravo",
        "zulu",
        "alpha",
        "yankee",
        "charlie",
        "mike",
    ]


def test_name_desc_sorts_within_validity_groups(table):
    """变量名降序：每组内部 Z→A，有效组仍在无效组之前。"""
    table.sortItems(0, Qt.SortOrder.DescendingOrder)

    assert _names(table) == [
        "zulu",
        "bravo",
        "yankee",
        "alpha",
        "mike",
        "charlie",
    ]


def test_index_asc_is_initial_order(table):
    """序号升序与初始加载顺序一致（首点结构性无变化，属既有设计）。"""
    before = _names(table)

    table.sortItems(2, Qt.SortOrder.AscendingOrder)

    assert _names(table) == before
    assert _names(table) == [
        "zulu",
        "bravo",
        "alpha",
        "yankee",
        "mike",
        "charlie",
    ]


def test_index_desc_sorts_within_validity_groups(table):
    """序号降序：只反转组内序号，无效组仍留在最后。"""
    table.sortItems(2, Qt.SortOrder.DescendingOrder)

    assert _names(table) == [
        "bravo",
        "zulu",
        "yankee",
        "alpha",
        "charlie",
        "mike",
    ]


def test_unit_asc_sorts_within_validity_groups(table):
    table.sortItems(1, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "bravo",
        "zulu",
        "alpha",
        "yankee",
        "charlie",
        "mike",
    ]


def test_unit_desc_sorts_within_validity_groups(table):
    table.sortItems(1, Qt.SortOrder.DescendingOrder)

    assert _names(table) == [
        "zulu",
        "bravo",
        "yankee",
        "alpha",
        "mike",
        "charlie",
    ]


def test_unknown_validity_stays_after_invalid(qapp):
    """UNKNOWN(-2) 自成一组且排在 INVALID 之后；组内仍按名称排序。"""
    widget = MyTableWidget(None)
    widget.populate(
        ["delta", "mike", "alpha"],
        {"delta": "x", "mike": "s", "alpha": "m"},
        {"delta": -2, "mike": -1, "alpha": 1},
    )
    try:
        widget.sortItems(0, Qt.SortOrder.AscendingOrder)
        assert _names(widget) == ["alpha", "mike", "delta"]
    finally:
        widget.deleteLater()


def test_missing_validity_sorts_last(table):
    """有效性为 None 的历史数据沿用 -999 语义，排在所有分组之后。"""
    alpha_row = _names(table).index("alpha")
    table.item(alpha_row, 0).setData(Qt.ItemDataRole.UserRole, None)

    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "bravo",
        "zulu",
        "yankee",
        "charlie",
        "mike",
        "alpha",
    ]


def test_filter_replay_keeps_hidden_set(table):
    """排序前后隐藏集合不变；可见行仍按“分组 + 组内排序”有序。"""
    assert table.hide_non_matching(["a"], [], UNITS) is True
    assert set(_hidden_names(table)) == {"zulu", "mike"}
    assert _visible_names(table) == ["bravo", "alpha", "yankee", "charlie"]

    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert set(_hidden_names(table)) == {"zulu", "mike"}
    assert _visible_names(table) == ["bravo", "alpha", "yankee", "charlie"]


def test_initial_indicator_hidden_and_state_empty(qapp):
    """新控件与首次 populate 后都不应指向不存在的 section 3。"""
    widget = MyTableWidget(None)
    try:
        header = widget.horizontalHeader()
        assert widget._column_sort_order == {}
        assert header.sortIndicatorSection() == -1

        widget.populate(NAMES, UNITS, VALID)

        assert widget._column_sort_order == {}
        assert header.sortIndicatorSection() == -1
    finally:
        widget.deleteLater()


def test_first_click_after_populate_is_ascending(table):
    header = table.horizontalHeader()
    header.sectionClicked.emit(0)
    assert table._column_sort_order[0] == Qt.SortOrder.AscendingOrder

    header.sectionClicked.emit(0)
    assert table._column_sort_order[0] == Qt.SortOrder.DescendingOrder


def test_populate_resets_sort_state_and_indicator(table):
    table.horizontalHeader().sectionClicked.emit(0)
    assert table._column_sort_order
    assert table.horizontalHeader().sortIndicatorSection() == 0

    table.populate(
        ["alpha", "bravo"],
        {"alpha": "m", "bravo": "s"},
        {"alpha": 1, "bravo": 0},
    )

    assert table._column_sort_order == {}
    assert table.horizontalHeader().sortIndicatorSection() == -1

    # 复位后首点重新从 Asc 开始
    table.horizontalHeader().sectionClicked.emit(0)
    assert table._column_sort_order[0] == Qt.SortOrder.AscendingOrder


def test_sort_keeps_selected_variable(qapp):
    """排序后选中行应仍指向同一个变量，而不是同一行号。"""
    widget = MyTableWidget(None)
    try:
        names = ["zulu", "alpha", "bravo"]
        widget.populate(
            names,
            {name: "m" for name in names},
            {name: 1 for name in names},
        )
        widget.selectRow(_names(widget).index("zulu"))
        assert widget._selected_var_names() == ["zulu"]

        widget.sortItems(0, Qt.SortOrder.AscendingOrder)

        assert _names(widget) == ["alpha", "bravo", "zulu"]
        selected_rows = {index.row() for index in widget.selectedIndexes()}
        assert len(selected_rows) == 1
        selected_name = widget.item(
            selected_rows.pop(), 0
        ).data(Qt.ItemDataRole.UserRole + 1)
        assert selected_name == "zulu"
    finally:
        widget.deleteLater()
