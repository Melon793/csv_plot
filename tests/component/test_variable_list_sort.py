"""变量列表排序语义回归测试。

覆盖：
- 变量名：非无效变量全局排序，INVALID(-1) 固定压底（升/降）；
- 序号：严格按文件原始列顺序，无效行也参与；
- 单位：保持既有"有效性分组 + 组内排序"；
- UNKNOWN(-2) 视为可用；缺失有效性按无效处理；
- 排序后过滤重放，隐藏集合不变。
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


def test_name_asc_global_with_invalid_tail(table):
    """可用段（valid + const）全局 A→Z；INVALID 段在尾部内部再排序。"""
    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "alpha",
        "bravo",
        "yankee",
        "zulu",
        "charlie",
        "mike",
    ]


def test_name_desc_keeps_invalid_tail(table):
    """降序时可用段 Z→A，INVALID 段仍在尾部（不能被整体 reverse 顶到头部）。"""
    table.sortItems(0, Qt.SortOrder.DescendingOrder)

    assert _names(table) == [
        "zulu",
        "yankee",
        "bravo",
        "alpha",
        "mike",
        "charlie",
    ]


def test_index_asc_is_file_order_including_invalid(table):
    """序号升序 = 文件原始列顺序 0..N-1，无效行不压底。"""
    table.sortItems(2, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "zulu",
        "alpha",
        "mike",
        "bravo",
        "yankee",
        "charlie",
    ]


def test_index_desc_is_reversed_file_order(table):
    table.sortItems(2, Qt.SortOrder.DescendingOrder)

    assert _names(table) == [
        "charlie",
        "yankee",
        "bravo",
        "mike",
        "alpha",
        "zulu",
    ]


def test_unit_keeps_validity_grouping(table):
    """单位列保持现状：每个有效性段内部按单位排序。"""
    table.sortItems(1, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "bravo",
        "zulu",
        "alpha",
        "yankee",
        "charlie",
        "mike",
    ]


def test_unknown_validity_is_usable(qapp):
    """UNKNOWN(-2)（MDF/BLF）视为可用，参与全局名称排序。"""
    widget = MyTableWidget(None)
    widget.populate(
        ["delta", "mike", "alpha"],
        {"delta": "x", "mike": "s", "alpha": "m"},
        {"delta": -2, "mike": -1, "alpha": 1},
    )
    try:
        widget.sortItems(0, Qt.SortOrder.AscendingOrder)
        assert _names(widget) == ["alpha", "delta", "mike"]
    finally:
        widget.deleteLater()


def test_missing_validity_is_treated_as_invalid(table):
    """有效性为 None 的历史数据按 INVALID 处理，不再伪造 -999 混入可用段。"""
    alpha_row = _names(table).index("alpha")
    table.item(alpha_row, 0).setData(Qt.ItemDataRole.UserRole, None)

    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert _names(table) == [
        "bravo",
        "yankee",
        "zulu",
        "alpha",
        "charlie",
        "mike",
    ]


def test_filter_replay_keeps_hidden_set(table):
    """排序前后隐藏集合不变；可见行仍按新语义有序。"""
    assert table.hide_non_matching(["a"], [], UNITS) is True
    assert set(_hidden_names(table)) == {"zulu", "mike"}
    assert _visible_names(table) == ["bravo", "alpha", "yankee", "charlie"]

    table.sortItems(0, Qt.SortOrder.AscendingOrder)

    assert set(_hidden_names(table)) == {"zulu", "mike"}
    assert _visible_names(table) == ["alpha", "bravo", "yankee", "charlie"]


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
