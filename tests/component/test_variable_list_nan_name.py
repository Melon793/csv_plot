"""变量列表对非 str 名称（float NaN）的兜底回归测试。

加载层已把空表头归一为 ``Unnamed: N``（见 test_loader_unnamed_header.py）；
这里直接构造 NaN 名称，验证 UI 的排序 / 筛选 / 绘制不会再抛异常，
防止历史缓存或其它 loader 再次把非 str 名称带进来。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from src.ui.variable_list import MyTableWidget

NAN = float("nan")


@pytest.fixture()
def table(qapp):
    widget = MyTableWidget(None)
    yield widget
    widget.deleteLater()


def _names(table):
    return [
        table.item(row, 0).data(Qt.ItemDataRole.UserRole + 1)
        for row in range(table.rowCount())
    ]


def test_name_sort_does_not_crash_on_float_name(table):
    table.populate([NAN, "alpha"], {"alpha": "m"}, {NAN: -1, "alpha": 1})

    table.sortItems(0, Qt.SortOrder.AscendingOrder)  # 修复前抛 AttributeError

    names = _names(table)
    assert names[0] == "alpha"
    assert isinstance(names[-1], float)
    assert str(names[-1]) == "nan"


def test_name_filter_does_not_crash_on_float_name(table):
    table.populate([NAN, "alpha"], {"alpha": "m"}, {NAN: -1, "alpha": 1})

    assert table.hide_non_matching(["alp"], [], {"alpha": "m"}) is True

    names = _names(table)
    alpha_row = next(row for row, name in enumerate(names) if name == "alpha")
    nan_row = next(row for row, name in enumerate(names) if isinstance(name, float))
    assert table.isRowHidden(alpha_row) is False
    assert table.isRowHidden(nan_row) is True


def test_paint_does_not_crash_on_float_name(table):
    table.populate([NAN, "alpha"], {"alpha": "m"}, {NAN: -1, "alpha": 1})
    table.resize(320, 120)

    pixmap = table.grab()  # 修复前 delegate 的 elidedText(float) 抛 TypeError

    assert not pixmap.isNull()


def test_unit_and_index_sort_still_work_with_float_name(table):
    table.populate(
        [NAN, "alpha", "beta"],
        {"alpha": "s", "beta": "m"},
        {NAN: -1, "alpha": 1, "beta": 0},
    )

    table.sortItems(1, Qt.SortOrder.AscendingOrder)  # 单位
    table.sortItems(2, Qt.SortOrder.AscendingOrder)  # 序号

    assert table.rowCount() == 3
