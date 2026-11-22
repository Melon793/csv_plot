"""表格程序化整行选中（免疫残留修饰键）。

``QAbstractItemView.selectRow`` 没有事件对象可读时会回退读应用级
``keyboardModifiers()``：用户真按着 Shift、或测试里合成事件残留 Shift 时，
"选中整行"会变成扩选甚至空选（见 tests/README §2.1）。程序化恢复选中
不应受键盘状态影响，统一走这里。
"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel


def select_table_row(view, row: int) -> None:
    """等价 ``view.selectRow(row)``，语义固定为"清空并整行选中 + 置当前行"。"""
    model = view.model()
    selection = view.selectionModel()
    if model is None or selection is None:
        return
    index = model.index(row, 0)
    if not index.isValid():
        return
    selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
    selection.select(
        index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )