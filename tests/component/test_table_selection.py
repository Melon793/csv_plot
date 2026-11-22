"""程序化整行选中：免疫合成事件残留的 Shift/Ctrl（见 tests/README §2.1）。"""

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from src.ui.table_selection import select_table_row
from src.ui.variable_list import MyTableWidget

NAMES = ["zulu", "alpha", "bravo"]


def _table() -> MyTableWidget:
    widget = MyTableWidget(None)
    widget.populate(
        NAMES, {name: "m" for name in NAMES}, {name: 1 for name in NAMES}
    )
    return widget


def _stuck_shift(widget) -> None:
    """复现 QTest 合成事件把 Shift 留在应用级修饰键缓存的状态。"""
    QTest.mousePress(
        widget.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    QTest.mouseRelease(
        widget.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier


def _reset_modifiers() -> None:
    QTest.mouseClick(
        QWidget(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )


def test_select_table_row_immune_to_stuck_shift(qapp):
    widget = _table()
    _stuck_shift(widget)
    try:
        select_table_row(widget, 0)
        assert widget._selected_var_names() == ["zulu"]
        assert widget.currentRow() == 0
        # 清空并重选（ClearAndSelect 语义，不扩成区间）
        select_table_row(widget, 2)
        assert widget._selected_var_names() == ["bravo"]
        assert widget.currentRow() == 2
    finally:
        _reset_modifiers()
        widget.deleteLater()


def test_select_table_row_ignores_invalid_row(qapp):
    widget = _table()
    try:
        select_table_row(widget, 99)
        assert widget._selected_var_names() == []
    finally:
        widget.deleteLater()