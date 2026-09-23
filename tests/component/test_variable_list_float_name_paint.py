"""NoHoverDelegate 绘制兜底回归（现场崩溃：float nan 列名进 elidedText）。

csv_plot.log 的 11 连崩根因之一：空表头单元格在 pandas 3.0 下保留 float
nan 列名，delegate 绘制时 elidedText(float) 抛 TypeError。①的兜底保证
任何非 str 值都不会再打穿绘制路径（②B 已从源头归一，此兜底防御
历史数据与其他 loader 家族）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
)

from src.ui.variable_list import NoHoverDelegate


def _paint_column_zero(table, value):
    delegate = NoHoverDelegate(table)
    item = QTableWidgetItem()
    item.setData(Qt.ItemDataRole.UserRole + 1, value)
    table.setItem(0, 0, item)
    img = QImage(120, 24, QImage.Format.Format_ARGB32)
    painter = QPainter(img)
    option = QStyleOptionViewItem()
    option.initFrom(table)
    option.rect = table.visualItemRect(item)
    try:
        delegate.paint(painter, option, table.indexFromItem(item))
    finally:
        painter.end()


def test_paint_with_float_nan_name_no_crash(qapp):
    table = QTableWidget(2, 2)
    _paint_column_zero(table, float("nan"))  # 修复前：TypeError 崩溃绘制
    table.deleteLater()


def test_paint_with_str_name_works(qapp):
    table = QTableWidget(2, 2)
    _paint_column_zero(table, "ENG01_SPEED")
    table.deleteLater()
