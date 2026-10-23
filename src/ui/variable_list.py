"""变量列表面板 —— MyTableWidget + NoHoverDelegate"""

from __future__ import annotations
from itertools import groupby
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtGui import QDrag, QPen, QColor, QAction
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QAbstractItemView,
    QStyledItemDelegate,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QHeaderView,
)
from src.core.logger import get_logger
from src.core.config import VAR_INFO_MAX_TABS, is_grouped_loader
from src.ui.drag_drop import build_var_mimedata, create_drag_pixmap
from src.ui.variable_actions import (
    add_variables_to_blank_plot,
    add_variables_to_data_table,
    normalize_var_list,
)

logger = get_logger(__name__)


class NoHoverDelegate(QStyledItemDelegate):
    """
    变量表自定义委托类

    功能：
    1. 禁用鼠标悬停和焦点的视觉反馈，避免干扰用户操作
    2. 在变量名列左侧绘制彩色方块标识符（绿色=有效，橙色=常数，红色=无效）
    3. 方块不占用文本显示空间，最大化变量名显示长度
    4. 确保选中行文本高对比度显示（白色文字）
    """

    def paint(self, painter, option, index):
        """
        自定义单元格绘制逻辑

        Args:
            painter: QPainter绘图对象
            option: QStyleOptionViewItem样式选项
            index: QModelIndex单元格索引
        """
        from PySide6.QtWidgets import QStyle

        # 移除悬停和焦点状态，避免出现白色块和焦点框
        option.state &= ~QStyle.StateFlag.State_MouseOver
        option.state &= ~QStyle.StateFlag.State_HasFocus

        # 绘制背景（选中状态用高亮色，未选中用基础色）
        self._draw_background(painter, option)

        # 变量名列（第0列）：绘制彩色方块 + 文本
        if index.column() == 0:
            self._draw_variable_name_column(painter, option, index)
        # 其他列（单位、序号）：仅绘制文本
        else:
            self._draw_text_column(painter, option, index)

    def _draw_background(self, painter, option):
        """绘制单元格背景"""
        from PySide6.QtWidgets import QStyle

        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())
        painter.restore()

    def _get_validity_color(self, valid):
        """
        根据有效性返回对应的颜色

        Args:
            valid: 有效性值（1=有效，0=常数，-1=无效）

        Returns:
            QColor或None
        """

        if valid == 1:
            return QColor(0, 200, 0)  # 鲜艳绿色（有效）
        elif valid == 0:
            return QColor(255, 140, 0)  # 橙色（常数）
        elif valid == -1:
            return QColor(255, 0, 0)  # 红色（无效）
        return None

    def _draw_variable_name_column(self, painter, option, index):
        """绘制变量名列（包含彩色方块标识符）"""

        # 获取有效性和颜色
        valid = index.data(Qt.ItemDataRole.UserRole)
        table_widget = self.parent()
        if getattr(table_widget, "_all_validity_unknown", False):
            color = None
        else:
            color = self._get_validity_color(valid) if valid is not None else None

        # 获取原始变量名（存储在UserRole+1中）
        original_name = index.data(Qt.ItemDataRole.UserRole + 1)
        if not original_name:
            return

        # 如果有效性标识，绘制彩色方块
        text_rect = option.rect
        if color:
            # 计算方块位置和大小
            square_size = min(option.rect.height() - 4, 12)
            square_x = option.rect.left() + 3
            square_y = option.rect.top() + (option.rect.height() - square_size) // 2

            # 绘制方块
            painter.save()
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            painter.drawRect(square_x, square_y, square_size, square_size)
            painter.restore()

            # 调整文本区域，为方块留出空间
            text_rect = QRect(option.rect)
            text_rect.setLeft(option.rect.left() + square_size + 8)
        else:
            # 无方块时，左侧留出小边距
            text_rect = option.rect.adjusted(6, 0, -6, 0)

        # 绘制文本
        self._draw_text(painter, option, original_name, text_rect)

    def _draw_text_column(self, painter, option, index):
        """绘制普通文本列（单位、序号）"""
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text is not None:
            text_rect = option.rect.adjusted(3, 0, -3, 0)
            self._draw_text(painter, option, str(text), text_rect)

    def _draw_text(self, painter, option, text, text_rect):
        """
        绘制文本，自动处理选中状态的颜色和文本省略

        Args:
            painter: QPainter对象
            option: 样式选项
            text: 要绘制的文本
            text_rect: 文本绘制区域
        """
        from PySide6.QtWidgets import QStyle

        painter.save()

        # 兜底：非 str 值（如历史数据里的 float NaN 列名）会让 elidedText 抛
        # TypeError 并中断整个绘制流程
        if not isinstance(text, str):
            text = str(text)

        # 选中时使用白色文字（高对比度），否则使用默认颜色
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(option.palette.text().color())

        # 绘制文本，自动省略过长部分（...）
        elided_text = painter.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided_text,
        )

        painter.restore()


class MyTableWidget(QTableWidget):
    """
    自定义表格控件类
    扩展QTableWidget功能，支持拖拽、右键菜单等自定义交互
    提供变量列表表格的增强显示和操作功能
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["变量名", "单位", "序号"])
        self.original_indices = {}  # 存储原始索引
        self._column_sort_order = {}  # 记录每列的当前排序状态：{column_index: order}
        self._row_var_names = []   # 当前每行对应的变量名（行号 → 变量名）
        self._is_full_table = False  # 是否已填充全量表
        self._current_name_keywords = []
        self._current_unit_keywords = []
        self._current_units = {}

        # 设置自定义委托，从绘制层面彻底禁用悬停和焦点效果
        self.setItemDelegate(NoHoverDelegate(self))

        # 字体
        # hdr = self.horizontalHeader()
        header_font = self.horizontalHeader().font()
        header_font.setBold(False)
        self.horizontalHeader().setFont(header_font)

        # 设置表格选择行为的样式
        # 保留未选中行的自定义背景色，仅在选中时使用高亮色
        self.setStyleSheet("""
            QTableWidget::item:selected {
                font-weight: normal;         /* 确保选中项字体也不加粗 */
            }
        """)

        # 默认列宽度：变量名:单位:序号 = 5:2:1
        total = 300
        self.setColumnWidth(0, int(total * 0.625))  # 变量名列
        self.setColumnWidth(1, int(total * 0.25))  # 单位列
        self.setColumnWidth(2, int(total * 0.125))  # 序号列

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)  # 关闭自动拉伸最后一列
        self.verticalHeader().setVisible(False)  # 隐藏行号
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )  # 允许多选
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # 禁用鼠标追踪和视口追踪，避免白色块跟随鼠标移动
        self.setMouseTracking(False)
        self.viewport().setMouseTracking(False)  # 同时禁用视口的鼠标跟踪

        # 禁用焦点指示器
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 禁用自动滚动（当鼠标向表格边缘移动时不会自动滚动）
        self.setAutoScroll(False)

        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)

        # 禁用Qt自动排序，使用自定义排序
        self.setSortingEnabled(False)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().sectionClicked.connect(self._handle_header_click)
        # 初始无排序：隐藏指向不存在 section 3 的默认指示器
        self._reset_sort_state()

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        # 设置字体大小
        # font = QFont()
        # font.setPointSize(12)  # 调小字体大小
        # self.setFont(font)

    def _reset_sort_state(self) -> None:
        """清空排序状态并隐藏表头指示器（section=-1 表示无排序）。"""
        self._column_sort_order.clear()
        self.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    def _handle_header_click(self, logicalIndex):
        """表头点击：首次升序、再次降序，按当前列语义排序"""
        # 如果该列之前没有排序过，或者上次是降序，则设为升序；否则设为降序
        if logicalIndex not in self._column_sort_order:
            new_order = Qt.SortOrder.AscendingOrder  # 第一次点击：升序
        else:
            current_order = self._column_sort_order[logicalIndex]
            # 切换排序顺序
            new_order = (
                Qt.SortOrder.DescendingOrder
                if current_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )

        # 记录当前列的排序状态
        self._column_sort_order[logicalIndex] = new_order

        # 直接调用自定义排序方法（不使用sortByColumn，因为setSortingEnabled=False）
        self.sortItems(logicalIndex, new_order)

    def _collect_sort_data(self, column: int, order: Qt.SortOrder) -> list[tuple]:
        """收集所有行数据并按列排序，返回排序后的 (name, unit, index, valid) 元组列表

        排序语义：
        - 变量名：非无效变量（VALID/CONST/UNKNOWN）全局排序，INVALID(-1) 固定压底；
        - 单位：保持既有"有效性分组 + 组内排序"；
        - 序号：按文件原始列顺序全局排序（无效行也参与）。
        """
        rows = []
        for row in range(self.rowCount()):
            name_item = self.item(row, 0)
            unit_item = self.item(row, 1)
            index_item = self.item(row, 2)

            if name_item and unit_item and index_item:
                valid = name_item.data(Qt.ItemDataRole.UserRole)
                rows.append(
                    (
                        name_item.data(Qt.ItemDataRole.UserRole + 1) or "",
                        unit_item.text(),
                        index_item.data(Qt.ItemDataRole.DisplayRole),
                        # 缺失有效性按无效处理，不再伪造 -999 混入可用段
                        -1 if valid is None else valid,
                    )
                )

        desc = order == Qt.SortOrder.DescendingOrder

        if column == 0:  # 变量名
            # 兜底：历史/异常路径可能留下非 str 名称（如 float NaN）
            return self._sort_usable_then_invalid(
                rows, key=lambda x: str(x[0]).lower(), descending=desc
            )

        if column == 2:  # 序号：严格按文件原始列顺序，不压底
            rows.sort(key=lambda x: x[2], reverse=desc)
            return rows

        # 单位：保持既有"有效性分组 + 组内排序"语义
        rows.sort(key=lambda x: -x[3])
        rows_sorted: list[tuple] = []
        for _valid_value, group in groupby(rows, key=lambda x: x[3]):
            group_list = list(group)
            if column == 1:
                group_list.sort(
                    key=lambda x: str(x[1]).lower(),
                    reverse=desc,
                )
            rows_sorted.extend(group_list)

        return rows_sorted

    @staticmethod
    def _sort_usable_then_invalid(
        rows: list[tuple], key, descending: bool
    ) -> list[tuple]:
        """两阶段排序：可用段全局排序，INVALID(-1) 整段压底。

        不能写成单键 ``(invalid_rank, key)`` 再整体 ``reverse=True``：
        降序时 invalid_rank=1 会先排，INVALID 会全部跑到头部。
        """
        usable = [r for r in rows if r[3] != -1]
        invalid = [r for r in rows if r[3] == -1]
        usable.sort(key=key, reverse=descending)
        invalid.sort(key=key, reverse=descending)
        return usable + invalid

    def _apply_sorted_order(self, sorted_data: list[tuple]) -> None:
        """按排序后的数据重新填充表格（整行移动，包含颜色信息）"""
        for row, data in enumerate(sorted_data):
            original_name, unit, index_val, valid_value = data

            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(unit)
            index_item = QTableWidgetItem()
            index_item.setData(Qt.ItemDataRole.DisplayRole, index_val)

            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.ItemDataRole.UserRole + 1, original_name)

            # 设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.ItemDataRole.UserRole, valid_value)
            unit_item.setData(Qt.ItemDataRole.UserRole, valid_value)
            index_item.setData(Qt.ItemDataRole.UserRole, valid_value)

            self.setItem(row, 0, name_item)
            self.setItem(row, 1, unit_item)
            self.setItem(row, 2, index_item)

    def sortItems(self, column, order=Qt.SortOrder.AscendingOrder):
        """重写排序方法；具体语义见 _collect_sort_data"""
        selected_names = self._selected_var_names()
        sorted_data = self._collect_sort_data(column, order)
        # 重建 item 前先清空旧行号选择，排序后按变量名恢复
        self.clearSelection()
        self._apply_sorted_order(sorted_data)
        self._restore_selection(selected_names)

        # 更新排序指示器
        self.horizontalHeader().setSortIndicator(column, order)

        self._row_var_names = [
            self.item(row, 0).data(Qt.ItemDataRole.UserRole + 1)
            for row in range(self.rowCount())
        ]

        # 排序后重新应用当前过滤条件
        if self._current_name_keywords or self._current_unit_keywords:
            self.hide_non_matching(
                self._current_name_keywords,
                self._current_unit_keywords,
                self._current_units,
            )

    def _selected_var_names(self) -> list:
        """返回当前选中的变量名（按行去重，保持选择顺序）。"""
        names = []
        for index in self.selectedIndexes():
            if index.column() != 0:
                continue
            name = self.item(index.row(), 0).data(Qt.ItemDataRole.UserRole + 1)
            if name not in names:
                names.append(name)
        return names

    def _restore_selection(self, names) -> None:
        """按变量名恢复行选择，避免排序后选中行指向另一个变量。"""
        if not names:
            return
        wanted = set(names)
        for row in range(self.rowCount()):
            name = self.item(row, 0).data(Qt.ItemDataRole.UserRole + 1)
            if name in wanted:
                self.setRangeSelected(
                    QTableWidgetSelectionRange(row, 0, row, self.columnCount() - 1),
                    True,
                )

    def _show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item = self.item(index.row(), 0)  # 变量名在第0列（索引0）
        # 获取原始变量名（不含彩色方框标识符）
        var_name = item.data(Qt.ItemDataRole.UserRole + 1)
        if not var_name:
            # 兼容旧数据
            var_name = item.text()

        selected_var_names = self._collect_selected_var_names()
        if var_name not in selected_var_names:
            selected_var_names = [var_name]

        menu = QMenu(self)

        # a. 添加至变量数值表
        act_add_table = QAction("添加至变量数值表", menu)
        act_add_table.triggered.connect(
            lambda: self._add_to_data_table(selected_var_names)
        )
        menu.addAction(act_add_table)

        # b. 添加至空白绘图区
        act_add_blank_plot = QAction("添加至空白绘图区", menu)
        act_add_blank_plot.triggered.connect(
            lambda: self._add_to_blank_plot(selected_var_names)
        )
        menu.addAction(act_add_blank_plot)

        # c. 复制变量名（复制所有选中变量名，用空格分隔）
        act_copy = QAction("复制变量名", menu)
        act_copy.triggered.connect(
            lambda: QApplication.clipboard().setText(" ".join(selected_var_names))
        )
        menu.addAction(act_copy)

        # d. 变量信息（多选时每个变量一个标签页）
        menu.addSeparator()
        act_var_info = QAction("变量信息", menu)
        act_var_info.triggered.connect(
            lambda: self._show_variable_info(selected_var_names)
        )
        menu.addAction(act_var_info)

        menu.exec(self.mapToGlobal(pos))

    def _show_variable_info(self, var_names):
        """打开变量信息窗口（全局单例 + 标签页累积）。

        多选时每个变量一个标签页；已存在的变量直接激活其标签页而不重建。
        元数据为纯内存读取且 O(1)（MDF 路径实测六组全属性约 99 μs；表格
        路径只读 pandas 的 dtype/categories/长度，不物化列数据），因此
        窗口可立即显示；统计特征（实测 18.8 ms/428k 点）由窗口内部的
        后台线程逐条回填。
        """
        var_list = normalize_var_list(var_names)
        if not var_list:
            return

        main_window = self.window()
        if not (main_window and getattr(main_window, "loader", None) is not None):
            QMessageBox.warning(self, "错误", "尚未加载数据文件")
            return

        if len(var_list) > VAR_INFO_MAX_TABS:
            QMessageBox.information(
                self,
                "提示",
                f"一次最多查看 {VAR_INFO_MAX_TABS} 个变量的信息，"
                f"已截断（原选择 {len(var_list)} 个）。",
            )
            var_list = var_list[:VAR_INFO_MAX_TABS]

        # 延迟 import：未使用此功能时不加载窗口模块
        from src.ui.dialogs.variable_info_dialog import VariableInfoDialog

        VariableInfoDialog.popup(var_list, parent=main_window)

    def _add_to_data_table(self, var_names):
        """添加至变量数值表 → 委托到 variable_actions"""
        add_variables_to_data_table(var_names, self.window())

    def _add_to_blank_plot(self, var_names):
        """添加至空白绘图区 → 委托到 variable_actions"""
        add_variables_to_blank_plot(var_names, self.window(), msg_parent=self)

    def _collect_selected_var_names(self) -> list[str]:
        """返回当前选中的变量原始名称列表"""
        selected_rows = sorted({index.row() for index in self.selectedIndexes()})
        result = []
        for row in selected_rows:
            item = self.item(row, 0)
            if item is None:
                continue
            var_name = item.data(Qt.ItemDataRole.UserRole + 1) or item.text()
            if var_name and var_name not in result:
                result.append(var_name)
        return result

    def _normalize_var_list(self, var_names) -> list[str]:
        """将输入标准化为不重复的变量名列表 → 委托到 variable_actions"""
        return normalize_var_list(var_names)

    def startDrag(self, supportedActions):
        """支持多选变量拖拽"""
        selected_rows = set()
        for index in self.selectedIndexes():
            selected_rows.add(index.row())

        if not selected_rows:
            return

        # 收集所有选中的变量名（使用原始变量名，不含彩色方框标识符）
        var_names = []
        for row in sorted(selected_rows):
            item = self.item(row, 0)  # 变量名在第0列
            if item is not None:
                # 获取原始变量名（UserRole+1存储了不含方框的原始名称）
                original_name = item.data(Qt.ItemDataRole.UserRole + 1)
                if original_name:
                    var_names.append(original_name)
                else:
                    # 兼容旧数据，如果没有存储原始名称，则使用显示文本
                    var_names.append(item.text())

        if not var_names:
            return

        drag = QDrag(self)
        drag.setMimeData(build_var_mimedata(var_names))

        preview_pixmap = create_drag_pixmap(var_names, self.font())
        if preview_pixmap:
            drag.setPixmap(preview_pixmap)
            hot_spot = QPoint(preview_pixmap.width() // 2, preview_pixmap.height() // 2)
            drag.setHotSpot(hot_spot)

        drag.exec(Qt.DropAction.MoveAction)

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            super().mouseDoubleClickEvent(event)
            return

        row = index.row()
        item = self.item(row, 0)  # 变量名在第0列
        # 获取原始变量名（不含彩色方框标识符）
        var_name = item.data(Qt.ItemDataRole.UserRole + 1)
        if not var_name:
            # 兼容旧数据
            var_name = item.text()

        main_window = self.window()
        if not hasattr(main_window, "loader") or main_window.loader is None:
            return

        if is_grouped_loader(main_window.loader):
            try:
                series = main_window.loader.get_series(var_name)
            except KeyError:
                logger.warning("MDF 变量 '%s' 在 loader 中不存在", var_name)
                return
        else:
            try:
                series = main_window.loader.df[var_name]
            except KeyError:
                logger.warning("CSV 变量 '%s' 在 DataFrame 中不存在", var_name)
                return

        # 弹出变量数值表
        from src.ui.table_dialog import DataTableDialog

        dlg = DataTableDialog.popup(var_name, series, parent=main_window)

        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: dlg.scroll_to_column(var_name))

        super().mouseDoubleClickEvent(event)

    def populate(self, var_names, units, validity):
        # 换文件/重载后不能沿用上一份文件的排序状态与箭头
        self._reset_sort_state()
        self.clearContents()
        self.clearSelection()  # 清除选择状态
        self.setRowCount(len(var_names))

        # 创建列表并排序: 先按validity降序, 然后按原顺序 (使用stable sort)
        items = list(
            zip(
                var_names,
                [units.get(v, "") for v in var_names],
                [validity.get(v, -1) for v in var_names],
            )
        )
        # 为了保持相同validity的原顺序, 我们用enumerate添加index
        indexed_items = [
            (valid, idx, name, unit) for idx, (name, unit, valid) in enumerate(items)
        ]  # idx asc for original order
        indexed_items.sort(
            key=lambda x: (-x[0], x[1])
        )  # valid desc (-valid), then original idx asc
        sorted_names = [name for valid, idx, name, unit in indexed_items]
        sorted_units = [unit for valid, idx, name, unit in indexed_items]
        sorted_indices = [idx for valid, idx, name, unit in indexed_items]

        # 保存原始索引映射（用于排序）
        for row, (name, idx) in enumerate(zip(sorted_names, sorted_indices)):
            self.original_indices[name] = idx
        sorted_valids = [valid for valid, idx, name, unit in indexed_items]

        self._all_validity_unknown = all(v == -2 for v in sorted_valids)

        for row, (idx, name, unit, valid) in enumerate(
            zip(sorted_indices, sorted_names, sorted_units, sorted_valids)
        ):
            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(unit)  # 单位列
            index_item = QTableWidgetItem()  # 序号列
            index_item.setData(
                Qt.ItemDataRole.DisplayRole, idx
            )  # 设置为整数，便于数字排序

            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.ItemDataRole.UserRole + 1, name)

            # 为所有item设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.ItemDataRole.UserRole, valid)
            unit_item.setData(Qt.ItemDataRole.UserRole, valid)
            index_item.setData(Qt.ItemDataRole.UserRole, valid)

            # 设置到正确的列
            self.setItem(row, 0, name_item)  # 第0列：变量名
            self.setItem(row, 1, unit_item)  # 第1列：单位
            self.setItem(row, 2, index_item)  # 第2列：序号

        self._row_var_names = sorted_names
        self._is_full_table = True

    def hide_non_matching(self, name_keywords, unit_keywords, units):
        """通过 setRowHidden 过滤行，不重建任何 QTableWidgetItem"""
        if not self._is_full_table:
            return False

        self._current_name_keywords = name_keywords
        self._current_unit_keywords = unit_keywords
        self._current_units = units

        self.setUpdatesEnabled(False)
        try:
            for row, name in enumerate(self._row_var_names):
                if name_keywords:
                    if not any(kw in str(name).lower() for kw in name_keywords):
                        self.setRowHidden(row, True)
                        continue
                if unit_keywords:
                    unit = str(units.get(name, "")).lower()
                    if not any(kw in unit for kw in unit_keywords):
                        self.setRowHidden(row, True)
                        continue
                self.setRowHidden(row, False)
        finally:
            self.setUpdatesEnabled(True)
        return True

    def show_all_rows(self):
        """取消所有行隐藏，恢复全量表显示"""
        if not self._is_full_table:
            return False
        self._current_name_keywords = []
        self._current_unit_keywords = []
        self.setUpdatesEnabled(False)
        try:
            for row in range(self.rowCount()):
                self.setRowHidden(row, False)
        finally:
            self.setUpdatesEnabled(True)
        return True
