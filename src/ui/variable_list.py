"""变量列表面板 —— MyTableWidget + NoHoverDelegate"""

from __future__ import annotations
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import QDrag, QPen, QColor, QPainter, QAction
from PyQt6.QtWidgets import (QApplication, QMenu, QAbstractItemView, QStyledItemDelegate, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)
from src.ui.drag_drop import build_var_mimedata, create_drag_pixmap

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
        from PyQt6.QtWidgets import QStyle
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QColor, QPen
        
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
        from PyQt6.QtWidgets import QStyle
        
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
        from PyQt6.QtGui import QColor
        
        if valid == 1:
            return QColor(0, 200, 0)      # 鲜艳绿色（有效）
        elif valid == 0:
            return QColor(255, 140, 0)    # 橙色（常数）
        elif valid == -1:
            return QColor(255, 0, 0)      # 红色（无效）
        return None
    
    def _draw_variable_name_column(self, painter, option, index):
        """绘制变量名列（包含彩色方块标识符）"""
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPen, QColor
        from PyQt6.QtWidgets import QStyle
        
        # 获取有效性和颜色
        valid = index.data(Qt.ItemDataRole.UserRole)
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
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QStyle
        
        painter.save()
        
        # 选中时使用白色文字（高对比度），否则使用默认颜色
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(option.palette.text().color())
        
        # 绘制文本，自动省略过长部分（...）
        elided_text = painter.fontMetrics().elidedText(
            text, 
            Qt.TextElideMode.ElideRight, 
            text_rect.width()
        )
        painter.drawText(
            text_rect, 
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided_text
        )
        
        painter.restore()


class MyTableWidget(QTableWidget):
    """
    自定义表格控件类
    扩展QTableWidget功能，支持拖拽、右键菜单等自定义交互
    提供数据表格的增强显示和操作功能
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["变量名", "单位", "序号"])
        self.original_indices = {}  # 存储原始索引
        self._column_sort_order = {}  # 记录每列的当前排序状态：{column_index: order}

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
        self.setColumnWidth(1, int(total * 0.25))   # 单位列
        self.setColumnWidth(2, int(total * 0.125))  # 序号列

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)  # 关闭自动拉伸最后一列
        self.verticalHeader().setVisible(False)  # 隐藏行号
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)  # 允许多选
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
          
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        # 设置字体大小
        # font = QFont()
        # font.setPointSize(12)  # 调小字体大小
        # self.setFont(font)
    
    def _handle_header_click(self, logicalIndex):
        """自定义排序处理，确保有效性始终是第一优先级"""
        # 如果该列之前没有排序过，或者上次是降序，则设为升序；否则设为降序
        if logicalIndex not in self._column_sort_order:
            new_order = Qt.SortOrder.AscendingOrder  # 第一次点击：升序
        else:
            current_order = self._column_sort_order[logicalIndex]
            # 切换排序顺序
            new_order = Qt.SortOrder.DescendingOrder if current_order == Qt.SortOrder.AscendingOrder else Qt.SortOrder.AscendingOrder
        
        # 记录当前列的排序状态
        self._column_sort_order[logicalIndex] = new_order
        
        # 直接调用自定义排序方法（不使用sortByColumn，因为setSortingEnabled=False）
        self.sortItems(logicalIndex, new_order)
    
    def sortItems(self, column, order=Qt.SortOrder.AscendingOrder):
        """重写排序方法，确保有效性始终是第一优先级"""
        # 收集所有行数据（整行移动）
        rows = []
        for row in range(self.rowCount()):
            name_item = self.item(row, 0)
            unit_item = self.item(row, 1)
            index_item = self.item(row, 2)
            
            if name_item and unit_item and index_item:
                valid = name_item.data(Qt.ItemDataRole.UserRole)
                # 获取原始变量名（存储在UserRole+1中）
                original_name = name_item.data(Qt.ItemDataRole.UserRole + 1)
                rows.append({
                    'name': original_name if original_name else '',
                    'unit': unit_item.text(),
                    'index': index_item.data(Qt.ItemDataRole.DisplayRole),
                    'valid': valid if valid is not None else -999,
                })
        
        # 排序逻辑：
        # Level 1: 有效性降序（1 → 0 → -1，即有效的在前）
        # Level 2: 按选择的列升序或降序
        
        # 使用分组排序：先按有效性分组，再在组内排序
        from itertools import groupby
        
        # 先按有效性降序排序（保证有效的在前）
        rows.sort(key=lambda x: -x['valid'])
        
        # 按有效性分组，然后在每组内按第二级字段排序
        rows_sorted = []
        for valid_value, group in groupby(rows, key=lambda x: x['valid']):
            group_list = list(group)
            
            # 在组内按选择的列排序
            if column == 0:  # 变量名
                group_list.sort(
                    key=lambda x: x['name'].lower(),
                    reverse=(order == Qt.SortOrder.DescendingOrder)
                )
            elif column == 1:  # 单位
                group_list.sort(
                    key=lambda x: x['unit'].lower(),
                    reverse=(order == Qt.SortOrder.DescendingOrder)
                )
            elif column == 2:  # 序号
                group_list.sort(
                    key=lambda x: x['index'],
                    reverse=(order == Qt.SortOrder.DescendingOrder)
                )
            
            rows_sorted.extend(group_list)
        
        rows = rows_sorted
        
        # 重新填充表格（整行移动，包括颜色）
        # 注意：不需要再次禁用排序，因为已经在__init__中禁用了
        
        for row, data in enumerate(rows):
            # 创建新的item（不含emoji，彩色方块由delegate绘制）
            valid_value = data['valid']
            original_name = data['name']
            
            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(data['unit'])
            index_item = QTableWidgetItem()
            index_item.setData(Qt.ItemDataRole.DisplayRole, data['index'])
            
            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.ItemDataRole.UserRole + 1, original_name)
            
            # 设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.ItemDataRole.UserRole, valid_value)
            unit_item.setData(Qt.ItemDataRole.UserRole, valid_value)
            index_item.setData(Qt.ItemDataRole.UserRole, valid_value)
            
            # 设置到表格
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, unit_item)
            self.setItem(row, 2, index_item)
        
        # 更新排序指示器
        self.horizontalHeader().setSortIndicator(column, order)

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
        
        # a. 添加至数值变量表
        act_add_table = QAction("添加至数值变量表", menu)
        act_add_table.triggered.connect(lambda: self._add_to_data_table(selected_var_names))
        menu.addAction(act_add_table)
        
        # b. 添加至空白绘图区
        act_add_blank_plot = QAction("添加至空白绘图区", menu)
        act_add_blank_plot.triggered.connect(lambda: self._add_to_blank_plot(selected_var_names))
        menu.addAction(act_add_blank_plot)
        
        # c. 复制变量名（复制时也使用原始名称，不含方框标识符）
        act_copy = QAction("复制变量名", menu)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(var_name))
        menu.addAction(act_copy)
        
        menu.exec(self.mapToGlobal(pos))

    def _add_to_data_table(self, var_names):
        var_list = self._normalize_var_list(var_names)
        if not var_list:
            return

        main_window = self.window()
        from src.ui.table_dialog import DataTableDialog
        DataTableDialog.add_variables(var_list, parent=main_window)

    def _add_to_blank_plot(self, var_names):
        var_list = self._normalize_var_list(var_names)
        if not var_list:
            return

        # 获取 MainWindow 实例
        main_window = self.window()
        if not (main_window and hasattr(main_window, 'loader')):
            QMessageBox.warning(self, '错误', '未找到主窗口实例')
            return
        loader = getattr(main_window, 'loader', None)

        if main_window is None:
            QMessageBox.warning(self, "错误", "未找到主窗口实例")
            return

        # 2. 在用户设置的当前布局(mxn)中查找空白绘图区，无论绘图区整体是否可见
        blank_plot = None
        rows, cols = main_window._plot_row_current, main_window._plot_col_current
        max_cols = main_window._plot_col_max_default # 这是完整网格的列数，用于计算索引

        for idx, container in enumerate(main_window.plot_widgets):
            # 根据一维索引计算其在完整网格(pxq)中的二维坐标(r, c)
            r = idx // max_cols
            c = idx % max_cols

            # 判断这个坐标是否在用户当前的(mxn)布局内
            if r < rows and c < cols:
                # 如果在布局内，再判断是否为空白
                if container.plot_widget.y_name == '' and container.plot_widget.curve is None:
                    blank_plot = container.plot_widget
                    break  # 找到第一个可用的就退出

        if blank_plot is None:
            QMessageBox.warning(self, "提示", "当前布局中已无空白绘图区")
            return

        # 3. 判断绘图区域整体是否被隐藏，并提示用户        
        _delay = 0
        if not main_window._plot_area_visible:
            # reply = QMessageBox.question(self, "确认", "绘图区域当前已隐藏，是否要显示它？",
            #                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            #                              QMessageBox.StandardButton.Yes)
            reply = QMessageBox.StandardButton.Yes
            if reply == QMessageBox.StandardButton.Yes:
                # 激活绘图区，同步按钮状态
                main_window.toggle_plot_btn.setChecked(False)
                _delay = 300
            else:
                return  # 用户选择不激活，则不执行后续操作
        def _job():
            # 4. 将变量添加至空白图中
            blank_plot.add_variables_to_plot(var_list)
            main_window.request_mark_stats_refresh()

        QTimer.singleShot(_delay, _job) 

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
        """将输入标准化为不重复的变量名列表"""
        if isinstance(var_names, str):
            candidates = [var_names]
        else:
            candidates = list(var_names) if var_names is not None else []

        normalized = []
        seen = set()
        for name in candidates:
            clean_name = (name or "").strip()
            if not clean_name or clean_name in seen:
                continue
            normalized.append(clean_name)
            seen.add(clean_name)
        return normalized

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
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            return

        if hasattr(main_window.loader, 'get_series'):
            series = main_window.loader.get_series(var_name)
        else:
            series = main_window.loader.df[var_name]

        # 弹出数值变量表
        from src.ui.table_dialog import DataTableDialog
        dlg = DataTableDialog.popup(var_name, series, parent=main_window)
        
        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: dlg.scroll_to_column(var_name))

        super().mouseDoubleClickEvent(event)

    def populate(self, var_names, units, validity):
        self.clearContents()
        self.clearSelection()  # 清除选择状态
        self.setRowCount(len(var_names))

        # 创建列表并排序: 先按validity降序, 然后按原顺序 (使用stable sort)
        items = list(zip(var_names, [units.get(v, '') for v in var_names], [validity.get(v, -1) for v in var_names]))
        # 为了保持相同validity的原顺序, 我们用enumerate添加index
        indexed_items = [(valid, idx, name, unit) for idx, (name, unit, valid) in enumerate(items)]  # idx asc for original order
        indexed_items.sort(key=lambda x: (-x[0], x[1]))  # valid desc (-valid), then original idx asc
        sorted_names = [name for valid, idx, name, unit in indexed_items]
        sorted_units = [unit for valid, idx, name, unit in indexed_items]
        sorted_indices = [idx for valid, idx, name, unit in indexed_items]
        
        # 保存原始索引映射（用于排序）
        for row, (name, idx) in enumerate(zip(sorted_names, sorted_indices)):
            self.original_indices[name] = idx
        sorted_valids = [valid for valid, idx, name, unit in indexed_items]

        for row, (idx, name, unit, valid) in enumerate(zip(sorted_indices, sorted_names, sorted_units, sorted_valids)):
            # 创建三列的item（不含emoji，彩色方块由delegate绘制）
            name_item = QTableWidgetItem()  # 变量名列（文本留空，由delegate绘制）
            unit_item = QTableWidgetItem(unit)  # 单位列
            index_item = QTableWidgetItem()  # 序号列
            index_item.setData(Qt.ItemDataRole.DisplayRole, idx)  # 设置为整数，便于数字排序
            
            # 存储原始变量名到UserRole+1（用于delegate绘制和所有操作）
            name_item.setData(Qt.ItemDataRole.UserRole + 1, name)
            
            # 为所有item设置有效性数据（用于排序和delegate绘制彩色方块）
            name_item.setData(Qt.ItemDataRole.UserRole, valid)
            unit_item.setData(Qt.ItemDataRole.UserRole, valid)
            index_item.setData(Qt.ItemDataRole.UserRole, valid)

            # 设置到正确的列
            self.setItem(row, 0, name_item)   # 第0列：变量名
            self.setItem(row, 1, unit_item)   # 第1列：单位
            self.setItem(row, 2, index_item)  # 第2列：序号



