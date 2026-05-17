"""绘图变量编辑器"""

from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QCheckBox, QColorDialog, QTableWidget, QTableWidgetItem, QHeaderView, QWidget)
import pyqtgraph as pg
from src.core.config import DEFAULT_LINE_WIDTH
from src.core.types import CurveInfo
from src.ui.drag_drop import parse_var_names_from_mimedata

class PlotVariableEditorDialog(QDialog):
    """
    绘图变量编辑器对话框类
    用于管理plot中的多个曲线，支持添加、删除、颜色自定义等功能
    """
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("绘图变量编辑器")
        self.setModal(False)  # 改为非模态，允许与主窗口交互
        self.resize(600, 400)
        self.setAcceptDrops(True)  # 启用拖拽功能
        
        # 高DPI支持 - PyQt6中不需要WA_UseHighDpiPixmaps
        # PyQt6默认支持高DPI，通过样式表控制字体大小
        
        self.setup_ui()
        self.load_current_curves()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("绘图变量编辑器")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title_label)
        
        # 创建表格
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["显示", "变量名", "颜色"])
        
        # 设置表格属性
        self.var_table.setDragDropMode(QTableWidget.DragDropMode.DropOnly)
        self.var_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.var_table.setAlternatingRowColors(True)
        self.var_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 6px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #000000;
            }
            QCheckBox {
                font-size: 12px;
            }
        """)
        
        # 设置列宽
        header = self.var_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # 显示列固定宽度
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # 变量名列自适应
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # 颜色列固定宽度
        self.var_table.setColumnWidth(0, 60)   # 显示列
        self.var_table.setColumnWidth(2, 80)   # 颜色列
        
        layout.addWidget(self.var_table)
        
        # 按钮区域
        button_layout = QHBoxLayout()

        # 上移/下移按钮
        self.move_up_btn = QPushButton("上移")
        self.move_up_btn.clicked.connect(lambda: self._move_selected_row(-1))
        self.move_up_btn.setEnabled(False)
        button_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("下移")
        self.move_down_btn.clicked.connect(lambda: self._move_selected_row(1))
        self.move_down_btn.setEnabled(False)
        button_layout.addWidget(self.move_down_btn)

        # 删除按钮
        self.remove_btn = QPushButton("删除选中")
        self.remove_btn.clicked.connect(self.remove_selected_variable)
        self.remove_btn.setEnabled(False)
        button_layout.addWidget(self.remove_btn)

        # 清空按钮
        self.clear_btn = QPushButton("清空所有")
        self.clear_btn.clicked.connect(self.clear_all_variables)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)

        # 重置颜色按钮
        self.reset_color_btn = QPushButton("重置颜色")
        self.reset_color_btn.clicked.connect(self.reset_curve_colors)
        self.reset_color_btn.setEnabled(False)
        button_layout.addWidget(self.reset_color_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 说明文本
        info_label = QLabel("提示：从变量表拖拽变量到此窗口可添加新变量")
        info_label.setStyleSheet("color: gray; font-size: 12px; margin-top: 10px;")
        layout.addWidget(info_label)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        
        # 连接信号
        self.var_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.var_table.cellClicked.connect(self.on_cell_clicked)
        # 不再需要itemChanged信号，因为使用QCheckBox控件
        
    def load_current_curves(self):
        """加载当前绘图中的曲线"""
        # 先清空表格
        self.var_table.setRowCount(0)
        
        # 检查多曲线模式
        if self.plot_widget.curves:
            # 有curves字典：从curves字典加载（无论是否是多曲线模式）
            for var_name, curve_info in self.plot_widget.curves.items():
                self._add_variable_to_table(var_name, curve_info)
        elif self.plot_widget.curve and self.plot_widget.y_name:
            # 单曲线模式：从curve和y_name加载
            var_name = self.plot_widget.y_name
            
            # 获取曲线的实际可见性状态
            curve_visible = True
            try:
                if hasattr(self.plot_widget.curve, 'isVisible'):
                    curve_visible = self.plot_widget.curve.isVisible()
            except Exception as e:
                print(f"获取曲线可见性失败: {e}")
            
            # 获取曲线的实际颜色
            curve_color = 'blue'
            try:
                if hasattr(self.plot_widget.curve, 'opts') and 'pen' in self.plot_widget.curve.opts:
                    pen = self.plot_widget.curve.opts['pen']
                    if hasattr(pen, 'color'):
                        curve_color = pen.color().name()
            except Exception as e:
                print(f"获取曲线颜色失败: {e}")
            
            curve_info = CurveInfo(
                var_name=var_name,
                curve=self.plot_widget.curve,
                x_data=np.array([]),
                y_data=np.array([]),
                color=curve_color,
                visible=curve_visible,
                y_format=self.plot_widget.y_format,
            )
            self._add_variable_to_table(var_name, curve_info)
        
        self.update_button_states()

    def _get_selected_row(self) -> int | None:
        """获取当前选中的行号"""
        selected_items = self.var_table.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].row()

    def _move_selected_row(self, offset: int):
        """移动选中行的位置"""
        if self.var_table.rowCount() <= 1 or not self.plot_widget.curves:
            return
        current_row = self._get_selected_row()
        if current_row is None:
            return
        target_row = current_row + offset
        if target_row < 0 or target_row >= self.var_table.rowCount():
            return

        order = []
        for row in range(self.var_table.rowCount()):
            name_item = self.var_table.item(row, 1)
            if name_item is not None:
                order.append(name_item.data(Qt.ItemDataRole.UserRole))

        if len(order) <= 1:
            return

        order[current_row], order[target_row] = order[target_row], order[current_row]
        self._apply_curve_order(order)
        self.load_current_curves()
        self.var_table.selectRow(target_row)

    def _apply_curve_order(self, new_order: list[str]):
        """根据给定顺序重排plot中的曲线"""
        if not self.plot_widget.curves:
            return
        reordered: dict[str, dict] = {}
        for name in new_order:
            if name in self.plot_widget.curves:
                reordered[name] = self.plot_widget.curves[name]
        # 附加遗漏的变量（理论上不会发生）
        for name, info in self.plot_widget.curves.items():
            if name not in reordered:
                reordered[name] = info
        self.plot_widget.curves = reordered
        self.plot_widget.update_legend()
        # 立即刷新光标显示顺序
        self.plot_widget._clear_cursor_items()
        if self.plot_widget.vline.isVisible():
            self.plot_widget.update_cursor_label()
    
    def _add_variable_to_table(self, var_name, curve_info):
        """添加变量到表格"""
        row = self.var_table.rowCount()
        self.var_table.insertRow(row)
        
        # 显示状态复选框 - 使用QCheckBox控件
        checkbox = QCheckBox()
        checkbox.setChecked(curve_info.visible)
        checkbox.stateChanged.connect(lambda state, name=var_name: self._on_checkbox_changed(name, state))
        self.var_table.setCellWidget(row, 0, checkbox)
        
        # 获取可见性状态
        is_visible = curve_info.visible
        
        # 变量名和单位
        unit = self.plot_widget.units.get(var_name, '')
        display_text = f"{var_name} ({unit})" if unit else var_name
        name_item = QTableWidgetItem(display_text)
        name_item.setData(Qt.ItemDataRole.UserRole, var_name)
        name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 1, name_item)
        
        # 颜色 - 使用QWidget显示真实颜色
        color = curve_info.color
        color_widget = QWidget()
        color_widget.setStyleSheet(f"background-color: {color}; border: 1px solid #333;")
        color_widget.setFixedSize(30, 20)
        self.var_table.setCellWidget(row, 2, color_widget)
        
        # 同时设置一个隐藏的item来存储数据
        color_item = QTableWidgetItem()
        color_item.setData(Qt.ItemDataRole.UserRole, var_name)
        color_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 2, color_item)
        
    def _on_checkbox_changed(self, var_name, state):
        """复选框状态变化处理"""
        is_visible = state == Qt.CheckState.Checked.value
        
        # 更新曲线可见性
        if self.plot_widget.curves and var_name in self.plot_widget.curves:
            self.plot_widget.curves[var_name].visible = is_visible
            ci = self.plot_widget.curves[var_name]
            if ci.curve is not None:
                curve_obj = ci.curve
                curve_obj.setVisible(is_visible)
            self.plot_widget.update_legend()
        elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
            if self.plot_widget.curve:
                self.plot_widget.curve.setVisible(is_visible)

    def on_selection_changed(self):
        """选择改变时的处理"""
        self.update_button_states()
    
    def on_cell_clicked(self, row, column):
        """单元格点击事件"""
        if column == 2:  # 颜色列
            self.set_variable_color(row)
    
    def toggle_variable_visibility(self, row):
        """切换变量显示状态"""
        var_name = self.var_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        visible_item = self.var_table.item(row, 0)
        is_visible = visible_item.checkState() == Qt.CheckState.Checked
        
        if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
            # 多曲线模式：更新curves字典中的可见性
            self.plot_widget.curves[var_name].visible = is_visible
            
            # 更新曲线显示
            ci = self.plot_widget.curves[var_name]
            if ci.curve is not None:
                try:
                    if ci.curve.scene() is not None:
                        ci.curve.setVisible(is_visible)
                    else:
                        # 曲线对象已经不在scene中，重新创建
                        self.plot_widget._recreate_curve(var_name)
                except Exception as e:
                    print(f"Warning: Error toggling curve visibility for {var_name}: {e}")
                    # 尝试重新创建曲线
                    self.plot_widget._recreate_curve(var_name)
            
            # 更新legend
            self.plot_widget.update_legend()
        elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
            # 单曲线模式：更新curve的可见性
            if self.plot_widget.curve:
                try:
                    if self.plot_widget.curve.scene() is not None:
                        self.plot_widget.curve.setVisible(is_visible)
                except Exception as e:
                    print(f"Warning: Error toggling single curve visibility: {e}")
        
    def update_button_states(self):
        """更新按钮状态"""
        has_selection = len(self.var_table.selectedItems()) > 0
        has_items = self.var_table.rowCount() > 0
        
        self.remove_btn.setEnabled(has_selection)
        self.clear_btn.setEnabled(has_items)
        self.reset_color_btn.setEnabled(has_items)

        selected_row = self._get_selected_row()
        can_move = (
            has_selection
            and self.var_table.rowCount() > 1
            and bool(self.plot_widget.curves)
        )
        self.move_up_btn.setEnabled(can_move and selected_row is not None and selected_row > 0)
        self.move_down_btn.setEnabled(
            can_move and selected_row is not None and selected_row < self.var_table.rowCount() - 1
        )
        
    def remove_selected_variable(self):
        """删除选中的变量"""
        selected_items = self.var_table.selectedItems()
        if not selected_items:
            return
        
        # 获取所有选中的行号
        selected_rows = set()
        for item in selected_items:
            selected_rows.add(item.row())
        
        # 记录最小的被删除行号，用于后续选中
        min_deleted_row = min(selected_rows)
        
        # 从后往前删除，避免行号变化
        for row in sorted(selected_rows, reverse=True):
            # 获取变量名 - 现在从第二列（变量名列）获取
            var_name_item = self.var_table.item(row, 1)
            if var_name_item is None:
                continue
            var_name = var_name_item.data(Qt.ItemDataRole.UserRole)
            
            if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
                # 多曲线模式：从curves字典中移除
                ci = self.plot_widget.curves[var_name]
                if ci.curve is not None and ci.curve.scene() is not None:
                    self.plot_widget.plot_item.removeItem(ci.curve)
                del self.plot_widget.curves[var_name]
            elif var_name in self.plot_widget.curves:
                # 单曲线模式但曲线在curves字典中：从curves字典中移除
                ci = self.plot_widget.curves[var_name]
                if ci.curve is not None and ci.curve.scene() is not None:
                    self.plot_widget.plot_item.removeItem(ci.curve)
                del self.plot_widget.curves[var_name]
            elif not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name:
                # 单曲线模式：清除整个plot
                self.plot_widget.clear_plot_item()
            
            # 从表格中移除
            self.var_table.removeRow(row)
        
        # 删除后自动选中下一条或上一条曲线
        row_count = self.var_table.rowCount()
        if row_count > 0:
            # 优先选中下一条（原来被删除行的位置）
            if min_deleted_row < row_count:
                next_row = min_deleted_row
            else:
                # 如果没有下一条，选中上一条
                next_row = row_count - 1
            
            # 选中整行
            self.var_table.selectRow(next_row)
        
        # 更新多曲线模式
        self.plot_widget.update_multi_curve_mode()
        
        # 更新vline bounds以反映移除变量后的数据范围
        self.plot_widget._update_vline_bounds_from_data()
        
        # 如果删除了所有曲线，确保完全清理
        if not self.plot_widget.curves:
            # 清理所有可能的残留
            if self.plot_widget.curve and self.plot_widget.curve.scene() is not None:
                self.plot_widget.plot_item.removeItem(self.plot_widget.curve)
            self.plot_widget.curve = None
            self.plot_widget.y_name = ''
            self.plot_widget.y_format = ''
            self.plot_widget.original_index_x = None
            self.plot_widget.original_y = None
            self.plot_widget.current_color_index = 0
            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.update_left_header("channel name")
            self.plot_widget.update_right_header("")
            
            # 清理所有plot item（先清除cursor items）
            # 清空所有变量时完全清除对象池，避免复用异常状态的items
            self.plot_widget._clear_cursor_items(hide_only=False)
            self.plot_widget._safe_clear_plot_items()

        self.plot_widget._recalc_max_point_density()
        main_window = self.plot_widget.window()
        if main_window is not None and hasattr(main_window, '_sync_min_xrange'):
            main_window._sync_min_xrange()

        self.update_button_states()
        
    def clear_all_variables(self):
        """清空所有变量"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有绘图变量吗？",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # 清空所有曲线
            if self.plot_widget.is_multi_curve_mode:
                # 多曲线模式：清空curves字典
                for var_name, ci in list(self.plot_widget.curves.items()):
                    if ci.curve is not None and ci.curve.scene() is not None:
                        self.plot_widget.plot_item.removeItem(ci.curve)
                self.plot_widget.curves.clear()
            else:
                # 单曲线模式：清空整个plot
                self.plot_widget.clear_plot_item()
            
            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.current_color_index = 0
            
            # 清空表格
            self.var_table.setRowCount(0)
            self.update_button_states()
            
            # 更新显示
            self.plot_widget.update_left_header("channel name")
            self.plot_widget.update_right_header("")
            
            # 重置vline bounds到默认值
            self.plot_widget._update_vline_bounds_from_data()
            
    def reset_curve_colors(self):
        """按照默认顺序重新分配曲线颜色"""
        row_count = self.var_table.rowCount()
        if row_count == 0:
            return
        color_cycle = getattr(self.plot_widget, 'curve_colors', ['blue'])
        if not color_cycle:
            return

        if self.plot_widget.curves:
            for idx, var_name in enumerate(self.plot_widget.curves.keys()):
                color_name = color_cycle[idx % len(color_cycle)]
                self._apply_color_to_curve(var_name, color_name)
        elif self.plot_widget.curve and self.plot_widget.y_name:
            self._apply_color_to_curve(self.plot_widget.y_name, color_cycle[0])

        # 重新加载表格以更新颜色显示
        selected_row = self._get_selected_row()
        self.load_current_curves()
        if selected_row is not None and selected_row < self.var_table.rowCount():
            self.var_table.selectRow(selected_row)

    def _apply_color_to_curve(self, var_name: str, color_name: str):
        """将指定变量的颜色更新为给定颜色"""
        updated = False
        if self.plot_widget.curves and var_name in self.plot_widget.curves:
            ci = self.plot_widget.curves[var_name]
            ci.color = color_name
            if ci.curve is not None:
                curve_obj = ci.curve
                old_pen = curve_obj.opts.get('pen')
                width = DEFAULT_LINE_WIDTH
                if hasattr(old_pen, 'widthF'):
                    width = old_pen.widthF()
                elif hasattr(old_pen, 'width'):
                    width = old_pen.width()
                curve_obj.setPen(pg.mkPen(color=color_name, width=width))

                # 如果curve当前有symbols，也需要更新symbol的颜色
                if hasattr(curve_obj, '_has_symbols') and curve_obj._has_symbols:
                    curve_obj.setSymbolPen(color_name)
                    curve_obj.setSymbolBrush(color_name)

                # 清除缓存标志，强制下次刷新时重新应用样式
                if hasattr(curve_obj, '_cached_pen_key'):
                    delattr(curve_obj, '_cached_pen_key')

            updated = True
        elif var_name == self.plot_widget.y_name and self.plot_widget.curve:
            old_pen = self.plot_widget.curve.opts.get('pen')
            width = DEFAULT_LINE_WIDTH
            if hasattr(old_pen, 'widthF'):
                width = old_pen.widthF()
            elif hasattr(old_pen, 'width'):
                width = old_pen.width()
            self.plot_widget.curve.setPen(pg.mkPen(color=color_name, width=width))

            # 如果curve当前有symbols，也需要更新symbol的颜色
            if hasattr(self.plot_widget.curve, '_has_symbols') and self.plot_widget.curve._has_symbols:
                self.plot_widget.curve.setSymbolPen(color_name)
                self.plot_widget.curve.setSymbolBrush(color_name)

            # 清除缓存标志，强制下次刷新时重新应用样式
            if hasattr(self.plot_widget.curve, '_cached_pen_key'):
                delattr(self.plot_widget.curve, '_cached_pen_key')

            updated = True

        if updated:
            self.plot_widget.update_legend()
            # 重新应用样式以确保symbol/线宽保持一致
            if hasattr(self.plot_widget, '_queue_ui_refresh'):
                self.plot_widget._queue_ui_refresh(immediate=True)
            self.plot_widget._clear_cursor_items()
            if hasattr(self.plot_widget, '_last_cursor_update_time'):
                self.plot_widget._last_cursor_update_time = 0
            if self.plot_widget.vline.isVisible():
                self.plot_widget.update_cursor_label()

    def set_variable_color(self, row=None):
        """设置变量颜色"""
        if row is None:
            # 从选中项获取行号
            selected_items = self.var_table.selectedItems()
            if not selected_items:
                return
            row = selected_items[0].row()
        
        # 获取变量名 - 现在从第二列（变量名列）获取
        var_name_item = self.var_table.item(row, 1)
        if var_name_item is None:
            return
        var_name = var_name_item.data(Qt.ItemDataRole.UserRole)
        
        # 打开颜色选择对话框
        current_color = 'blue'  # 默认颜色
        if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
            current_color = self.plot_widget.curves[var_name].get('color', 'blue')
        
        color = QColorDialog.getColor(QColor(current_color), self, "选择颜色")
        
        if color.isValid():
            self._apply_color_to_curve(var_name, color.name())
            # 更新表格项颜色
            color_widget = self.var_table.cellWidget(row, 2)
            if color_widget:
                color_widget.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #333;")
                
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasText():
            var_names = self.plot_widget._extract_var_names_from_text(event.mimeData().text())
            self.plot_widget._notify_drag_indicator(
                var_names,
                hide=False,
                source_widget=self,
                indicator_text="释放以添加"
            )
            event.acceptProposedAction()
        else:
            self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
            event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        if event.mimeData().hasText():
            var_names = self.plot_widget._extract_var_names_from_text(event.mimeData().text())
            self.plot_widget._notify_drag_indicator(
                var_names,
                hide=False,
                source_widget=self,
                indicator_text="释放以添加"
            )
            event.acceptProposedAction()
        else:
            self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
        event.accept()
            
    def dropEvent(self, event):
        """拖拽放下事件，支持单个或多个变量同时拖入"""
        if event.mimeData().hasText():
            var_names = parse_var_names_from_mimedata(event.mimeData())

            if len(var_names) > 1:
                # 多个变量：批量添加
                failed_vars = []
                success_count = 0

                for var_name in var_names:
                    # 检查变量是否已存在
                    if (self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves) or \
                       (not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name):
                        failed_vars.append(f"{var_name} (已存在)")
                        continue

                    # 添加变量到绘图
                    success = self.plot_widget.add_variable_to_plot(var_name)
                    if success:
                        success_count += 1
                    else:
                        failed_vars.append(var_name)

                # 重新加载列表以显示新添加的变量
                if success_count > 0:
                    self.load_current_curves()

                # 显示结果消息（只在有失败时提示）
                if failed_vars:
                    QMessageBox.warning(self, "批量添加结果",
                                      f"成功添加 {success_count} 个变量\n失败的变量: {', '.join(failed_vars)}")
            else:
                # 单个变量：原有逻辑
                var_name = var_names[0] if var_names else ""
                if not var_name:
                    event.ignore()
                    self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
                    return

                # 检查变量是否已存在
                if (self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves) or \
                   (not self.plot_widget.is_multi_curve_mode and var_name == self.plot_widget.y_name):
                    QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return

                # 添加变量到绘图
                success = self.plot_widget.add_variable_to_plot(var_name)
                if success:
                    # 重新加载列表以显示新添加的变量
                    self.load_current_curves()
                else:
                    QMessageBox.warning(self, "错误", f"无法添加变量 {var_name}")

            event.acceptProposedAction()
        else:
            event.ignore()
        self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)

