"""绘图变量编辑器"""

from __future__ import annotations
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QCheckBox,
    QColorDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
)
import pyqtgraph as pg
from src.core.config import DEFAULT_LINE_WIDTH
from src.core.data_types import CurveInfo
from src.core.logger import get_logger
from src.ui.drag_drop import parse_var_names_from_mimedata
from src.ui.widgets.variable_search_bar import VariableSearchBar

logger = get_logger(__name__)


class PlotVariableEditorDialog(QDialog):
    """
    绘图变量编辑器对话框类
    用于管理plot中的多个曲线，支持添加、删除、颜色自定义等功能
    """

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("绘图变量编辑器")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.resize(600, 600)  # 高度从 400 调到 600，留出候选列表空间
        self.setAcceptDrops(True)

        # 高DPI支持 - PySide6中不需要WA_UseHighDpiPixmaps
        # PySide6默认支持高DPI，通过样式表控制字体大小

        self.setup_ui()
        self.load_current_curves()

    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("绘图变量编辑器")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; margin-bottom: 15px;"
        )
        layout.addWidget(title_label)

        # ===== 内嵌搜索栏 =====
        # 数据源：从主窗口获取 var_names 与 data_validity；units 用 plot_widget.units（与现有 _add_variable_to_table 一致）
        main_window = self.plot_widget.window()
        if (
            main_window is not None
            and getattr(main_window, "loader", None) is not None
        ):
            self.search_bar = VariableSearchBar(
                var_names=list(main_window.loader.var_names),
                units=self.plot_widget.units or {},
                validity=getattr(main_window, "data_validity", None) or {},
                plot_widget=self.plot_widget,
                parent=self,
            )
            self.search_bar.variable_selected.connect(self._on_variable_selected)
            self.search_bar.variable_removed.connect(self._on_variable_removed)
            # 搜索框为空时按 Esc → 焦点切回表格（取消聚焦）
            self.search_bar.escape_pressed.connect(self._on_search_escape)
            layout.addWidget(self.search_bar)

            # 快捷键：焦点切到搜索栏
            # 1) StandardKey.Find：mac=⌘F，Win/Linux=Ctrl+F（符合平台查找习惯）
            self._search_shortcut_find = QShortcut(
                QKeySequence(QKeySequence.StandardKey.Find), self
            )
            self._search_shortcut_find.activated.connect(self._focus_search_bar)
            # 2) Insert：仅非 mac 平台注册（macBook 无独立 Insert 键，注册也无响应）
            if sys.platform != "darwin":
                self._search_shortcut_insert = QShortcut(
                    QKeySequence(Qt.Key.Key_Insert), self
                )
                self._search_shortcut_insert.activated.connect(self._focus_search_bar)
        else:
            self.search_bar = None
        # ======================

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
        self.var_table.setColumnWidth(0, 60)  # 显示列
        self.var_table.setColumnWidth(2, 80)  # 颜色列

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
        # 拖拽添加等外部变化时刷新表格（必需，否则拖拽后表格不更新）
        self.plot_widget.curves_changed.connect(self.load_current_curves)

    def _focus_search_bar(self):
        """⌘F / Ctrl+F / Insert：焦点切到搜索栏并全选文本"""
        if self.search_bar is not None:
            self.search_bar.focus_search_edit()

    def _on_search_escape(self):
        """搜索框为空时按 Esc：取消聚焦，焦点切回表格"""
        self.var_table.setFocus()

    def _on_variable_selected(self, var_name: str):
        """处理搜索栏中选中（未添加）变量的添加请求

        - show_duplicate_warning=False：搜索栏已通过 select_first 跳过已添加项
          保证不会重复添加，此处抑制底层 warning 避免双重提示（双保险）
        - load_current_curves 由 curves_changed 信号触发，无需显式调用
        """
        success = self.plot_widget.add_variable_to_plot(
            var_name, show_duplicate_warning=False
        )
        if success:
            # 标记为"本次刚添加"：置灰但保持原位置，避免连续添加时列表频繁跳动
            if self.search_bar is not None:
                self.search_bar.mark_added(var_name)
        # 失败分支保持静默：搜索栏已通过 select_first 跳过已添加项保证不会重复添加

    def _on_variable_removed(self, var_name: str):
        """处理搜索栏中移除已添加变量的请求

        复用 _remove_selected_variable_impl 的删除逻辑，但不调用 reset_session：
        搜索栏移除是单次操作，不应清空 _last_op_var 锚点（否则后续键盘 Enter
        无法从被移除项往后定位下一个未添加项）。

        流程：
        1. 按变量名在表格中定位行
        2. selectRow 该行
        3. 调用 _remove_selected_variable_impl()（curves 删除/表格行删除/归一化/清理，不含会话重置）
        4. 调用 search_bar.mark_removed(var_name) 记录锚点（变量回原位）

        会话锚点清空仍由 Esc/关键词变化/表格删除按钮/清空 触发，搜索栏移除不在此列。
        """
        # 按变量名定位表格行（变量名在第 1 列，索引 1）
        target_row = None
        for row in range(self.var_table.rowCount()):
            item = self.var_table.item(row, 1)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == var_name:
                target_row = row
                break
        if target_row is None:
            # 变量不在表格中（理论上不会发生，防御性兜底）
            # 清空 _pending_mouse_top，避免搜索栏 _on_item_clicked emit 前已设置的
            # pending 值遗留，被后续 refresh 误用导致滚动条跳到错误位置
            if self.search_bar is not None:
                self.search_bar._pending_mouse_top = None
            return
        # 选中该行并复用核心删除逻辑（不含会话重置）
        self.var_table.selectRow(target_row)
        self._remove_selected_variable_impl()
        self.update_button_states()
        # 标记为"本次刚移除"：记录锚点 _last_op_var，变量回原位（不清空锚点，供后续 select_first 定位）
        if self.search_bar is not None:
            self.search_bar.mark_removed(var_name)
            # 焦点恢复：上方 selectRow / removeRow 会让 var_table 抢走焦点，
            # 移除完成后把焦点还给搜索框，用户可继续输入关键词或 Enter 操作下一项
            # 不用 focus_search_edit()（会 selectAll 打断输入），直接 setFocus
            self.search_bar.search_edit.setFocus()

    def load_current_curves(self):
        """加载当前绘图中的曲线"""
        import numpy as np
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
                if hasattr(self.plot_widget.curve, "isVisible"):
                    curve_visible = self.plot_widget.curve.isVisible()
            except Exception:
                logger.debug("curve.isVisible() 异常，跳过 visible 状态读取", exc_info=True)

            # 获取曲线的实际颜色
            curve_color = "blue"
            try:
                if (
                    hasattr(self.plot_widget.curve, "opts")
                    and "pen" in self.plot_widget.curve.opts
                ):
                    pen = self.plot_widget.curve.opts["pen"]
                    if hasattr(pen, "color"):
                        curve_color = pen.color().name()
            except Exception:
                logger.debug("pen.color() 异常，使用默认颜色", exc_info=True)

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
        checkbox.stateChanged.connect(
            lambda state, name=var_name: self._on_checkbox_changed(name, state)
        )
        self.var_table.setCellWidget(row, 0, checkbox)

        # 变量名和单位
        unit = self.plot_widget.units.get(var_name, "")
        display_text = f"{var_name} ({unit})" if unit else var_name
        name_item = QTableWidgetItem(display_text)
        name_item.setData(Qt.ItemDataRole.UserRole, var_name)
        name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 1, name_item)

        # 颜色 - 使用QWidget显示真实颜色
        color = curve_info.color
        color_widget = QWidget()
        color_widget.setStyleSheet(
            f"background-color: {color}; border: 1px solid #333;"
        )
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
        elif (
            not self.plot_widget.is_multi_curve_mode
            and var_name == self.plot_widget.y_name
        ):
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
                        self.plot_widget._recreate_curve(var_name)
                except Exception:
                    logger.debug("_recreate_curve(%s) 异常，跳过", var_name, exc_info=True)

            # 更新legend
            self.plot_widget.update_legend()
        elif (
            not self.plot_widget.is_multi_curve_mode
            and var_name == self.plot_widget.y_name
        ):
            # 单曲线模式：更新curve的可见性
            if self.plot_widget.curve:
                try:
                    if self.plot_widget.curve.scene() is not None:
                        self.plot_widget.curve.setVisible(is_visible)
                except Exception:
                    logger.debug("curve.setVisible(%s) 异常，跳过", is_visible, exc_info=True)
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
        self.move_up_btn.setEnabled(
            can_move and selected_row is not None and selected_row > 0
        )
        self.move_down_btn.setEnabled(
            can_move
            and selected_row is not None
            and selected_row < self.var_table.rowCount() - 1
        )

    def closeEvent(self, event):
        """关闭时断开 curves_changed 信号，避免重复打开编辑器导致信号连接累积"""
        try:
            self.plot_widget.curves_changed.disconnect(self.load_current_curves)
        except (TypeError, RuntimeError):
            pass
        if self.search_bar is not None:
            try:
                self.plot_widget.curves_changed.disconnect(self.search_bar._on_curves_changed)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)

    def remove_selected_variable(self):
        """表格"删除选中"按钮入口：删除选中行 + 重置搜索会话

        表格删除是显式会话结束动作（用户心智：删除 = 这批操作到此为止），
        因此末尾调用 reset_session 清空搜索栏的 _last_op_var 锚点。

        注意：搜索栏内移除变量不应调本方法（会清空锚点，影响后续 select_first 定位），
        应改调 _remove_selected_variable_impl + mark_removed（见 _on_variable_removed）。
        """
        self._remove_selected_variable_impl()
        self.update_button_states()
        # 删除变量 → 重置搜索会话（清锚点 + 刷新已添加样式）
        if self.search_bar is not None:
            self.search_bar.reset_session()

    def _remove_selected_variable_impl(self):
        """删除选中行的核心逻辑（不含会话重置）

        供 remove_selected_variable（表格删除按钮）和 _on_variable_removed
        （搜索栏移除）共用。调用方负责后续的会话状态处理：
        - 表格删除：调 reset_session 清锚点
        - 搜索栏移除：调 mark_removed 保持锚点（保持原位 + select_first 定位）
        """
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

            if (
                self.plot_widget.is_multi_curve_mode
                and var_name in self.plot_widget.curves
            ):
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
            elif (
                not self.plot_widget.is_multi_curve_mode
                and var_name == self.plot_widget.y_name
            ):
                # 单曲线模式：清除整个plot
                self.plot_widget.clear_plot_item()

            # 修复：del curves[var_name] 只删字典项，不会清 y_name 残留
            # （add_variable_to_plot 单→多切换时 y_name 未清空，仍指向已删除变量）
            # 搜索栏 _get_existing_set 会把 y_name 误判为"已添加"，导致样式不刷新
            # 此处统一清理：第三个分支 clear_plot_item 已清 y_name，此条件对其为 False，无副作用
            if self.plot_widget.y_name == var_name:
                self.plot_widget.y_name = ""
                self.plot_widget.y_format = ""

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

        # 修复：多曲线模式下 y_name 可能是历史残留（add_variable_to_plot 单→多切换时未清空）
        # 删除变量后若 y_name 指向已不存在的变量，清空避免搜索栏 _get_existing_set 误判为已添加
        if self.plot_widget.is_multi_curve_mode and self.plot_widget.curves:
            if self.plot_widget.y_name and self.plot_widget.y_name not in self.plot_widget.curves:
                self.plot_widget.y_name = ""
                self.plot_widget.y_format = ""

        # 修复：从多曲线降至1条时，归一化为单曲线状态
        if len(self.plot_widget.curves) == 1 and not self.plot_widget.is_multi_curve_mode:
            single_ci = next(iter(self.plot_widget.curves.values()))
            saved_color = single_ci.color
            self.plot_widget.plot_variable(single_ci.var_name)
            # 恢复用户自定义颜色（plot_variable 会重置为蓝色）
            try:
                if self.plot_widget.curve is not None and hasattr(self.plot_widget.curve, 'opts'):
                    pen = pg.mkPen(color=saved_color, width=DEFAULT_LINE_WIDTH)
                    self.plot_widget.curve.setPen(pen)
            except Exception:
                logger.debug("删除归一化后恢复曲线颜色失败", exc_info=True)

            # 归一化的 plot_variable 会 emit curves_changed → load_current_curves
            # 重建表格，清除上方 selectRow(next_row) 设置的选中。归一化后表格仅 1 行，重选行 0
            if self.var_table.rowCount() > 0:
                self.var_table.selectRow(0)

        # 更新vline bounds以反映移除变量后的数据范围
        self.plot_widget._update_vline_bounds_from_data()

        # 刷新 cursor 标签以反映曲线删除
        if self.plot_widget.vline.isVisible():
            self.plot_widget.update_cursor_label()

        # 如果删除了所有曲线，确保完全清理
        # 注意：2→1 场景归一化块（上方）会把唯一曲线从 curves 迁移到 y_name，
        # 此时 curves 为空但 y_name 有值，不应进入全清理块（否则会把归一化保留的
        # y_name 也清掉，导致搜索栏 _get_existing_set 误判所有变量为"未添加"）
        if not self.plot_widget.curves and not self.plot_widget.y_name:
            # 清理所有可能的残留
            if self.plot_widget.curve and self.plot_widget.curve.scene() is not None:
                self.plot_widget.plot_item.removeItem(self.plot_widget.curve)
            self.plot_widget.curve = None
            self.plot_widget.y_name = ""
            self.plot_widget.y_format = ""
            self.plot_widget.original_index_x = None
            self.plot_widget.original_y = None
            self.plot_widget.current_color_index = 0
            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.update_left_header("channel name")
            self.plot_widget.update_right_header("")

            # 先重置 Y 轴范围（与 clear_plot_item 顺序一致：先 reset 再 clear）
            self.plot_widget._reset_plot_limits()
            # 清理所有plot item（先清除cursor items）
            # 清空所有变量时完全清除对象池，避免复用异常状态的items
            self.plot_widget._clear_cursor_items(hide_only=False)
            self.plot_widget._safe_clear_plot_items()
            # 通知其他 widget 曲线已清空（与 clear_plot_item 的 emit 行为一致）
            self.plot_widget.curves_changed.emit()

        self.plot_widget._recalc_max_point_density()
        main_window = self.plot_widget.window()
        if main_window is not None and hasattr(main_window, "cursor_sync_manager"):
            main_window.cursor_sync_manager._sync_min_xrange()

    def clear_all_variables(self):
        """清空所有变量"""
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要清空所有绘图变量吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 清空所有曲线
            if self.plot_widget.is_multi_curve_mode:
                # 多曲线模式：清空curves字典
                for var_name, ci in list(self.plot_widget.curves.items()):
                    if ci.curve is not None and ci.curve.scene() is not None:
                        self.plot_widget.plot_item.removeItem(ci.curve)
                self.plot_widget.curves.clear()
                # 清空单曲线残留状态（y_name 等）
                # 否则搜索栏 _get_existing_set() 会误判 y_name 仍为"已添加"
                self.plot_widget.y_name = ""
                self.plot_widget.y_format = ""
                self.plot_widget.curve = None
                self.plot_widget.original_index_x = None
                self.plot_widget.original_y = None
                # 重置 Y 轴范围到默认 0-1（与单曲线分支 clear_plot_item 行为一致）
                self.plot_widget._reset_plot_limits()
            else:
                # 单曲线模式：清空整个plot（clear_plot_item 内部会清空 y_name 等）
                self.plot_widget.clear_plot_item()

            self.plot_widget.is_multi_curve_mode = False
            self.plot_widget.current_color_index = 0

            # 清空表格
            self.var_table.setRowCount(0)
            self.update_button_states()
            # 清空所有 → 重置搜索会话（清锚点 + 刷新已添加样式）
            if self.search_bar is not None:
                self.search_bar.reset_session()

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
        color_cycle = getattr(self.plot_widget, "curve_colors", ["blue"])
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
                old_pen = curve_obj.opts.get("pen")
                width = DEFAULT_LINE_WIDTH
                if hasattr(old_pen, "widthF"):
                    width = old_pen.widthF()
                elif hasattr(old_pen, "width"):
                    width = old_pen.width()
                curve_obj.setPen(pg.mkPen(color=color_name, width=width))

                # 如果curve当前有symbols，也需要更新symbol的颜色
                if hasattr(curve_obj, "_has_symbols") and curve_obj._has_symbols:
                    curve_obj.setSymbolPen(color_name)
                    curve_obj.setSymbolBrush(color_name)

                # 清除缓存标志，强制下次刷新时重新应用样式
                if hasattr(curve_obj, "_cached_pen_key"):
                    delattr(curve_obj, "_cached_pen_key")

            updated = True
        elif var_name == self.plot_widget.y_name and self.plot_widget.curve:
            old_pen = self.plot_widget.curve.opts.get("pen")
            width = DEFAULT_LINE_WIDTH
            if hasattr(old_pen, "widthF"):
                width = old_pen.widthF()
            elif hasattr(old_pen, "width"):
                width = old_pen.width()
            self.plot_widget.curve.setPen(pg.mkPen(color=color_name, width=width))

            # 如果curve当前有symbols，也需要更新symbol的颜色
            if (
                hasattr(self.plot_widget.curve, "_has_symbols")
                and self.plot_widget.curve._has_symbols
            ):
                self.plot_widget.curve.setSymbolPen(color_name)
                self.plot_widget.curve.setSymbolBrush(color_name)

            # 清除缓存标志，强制下次刷新时重新应用样式
            if hasattr(self.plot_widget.curve, "_cached_pen_key"):
                delattr(self.plot_widget.curve, "_cached_pen_key")

            updated = True

        if updated:
            self.plot_widget.update_legend()
            # 重新应用样式以确保symbol/线宽保持一致
            if hasattr(self.plot_widget, "_queue_ui_refresh"):
                self.plot_widget._queue_ui_refresh(immediate=True)
            self.plot_widget._clear_cursor_items()
            if hasattr(self.plot_widget, "_last_cursor_update_time"):
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
        current_color = "blue"  # 默认颜色
        if self.plot_widget.is_multi_curve_mode and var_name in self.plot_widget.curves:
            current_color = self.plot_widget.curves[var_name].color

        color = QColorDialog.getColor(QColor(current_color), self, "选择颜色")

        if color.isValid():
            self._apply_color_to_curve(var_name, color.name())
            # 更新表格项颜色
            color_widget = self.var_table.cellWidget(row, 2)
            if color_widget:
                color_widget.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #333;"
                )

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasText():
            var_names = self.plot_widget._extract_var_names_from_text(
                event.mimeData().text()
            )
            self.plot_widget._notify_drag_indicator(
                var_names, hide=False, source_widget=self, indicator_text="释放以添加"
            )
            event.acceptProposedAction()
        else:
            self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
            event.ignore()

    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        if event.mimeData().hasText():
            var_names = self.plot_widget._extract_var_names_from_text(
                event.mimeData().text()
            )
            self.plot_widget._notify_drag_indicator(
                var_names, hide=False, source_widget=self, indicator_text="释放以添加"
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
                    if (
                        self.plot_widget.is_multi_curve_mode
                        and var_name in self.plot_widget.curves
                    ) or (
                        not self.plot_widget.is_multi_curve_mode
                        and var_name == self.plot_widget.y_name
                    ):
                        failed_vars.append(f"{var_name} (已存在)")
                        continue

                    # 添加变量到绘图
                    success = self.plot_widget.add_variable_to_plot(var_name)
                    if success:
                        success_count += 1
                    else:
                        failed_vars.append(var_name)

                # 重新加载列表以显示新添加的变量
                # load_current_curves 由 curves_changed 信号触发（每次 add_variable_to_plot 成功都 emit）
                # 注意：批量添加时信号会多次触发 load_current_curves，可接受（表格行数小）

                # 显示结果消息（只在有失败时提示）
                if failed_vars:
                    QMessageBox.warning(
                        self,
                        "批量添加结果",
                        f"成功添加 {success_count} 个变量\n失败的变量: {', '.join(failed_vars)}",
                    )
            else:
                # 单个变量：原有逻辑
                var_name = var_names[0] if var_names else ""
                if not var_name:
                    event.ignore()
                    self.plot_widget._notify_drag_indicator(
                        hide=True, source_widget=self
                    )
                    return

                # 检查变量是否已存在
                if (
                    self.plot_widget.is_multi_curve_mode
                    and var_name in self.plot_widget.curves
                ) or (
                    not self.plot_widget.is_multi_curve_mode
                    and var_name == self.plot_widget.y_name
                ):
                    QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return

                # 添加变量到绘图
                success = self.plot_widget.add_variable_to_plot(var_name)
                # load_current_curves 由 curves_changed 信号触发，无需显式调用
                if not success:
                    QMessageBox.warning(self, "错误", f"无法添加变量 {var_name}")

            event.acceptProposedAction()
        else:
            event.ignore()
        self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
