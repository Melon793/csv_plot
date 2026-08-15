"""绘图变量编辑器"""

from __future__ import annotations
import sys
from PySide6.QtCore import Qt, QSignalBlocker
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
        # 性能优化（方案 A）：改走增量更新，仅增删变化行；复杂场景内部回退全量重建
        self.plot_widget.curves_changed.connect(self._on_curves_changed_incremental)

    def _focus_search_bar(self):
        """⌘F / Ctrl+F / Insert：焦点切到搜索栏并全选文本"""
        if self.search_bar is not None:
            self.search_bar.focus_search_edit()

    def _on_search_escape(self):
        """搜索框为空时按 Esc：取消聚焦，焦点切回表格"""
        self.var_table.setFocus()

    def refresh_data_source(self):
        """数据重载/新加载后就地更新搜索栏数据源（保留搜索词）

        由 file_loader_manager 加载完成路径通过
        main_window.findChildren(PlotVariableEditorDialog) 逐个调用。

        存活检查：subplot 关闭/重建场景下 plot_widget 的 C++ 对象可能已销毁，
        此时静默跳过本实例，避免阻断其他实例的更新。
        """
        if self.search_bar is None:
            return
        pw = self.plot_widget
        if pw is None:
            return
        main_window = self.window()
        loader = getattr(main_window, "loader", None)
        if loader is None:
            return
        try:
            units = pw.units or {}
        except RuntimeError:
            # C++ 对象已销毁（subplot 关闭/重建）：静默跳过
            logger.debug("refresh_data_source: plot_widget 已销毁，跳过更新")
            return
        self.search_bar.update_data_source(
            var_names=list(loader.var_names),
            units=units,
            validity=getattr(main_window, "data_validity", None) or {},
        )

    def _on_variable_selected(self, var_name: str):
        """处理搜索栏中选中（未添加）变量的添加请求

        - show_duplicate_warning=False：搜索栏已跳过已添加项
          保证不会重复添加，此处抑制底层 warning 避免双重提示（双保险）
        - 表格刷新由 curves_changed 信号触发增量更新，无需显式调用
        """
        success = self.plot_widget.add_variable_to_plot(
            var_name, show_duplicate_warning=False
        )
        if success:
            # 标记为"本次刚添加"：增量更新样式（置灰），列表不重建不跳动
            if self.search_bar is not None:
                self.search_bar.mark_added(var_name)
        else:
            # 失败分支仅记录日志（不弹框）：正常路径下搜索栏已跳过已添加项，
            # 失败多见于陈旧快照等异常场景（如重载中间态），便于排查
            logger.debug("搜索栏添加变量失败: %s", var_name)

    def _on_variable_removed(self, var_name: str):
        """处理搜索栏中移除已添加变量的请求

        复用 _remove_selected_variable_impl 的删除逻辑，但不调用 reset_session：
        搜索栏移除是单次操作，移除后选中停留原项（误删可立即重新 Enter 添加）。

        流程：
        1. 按变量名在表格中定位行
        2. selectRow 该行
        3. 调用 _remove_selected_variable_impl()（curves 删除/表格行删除/归一化/清理，不含会话重置）
        4. 调用 search_bar.mark_removed(var_name) 增量更新标记（变量回原位）
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
            return
        # 选中该行并复用核心删除逻辑（不含会话重置）
        self.var_table.selectRow(target_row)
        self._remove_selected_variable_impl()
        self.update_button_states()
        # 标记为"本次刚移除"：增量更新标记，变量回原位
        if self.search_bar is not None:
            self.search_bar.mark_removed(var_name)
            # 焦点恢复：上方 selectRow / removeRow 会让 var_table 抢走焦点，
            # 移除完成后把焦点还给搜索框，用户可继续输入关键词或 Enter 操作下一项
            # 不用 focus_search_edit()（会 selectAll 打断输入），直接 setFocus
            self.search_bar.search_edit.setFocus()

    def load_current_curves(self):
        """加载当前绘图中的曲线（统一版：始终从 curves 字典加载）"""
        # 先清空表格
        self.var_table.setRowCount(0)

        for var_name, curve_info in self.plot_widget.curves.items():
            self._add_variable_to_table(var_name, curve_info)

        self.update_button_states()

    def _on_curves_changed_incremental(self):
        """curves_changed 增量表格更新（性能优化方案 A）

        对表格行与 curves 字典做 diff，按场景分流：
        - 名称+顺序完全一致：轻量就地刷新（可见性/颜色），不重建任何控件；
        - 纯添加且新曲线全部在末尾：仅追加新行（Enter 连续添加的主场景）；
        - 纯删除且剩余顺序一致：仅删除对应行；
        - 其余复杂场景（重排、中间插入、混合增删）：回退 load_current_curves 全量重建。

        回退兜底保证正确性：增量路径出错的最坏结果也只是退回全量重建。
        """
        table = self.var_table
        table_names: list[str] = []
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            table_names.append(
                name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
            )
        curve_names = list(self.plot_widget.curves.keys())
        table_set = set(table_names)
        curve_set = set(curve_names)
        removed = table_set - curve_set
        added = curve_set - table_set

        path = "full_rebuild"
        try:
            if not removed and not added and table_names == curve_names:
                # 场景 1：集合与顺序均未变 → 就地刷新可见性/颜色
                self._refresh_rows_in_place()
                path = "in_place"
            elif not removed and added and curve_names[:-len(added)] == table_names:
                # 场景 2：纯添加且全部在末尾 → 追加行
                for name in curve_names[-len(added):]:
                    self._add_variable_to_table(name, self.plot_widget.curves[name])
                path = "append"
            elif not added and removed:
                # 场景 3：纯删除 → 删除对应行（倒序删除避免行号漂移）
                survivors = [n for n in table_names if n in curve_set]
                if survivors == curve_names:
                    for row in range(table.rowCount() - 1, -1, -1):
                        if table_names[row] in removed:
                            table.removeRow(row)
                    path = "remove"
        except Exception:
            logger.debug("增量表格更新异常，回退全量重建", exc_info=True)
            path = "full_rebuild"

        if path == "full_rebuild":
            self.load_current_curves()
            return

        self.update_button_states()

    def _refresh_rows_in_place(self):
        """就地刷新表格行的可见性与颜色（不重建控件）

        用于 curves 集合与顺序均未变的场景（如颜色修改、外部可见性变更）。
        """
        table = self.var_table
        for row in range(table.rowCount()):
            var_name = table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            curve_info = self.plot_widget.curves.get(var_name)
            if curve_info is None:
                continue
            checkbox = table.cellWidget(row, 0)
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(curve_info.visible)
            color_widget = table.cellWidget(row, 2)
            if color_widget is not None:
                color_widget.setStyleSheet(
                    f"background-color: {curve_info.color}; border: 1px solid #333;"
                )

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
        """复选框状态变化处理（统一版）"""
        is_visible = state == Qt.CheckState.Checked.value
        logger.debug(
            "[EDITOR] _on_checkbox_changed: var=%s, visible=%s", var_name, is_visible
        )

        if var_name in self.plot_widget.curves:
            self.plot_widget.curves[var_name].visible = is_visible
            ci = self.plot_widget.curves[var_name]
            if ci.curve is not None:
                ci.curve.setVisible(is_visible)
            self.plot_widget._update_header_for_curves()

    def on_selection_changed(self):
        """选择改变时的处理"""
        self.update_button_states()

    def on_cell_clicked(self, row, column):
        """单元格点击事件"""
        if column == 2:  # 颜色列
            self.set_variable_color(row)

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
            self.plot_widget.curves_changed.disconnect(self._on_curves_changed_incremental)
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
        因此末尾调用 reset_session 增量刷新搜索栏的"已添加"样式。

        注意：搜索栏内移除变量不应调本方法（reset_session 后选中停留原位，
        与搜索栏移除的键盘行为不一致），
        应改调 _remove_selected_variable_impl + mark_removed（见 _on_variable_removed）。
        """
        self._remove_selected_variable_impl()
        self.update_button_states()
        # 删除变量 → 重置搜索会话（增量刷新已添加样式，不跳顶）
        if self.search_bar is not None:
            self.search_bar.reset_session()

    def _remove_selected_variable_impl(self):
        """删除选中行的核心逻辑（不含会话重置）

        供 remove_selected_variable（表格删除按钮）和 _on_variable_removed
        （搜索栏移除）共用。调用方负责后续的会话状态处理：
        - 表格删除：调 reset_session 增量刷新样式
        - 搜索栏移除：调 mark_removed 增量刷新标记（选中停留原项）

        plot 侧删除委托 remove_variables_from_plot（批量、单次刷新链、
        不 emit，防信号回环），编辑器仅保留表格行删除/选中迁移/删空 emit。
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

        # 收集待删变量名（按行号升序，与 curves 字典顺序一致）
        selected_names: list[str] = []
        for row in sorted(selected_rows):
            var_name_item = self.var_table.item(row, 1)
            if var_name_item is None:
                continue
            var_name = var_name_item.data(Qt.ItemDataRole.UserRole)
            if var_name:
                selected_names.append(var_name)

        # plot 侧批量移除（轻量删除 + 末尾一次完整刷新链，不 emit）
        self.plot_widget.remove_variables_from_plot(selected_names)

        # 从表格中移除（从后往前，避免行号变化）
        for row in sorted(selected_rows, reverse=True):
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

        # 删空时才通知其他 widget（与改造前 emit 行为一致；
        # 部分删除不 emit，避免与编辑器自身行删除双刷新形成回环）
        if not self.plot_widget.curves:
            self.plot_widget.curves_changed.emit()

    def clear_all_variables(self):
        """清空所有变量（统一版）

        plot 侧删除委托 remove_variables_from_plot（含完整清理链：
        legend/光标 item 池/密度/_sync_min_xrange，顺带修复原实现
        缺失的 4 项清理，设计 §3.5 S7）；编辑器仅保留确认框、
        表格清空与删空 emit（批量 API 不 emit，维持显式触发现状）。
        """
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要清空所有绘图变量吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            cleared_count = len(self.plot_widget.curves)
            logger.debug("[EDITOR] clear_all_variables: 清除 %d 条曲线", cleared_count)
            # 批量移除（单次刷新链，不 emit）
            self.plot_widget.remove_variables_from_plot(
                list(self.plot_widget.curves)
            )

            # 清空表格
            self.var_table.setRowCount(0)
            self.update_button_states()
            # 清空所有 → 重置搜索会话（增量刷新已添加样式，不跳顶）
            if self.search_bar is not None:
                self.search_bar.reset_session()

            # 通知其他组件曲线已清空
            self.plot_widget.curves_changed.emit()

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

        # 重新加载表格以更新颜色显示
        selected_row = self._get_selected_row()
        self.load_current_curves()
        if selected_row is not None and selected_row < self.var_table.rowCount():
            self.var_table.selectRow(selected_row)

    def _apply_color_to_curve(self, var_name: str, color_name: str):
        """将指定变量的颜色更新为给定颜色（统一版）"""
        updated = False
        if var_name in self.plot_widget.curves:
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
        if var_name in self.plot_widget.curves:
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
                    # 检查变量是否已存在（统一版：始终检查 curves 字典）
                    if var_name in self.plot_widget.curves:
                        failed_vars.append(f"{var_name} (已存在)")
                        continue

                    # 添加变量到绘图
                    success = self.plot_widget.add_variable_to_plot(var_name)
                    if success:
                        success_count += 1
                    else:
                        failed_vars.append(var_name)

                # 重新加载列表以显示新添加的变量
                # 表格刷新由 curves_changed 信号触发增量更新（每次 add_variable_to_plot 成功都 emit）
                # 批量添加时信号多次触发，增量路径每次仅追加 1 行，开销可忽略

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

                # 检查变量是否已存在（统一版）
                if var_name in self.plot_widget.curves:
                    QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return

                # 添加变量到绘图
                success = self.plot_widget.add_variable_to_plot(var_name)
                # 表格刷新由 curves_changed 信号触发增量更新，无需显式调用
                if not success:
                    QMessageBox.warning(self, "错误", f"无法添加变量 {var_name}")

            event.acceptProposedAction()
        else:
            event.ignore()
        self.plot_widget._notify_drag_indicator(hide=True, source_widget=self)
