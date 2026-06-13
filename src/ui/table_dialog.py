"""数据表格对话框 —— DataTableDialog + 相关辅助类"""

from __future__ import annotations
import weakref
from threading import Lock
from PySide6.QtCore import Qt, QTimer, QEvent, QObject, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QFontMetrics, QColor, QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QAbstractItemView,
    QLabel,
    QTableView,
    QStyledItemDelegate,
    QMessageBox,
    QDialog,
    QSplitter,
    QMenu,
)
from src.core.config import FROZEN_VIEW_WIDTH_DEFAULT, BLINK_PULSE
from src.ui.drag_drop import parse_var_names_from_mimedata


class DropOverlay(QWidget):
    """
    拖拽覆盖层类
    在文件拖拽到应用程序时显示半透明的覆盖层，提供视觉反馈
    """

    def __init__(self, parent=None):
        import pandas as pd
        globals()['pd'] = pd
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setStyleSheet("""
            background:rgba(255,255,255,200);   
            border:none;
        """)

        self.label = QLabel("请丢入数据", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            background-color: rgba(168, 168, 168, 255);
            color:#333;
            font-size:36px;
            border-radius:12px;
            padding:20px 40px;
            color: rgba(128, 128, 128, 200);
        """)
        self.hide()

    def adjust_text(self, file_type_supported=True):
        if file_type_supported:
            self.label.setText("请丢入数据")
        else:
            self.label.setText("数据格式不支持")

    def adjust_font(self):
        # 根据 label 当前尺寸动态字号
        side = min(self.label.width(), self.label.height())
        font_size = max(12, min(int(side * 0.3), 128))
        font = self.label.font()  # QFont()
        font.setPixelSize(font_size)
        font.setBold(True)
        self.label.setFont(font)

    def resizeEvent(self, event):
        # self.label.adjustSize()
        w_half = self.width()
        h_half = self.height()
        self.label.setFixedSize(w_half, h_half)
        self.adjust_font()

        self.label.move(
            (self.width() - self.label.width()) // 2,
            (self.height() - self.label.height()) // 2,
        )


class PandasTableModel(QAbstractTableModel):
    """
    Pandas数据表格模型类
    只读官方虚拟模型，支持千万行秒开
    将pandas DataFrame数据适配到Qt的表格视图中，提供高效的数据访问功能
    """

    def __init__(self, df: pd.DataFrame, units: dict[str, str], parent=None):
        super().__init__(parent)
        self._df = df
        self._units = units

    # 三个必须实现的纯虚函数
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._df.iloc[index.row(), index.column()]
        return str(value) if pd.notnull(value) else ""

    def headerData(self, section, orientation, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if self._df.columns.empty:
            return None
        if orientation == Qt.Orientation.Horizontal:
            col_name = str(self._df.columns[section])
            unit = self._units.get(col_name, "")
            return f"{col_name}\n({unit})" if unit else f"{col_name}\n()"
        return str(section + 1)  # 行号 1-based

    def removeColumns(self, column, count, parent=QModelIndex()):
        if column < 0 or column + count > self.columnCount():
            return False
        self.beginRemoveColumns(parent, column, column + count - 1)
        self._df.drop(self._df.columns[column : column + count], axis=1, inplace=True)
        self.endRemoveColumns()
        return True


class CustomDelegate(QStyledItemDelegate):
    """
    自定义表格项委托类
    为表格单元格提供自定义的显示和编辑功能
    支持数据格式化和特殊显示效果
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_rows = set()
        self.selected_cols = set()
        self.highlighted_rows = set()  # 新增：用于存储需要高亮的行（来自另一个视图）
        self.highlighted_cols = set()  # 新增：用于存储需要高亮的列（用于闪烁效果）

    def paint(self, painter, option, index):

        painter.save()

        # 判断单元格是否被选中（同时在被选中的行和列中）
        is_selected_cell = (
            index.row() in self.selected_rows and index.column() in self.selected_cols
        )

        # 判断单元格是否只在被选中的行或列中（但不是同时）
        is_in_selected_row = index.row() in self.selected_rows
        is_in_selected_col = index.column() in self.selected_cols
        is_in_selected_row_or_col = (
            is_in_selected_row or is_in_selected_col
        ) and not is_selected_cell

        # 被选中的单元格本身：使用系统高亮颜色（和主界面变量列表一致），50%透明度
        if is_selected_cell:
            highlight_color = option.palette.highlight().color()
            # 设置50%透明度（alpha = 128）
            highlight_color.setAlpha(128)
            painter.fillRect(option.rect, highlight_color)
        # 被选中的单元格所在的行或列：使用浅蓝色，提高透明度
        elif is_in_selected_row_or_col:
            painter.fillRect(
                option.rect, QColor(200, 200, 255, 32)
            )  # 浅蓝高亮，更透明（从64降低到32）

        # 新增：高亮来自另一个视图的行
        if index.row() in self.highlighted_rows:
            painter.fillRect(
                option.rect, QColor(255, 200, 200, 64)
            )  # 淡红色高亮，更透明

        # 新增：高亮指定的列（用于闪烁）
        if index.column() in self.highlighted_cols:
            painter.fillRect(
                option.rect, QColor(200, 200, 255, 128)
            )  # 淡蓝色高亮，半透明

        super().paint(painter, option, index)
        painter.restore()


class XYScatterPlotDialog(QDialog):
    """
    XY散点图对话框类
    用于创建和配置XY散点图，允许用户选择X轴和Y轴变量
    提供图形参数设置和预览功能
    """

    def __init__(self, x_data, y_data, x_name, y_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("X/Y 散点图")
        self.resize(500, 500)

        # 设置窗口在关闭时释放内存
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)

        import pyqtgraph as pg

        # 创建 pyqtgraph 绘图组件
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # 绘制散点图
        scatter = pg.ScatterPlotItem(x=x_data, y=y_data, pen="r", brush="r", size=5)
        self.plot_widget.addItem(scatter)
        self.plot_widget.setBackground("w")
        black_pen = pg.mkPen(color="k", width=2)
        self.plot_widget.getViewBox().setBorder(black_pen)  # 外框黑色

        # 文字加粗，但字体家族用系统默认
        bold_font = QFont()
        bold_font.setBold(True)

        # 设置坐标轴标签和标题
        self.plot_widget.setLabel("bottom", text=x_name, color="k", font=bold_font)
        self.plot_widget.setLabel("left", text=y_name, color="k", font=bold_font)

        # 直接设置标签字体
        axis_bottom = self.plot_widget.getAxis("bottom")
        axis_left = self.plot_widget.getAxis("left")
        axis_bottom.label.setFont(bold_font)
        axis_left.label.setFont(bold_font)

        # self.plot_widget.setTitle(f"{y_name} vs. {x_name}")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # 拿到 AxisItem 句柄
        axis_bottom = self.plot_widget.getAxis("bottom")
        axis_left = self.plot_widget.getAxis("left")

        for ax in (axis_bottom, axis_left):
            # 设置轴线和刻度线的颜色为黑色
            ax.setPen("k")
            # 设置刻度文字颜色为黑色
            ax.setTextPen("k")
            # 设置刻度文字的字体
            ax.setTickFont(QFont())


class DataTableDialog(QMainWindow):
    """
    数据表格对话框类
    以独立窗口形式显示完整的数据表格
    支持数据查看、搜索、排序和导出功能
    使用单例模式确保只有一个表格窗口实例
    """

    _instance = None
    _saved_scroll_pos = None  # 类级变量存储滚动位置

    @classmethod
    def popup(cls, var_name: str, data, parent=None):
        if cls._instance is None:
            cls._instance = cls(parent)
        else:
            cls._instance._update_owner_from_widget(parent)

        dlg = cls._instance
        dlg.save_geom()
        if dlg.has_column(var_name):
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            # return dlg
        else:
            cls._saved_scroll_pos = (
                dlg.main_view.verticalScrollBar().value() if dlg.main_view else None
            )
            dlg.load_geom()
            dlg._add_variable_to_table(var_name, data)  # 使用内部函数
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

        # 闪烁
        QTimer.singleShot(100, lambda: dlg._blink_column(var_name, pulse=BLINK_PULSE))
        return dlg

    @classmethod
    def add_variables(cls, var_names, parent=None):
        """批量添加变量至数值变量表，复用拖拽逻辑"""
        if isinstance(var_names, str):
            candidates = [var_names]
        else:
            candidates = [name for name in (var_names or []) if isinstance(name, str)]

        normalized = []
        seen = set()
        for name in candidates:
            clean = name.strip()
            if not clean or clean in seen:
                continue
            normalized.append(clean)
            seen.add(clean)

        if not normalized:
            return

        if cls._instance is None:
            cls._instance = cls(parent)
        else:
            cls._instance._update_owner_from_widget(parent)

        dlg = cls._instance
        dlg.save_geom()
        dlg.load_geom()
        dlg.show()
        if dlg.isMinimized():
            dlg.showNormal()
        dlg.raise_()
        dlg.activateWindow()
        dlg._handle_dropped_variables(normalized)

    def _update_owner_from_widget(self, widget):
        window = None
        if isinstance(widget, QWidget):
            window = widget.window()
        self._owner_window_ref = weakref.ref(window) if window else None

    def _get_owner_window(self):
        if self._owner_window_ref:
            window = self._owner_window_ref()
            if window:
                return window
        active = QApplication.activeWindow()
        if active and isinstance(active, QMainWindow) and hasattr(active, "loader"):
            return active
        return None

    def _resolve_loader(self):
        owner = self._get_owner_window()
        if owner and hasattr(owner, "loader"):
            return owner.loader
        return None

    def __init__(self, parent=None):
        import pandas as pd
        globals()['pd'] = pd
        super().__init__(parent)
        self.setWindowTitle("变量数值表")
        self.window_geometry = None
        self.scatter_plot_windows = []
        self._skip_close_confirmation = False
        self.frozen_columns = []
        self._owner_window_ref = None
        self._update_owner_from_widget(parent)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        # Initialize user-preferred left width
        self.user_left_width = (
            FROZEN_VIEW_WIDTH_DEFAULT  # Initial fixed width for frozen_view
        )

        # Connect splitterMoved to update user preference when handle is dragged
        self.splitter.splitterMoved.connect(self._update_user_left_width)

        self.frozen_view = QTableView(self)
        self.frozen_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # self.frozen_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.frozen_view.verticalHeader().setVisible(True)
        self.frozen_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.frozen_view.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.frozen_view.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # self.frozen_view.setStyleSheet("QTableView { background-color: rgba(245,245,245,128); }")
        self.frozen_view.horizontalHeader().customContextMenuRequested.connect(
            self._on_frozen_header_right_click
        )
        self.frozen_view.horizontalHeader().setSectionsMovable(True)
        self.frozen_view.horizontalHeader().setDragEnabled(True)
        self.frozen_view.horizontalHeader().setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.frozen_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.frozen_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frozen_view.customContextMenuRequested.connect(
            self._show_table_context_menu
        )

        self.main_view = QTableView(self)
        self.main_view.setSortingEnabled(False)
        self.main_view.verticalHeader().setVisible(False)
        self.main_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.main_view.horizontalHeader().setSectionsMovable(True)
        self.main_view.horizontalHeader().setDragEnabled(True)
        self.main_view.horizontalHeader().setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.main_view.horizontalHeader().setDragDropOverwriteMode(False)

        self.main_view.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.main_view.horizontalHeader().customContextMenuRequested.connect(
            self._on_main_header_right_click
        )
        self.main_view.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.main_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.main_view.customContextMenuRequested.connect(self._show_table_context_menu)

        fm = QFontMetrics(self.main_view.font())
        safe_height = int(fm.height() * 1.6)
        self.main_view.verticalHeader().setDefaultSectionSize(safe_height)
        self.frozen_view.verticalHeader().setDefaultSectionSize(safe_height)

        self.main_view.setWordWrap(False)
        self.frozen_view.setWordWrap(False)

        splitter.addWidget(self.frozen_view)
        splitter.addWidget(self.main_view)
        splitter.setSizes([self.user_left_width, 400])

        main_layout.addWidget(splitter)

        self._df = pd.DataFrame()
        self._df_lock = Lock()
        self.model = None
        self.units = {}

        font = self.main_view.horizontalHeader().font()
        font.setBold(True)
        self.main_view.horizontalHeader().setFont(font)
        self.frozen_view.horizontalHeader().setFont(font)

        self._syncing_vertical_scroll = False
        self._syncing_row_height = False
        self.main_view.verticalScrollBar().valueChanged.connect(
            self._on_main_vertical_scroll
        )
        self.frozen_view.verticalScrollBar().valueChanged.connect(
            self._on_frozen_vertical_scroll
        )
        self.main_view.verticalHeader().sectionResized.connect(self._sync_row_heights)
        self.frozen_view.verticalHeader().sectionResized.connect(self._sync_row_heights)
        self.main_view.horizontalHeader().setResizeContentsPrecision(
            1
        )  # 0: BalanceSpeedAndAccuracy, 試1 (Speed)
        self.frozen_view.horizontalHeader().setResizeContentsPrecision(1)

        self.delegate_frozen = CustomDelegate(self)
        self.delegate_main = CustomDelegate(self)
        self.delegate_frozen.highlighted_rows = set()
        self.delegate_main.highlighted_rows = set()
        self.frozen_view.setItemDelegate(self.delegate_frozen)
        self.main_view.setItemDelegate(self.delegate_main)

        # 添加当前焦点视图跟踪
        self.current_focused_view = None

        # 为两个视图安装焦点事件过滤器
        self.frozen_view.installEventFilter(self)
        self.main_view.installEventFilter(self)

        # 启用拖放功能
        self.setAcceptDrops(True)
        self.main_view.setAcceptDrops(True)
        self.frozen_view.setAcceptDrops(True)

        # 安装事件过滤器处理视图的拖放事件
        self.drop_filter = self.DropFilter(self)
        self.main_view.viewport().installEventFilter(self.drop_filter)
        self.frozen_view.viewport().installEventFilter(self.drop_filter)

        if (
            self.parent()
            and hasattr(self.parent(), "data_table_geometry")
            and self.parent().data_table_geometry
        ):
            self.restoreGeometry(self.parent().data_table_geometry)
        else:
            self.resize(600, 400)
            screen = QApplication.primaryScreen().availableGeometry()
            size = self.geometry()
            x = (screen.width() - size.width()) // 2
            y = (screen.height() - size.height()) // 2
            self.move(x, y)

    # 事件过滤器类处理拖放事件
    class DropFilter(QObject):
        def __init__(self, parent_dialog):
            super().__init__(parent_dialog)
            self.parent_dialog = parent_dialog

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.DragMove:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.Drop:
                if event.mimeData().hasText():
                    var_names = parse_var_names_from_mimedata(event.mimeData())
                    self.parent_dialog._handle_dropped_variables(var_names)
                    event.acceptProposedAction()
                    return True
            return super().eventFilter(obj, event)

    def _update_user_left_width(self, pos, index):
        if index == 1:  # Handle for the first splitter section
            self.user_left_width = self.splitter.sizes()[0]

    def _on_main_vertical_scroll(self, value: int):
        self._sync_vertical_scrollbars(self.frozen_view.verticalScrollBar(), value)

    def _on_frozen_vertical_scroll(self, value: int):
        self._sync_vertical_scrollbars(self.main_view.verticalScrollBar(), value)

    def _sync_vertical_scrollbars(self, target_scrollbar, value: int):
        if self._syncing_vertical_scroll:
            return
        self._syncing_vertical_scroll = True
        try:
            if target_scrollbar.value() != value:
                target_scrollbar.setValue(value)
        finally:
            self._syncing_vertical_scroll = False

    def _cancel_plot_drag_indicator(self):
        main_window = self._get_owner_window()
        if not main_window:
            return
        container = getattr(main_window, "_active_drag_container", None)
        if container and getattr(container, "plot_widget", None):
            main_window._hide_drag_indicator_for_plot(container.plot_widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # On window resize, fix left width to user preference, stretch right
        total_width = sum(self.splitter.sizes())
        self.splitter.setSizes(
            [self.user_left_width, total_width - self.user_left_width]
        )

    # 拖放相关方法
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            self._cancel_plot_drag_indicator()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            self._cancel_plot_drag_indicator()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self._cancel_plot_drag_indicator()
        var_names = parse_var_names_from_mimedata(event.mimeData())
        self._handle_dropped_variables(var_names)
        event.acceptProposedAction()

    def _blink_step_on(self, delegate, col_idx, view):
        # 步骤1: 高亮 (持续0.5s)
        delegate.highlighted_cols.add(col_idx)
        view.viewport().update()

    def _blink_step_off(self, delegate, col_idx, view):
        # 步骤2: 取消高亮 (持续0.5s)
        delegate.highlighted_cols.remove(col_idx)
        view.viewport().update()

    def _blink_column(self, var_name, pulse: int = 800):
        if self.has_column(var_name):
            # 启动闪烁动画：淡蓝色底色闪烁2次，频率1次/秒（每个周期1s：高亮0.5s + 正常0.5s）
            col_idx = self._df.columns.get_loc(var_name)  # 获取逻辑列索引
            if var_name in self.frozen_columns:
                delegate = self.delegate_frozen
                view = self.frozen_view
            else:
                delegate = self.delegate_main
                view = self.main_view

            # 步骤1: 高亮 (持续0.5s)
            self._blink_step_on(delegate, col_idx, view)
            QTimer.singleShot(
                pulse, lambda: self._blink_step_off(delegate, col_idx, view)
            )
        return

    # 内部函数：处理拖放的多个变量
    def _handle_dropped_variables(self, var_names: list[str]):
        """
        处理拖放的多个变量，添加到非冻结区

        支持单个或多个变量同时拖入
        对于多个变量，批量添加并显示结果

        Args:
            var_names: 要添加的变量名称列表
        """
        if not var_names:
            return

        if len(var_names) == 1:
            # 单个变量：使用原有逻辑
            self._handle_dropped_variable(var_names[0])
            return

        # 多个变量：批量处理
        loader = self._resolve_loader()
        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return

        existing_vars = []
        invalid_vars = []
        added_vars = []

        # 保存当前的垂直滚动位置
        self._saved_scroll_pos = (
            self.main_view.verticalScrollBar().value() if self.main_view else None
        )

        for var_name in var_names:
            # 检查变量是否已存在
            if self.has_column(var_name):
                existing_vars.append(var_name)
                continue

            # 检查变量是否在数据中存在
            is_mdf_loader = getattr(loader, "LOADER_TYPE", "") == "mdf"
            if not is_mdf_loader and var_name not in loader.df.columns:
                invalid_vars.append(var_name)
                continue

            try:
                if is_mdf_loader:
                    series = loader.get_series(var_name)
                else:
                    series = loader.df[var_name]
                self._add_variable_to_table(var_name, series)
                added_vars.append(var_name)
            except Exception as e:
                invalid_vars.append(f"{var_name} (错误: {str(e)})")

        # 显示结果消息（只在有问题时提示）
        msg_parts = []
        if added_vars:
            # 滚动到最后添加的变量
            last_var = added_vars[-1]
            QTimer.singleShot(100, lambda: self.scroll_to_column(last_var))
            QTimer.singleShot(
                100, lambda: self._blink_column(last_var, pulse=BLINK_PULSE)
            )

        # 只在有错误或已存在变量时显示提示
        if existing_vars or invalid_vars:
            if added_vars:
                msg_parts.append(f"成功添加 {len(added_vars)} 个变量")

            if existing_vars:
                msg_parts.append(f"已存在 {len(existing_vars)} 个变量")

            if invalid_vars:
                msg_parts.append(
                    f"无效变量: {', '.join(invalid_vars[:5])}"
                )  # 最多显示5个
                if len(invalid_vars) > 5:
                    msg_parts.append(f"等共 {len(invalid_vars)} 个")

            if invalid_vars:
                QMessageBox.warning(self, "批量添加结果", "\n".join(msg_parts))
            else:
                QMessageBox.information(self, "批量添加结果", "\n".join(msg_parts))

    # 内部函数：处理拖放的变量
    def _handle_dropped_variable(self, var_name: str):
        """
        处理拖放的变量，添加到非冻结区

        处理从变量列表拖拽到表格的变量
        检查变量是否已存在，如果存在则高亮显示，否则添加到表格

        Args:
            var_name: 要添加的变量名称
        """
        # 检查变量是否已存在
        if self.has_column(var_name):
            self.scroll_to_column(var_name)
            self._blink_column(var_name, pulse=BLINK_PULSE)
            return

        # 获取主窗口loader
        loader = self._resolve_loader()

        if loader is None:
            QMessageBox.warning(self, "错误", "没有加载数据")
            return

        is_mdf_loader = getattr(loader, "LOADER_TYPE", "") == "mdf"
        if not is_mdf_loader and var_name not in loader.df.columns:  # 改为 loader
            QMessageBox.warning(self, "错误", f"变量 '{var_name}' 不存在")
            return

        # 保存当前的垂直滚动位置，避免添加变量后列表位置变化
        self._saved_scroll_pos = (
            self.main_view.verticalScrollBar().value() if self.main_view else None
        )

        if is_mdf_loader:
            series = loader.get_series(var_name)
        else:
            series = loader.df[var_name]  # 改为 loader
        self._add_variable_to_table(var_name, series)

        # 滚动到新添加的列
        QTimer.singleShot(100, lambda: self.scroll_to_column(var_name))
        QTimer.singleShot(100, lambda: self._blink_column(var_name, pulse=BLINK_PULSE))

    # 内部函数：添加变量到表格
    def _add_variable_to_table(self, var_name: str, data: pd.Series):
        """
        内部函数：将变量添加到表格的非冻结区

        将新变量添加到数据表格中，更新模型和视图
        保持滚动位置和焦点状态

        Args:
            var_name: 变量名称
            data: 变量数据序列
        """
        self._df[var_name] = data.reset_index(drop=True)
        max_len = max(len(self._df), len(data))
        if len(self._df) < max_len:
            self._df = self._df.reindex(range(max_len))
        loader = self._resolve_loader()

        if loader:
            self.units = loader.units

        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()
        if self._saved_scroll_pos is not None:
            QTimer.singleShot(
                0,
                lambda: self.main_view.verticalScrollBar().setValue(
                    self._saved_scroll_pos
                ),
            )

    def eventFilter(self, obj, event):
        # 处理焦点变化事件
        if event.type() == QEvent.Type.FocusIn:
            if obj in [self.frozen_view, self.main_view]:
                self.current_focused_view = obj
                self._update_highlights_on_focus_change()

        return super().eventFilter(obj, event)

    def _update_highlights_on_focus_change(self):
        # 根据当前焦点视图更新高亮
        if self.current_focused_view == self.frozen_view:
            # 清除主视图的同步高亮
            self.delegate_main.highlighted_rows = set()
            self.delegate_frozen.highlighted_rows = set()
            selection_model = self.frozen_view.selectionModel()
            if selection_model is None:
                return
            selected_indexes = selection_model.selectedIndexes()
            self.delegate_frozen.selected_rows = set(
                idx.row() for idx in selected_indexes
            )
            self.delegate_frozen.selected_cols = set(
                idx.column() for idx in selected_indexes
            )

            # 设置主视图的高亮行
            self.delegate_main.highlighted_rows = self.delegate_frozen.selected_rows

            # 清除主视图的选中状态（只保留高亮行）
            self.delegate_main.selected_rows = set()
            self.delegate_main.selected_cols = set()

        elif self.current_focused_view == self.main_view:
            # 清除冻结视图的同步高亮
            self.delegate_frozen.highlighted_rows = set()
            self.delegate_main.highlighted_rows = set()
            selection_model = self.main_view.selectionModel()
            if selection_model is None:
                return
            selected_indexes = selection_model.selectedIndexes()
            self.delegate_main.selected_rows = set(
                idx.row() for idx in selected_indexes
            )
            self.delegate_main.selected_cols = set(
                idx.column() for idx in selected_indexes
            )

            # 设置冻结视图的高亮行
            self.delegate_frozen.highlighted_rows = self.delegate_main.selected_rows

            # 清除冻结视图的选中状态（只保留高亮行）
            self.delegate_frozen.selected_rows = set()
            self.delegate_frozen.selected_cols = set()

        else:
            # 没有焦点，清空所有高亮
            self.delegate_frozen.selected_rows = set()
            self.delegate_frozen.selected_cols = set()
            self.delegate_frozen.highlighted_rows = set()

            self.delegate_main.selected_rows = set()
            self.delegate_main.selected_cols = set()
            self.delegate_main.highlighted_rows = set()

        # 更新视图
        self.frozen_view.viewport().update()
        self.main_view.viewport().update()

    def _update_highlights_frozen(self, selected, deselected):
        # 设置当前焦点视图为冻结视图
        self.current_focused_view = self.frozen_view
        self._update_highlights_on_focus_change()

    def _update_highlights_main(self, selected, deselected):
        # 设置当前焦点视图为主视图
        self.current_focused_view = self.main_view
        self._update_highlights_on_focus_change()

    def focusInEvent(self, event):
        # 当对话框获得焦点时，更新高亮
        super().focusInEvent(event)
        self._update_highlights_on_focus_change()

    def focusOutEvent(self, event):
        # 当对话框失去焦点时，清除所有高亮
        super().focusOutEvent(event)
        self.current_focused_view = None
        self._update_highlights_on_focus_change()

    def _show_table_context_menu(self, pos):
        """
        根据视觉顺序判断是否显示绘图菜单，并传递正确的列索引。
        """
        view = self.sender()
        if not isinstance(view, QTableView):
            return

        selected_indexes = view.selectionModel().selectedIndexes()
        if not selected_indexes:
            return

        # 计算两侧选择与列集合
        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        if view == self.main_view:
            other_view = self.frozen_view
        else:
            other_view = self.main_view

        other_selected = other_view.selectionModel().selectedIndexes()
        all_selected = selected_indexes + other_selected

        # 构建每列的选中行集合（基于两侧合并选择）
        rows_per_col_all: dict[int, set[int]] = {}
        for idx in all_selected:
            rows_per_col_all.setdefault(idx.column(), set()).add(idx.row())

        total_cols = set(rows_per_col_all.keys())

        # 计算复制可行性与顺序
        can_copy = False
        ordered_cols: list[int] = []
        rows_order: list[int] = []

        if len(total_cols) == 1:
            # 单列：允许复制（支持多段选择）
            only_col = next(iter(total_cols))
            ordered_cols = [only_col]
            rows_order = sorted(rows_per_col_all[only_col])
            can_copy = len(rows_order) > 0
        elif len(total_cols) >= 2:
            # 多列：几列的行号集合需完全相同
            cols_list = list(total_cols)
            base_rows = rows_per_col_all[cols_list[0]] if cols_list else set()
            if base_rows and all(
                rows_per_col_all[c] == base_rows for c in cols_list[1:]
            ):
                # 左到右的可视列顺序：先冻结区，再主区
                frozen_header = self.frozen_view.horizontalHeader()
                main_header = self.main_view.horizontalHeader()
                frozen_selected_cols = [c for c in total_cols if c in frozen_cols]
                main_selected_cols = [c for c in total_cols if c not in frozen_cols]
                frozen_selected_cols.sort(key=lambda c: frozen_header.visualIndex(c))
                main_selected_cols.sort(key=lambda c: main_header.visualIndex(c))
                ordered_cols = frozen_selected_cols + main_selected_cols
                rows_order = sorted(base_rows)
                can_copy = True

        # 计算绘图相关（尽量保持原有逻辑）
        plot_enabled = False
        x_col = y_col = None
        plot_rows: list[int] = []

        all_rows = set()
        for rows in rows_per_col_all.values():
            all_rows.update(rows)

        if len(total_cols) == 2 and len(all_rows) >= 2:
            cols_list = list(total_cols)
            frozen_sel = [c for c in cols_list if c in frozen_cols]
            main_sel = [c for c in cols_list if c not in frozen_cols]

            if len(frozen_sel) == 2:
                header = self.frozen_view.horizontalHeader()
                frozen_sel.sort(key=lambda c: header.visualIndex(c))
                x_col, y_col = frozen_sel[0], frozen_sel[1]
            elif len(main_sel) == 2:
                header = self.main_view.horizontalHeader()
                main_sel.sort(key=lambda c: header.visualIndex(c))
                x_col, y_col = main_sel[0], main_sel[1]
            elif len(frozen_sel) == 1 and len(main_sel) == 1:
                x_col, y_col = frozen_sel[0], main_sel[0]

            if x_col is not None and y_col is not None:
                plot_rows = sorted(all_rows)
                plot_enabled = True

        menu = QMenu(self)

        # 绘图菜单（仅在正好两列被选中时展示；保持原行为）
        scatter_actions_added = False
        plot_candidate_cols = total_cols
        if len(plot_candidate_cols) == 2:
            # 获取列名用于展示
            cols_list = sorted(list(plot_candidate_cols))
            if plot_enabled and x_col is not None and y_col is not None:
                x_show, y_show = x_col, y_col
            else:
                x_show, y_show = cols_list[0], cols_list[1]
            x_name = self.model.headerData(
                x_show, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            ).replace("\n", " ")
            y_name = self.model.headerData(
                y_show, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            ).replace("\n", " ")

            act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
            act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
            if plot_enabled and x_col is not None and y_col is not None:
                act1.triggered.connect(
                    lambda _checked=False, rows=plot_rows, x=x_col, y=y_col: self._plot_xy_scatter(
                        x, y, rows
                    )
                )
                act2.triggered.connect(
                    lambda _checked=False, rows=plot_rows, x=x_col, y=y_col: self._plot_xy_scatter(
                        y, x, rows
                    )
                )
                act1.setEnabled(True)
                act2.setEnabled(True)
                # 若可激活绘图，则先放绘图菜单
                menu.addAction(act1)
                menu.addAction(act2)
                scatter_actions_added = True
            else:
                act1.setEnabled(False)
                act2.setEnabled(False)
                # 若无法激活绘图，则稍后把它们放在复制项之后

        # 复制到剪贴板（两个按钮）
        act_copy_selected = QAction("复制所选数据到剪贴板", menu)
        act_copy_selected.setEnabled(can_copy)
        if can_copy:
            act_copy_selected.triggered.connect(
                lambda: self._copy_selected_to_clipboard(ordered_cols, rows_order)
            )

        act_copy_all = QAction("复制表内所有数据到剪贴板", menu)
        enable_all = (
            self._df is not None and self._df.shape[0] > 0 and self._df.shape[1] > 0
        )
        act_copy_all.setEnabled(enable_all)
        if enable_all:
            act_copy_all.triggered.connect(self._copy_all_to_clipboard)

        if scatter_actions_added:
            menu.addSeparator()
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
        else:
            # 若无法激活x/y散点图，则优先展示复制功能
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
            # 若存在两列但不可激活，也追加禁用的绘图项在其后
            if len(plot_candidate_cols) == 2:
                menu.addSeparator()
                act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
                act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
                act1.setEnabled(False)
                act2.setEnabled(False)
                menu.addAction(act1)
                menu.addAction(act2)

        menu.exec(view.mapToGlobal(pos))

    def _show_plot_menu(self, pos, view, x_col, y_col, rows, enabled=True):
        x_name = self.model.headerData(
            x_col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        ).replace("\n", " ")
        y_name = self.model.headerData(
            y_col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        ).replace("\n", " ")
        menu = QMenu(self)
        act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
        act1.triggered.connect(
            lambda _checked=False, rows=rows, x=x_col, y=y_col: self._plot_xy_scatter(
                x, y, rows
            )
        )
        act1.setEnabled(enabled)
        act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
        act2.triggered.connect(
            lambda _checked=False, rows=rows, x=x_col, y=y_col: self._plot_xy_scatter(
                y, x, rows
            )
        )
        act2.setEnabled(enabled)
        menu.addAction(act1)
        menu.addAction(act2)
        menu.exec(view.mapToGlobal(pos))

    def _plot_xy_scatter(
        self, x_col_idx, y_col_idx, rows=None, start_row=None, num_rows=None
    ):
        """
        接收已按视觉顺序确定的逻辑列索引进行绘图。
        """
        try:
            if rows is None:
                if start_row is None or num_rows is None:
                    return
                row_indexer = slice(start_row, start_row + num_rows)
            else:
                row_indexer = rows

            # 直接使用正确的逻辑索引提取数据
            x_data_series = pd.to_numeric(
                self._df.iloc[row_indexer, x_col_idx], errors="coerce"
            )
            y_data_series = pd.to_numeric(
                self._df.iloc[row_indexer, y_col_idx], errors="coerce"
            )

            # 验证1：检查是否有非数值数据
            if x_data_series.isnull().any() or y_data_series.isnull().any():
                QMessageBox.warning(
                    self, "绘图错误", "选中区域包含无法转换为数字的单元格。"
                )
                return

            # 获取清理后的列标题
            x_header = self.model.headerData(
                x_col_idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            ).replace("\n", " ")
            y_header = self.model.headerData(
                y_col_idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            ).replace("\n", " ")

            # 创建并显示绘图窗口
            plot_dialog = XYScatterPlotDialog(
                x_data_series.to_numpy(),
                y_data_series.to_numpy(),
                x_header,
                y_header,
                self,
            )
            self.scatter_plot_windows.append(plot_dialog)
            plot_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "未知错误", f"绘图时发生错误: {e}")

    def _copy_selected_to_clipboard(
        self, ordered_cols: list[int], rows_order: list[int]
    ):
        """将选中区域的数据复制到剪贴板。

        第一行：变量名；第二行：单位；第三行开始为数据。
        """
        if not ordered_cols or not rows_order:
            return
        # 变量名与单位
        var_names = [str(self._df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, "") for name in var_names]

        # 组装数据（按行）
        lines = []
        lines.append("\t".join(var_names))
        lines.append("\t".join(units))

        for r in rows_order:
            row_vals = []
            for c in ordered_cols:
                val = self._df.iloc[r, c]
                if pd.isna(val):
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            lines.append("\t".join(row_vals))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

    def _copy_all_to_clipboard(self):
        """复制表内所有数据到剪贴板，列顺序按可视顺序（先冻结区再主区）。"""
        if self._df is None or self._df.shape[0] == 0 or self._df.shape[1] == 0:
            return

        # 计算可视列顺序：先冻结区，再主区
        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        frozen_header = self.frozen_view.horizontalHeader()
        main_header = self.main_view.horizontalHeader()

        all_cols = list(range(self._df.shape[1]))
        frozen_list = [c for c in all_cols if c in frozen_cols]
        main_list = [c for c in all_cols if c not in frozen_cols]
        frozen_list.sort(key=lambda c: frozen_header.visualIndex(c))
        main_list.sort(key=lambda c: main_header.visualIndex(c))
        ordered_cols = frozen_list + main_list

        # 变量名与单位
        var_names = [str(self._df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, "") for name in var_names]

        lines = []
        lines.append("\t".join(var_names))
        lines.append("\t".join(units))

        for r in range(self._df.shape[0]):
            row_vals = []
            for c in ordered_cols:
                val = self._df.iloc[r, c]
                if pd.isna(val):
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            lines.append("\t".join(row_vals))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

    def _connect_signals(self):
        if self.model:
            self.main_view.selectionModel().selectionChanged.connect(
                self._update_highlights_main
            )
            self.frozen_view.selectionModel().selectionChanged.connect(
                self._update_highlights_frozen
            )

    def save_geom(self):
        """
        保存窗口几何信息

        将当前窗口的位置和大小保存到父窗口的几何信息中
        用于下次打开时恢复窗口状态
        """
        if self.parent() and hasattr(self.parent(), "data_table_geometry"):
            self.parent().data_table_geometry = self.saveGeometry()

    def load_geom(self):
        """
        加载窗口几何信息

        从父窗口的几何信息中恢复窗口的位置和大小
        提供用户界面状态的持久化
        """
        if (
            self.parent()
            and hasattr(self.parent(), "data_table_geometry")
            and self.parent().data_table_geometry is not None
        ):
            geom = self.parent().data_table_geometry
            self.restoreGeometry(geom)

    def clear_all_columns(self):
        """重载数据时：清空 _df，释放持有的所有 numpy 数组。"""
        if hasattr(self, "_df") and self._df is not None and not self._df.empty:
            self._df = pd.DataFrame()
        if hasattr(self, "model"):
            self.model = None

    def closeEvent(self, event):
        for win in self.scatter_plot_windows[:]:
            try:
                # 尝试访问窗口属性来检查是否有效
                if hasattr(win, "isVisible"):
                    win.close()
            except RuntimeError:
                # 窗口已经被删除，跳过
                pass
        if not (self._skip_close_confirmation) and (len(self._df.columns) >= 4):
            reply = QMessageBox.question(
                self,
                "确认关闭",
                "是否清除所有列表，并关闭数值变量表窗口？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            # if user did not confirm to close the window
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self.set_skip_close_confirmation(False)

        self.scatter_plot_windows.clear()

        # 其他清理代码保持不变...
        self.save_geom()
        self._df = pd.DataFrame()
        self.main_view.setModel(None)
        self.frozen_view.setModel(None)
        self._instance = None
        self._saved_scroll_pos = None
        self.frozen_columns = []
        self.hide()
        event.accept()

    def set_skip_close_confirmation(self, status: bool):
        self._skip_close_confirmation = status

    def has_column(self, var_name: str) -> bool:
        return var_name in self._df.columns

    def add_series(self, var_name: str, data: pd.Series):
        self._add_variable_to_table(var_name, data)

    def _update_views(self):
        if self.model is None:
            return
        frozen_count = 0
        for col in range(self.model.columnCount()):
            var_name = self._df.columns[col]
            if var_name in self.frozen_columns:
                self.main_view.setColumnHidden(col, True)
                self.frozen_view.setColumnHidden(col, False)
                frozen_count += 1
            else:
                self.main_view.setColumnHidden(col, False)
                self.frozen_view.setColumnHidden(col, True)
        if frozen_count == 0:
            self.frozen_view.hide()
        else:
            self.frozen_view.show()
        if frozen_count > 0:
            self.frozen_view.verticalHeader().setVisible(True)
            self.main_view.verticalHeader().setVisible(False)
        else:
            self.frozen_view.verticalHeader().setVisible(False)
            self.main_view.verticalHeader().setVisible(True)

    def _get_full_visual_order(self):
        """获取所有列的完整视觉顺序（从左到右）"""
        full_order = []

        # 获取冻结区的视觉顺序
        frozen_header = self.frozen_view.horizontalHeader()
        for visual_idx in range(frozen_header.count()):
            logical_idx = frozen_header.logicalIndex(visual_idx)
            if not self.frozen_view.isColumnHidden(logical_idx):
                col_name = self._df.columns[logical_idx]
                full_order.append(col_name)

        # 获取非冻结区的视觉顺序
        main_header = self.main_view.horizontalHeader()
        for visual_idx in range(main_header.count()):
            logical_idx = main_header.logicalIndex(visual_idx)
            if not self.main_view.isColumnHidden(logical_idx):
                col_name = self._df.columns[logical_idx]
                full_order.append(col_name)

        return full_order

    def _restore_visual_order_after_model_change(
        self, old_visual_order, new_logical_order
    ):
        """在模型改变后恢复视觉顺序"""
        # 创建从列名到新逻辑索引的映射
        name_to_new_logical = {name: idx for idx, name in enumerate(new_logical_order)}

        # 获取冻结区和非冻结区的表头
        frozen_header = self.frozen_view.horizontalHeader()
        main_header = self.main_view.horizontalHeader()

        # 按照旧的视觉顺序重新排列列
        current_visual_index = 0

        # 处理冻结区的列
        for col_name in old_visual_order:
            if col_name in self.frozen_columns:
                logical_idx = name_to_new_logical[col_name]
                current_visual_idx = frozen_header.visualIndex(logical_idx)
                if current_visual_idx != current_visual_index:
                    frozen_header.moveSection(current_visual_idx, current_visual_index)
                current_visual_index += 1

        # 重置视觉索引计数器，开始处理非冻结区
        current_visual_index = 0

        # 处理非冻结区的列
        for col_name in old_visual_order:
            if col_name not in self.frozen_columns:
                logical_idx = name_to_new_logical[col_name]
                current_visual_idx = main_header.visualIndex(logical_idx)
                if current_visual_idx != current_visual_index:
                    main_header.moveSection(current_visual_idx, current_visual_index)
                current_visual_index += 1

    def freeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]

        if var_name not in self.frozen_columns:
            # 调整splitter大小的代码保持不变
            col_width = self.main_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            if not self.frozen_columns:
                new_frozen_width = FROZEN_VIEW_WIDTH_DEFAULT
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                new_frozen_width = frozen_width + col_width
                new_main_width = main_width - col_width

            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

            # 获取当前所有列的完整视觉顺序
            full_visual_order = self._get_full_visual_order()

            # 将要冻结的列添加到冻结列列表
            self.frozen_columns.append(var_name)

            # 重新构建列顺序
            new_column_order = []

            # 按照完整视觉顺序添加列，但冻结列在前，非冻结列在后
            for col in full_visual_order:
                if col in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)

            for col in full_visual_order:
                if col not in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)

            # 重新排列DataFrame
            self._df = self._df[new_column_order]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()

            # 重新设置模型后，恢复用户调整的视觉顺序
            self._restore_visual_order_after_model_change(
                full_visual_order, new_column_order
            )

    def unfreeze_column(self, logical_col):
        var_name = self._df.columns[logical_col]

        if var_name in self.frozen_columns:
            # 调整splitter大小的代码保持不变
            col_width = self.frozen_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            if len(self.frozen_columns) == 2:
                new_frozen_width = FROZEN_VIEW_WIDTH_DEFAULT
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                new_frozen_width = max(0, frozen_width - col_width)
                new_main_width = main_width + col_width

            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

            # 获取当前所有列的完整视觉顺序
            full_visual_order = self._get_full_visual_order()

            # 将要解冻的列从冻结列列表中移除
            self.frozen_columns.remove(var_name)

            # 重新构建列顺序
            new_column_order = []

            # 按照完整视觉顺序添加列，但冻结列在前，非冻结列在后
            for col in full_visual_order:
                if col in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)

            for col in full_visual_order:
                if col not in self.frozen_columns and col not in new_column_order:
                    new_column_order.append(col)

            # 重新排列DataFrame
            self._df = self._df[new_column_order]
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
            self._update_views()

            # 重新设置模型后，恢复用户调整的视觉顺序
            self._restore_visual_order_after_model_change(
                full_visual_order, new_column_order
            )

    def _sync_row_heights(self, logicalIndex, oldSize, newSize):
        if self._syncing_row_height:
            return
        sender = self.sender()
        target_header = None
        target_view = None
        if sender == self.main_view.verticalHeader():
            target_header = self.frozen_view.verticalHeader()
            target_view = self.frozen_view
        elif sender == self.frozen_view.verticalHeader():
            target_header = self.main_view.verticalHeader()
            target_view = self.main_view
        if target_view is None:
            return
        self._syncing_row_height = True
        try:
            current_size = target_header.sectionSize(logicalIndex)
            if current_size != newSize:
                target_view.setRowHeight(logicalIndex, newSize)
        finally:
            self._syncing_row_height = False

    def _on_frozen_header_right_click(self, pos):
        self._on_header_right_click(pos, self.frozen_view)

    def _on_main_header_right_click(self, pos):
        self._on_header_right_click(pos, self.main_view)

    def _on_header_right_click(self, pos, view):
        header = view.horizontalHeader()
        logical_col = header.logicalIndexAt(pos)
        if logical_col < 0:
            return

        var_name = self._df.columns[logical_col]

        menu = QMenu(self)
        act_delete = menu.addAction(f'删除列 "{var_name}"')
        if var_name in self.frozen_columns:
            act_freeze = menu.addAction("解除冻结列")
        else:
            act_freeze = menu.addAction("冻结列")

        # 新增: 复制变量名
        act_copy = menu.addAction("复制变量名")
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(var_name))

        # 新增: 清空列表（全局操作，不依赖具体列）
        act_clear = menu.addAction("清空列表")
        act_clear.triggered.connect(self._clear_all_columns)

        selected = menu.exec(header.mapToGlobal(pos))
        if selected == act_delete:
            self._remove_column(logical_col)
        elif selected == act_freeze:
            if var_name in self.frozen_columns:
                self.unfreeze_column(logical_col)
            else:
                self.freeze_column(logical_col)

    def _clear_all_columns(self):
        reply = QMessageBox.question(
            self,
            "确认",
            "是否清空所有列？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # 清空所有列
        while self.model.columnCount() > 0:
            self.model.removeColumns(0, 1)
        self._df = pd.DataFrame()
        self.frozen_columns = []
        self._update_views()

    def scroll_to_column(self, var_name: str):
        """滚动到指定变量名的列（可能在冻结区或普通区），不影响垂直滚动位置"""
        if var_name not in self._df.columns:
            return False

        # 获取列的索引
        col_idx = self._df.columns.get_loc(var_name)

        # 确定列在哪个视图（冻结区或普通区）
        if var_name in self.frozen_columns:
            view = self.frozen_view
        else:
            view = self.main_view

        # 获取水平头部
        header = view.horizontalHeader()

        # 获取列的视觉位置
        visual_idx = header.visualIndex(col_idx)

        # 计算列的位置和大小
        col_pos = 0
        for i in range(visual_idx):
            col_pos += header.sectionSize(header.logicalIndex(i))

        col_width = header.sectionSize(col_idx)

        # 获取当前水平滚动位置
        scroll_pos = view.horizontalScrollBar().value()

        # 计算需要的滚动位置，使列在视图中可见
        viewport_width = view.viewport().width()

        # 如果列在视图左侧之外
        if col_pos < scroll_pos:
            view.horizontalScrollBar().setValue(col_pos)
        # 如果列在视图右侧之外
        elif col_pos + col_width > scroll_pos + viewport_width:
            view.horizontalScrollBar().setValue(col_pos + col_width - viewport_width)

        return True

    def _remove_column(self, logical_col):
        var_name = self._df.columns[logical_col]

        # 如果要删除的列在冻结区，则执行与解冻相同的宽度调整策略
        if var_name in self.frozen_columns:
            # 1. 从 frozen_view 获取列宽
            col_width = self.frozen_view.columnWidth(logical_col)
            current_sizes = self.splitter.sizes()
            frozen_width, main_width = current_sizes[0], current_sizes[1]

            # 2. 应用特殊宽度逻辑：
            #    如果删除后只剩一列（即删除前有两列），则将剩余的冻结区宽度设为 150
            if len(self.frozen_columns) == 2:
                new_frozen_width = 150
                total_width = frozen_width + main_width
                new_main_width = total_width - new_frozen_width
            else:
                #    否则，直接减去被删除列的宽度
                new_frozen_width = max(0, frozen_width - col_width)
                new_main_width = main_width + col_width

            # 3. 应用新尺寸并更新用户偏好宽度
            self.splitter.setSizes([new_frozen_width, new_main_width])
            self.user_left_width = new_frozen_width

        # 从冻结列表中移除
        if var_name in self.frozen_columns:
            self.frozen_columns.remove(var_name)

        # 从DataFrame中删除列
        self._df.drop(columns=[var_name], inplace=True)

        # 刷新模型和视图
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()
        self._update_views()

    def update_data(self, loader):
        """
        当主窗口重载数据时，更新此对话框中的数据

        同步数据表格与主窗口的数据状态
        保持用户界面的一致性和数据完整性

        Args:
            loader: 数据加载器实例
        """
        if self.model is None or self._df.empty:
            return

        scroll_pos = self.main_view.verticalScrollBar().value()
        frozen_cols = self.frozen_columns.copy()
        current_cols = list(self._df.columns)

        # --- BUG修复 START ---

        # 创建一个新的DataFrame来保存更新后的数据
        new_df = pd.DataFrame()
        removed = []

        # 遍历当前表中的列
        is_mdf_loader = getattr(loader, "LOADER_TYPE", "") == "mdf"
        for col in current_cols:
            if is_mdf_loader:
                try:
                    series = loader.get_series(col)
                    new_df[col] = series.reset_index(drop=True)
                except KeyError:
                    removed.append(col)
            elif col in loader.df.columns:
                # 从新的加载器数据中复制完整的列
                # 这是关键修复：确保新DataFrame获得完整行数
                new_df[col] = loader.df[col]
            else:
                # 该列已从源文件中移除
                removed.append(col)

        # 用新的、行数正确的DataFrame替换旧的
        self._df = new_df

        # --- BUG修复 END ---

        self.units = loader.units
        self.model = PandasTableModel(self._df, self.units)
        self.main_view.setModel(self.model)
        self.frozen_view.setModel(self.model)
        self._connect_signals()

        # 重新应用冻结列，确保它们仍然存在
        self.frozen_columns = [col for col in frozen_cols if col in self._df.columns]

        self._update_views()
        QTimer.singleShot(
            0, lambda: self.main_view.verticalScrollBar().setValue(scroll_pos)
        )

        if removed:
            msg = f"以下变量已从数据中移除：{', '.join(removed)}"
            QMessageBox.information(self, "更新通知", msg)

        if self._df.empty:
            # 增加这行，避免在表格变空并关闭时弹出烦人的确认框
            self.set_skip_close_confirmation(True)
            self.close()
