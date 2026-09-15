"""变量数值表对话框 —— DataTableDialog + 相关辅助类"""

from __future__ import annotations
import weakref
from dataclasses import dataclass
from threading import Lock

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    Qt,
    QTimer,
    QEvent,
    QObject,
    QAbstractTableModel,
    QModelIndex,
    QItemSelectionModel,
    QPoint,
)
from PySide6.QtGui import QFontMetrics, QColor, QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QAbstractItemView,
    QLabel,
    QTableView,
    QStyledItemDelegate,
    QMessageBox,
    QDialog,
    QSplitter,
    QMenu,
    QTabWidget,
    QComboBox,
    QCompleter,
)
from src.core.config import FROZEN_VIEW_WIDTH_DEFAULT, BLINK_PULSE
from src.ui.drag_drop import parse_var_names_from_mimedata
from src.core.logger import get_logger

logger = get_logger("ui.table_dialog")


@dataclass
class _GroupTabState:
    """MDF tab 模式下，每个 channel group 对应的 tab 状态。"""

    group_index: int
    view: QTableView
    df: pd.DataFrame
    model: PandasTableModel | None = None
    scroll_pos: int = 0
    widget_index: int = -1  # 在 QTabWidget 中的索引


# 无时间轴时的共用空数组（避免每次判定都新建 ndarray）
_EMPTY_TIME = np.array([], dtype=np.float64)


def _nearest_time_row(time_array: np.ndarray, target_time: float) -> int:
    """二分查找 time_array 中与 target_time 最接近的行号。

    tab 切换的时间锚点同步与 jump_to_data 共用同一实现，避免两处取整
    逻辑漂移。端点钳制由调用方处理（空数组返回 0，由调用方先行判空）。
    """
    row = int(np.searchsorted(time_array, target_time, side="left"))
    row = max(0, min(row, len(time_array) - 1))
    if row > 0 and abs(time_array[row - 1] - target_time) < abs(
        time_array[row] - target_time
    ):
        row -= 1
    return row


def _state_time_array(state: _GroupTabState) -> np.ndarray:
    """取 tab 的时间轴：唯一真相是 state.df 的 'time' 列。

    不在 state 上另存一份 loader 缓存数组本体：tab 生命周期等于会话，那份
    引用会把旧 loader 的 _time_cache 钉住不让 LRU 逐出（旧 MDF 文件常驻内存）；
    且 df 行数可能因 _align_len 填充或 update_data 重建而变化，另存一份必然
    与 df 错位。0 行空 tab（cycles_nr=0 的 group）无时间轴，返回空数组，
    调用方自行跳过锚点计算。
    """
    if "time" in state.df.columns and len(state.df) > 0:
        # float64 列的 to_numpy() 是视图而非拷贝（实测 1e7 点×5 次 0.0ms）
        return state.df["time"].to_numpy()
    return _EMPTY_TIME


class DropOverlay(QWidget):
    """
    拖拽覆盖层类
    在文件拖拽到应用程序时显示半透明的覆盖层，提供视觉反馈
    """

    def __init__(self, parent=None):
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
    变量数值表对话框类
    以独立窗口形式显示完整的变量数值表
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
        dlg._ensure_loader_consistent()
        dlg.save_geom()
        if dlg.has_column(var_name):
            # 变量已存在：在 tab 模式下切换到对应 tab
            if dlg._tab_mode:
                for state in dlg._group_tabs.values():
                    if var_name in state.df.columns and dlg._tab_widget:
                        dlg._tab_widget.setCurrentIndex(state.widget_index)
                        break
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        else:
            cls._saved_scroll_pos = (
                dlg.main_view.verticalScrollBar().value()
                if (dlg.main_view and not dlg._tab_mode)
                else None
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
        """批量添加变量至变量数值表，复用拖拽逻辑"""
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
        dlg._ensure_loader_consistent()
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

        # ---- MDF tab 模式 UI（初始隐藏，首次添加 MDF 变量时激活）----
        self._tab_mode = False
        self._tab_widget: QTabWidget | None = None
        self._group_tabs: dict[int, _GroupTabState] = {}
        self._var_locator: QComboBox | None = None  # 已添加变量的定位下拉框
        self._tab_container_widget: QWidget | None = None
        self._current_time_anchor: float | None = None  # 当前可见首行时刻
        self._tab_sync_depth = 0  # 程序化滚动嵌套深度：>0 时抑制锚点回写
        self._prev_tab_index = -1  # 上一个当前 tab 的 widget 索引
        self._gen = 0  # 代际令牌：重置后作废在途 QTimer 回调
        self._tab_loader_key: tuple | None = None  # 建 tab 时的数据源标识
        self._table_vars_snapshot: list[str] = []  # 重载前的单表变量清单
        self._tab_vars_snapshot: dict[int, list[str]] = {}  # 重载前的 tab 变量清单
        self._setup_tab_mode_ui()

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

    # ------------------------------------------------------------------
    # MDF tab 模式：UI 创建与切换
    # ------------------------------------------------------------------

    def _setup_tab_mode_ui(self):
        """创建 tab 模式 UI 组件（变量定位下拉框 + QTabWidget），初始隐藏。"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 变量定位下拉框：只列出已添加到表中的变量，选中仅定位不添加。
        # 默认 completer 是前缀匹配（MatchStartsWith），必须改成 MatchContains
        # 才能实现“输入任意子串即过滤”；caseSensitivity 另设，两开关独立。
        self._var_locator = QComboBox()
        self._var_locator.setEditable(True)
        self._var_locator.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = self._var_locator.completer()
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._var_locator.setMaxVisibleItems(15)
        self._var_locator.lineEdit().setPlaceholderText("定位变量... (Ctrl+F)")
        self._var_locator.setToolTip(
            "列出已添加到变量数值表的变量；输入关键字过滤，回车或点击选中项定位到对应列。"
        )
        self._var_locator.activated.connect(self._on_var_locator_activated)
        loc_layout = QHBoxLayout()
        loc_layout.setContentsMargins(0, 0, 0, 0)
        loc_layout.addWidget(self._var_locator)
        layout.addLayout(loc_layout)

        # Ctrl+F 快捷键
        shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        shortcut.activated.connect(self._focus_search)

        # Tab widget
        self._tab_widget = QTabWidget()
        self._tab_widget.setUsesScrollButtons(True)
        self._tab_widget.setTabsClosable(False)
        self._tab_widget.setMovable(False)
        self._tab_widget.currentChanged.connect(self._on_tab_switched)
        layout.addWidget(self._tab_widget)

        # 添加到主布局（在 splitter 之后）
        main_layout = self.centralWidget().layout()
        main_layout.addWidget(container)
        container.setVisible(False)
        self._tab_container_widget = container

    @staticmethod
    def _loader_identity(loader):
        """数据源身份标识：类型+路径+变量数。用于检测 tab 建立后 loader 被替换。"""
        if loader is None:
            return None
        return (
            getattr(loader, "LOADER_TYPE", ""),
            str(getattr(loader, "path", "") or ""),
            len(getattr(loader, "var_names", None) or ()),
        )

    def _ensure_loader_consistent(self):
        """数据源已变更但未走过置路径时，在入口处重置 tab 状态。

        正常重载路径由 update_data 负责重置；但存在绕过它的入口
        （如直接拖拽/双击），若不拦截会把新数据的变量往旧数据的 tab 里
        添加（group 索引语义不同，行对齐错乱）。
        """
        if not self._tab_mode and not self._group_tabs:
            return
        loader = self._resolve_loader()
        if (
            self._tab_loader_key is not None
            and self._loader_identity(loader) != self._tab_loader_key
        ):
            logger.info("DataTableDialog 检测到数据源变更，重置 tab 状态")
            self._reset_tab_mode()

    def has_table_content(self) -> bool:
        """表内是否有数据（单表或 tab 模式），作为 show/关闭判据的统一口径。

        旧判据只看 `_df.empty`：tab 模式下 `_df` 恒空，会把刚重建好的 tab
        窗口直接关掉（file_loader_manager）或误判弹窗内容已清空。
        """
        if self._group_tabs:
            # 空 group（cycles_nr=0）也会生成 tab，只判 dict 非空会把 0 行
            # 的空 tab 误判为“有内容”，重载后弹出一个空白窗口而非自动关闭
            return any(len(st.df) > 0 for st in self._group_tabs.values())
        return self._df is not None and not self._df.empty

    def _switch_to_tab_mode(self):
        """从单表模式切换到 MDF tab 模式。"""
        if self._tab_mode:
            return
        # 进入 tab 模式前清空单表残留（旧 CSV/Excel 列会以隐藏表形式滞留）
        if not self._df.empty:
            self._df = pd.DataFrame()
            self.model = None
            self.frozen_columns = []
            self.main_view.setModel(None)
            self.frozen_view.setModel(None)
        self._tab_mode = True
        loader = self._resolve_loader()
        self._tab_loader_key = self._loader_identity(loader)
        # units 供定位框/复制头行查单位；旧单表分支不会在 tab 模式下刷新它
        if loader is not None and hasattr(loader, "units"):
            self.units = loader.units

        # 隐藏单表模式 UI
        self.splitter.setVisible(False)

        # 显示 tab 模式 UI
        if self._tab_container_widget:
            self._tab_container_widget.setVisible(True)


    def _focus_search(self):
        """聚焦变量定位下拉框并全选，便于直接输入过滤。"""
        if self._tab_mode and self._var_locator:
            self._var_locator.setFocus()
            self._var_locator.lineEdit().selectAll()

    def _refresh_var_locator_items(self):
        """按表内已添加变量重建定位下拉框条目（只定位，不添加新变量）。"""
        combo = self._var_locator
        if combo is None:
            return
        text = combo.lineEdit().text()
        combo.clear()
        # 单位表优先用已缓存的 self.units：loader.units 是每次重建全量 dict 的
        # property（文件变量上千时），而本函数每加一列就被调一次
        units = self.units
        if not units:
            loader = self._resolve_loader()
            units = getattr(loader, "units", None) or {} if loader is not None else {}
        for gi in sorted(self._group_tabs.keys()):
            state = self._group_tabs[gi]
            for col in state.df.columns:
                if col == "time":
                    continue
                unit = units.get(col, "")
                unit_text = f" [{unit}]" if unit and unit != "-" else ""
                combo.addItem(f"{col}{unit_text} — G{gi}", (col, gi))
        combo.lineEdit().setText(text)

    def _on_var_locator_activated(self, index: int):
        """选中定位条目（点选或回车）：切换到对应 tab 并滚动到该列。"""
        if index < 0 or self._var_locator is None:
            return
        data = self._var_locator.itemData(index)
        if not data:
            return
        var_name, group_index = data
        state = self._group_tabs.get(group_index)
        if state is None:
            return
        if self._tab_widget:
            self._tab_widget.setCurrentIndex(state.widget_index)
        self._scroll_tab_to_column(state, var_name)
        self._var_locator.setCurrentIndex(-1)
        self._var_locator.lineEdit().setText("")

    def _scroll_tab_to_column(
        self, tab_state: _GroupTabState, var_name: str, blink: bool = True
    ):
        """在指定 tab 中水平滚动到目标列，保持垂直位置不动，可选闪烁高亮。"""
        if var_name not in tab_state.df.columns:
            return
        col_idx = tab_state.df.columns.get_loc(var_name)
        if tab_state.model:
            # scrollTo 的 PositionAtCenter 是二维的：若用第 0 行构造 index，
            # 垂直方向也会被拉回 0（实测 value 12345→0），毁掉时间锚点。
            # 改用"当前首可见行 + 半页"的行居中：居中后新 value == 原 value，
            # 垂直零位移。
            # 取首可见行行号，计算视口中间行，构造 index 做 scrollTo
            top_idx = tab_state.view.indexAt(QPoint(0, 0))
            top_row = top_idx.row() if top_idx.isValid() else 0
            page_rows = max(1, tab_state.view.viewport().height() // max(1, tab_state.view.rowHeight(0)))
            center_row = min(top_row + page_rows // 2, tab_state.model.rowCount() - 1)
            qindex = tab_state.model.index(center_row, col_idx)
            tab_state.view.scrollTo(
                qindex, QAbstractItemView.ScrollHint.PositionAtCenter
            )
        if not blink:
            return
        # 闪烁高亮（代际令牌：重置后 tab_state 已释放，过期回调自杀）
        gen = self._gen

        def _do_blink():
            if gen == self._gen:
                self._blink_tab_column(tab_state, var_name)

        QTimer.singleShot(100, _do_blink)

    def _blink_tab_column(
        self, tab_state: _GroupTabState, var_name: str, pulse: int = 800
    ):
        """在 tab 模式中闪烁高亮指定列。"""
        if var_name not in tab_state.df.columns:
            return
        col_idx = tab_state.df.columns.get_loc(var_name)
        if tab_state.model is None:
            return
        delegate = tab_state.view.itemDelegate()
        if isinstance(delegate, CustomDelegate):
            self._blink_step_on(delegate, col_idx, tab_state.view)
            # off 回调同样需要代际令牌：闪烁窗口期（800ms）内若发生
            # _reset_tab_mode，该 view 已被 deleteLater，过期回调访问
            # 已销毁的 C++ 对象会抛 RuntimeError
            gen = self._gen
            QTimer.singleShot(
                pulse,
                lambda g=gen, d=delegate, c=col_idx, v=tab_state.view: (
                    self._blink_step_off(d, c, v) if g == self._gen else None
                ),
            )

    def _group_state_by_widget_index(self, widget_index: int):
        if widget_index < 0:
            return None
        for state in self._group_tabs.values():
            if state.widget_index == widget_index:
                return state
        return None

    def _group_state_by_view(self, view):
        """按视图反查 tab state（页序变动时 widget_index 会陈旧，view 不会）。"""
        for state in self._group_tabs.values():
            if state.view is view:
                return state
        return None

    def _bind_tab_scroll_sync(self, state: _GroupTabState):
        """监听tab 视图垂直滚动，实时更新全局时间锚点。
    
        Qt6/PySide6 QTableView 默认 ScrollPerPixel，
        verticalScrollBar().value() 返回像素位置而非行号，
        必须用 view.rowAt() 转换为行号后再传给 _update_time_anchor。
        """
        gen = self._gen
    
        def on_scroll(_pixel: int, st=state, g=gen):
            if g != self._gen:
                return  # tab 已重置，过期回调自杀
            # 取首可见行行号：indexAt(QPoint(0,0)) 返回视口左上角处的模型索引
            idx = st.view.indexAt(QPoint(0, 0))
            row = idx.row() if idx.isValid() else 0
            self._update_time_anchor(st, row)

        state.view.verticalScrollBar().valueChanged.connect(on_scroll)

    def _update_time_anchor(self, state: _GroupTabState, row: int):
        """记录"当前可见首行时刻"作为切 tab 同步锚点。
    
        调用方负责将滚动位置转换为行号（Qt6 ScrollPerPixel 下
        scrollbar value 是像素，需用 indexAt 转换）。
        程序化滚动（guard 生效中）不回写，避免跳变覆盖用户真实视线。
        只认当前可见 tab 的滚动：隐藏 tab 的 valueChanged（hide/show 时
        viewport 尺寸变化引发的范围重算/值钳制微调、残留程序化滚动）会把
        全局锚点污染成其停留位置的时刻（实测：G0 滚到 150s 后切 G1，
        锚点被隐藏 G1 的滚动信号覆写为其末尾时刻 → 切换直接跳末尾）。
        """
        if self._tab_sync_depth:
            return
        _cur = (
            self._group_state_by_widget_index(self._tab_widget.currentIndex())
            if self._tab_widget is not None
            else None
        )
        if _cur is not state:
            return
        t = _state_time_array(state)
        if len(t) == 0:
            return
        _idx = min(row, len(t) - 1)
        self._current_time_anchor = float(t[_idx])

    def _on_tab_switched(self, index: int):
        """tab 切换：保存旧 tab 行位置，目标 tab 对齐到当前时间锚点。

        锚点存在且被目标时间轴覆盖时按时间二分映射到目标 tab 的行
        （不同 group 采样率不同，相同的行号不等于相同的时刻）；目标
        时间轴不覆盖锚点（该表没有对应时刻的数据）时恢复其自己上次
        位置、锚点不动——旧实现钳到最近端点，表首/表尾行的大时刻
        在用户视角就是“莫名跳到表末尾”。
        """
        if not self._tab_mode:
            self._prev_tab_index = index
            return
        # 只保存上一个 tab 的位置（全量覆写会把新 tab 当前值误存为其历史位置）
        prev = self._group_state_by_widget_index(self._prev_tab_index)
        if prev is not None and prev.view is not None:
            prev.scroll_pos = prev.view.verticalScrollBar().value()

        target = self._group_state_by_widget_index(index)
        self._prev_tab_index = index
        if target is None or target.view is None or target.model is None:
            return

        anchor = self._current_time_anchor
        t_target = _state_time_array(target)
        self._tab_sync_depth += 1
        try:
            in_range = False
            if anchor is not None and len(t_target) > 0:
                # 用 min/max 而非 t[0]/t[-1] 判覆盖：对单调轴两者等价，
                # 还能兼容 asammdf 跨块拼装出的非单调轴
                in_range = float(t_target.min()) <= anchor <= float(t_target.max())
            if in_range:
                row = _nearest_time_row(t_target, anchor)
                # scrollTo 而非 sb.setValue：后者在尾部 pageStep-1 行被钓制，
                # 且页面首次布局时 max 会抖动
                target.view.scrollTo(
                    target.model.index(row, 0),
                    QAbstractItemView.ScrollHint.PositionAtTop,
                )
            else:
                # 锚点越界（或无锚点）：保持该表自己的历史位置
                target.view.verticalScrollBar().setValue(target.scroll_pos)
        finally:
            self._tab_sync_depth -= 1

    def locate_time(
        self, group_index: int, target_time: float, var_name: str | None = None
    ) -> bool:
        """跳转到指定 group 的 tab 并定位到 target_time 最近行（jump_to_data 用）。

        返回是否成功找到目标 tab。先把锚点设为 target_time 并用 guard 拑住
        程序化滚动的回写，避免切换瞬间旧锚点把本次跳转目标覆盖。

        Args:
            var_name: 需要高亮/居中的目标变量列；None 或该变量不在本 tab 时
                回退到 time 列（col 0）。
        """
        state = self._group_tabs.get(group_index)
        if (
            state is None
            or state.view is None
            or state.model is None
            or len(_state_time_array(state)) == 0
        ):
            return False
        row = _nearest_time_row(_state_time_array(state), target_time)
        self._current_time_anchor = float(target_time)
        if self._tab_widget:
            self._tab_sync_depth += 1
            try:
                self._tab_widget.setCurrentIndex(state.widget_index)
            finally:
                self._tab_sync_depth -= 1
        # 选中目标变量列而不是总在第 0 列的 time：多变量跳一时能看出落点
        col = 0
        if var_name is not None and var_name in state.df.columns:
            col = int(state.df.columns.get_loc(var_name))
        qindex = state.model.index(row, col)
        gen = self._gen

        def _do():
            if gen != self._gen or state.model is None:
                return
            self._tab_sync_depth += 1
            try:
                state.view.scrollTo(
                    qindex, QAbstractItemView.ScrollHint.PositionAtCenter
                )
                sm = state.view.selectionModel()
                if sm is not None:
                    sm.select(
                        qindex,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect,
                    )
            finally:
                self._tab_sync_depth -= 1
                # scroll_pos 统一存像素值（与 _on_tab_switched 一致）
                state.scroll_pos = state.view.verticalScrollBar().value()

        QTimer.singleShot(0, _do)
        return True

    def _reset_tab_mode(self):
        """退出 tab 模式并完整释放所有 group tab 的视图/模型/数据。

        顺序敏感：先降级 _tab_mode 并摘走 _group_tabs（使信号回调看到一致的
        空态），断开 currentChanged 防止 clear() 触发 index=-1 回灌，升代际
        令牌作废在途 QTimer 闭包，再逐个摘事件过滤器、解绑模型、deleteLater
        视图，最后清空快照。时间轴不在 state 上另存副本（见
        _state_time_array），旧 loader 的 _time_cache 数组本体因此可被 LRU
        正常逐出，不必等整个 tab 关闭。定位下拉框是容器
        复用控件，只清条目不销毁。
        """
        if not self._tab_mode and not self._group_tabs:
            return
        self._tab_mode = False
        tabs, self._group_tabs = self._group_tabs, {}
        self._gen += 1
        self._current_time_anchor = None
        self._prev_tab_index = -1
        self._tab_loader_key = None
        if self._tab_widget is not None:
            try:
                self._tab_widget.currentChanged.disconnect(self._on_tab_switched)
            except (RuntimeError, TypeError):
                pass
        for state in tabs.values():
            try:
                state.view.viewport().removeEventFilter(self.drop_filter)
            except RuntimeError:
                pass
            state.view.setModel(None)
            state.view.deleteLater()
            state.df = pd.DataFrame()
            state.model = None
            state.scroll_pos = 0
        if self._tab_widget is not None:
            self._tab_widget.clear()
            self._tab_widget.currentChanged.connect(self._on_tab_switched)
        if self._var_locator is not None:
            self._var_locator.clear()
        if self._tab_container_widget is not None:
            self._tab_container_widget.setVisible(False)
        self.splitter.setVisible(True)
        if self.model is None:
            # 进 tab 模式时 _switch_to_tab_mode 把单表模型置了 None；删空最后一个
            # tab 退回时若不补一个空模型，_update_views 会直接 return，窗口就是
            # 左右两块白板（frozen_view 也得不到隐藏）
            self._df = pd.DataFrame()
            self.model = PandasTableModel(self._df, self.units)
            self.main_view.setModel(self.model)
            self.frozen_view.setModel(self.model)
            self._connect_signals()
        self._update_views()

    @staticmethod
    def _align_len(values: np.ndarray, n: int) -> np.ndarray:
        """列数据长度对齐时间轴：过长截断、过短补空值、空列填 None。

        MDF 重载/异常数据下 y 长度与 master 时间轴可能不一致，旧写法
        `y[:n]` 只防变长不防变短，赋列时抛 ValueError。

        短缺部分补空值而不是重复尾值：重复尾值会在数据缺失区间伪造出一条
        平线，行数和数值看起来都正常、时刻却全错，用户无从察觉；NaN/None
        在模型里渲染为空单元格（见 PandasTableModel.data 的 notnull 分支）。
        整型/字符串列无法承载 NaN，转 object 列填 None。
        """
        if n <= 0:
            return values[:0]
        if len(values) >= n:
            return values[:n]
        if len(values) == 0:
            return np.full(n, None, dtype=object)
        logger.warning(
            "列数据长度 %d 短于时间轴 %d，尾部 %d 行按空值处理（数据缺失）",
            len(values),
            n,
            n - len(values),
        )
        if values.dtype.kind == "f":
            pad = np.full(n - len(values), np.nan, dtype=values.dtype)
            return np.concatenate([values, pad])
        pad = np.full(n - len(values), None, dtype=object)
        return np.concatenate([values.astype(object), pad])

    @staticmethod
    def _row_header_width(n_rows: int, font: QFont) -> int:
        """行号列定宽：按最大行号的位数估算，下限 48px。

        行号表头右对齐，宽度不够时被裁的是左边的**高位**数字：实测 48px 只
        容得下 5 位，100Hz 长时程文件的 6 位行号会显示成“03580”（实为 103580），
        用户无从察觉。不用 ResizeToContents：万行表逐节测量代价大。
        取 "9"×位数而非 str(n_rows)：“9” 是最宽的字形，估计偏保守。
        """
        digits = max(4, len(str(max(1, n_rows))))
        return max(48, QFontMetrics(font).horizontalAdvance("9" * digits) + 10)

    def _add_variable_to_tab(
        self,
        var_name: str,
        group_index: int,
        loader=None,
    ) -> _GroupTabState | None:
        """将变量添加到指定 group 的 tab。返回对应的 _GroupTabState。

        由 _add_variable_to_table（MDF 分支）与 update_data 重建路径调用，
        内部自带 _switch_to_tab_mode，保证任何入口下 _tab_mode 与
        _group_tabs 一致（旧版存在“建了 tab 但未切模式→加了看不见”的缝）。

        Args:
            var_name: 变量名
            group_index: 目标 channel group 索引（内部会按名校正）
            loader: 取数用的 loader；缺省回退到 owner 的当前 loader。
                update_data 必须显式下传入参 loader，否则会出现
                “按新 loader 查 group、按旧 loader 取数”的静默串数据。
        """
        if loader is None:
            loader = self._resolve_loader()
        if loader is None:
            return None
        if not self._tab_mode:
            self._switch_to_tab_mode()

        # group 归属以变量名为准：调用方（update_data 拿的是旧文件的 group
        # 快照）传入的 group_index 在新数据里语义可能不同，直接沿用会把变量
        # 塞进错误的时间轴——行数以该轴为准做截断/填充，数据看起来正常但时刻
        # 全错，用户无从察觉
        get_gi = getattr(loader, "get_var_group_index", None)
        if callable(get_gi):
            try:
                real_gi = get_gi(var_name)
            except KeyError:
                real_gi = group_index  # 变量不存在，交给下方 get_series 报错
            if real_gi != group_index:
                logger.warning(
                    "变量 '%s' 实属 G%d，却请求加入 G%d tab，已按名归位",
                    var_name,
                    real_gi,
                    group_index,
                )
                group_index = real_gi

        # 获取 group 时间数组
        try:
            time_array = loader.get_group_time_array(group_index)
        except Exception:
            logger.warning("获取 group %d 时间数组失败", group_index, exc_info=True)
            return None

        # 获取变量数据
        try:
            series = loader.get_series(var_name)
        except KeyError:
            logger.warning("变量 '%s' 不存在", var_name)
            return None

        y_data = series.to_numpy()

        if group_index in self._group_tabs:
            # 已有 tab，添加列
            state = self._group_tabs[group_index]
            if var_name in state.df.columns:
                return state  # 已存在
            state.df[var_name] = self._align_len(y_data, len(state.df))
            state.model = PandasTableModel(state.df, self.units)
            state.view.setModel(state.model)
            # 与新 tab 分支对齐：切到该列所在页。否则向非当前 tab 加列时
            # “加了看不见”，且调用方随后打在隐藏页上的闪烁用户完全看不到
            if self._tab_widget is not None:
                self._tab_widget.setCurrentIndex(state.widget_index)
            self._refresh_var_locator_items()
            return state
        else:
            # 创建新 tab，time 作为首列（列名避开变量命名空间）
            # DataFrame 构造会拷贝一份时间轴（实测 pandas 3.0 不与入参共享内存），
            # 因此 tab 持有的是自己那一份，不会钉住 loader 的 _time_cache
            tab_df = pd.DataFrame({"time": np.asarray(time_array)})
            tab_df[var_name] = self._align_len(y_data, len(tab_df))
            tab_model = PandasTableModel(tab_df, self.units)

            tab_view = QTableView()
            tab_view.setModel(tab_model)
            # 行号列：PandasTableModel.headerData 的 Vertical 分支提供 1-based
            # 行号，打开内建表头即可，不要 ResizeToContents（万行表测量代价大）
            vh = tab_view.verticalHeader()
            vh.setVisible(True)
            # 宽度跟着该 group 的行数走：固定 48px 在 6 位以上行号会裁掉高位
            vh_width = self._row_header_width(len(tab_df), tab_view.font())
            vh.setMinimumWidth(vh_width)
            vh.setMaximumWidth(vh_width)
            vh.setDefaultAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            tab_view.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectItems
            )
            tab_view.horizontalHeader().setSectionsMovable(True)
            tab_view.horizontalHeader().setDragEnabled(True)
            tab_view.horizontalHeader().setDragDropMode(
                QAbstractItemView.DragDropMode.InternalMove
            )
            tab_view.horizontalHeader().setDragDropOverwriteMode(False)
            tab_view.horizontalHeader().setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            tab_view.setWordWrap(False)
            tab_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tab_view.customContextMenuRequested.connect(
                self._show_table_context_menu
            )
            # 表头右键菜单与单表模式对齐：删除列/复制变量名/清空列表在 tab
            # 模式下原本无处可用（冻结列不适用，置灰而非静默无效）
            tab_view.horizontalHeader().setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            tab_view.horizontalHeader().customContextMenuRequested.connect(
                lambda pos, v=tab_view: self._on_header_right_click(pos, v)
            )
            tab_view.setAcceptDrops(True)
            tab_view.viewport().installEventFilter(self.drop_filter)

            # 表头字体加粗
            font = tab_view.horizontalHeader().font()
            font.setBold(True)
            tab_view.horizontalHeader().setFont(font)

            # 行高
            fm = QFontMetrics(tab_view.font())
            safe_height = int(fm.height() * 1.6)
            tab_view.verticalHeader().setDefaultSectionSize(safe_height)

            # 委托
            delegate = CustomDelegate(tab_view)
            delegate.highlighted_rows = set()
            tab_view.setItemDelegate(delegate)

            # 添加到 tab widget
            if self._tab_widget is None:
                return None
            widget_index = self._tab_widget.addTab(tab_view, f"G{group_index}")
            # 设置 tooltip
            try:
                label = loader.get_group_label(group_index)
                self._tab_widget.setTabToolTip(widget_index, label)
            except Exception:
                self._tab_widget.setTabToolTip(widget_index, f"Group {group_index}")

            state = _GroupTabState(
                group_index=group_index,
                view=tab_view,
                df=tab_df,
                model=tab_model,
                widget_index=widget_index,
            )
            # 注册与滚动监听必须先于切页：setCurrentIndex 同步触发
            # _on_tab_switched，而它第一步就按 widget_index 反查 state，
            # 查不到直接 return → 新建 tab 停在第 0 行、时间锚点同步失效
            # （_bind 前置也安全：程序化滚动仍被 _tab_sync_depth 拑住）
            self._group_tabs[group_index] = state
            self._bind_tab_scroll_sync(state)
            self._refresh_var_locator_items()
            # 新 tab 自动切换为当前页（旧版不切页：双击新 group 变量时
            # 闪烁打在隐藏页，用户视角“加了看不见”）
            self._tab_widget.setCurrentIndex(widget_index)
            return state

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
        if not main_window or not hasattr(main_window, 'layout_manager'):
            return
        container = getattr(main_window, "_active_drag_container", None)
        if container and getattr(container, "plot_widget", None):
            main_window.layout_manager._hide_drag_indicator_for_plot(container.plot_widget)

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
        # discard 而非 remove：highlighted_cols 是 set，同一列在 pulse 窗口
        # 内被重复闪烁时第二次 add 被去重，但仍会排入等量的 off 回调，
        # 后到的 remove 会因元素已被删而抛 KeyError
        delegate.highlighted_cols.discard(col_idx)
        view.viewport().update()

    def _blink_column(self, var_name, pulse: int = 800):
        if self._tab_mode:
            # tab 模式：在对应 tab 中闪烁
            for state in self._group_tabs.values():
                if var_name in state.df.columns:
                    self._blink_tab_column(state, var_name, pulse)
                    break
            return

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

        self._ensure_loader_consistent()
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
        self._ensure_loader_consistent()
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
        内部函数：将变量添加到表格

        MDF 数据自动切换到 tab 模式，按 channel group 分 tab 展示。
        CSV/Excel 数据保持原有单表行为。

        Args:
            var_name: 变量名称
            data: 变量数据序列
        """
        loader = self._resolve_loader()
        is_mdf = loader is not None and getattr(loader, "LOADER_TYPE", "") == "mdf"

        if is_mdf:
            # MDF 数据：切换到 tab 模式并按 group 添加（数据获取在
            # _add_variable_to_tab 内部完成，不消费 data 参数）
            try:
                group_index = loader.get_var_group_index(var_name)
            except KeyError:
                logger.warning("MDF 变量 '%s' 无法获取 group 信息", var_name)
                return

            self._add_variable_to_tab(var_name, group_index)
            return

        # ---- 非 MDF 数据：原有单表逻辑 ----
        self._df[var_name] = data.reset_index(drop=True)
        max_len = max(len(self._df), len(data))
        if len(self._df) < max_len:
            self._df = self._df.reindex(range(max_len))

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

    def _update_highlights_frozen(self, _selected, _deselected):
        # selected/deselected 由 Qt selectionModel 信号传入，此处不需要
        self.current_focused_view = self.frozen_view
        self._update_highlights_on_focus_change()

    def _update_highlights_main(self, _selected, _deselected):
        # selected/deselected 由 Qt selectionModel 信号传入，此处不需要
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

        analysis = self._analyze_selection(view)
        if analysis is None:
            return

        menu = QMenu(self)
        scatter_added = self._build_plot_menu(menu, analysis)
        self._build_copy_menu(menu, analysis, scatter_added)
        menu.exec(view.mapToGlobal(pos))

    def _analyze_selection(self, view):
        """分析当前视图及另一视图的选中内容，返回分析结果字典；无选中时返回 None。"""
        # tab 模式的每个 tab 有独立 df/model、无冻结区概念，单独分析。
        # （旧版直接引用无 model 的 main_view/_df，tab 模式右键直接崩溃）
        if self._tab_mode or self._group_tabs:
            state = self._group_state_by_view(view)
            if state is not None:
                return self._analyze_tab_selection(state, view)

        sm = view.selectionModel()
        if sm is None:
            return None
        selected_indexes = sm.selectedIndexes()
        if not selected_indexes:
            return None

        frozen_cols = set(self._df.columns.get_loc(col) for col in self.frozen_columns)
        if view == self.main_view:
            other_view = self.frozen_view
        else:
            other_view = self.main_view

        other_sm = other_view.selectionModel()
        other_selected = other_sm.selectedIndexes() if other_sm else []
        all_selected = list(selected_indexes) + list(other_selected)

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
            only_col = next(iter(total_cols))
            ordered_cols = [only_col]
            rows_order = sorted(rows_per_col_all[only_col])
            can_copy = len(rows_order) > 0
        elif len(total_cols) >= 2:
            cols_list = list(total_cols)
            base_rows = rows_per_col_all[cols_list[0]] if cols_list else set()
            if base_rows and all(
                rows_per_col_all[c] == base_rows for c in cols_list[1:]
            ):
                frozen_header = self.frozen_view.horizontalHeader()
                main_header = self.main_view.horizontalHeader()
                frozen_selected_cols = [c for c in total_cols if c in frozen_cols]
                main_selected_cols = [c for c in total_cols if c not in frozen_cols]
                frozen_selected_cols.sort(key=lambda c: frozen_header.visualIndex(c))
                main_selected_cols.sort(key=lambda c: main_header.visualIndex(c))
                ordered_cols = frozen_selected_cols + main_selected_cols
                rows_order = sorted(base_rows)
                can_copy = True

        # 计算绘图可行性
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

        return {
            "total_cols": total_cols,
            "can_copy": can_copy,
            "ordered_cols": ordered_cols,
            "rows_order": rows_order,
            "plot_enabled": plot_enabled,
            "x_col": x_col,
            "y_col": y_col,
            "plot_rows": plot_rows,
        }

    def _analyze_tab_selection(self, state: _GroupTabState, view):
        """tab 视图的选中分析：单视图、无冻结区，结果附带 df/model 供消费方透传。"""
        sm = view.selectionModel()
        if sm is None:
            return None
        selected = sm.selectedIndexes()
        if not selected:
            return None

        rows_per_col: dict[int, set[int]] = {}
        for idx in selected:
            rows_per_col.setdefault(idx.column(), set()).add(idx.row())
        total_cols = set(rows_per_col.keys())
        header = view.horizontalHeader()

        can_copy = False
        ordered_cols: list[int] = []
        rows_order: list[int] = []
        if len(total_cols) == 1:
            only = next(iter(total_cols))
            ordered_cols = [only]
            rows_order = sorted(rows_per_col[only])
            can_copy = bool(rows_order)
        elif len(total_cols) >= 2:
            cols_list = list(total_cols)
            base = rows_per_col[cols_list[0]]
            if base and all(rows_per_col[c] == base for c in cols_list[1:]):
                ordered_cols = sorted(cols_list, key=lambda c: header.visualIndex(c))
                rows_order = sorted(base)
                can_copy = True

        plot_enabled = False
        x_col = y_col = None
        plot_rows: list[int] = []
        all_rows: set[int] = set()
        for rows in rows_per_col.values():
            all_rows.update(rows)
        if len(total_cols) == 2 and len(all_rows) >= 2:
            ordered2 = sorted(total_cols, key=lambda c: header.visualIndex(c))
            x_col, y_col = ordered2[0], ordered2[1]
            plot_rows = sorted(all_rows)
            plot_enabled = True

        return {
            "total_cols": total_cols,
            "can_copy": can_copy,
            "ordered_cols": ordered_cols,
            "rows_order": rows_order,
            "plot_enabled": plot_enabled,
            "x_col": x_col,
            "y_col": y_col,
            "plot_rows": plot_rows,
            "df": state.df,
            "model": state.model,
        }

    def _build_plot_menu(self, menu, analysis):
        """构建绘图子菜单。返回是否已添加可用的散点图动作。"""
        total_cols = analysis["total_cols"]
        if len(total_cols) != 2:
            return False

        plot_enabled = analysis["plot_enabled"]
        x_col = analysis["x_col"]
        y_col = analysis["y_col"]
        plot_rows = analysis["plot_rows"]

        cols_list = sorted(list(total_cols))
        if plot_enabled and x_col is not None and y_col is not None:
            x_show, y_show = x_col, y_col
        else:
            x_show, y_show = cols_list[0], cols_list[1]
        # 模型路由：tab 分析结果自带 model，单表回退 self.model
        model = analysis.get("model") or self.model
        if model is None:
            return False
        x_name = model.headerData(
            x_show, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        ).replace("\n", " ")
        y_name = model.headerData(
            y_show, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        ).replace("\n", " ")

        # 存储列名供 _build_copy_menu 在禁用场景下复用
        analysis["x_name"] = x_name
        analysis["y_name"] = y_name

        act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
        act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
        if plot_enabled and x_col is not None and y_col is not None:
            act1.triggered.connect(
                lambda _checked=False, rows=plot_rows, x=x_col, y=y_col,
                d=analysis.get("df"), m=analysis.get("model"): self._plot_xy_scatter(
                    x, y, rows=rows, df=d, model=m
                )
            )
            act2.triggered.connect(
                lambda _checked=False, rows=plot_rows, x=x_col, y=y_col,
                d=analysis.get("df"), m=analysis.get("model"): self._plot_xy_scatter(
                    y, x, rows=rows, df=d, model=m
                )
            )
            act1.setEnabled(True)
            act2.setEnabled(True)
            menu.addAction(act1)
            menu.addAction(act2)
            return True
        else:
            act1.setEnabled(False)
            act2.setEnabled(False)
            return False

    def _build_copy_menu(self, menu, analysis, scatter_added):
        """构建复制子菜单（含禁用绘图项的补位逻辑）。"""
        can_copy = analysis["can_copy"]
        ordered_cols = analysis["ordered_cols"]
        rows_order = analysis["rows_order"]
        total_cols = analysis["total_cols"]

        act_copy_selected = QAction("复制所选数据到剪贴板", menu)
        act_copy_selected.setEnabled(can_copy)
        if can_copy:
            act_copy_selected.triggered.connect(
                lambda checked=False, cols=ordered_cols, rows=rows_order,
                d=analysis.get("df"): self._copy_selected_to_clipboard(cols, rows, d)
            )

        act_copy_all = QAction("复制表内所有数据到剪贴板", menu)
        cur_df = analysis.get("df")
        if cur_df is None:
            cur_df = self._df
        enable_all = (
            cur_df is not None and cur_df.shape[0] > 0 and cur_df.shape[1] > 0
        )
        act_copy_all.setEnabled(enable_all)
        if enable_all:
            act_copy_all.triggered.connect(
                lambda checked=False, d=cur_df: self._copy_all_to_clipboard(d)
            )

        if scatter_added:
            menu.addSeparator()
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
        else:
            menu.addAction(act_copy_selected)
            menu.addAction(act_copy_all)
            if len(total_cols) == 2:
                menu.addSeparator()
                x_name = analysis.get("x_name", "")
                y_name = analysis.get("y_name", "")
                act1 = QAction(f"绘制x/y图，x={x_name}，y={y_name}", menu)
                act2 = QAction(f"绘制x/y图，x={y_name}，y={x_name}", menu)
                act1.setEnabled(False)
                act2.setEnabled(False)
                menu.addAction(act1)
                menu.addAction(act2)

    def _plot_xy_scatter(
        self,
        x_col_idx,
        y_col_idx,
        rows=None,
        start_row=None,
        num_rows=None,
        df=None,
        model=None,
    ):
        """
        接收已按视觉顺序确定的逻辑列索引进行绘图。

        df/model 由右键分析结果透传（tab 模式指向对应 tab 的数据）；
        省略时回退单表 self._df/self.model。
        """
        try:
            if df is None:
                df = self._df
            if model is None:
                model = self.model
            if rows is None:
                if start_row is None or num_rows is None:
                    return
                row_indexer = slice(start_row, start_row + num_rows)
            else:
                row_indexer = rows

            # 直接使用正确的逻辑索引提取数据
            x_data_series = pd.to_numeric(
                df.iloc[row_indexer, x_col_idx], errors="coerce"
            )
            y_data_series = pd.to_numeric(
                df.iloc[row_indexer, y_col_idx], errors="coerce"
            )

            # 验证1：检查是否有非数值数据
            if x_data_series.isnull().any() or y_data_series.isnull().any():
                QMessageBox.warning(
                    self, "绘图错误", "选中区域包含无法转换为数字的单元格。"
                )
                return

            # 获取清理后的列标题
            x_header = model.headerData(
                x_col_idx, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            ).replace("\n", " ")
            y_header = model.headerData(
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
        self, ordered_cols: list[int], rows_order: list[int], df=None
    ):
        """将选中区域的数据复制到剪贴板。

        第一行：变量名；第二行：单位；第三行开始为数据。
        df 省略时用单表 self._df（tab 模式由右键分析结果透传）。
        """
        if not ordered_cols or not rows_order:
            return
        if df is None:
            df = self._df
        # 变量名与单位
        var_names = [str(df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, "") for name in var_names]

        # 组装数据（按行）
        lines = []
        lines.append("\t".join(var_names))
        lines.append("\t".join(units))

        for r in rows_order:
            row_vals = []
            for c in ordered_cols:
                val = df.iloc[r, c]
                if pd.isna(val):
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            lines.append("\t".join(row_vals))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

    def _copy_all_to_clipboard(self, df=None):
        """复制表内所有数据到剪贴板，列顺序按可视顺序（先冻结区再主区）。

        tab 模式传入的 df 属于单个 tab，无冻结区概念，按 df 列序直接复制。
        """
        if df is None:
            df = self._df
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            return

        if df is not self._df:
            # tab 模式：单视图无冻结分割，按当前 tab 的表头视觉顺序
            ordered_cols = list(range(df.shape[1]))
        else:
            # 计算可视列顺序：先冻结区，再主区
            frozen_cols = set(
                self._df.columns.get_loc(col) for col in self.frozen_columns
            )
            frozen_header = self.frozen_view.horizontalHeader()
            main_header = self.main_view.horizontalHeader()

            all_cols = list(range(df.shape[1]))
            frozen_list = [c for c in all_cols if c in frozen_cols]
            main_list = [c for c in all_cols if c not in frozen_cols]
            frozen_list.sort(key=lambda c: frozen_header.visualIndex(c))
            main_list.sort(key=lambda c: main_header.visualIndex(c))
            ordered_cols = frozen_list + main_list

        # 变量名与单位
        var_names = [str(df.columns[c]) for c in ordered_cols]
        units = [self.units.get(name, "") for name in var_names]

        lines = []
        lines.append("\t".join(var_names))
        lines.append("\t".join(units))

        for r in range(df.shape[0]):
            row_vals = []
            for c in ordered_cols:
                val = df.iloc[r, c]
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
        """重载数据时：先快照当前变量清单，再清空表格数据与 tab 状态。

        快照必须先于清空：file_loader_manager 在调 update_data 前会经本
        方法清掉 _df，若不留清单，update_data 无从得知该用新数据重建哪些
        变量（旧版因此在全路径上提前 return，tab 数据原样残留界面）。
        """
        self._table_vars_snapshot = self.get_column_names()
        self._tab_vars_snapshot = {
            gi: [c for c in st.df.columns if c != "time"]
            for gi, st in self._group_tabs.items()
        }
        if self._group_tabs or self._tab_mode:
            self._reset_tab_mode()
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
        if not self._skip_close_confirmation and self._count_table_vars() >= 4:
            reply = QMessageBox.question(
                self,
                "确认关闭",
                "是否清除所有列表，并关闭变量数值表窗口？",
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
        self._reset_tab_mode()
        self._table_vars_snapshot = []
        self._tab_vars_snapshot = {}
        self.save_geom()
        self._df = pd.DataFrame()
        self.main_view.setModel(None)
        self.frozen_view.setModel(None)
        # 必须经类名赋值：旧写法 self._instance = None 只创建了遮蔽类属性的
        # 实例属性，DataTableDialog._instance 永远指着已关闭窗口，导致单例
        # 泄漏且下次 popup 复用陈旧窗口（跨文件数据串显）。
        DataTableDialog._instance = None
        DataTableDialog._saved_scroll_pos = None
        self.frozen_columns = []
        self.hide()
        event.accept()

    def set_skip_close_confirmation(self, status: bool):
        self._skip_close_confirmation = status

    def _count_table_vars(self) -> int:
        """表内变量数（tab 模式不含 time 列），用于关闭确认门槛。"""
        return len(self.get_column_names())

    def has_column(self, var_name: str) -> bool:
        # 判据用 `_tab_mode or _group_tabs` 而非只看 _tab_mode：重置中间态
        # 或两者不一致时，只看 _tab_mode 会去查空的 _df 导致误判“变量不存在”
        if self._tab_mode or self._group_tabs:
            return any(
                var_name in state.df.columns
                for state in self._group_tabs.values()
            )
        return var_name in self._df.columns

    def get_column_names(self) -> list[str]:
        """返回表内全部变量名（公开接口）；tab 模式聚合各 tab 并排除 time 列。"""
        if self._tab_mode or self._group_tabs:
            # dict.fromkeys 去重：保序且 O(n)；`c not in names` 列表查重是 O(n²)
            return list(
                dict.fromkeys(
                    c
                    for state in self._group_tabs.values()
                    for c in state.df.columns
                    if c != "time"
                )
            )
        return self._df.columns.tolist()

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

    def _rebuild_frozen_column_order(self, full_visual_order: list[str]) -> list[str]:
        """根据 frozen_columns 重建完整视觉列顺序（冻结列在前）"""
        new_order: list[str] = []
        # 冻结列在前
        for col in full_visual_order:
            if col in self.frozen_columns and col not in new_order:
                new_order.append(col)
        # 非冻结列在后
        for col in full_visual_order:
            if col not in self.frozen_columns and col not in new_order:
                new_order.append(col)
        return new_order

    def freeze_column(self, logical_col):
        # tab 模式无冻结区概念（time 首列常驻，水平滚动即可），直接拒绝
        if self._tab_mode or self._group_tabs:
            return
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

            new_column_order = self._rebuild_frozen_column_order(full_visual_order)

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
        if self._tab_mode or self._group_tabs:
            return
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

            new_column_order = self._rebuild_frozen_column_order(full_visual_order)

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
        """表头右键菜单（单表与 MDF tab 模式共用一个入口）。

        tab 模式按 view 反查所属 state，删除列作用于该 tab 的 df。冻结列在
        tab 模式无意义（time 首列常驻、水平滚动即达），但菜单项仍列出并置灰：
        旧写法是 freeze_column 里静默 return，用户点了没有任何反馈。
        """
        header = view.horizontalHeader()
        logical_col = header.logicalIndexAt(pos)
        if logical_col < 0:
            return

        state = self._group_state_by_view(view)
        df = state.df if state is not None else self._df
        if logical_col >= len(df.columns):
            return
        var_name = str(df.columns[logical_col])

        menu = QMenu(self)
        act_delete = menu.addAction(f'删除列 "{var_name}"')
        if state is not None:
            # time 列是整页的行→时刻映射，删掉它该 tab 就没法看了
            act_delete.setEnabled(var_name != "time")

        act_freeze = None
        if state is not None:
            act_freeze = menu.addAction("冻结列（tab 模式不支持）")
            act_freeze.setEnabled(False)
            act_freeze.setToolTip("tab 模式下 time 首列常驻，水平滚动即可查看")
        elif var_name in self.frozen_columns:
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
            if state is not None:
                self._remove_tab_column(state, var_name)
            else:
                self._remove_column(logical_col)
        elif state is None and selected == act_freeze:
            if var_name in self.frozen_columns:
                self.unfreeze_column(logical_col)
            else:
                self.freeze_column(logical_col)

    def _remove_tab_column(self, state: _GroupTabState, var_name: str):
        """tab 模式删除一列；该组已无变量列时整页移除。"""
        if var_name == "time" or var_name not in state.df.columns:
            return
        state.df.drop(columns=[var_name], inplace=True)
        if any(c != "time" for c in state.df.columns):
            # setModel 只换模型不换视图，垂直滚动条对象不变 → 锚点监听仍有效
            state.model = PandasTableModel(state.df, self.units)
            state.view.setModel(state.model)
            self._refresh_var_locator_items()
            return
        self._remove_tab(state)

    def _remove_tab(self, state: _GroupTabState):
        """整页移除一个 tab。

        removeTab 会让后续页的 widget_index 整体前移，必须按 view 反查重映射；
        否则 _group_state_by_widget_index 会查到错误的 state：切页时把 A 组的
        历史位置写进 B 组、锚点对齐跳到无关的行。
        """
        self._group_tabs.pop(state.group_index, None)
        tw = self._tab_widget
        was_current = tw is not None and tw.indexOf(state.view) == tw.currentIndex()
        if tw is not None:
            try:
                tw.currentChanged.disconnect(self._on_tab_switched)
            except (RuntimeError, TypeError):
                pass
            idx = tw.indexOf(state.view)
            if idx >= 0:
                tw.removeTab(idx)
            for i in range(tw.count()):
                target = self._group_state_by_view(tw.widget(i))
                if target is not None:
                    target.widget_index = i
            self._prev_tab_index = tw.currentIndex()
            tw.currentChanged.connect(self._on_tab_switched)
        try:
            state.view.viewport().removeEventFilter(self.drop_filter)
        except RuntimeError:
            pass
        state.view.setModel(None)
        state.view.deleteLater()
        state.df = pd.DataFrame()
        state.model = None
        state.scroll_pos = 0
        if not self._group_tabs:
            # 最后一个 tab 被删空：退回单表模式 UI，不留一个空标签栏
            self._reset_tab_mode()
            return
        if was_current and tw is not None and tw.currentIndex() >= 0:
            # 删的是当前页：补做一次切页，让顶上来的页对齐全局时间锚点
            self._on_tab_switched(tw.currentIndex())
        self._refresh_var_locator_items()

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
        if self._tab_mode or self._group_tabs:
            # tab 模式没有 self.model/_df（单表模型为 None），原写法直接
            # AttributeError；整体重置并清掉重建快照，否则下次重载会把用户
            # 刚刚清空的列“复活”
            self._reset_tab_mode()
            self._df = pd.DataFrame()
            self._table_vars_snapshot = []
            self._tab_vars_snapshot = {}
            return
        if self.model is None:
            return
        # 清空所有列
        while self.model.columnCount() > 0:
            self.model.removeColumns(0, 1)
        self._df = pd.DataFrame()
        self.frozen_columns = []
        self._update_views()

    def scroll_to_column(self, var_name: str):
        """滚动到指定变量名的列，不影响垂直滚动位置。

        tab 模式：切到变量所在 tab 并水平滚动（旧版只查 _df，tab 模式下
        恒为静默 no-op，variable_list 双击定位失效）。
        """
        if self._tab_mode or self._group_tabs:
            for state in self._group_tabs.values():
                if var_name in state.df.columns:
                    if self._tab_widget:
                        self._tab_widget.setCurrentIndex(state.widget_index)
                    self._scroll_tab_to_column(state, var_name, blink=False)
                    return True
            return False
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
        """主窗口加载/重载新数据后，用新 loader 重建本对话框内容。

        重建依据是 clear_all_columns 留下的变量清单快照（旧版以 `_df` 为
        唯一真相，而 `_df` 在本方法前已被清掉 → 全路径提前 return，成为
        死代码，tab 模式状态残留界面）。

        - 新数据 MDF：逐变量先按名反查 group、再经 _add_variable_to_tab 重建
          tab（内部自动切 tab 模式；旧 group 索引在新文件里语义可能不同，
          不可沿用）
        - 新数据 CSV/Excel：单表逐列重建
        新数据中不存在的列计入 removed 并提示。

        Args:
            loader: 数据加载器实例
        """
        if loader is None:
            return
        old_flat = self._table_vars_snapshot or self.get_column_names()
        old_by_group = self._tab_vars_snapshot or {}
        self._table_vars_snapshot = []
        self._tab_vars_snapshot = {}

        if self._tab_mode or self._group_tabs:
            self._reset_tab_mode()

        is_mdf = getattr(loader, "LOADER_TYPE", "") == "mdf"
        removed: list[str] = []

        if is_mdf:
            # 单位表跟着新文件走：_switch_to_tab_mode 只在首次进 tab 模式时
            # 抓 units，此处不刷新的话重建出的列会沿用旧文件的单位
            self.units = getattr(loader, "units", {}) or {}
            # 一律按名反查 group：快照里的 gi 属于旧文件，新文件里同一序号
            # 可能指向另一个 channel group（换设备/换工步很常见），沿用会
            # 把变量对齐到错误时间轴。合并两个来源后按新 group 排序，
            # 保证 tab 创建顺序仍按 group 递增
            rebuild_vars = [*old_flat]
            seen = set(rebuild_vars)
            for cols in old_by_group.values():
                for col in cols:
                    if col not in seen:
                        seen.add(col)
                        rebuild_vars.append(col)
            keyed: list[tuple[int, str]] = []
            for col in rebuild_vars:
                try:
                    keyed.append((loader.get_var_group_index(col), col))
                except KeyError:
                    removed.append(col)
            for gi, col in sorted(keyed):
                if self._add_variable_to_tab(col, gi, loader=loader) is None:
                    removed.append(col)
        else:
            new_df = pd.DataFrame()
            for col in old_flat:
                if loader.df is not None and col in loader.df.columns:
                    new_df[col] = loader.df[col]
                else:
                    removed.append(col)
            self._df = new_df
            self.units = loader.units
            if not self._df.empty:
                self.model = PandasTableModel(self._df, self.units)
                self.main_view.setModel(self.model)
                self.frozen_view.setModel(self.model)
                self._connect_signals()
            self.frozen_columns = [
                c for c in self.frozen_columns if c in self._df.columns
            ]
            if self.model is not None:
                self._update_views()

        self._tab_loader_key = self._loader_identity(loader)
        self._refresh_var_locator_items()

        if removed:
            shown = ", ".join(removed[:10]) + (" …" if len(removed) > 10 else "")
            QMessageBox.information(self, "更新通知", f"以下变量已从数据中移除：{shown}")

        if not self.has_table_content():
            # 表已空：直接关闭，不弹确认框
            self.set_skip_close_confirmation(True)
            self.close()
