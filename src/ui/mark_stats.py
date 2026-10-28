"""标记统计窗口 —— QTableView + 筛选/排序/列客制化（方案 C 重构）。

数据流：mark_region_manager 逐曲线产出 MarkStatEntry → layout_manager
汇总所有页 × 可见子图 → 本窗口 update_stats() 填充扁平表格。

设计要点（见 tmp 设计文档 §5）：
- 指标注册表 STAT_COLUMNS 是唯一事实来源，增删指标只改注册表；
- 列 0-3 固定上下文列（页/子图/曲线/单位），其后为用户已选指标列；
- 双路刷新：行 key 集不变走 dataChanged（游标拖动高频路径），行集变化
  才 reset 并手动恢复 selection/滚动/sort（按 key 重映射）；
- 列配置与表头状态持久化到 AppSettings，写入即 sync。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QSignalBlocker,
    QItemSelectionModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
)
from shiboken6 import isValid

from src.core.settings import AppSettings

if TYPE_CHECKING:
    from src.core.data_types import MarkStatEntry

# 排序/过滤取值用的自定义 role（数值排序，NaN→-inf 钉底见 model.data）
SORT_ROLE = Qt.ItemDataRole.UserRole + 1


# ---------------------------------------------------------------------------
# 指标注册表（§5.2 单一事实来源）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatColumn:
    """一列统计指标的定义。

    key 是持久化用稳定标识，永不改名；getter 从 MarkStatEntry 取值，
    NaN/None 由格式化层统一显示为 "-"。
    """

    key: str
    title: str
    tooltip: str
    getter: Callable[["MarkStatEntry"], object]
    default_on: bool = False


_STAT_COLUMN_DEFS: list[tuple[str, str, str, Callable, bool]] = [
    # 标题格式：中文名 + ASCII 符号（符号沿用旧版表头，新增指标用紧凑缩写；
    # 全部保持 ASCII，避免非 ASCII 字符的字体渲染宽度不可预期）
    ("x1", "起点 x1", "区间左边界采样点的 x 值", lambda e: e.x1, False),
    ("x2", "终点 x2", "区间右边界采样点的 x 值", lambda e: e.x2, False),
    ("y1", "起点 y1", "区间左边界采样点的 y 值", lambda e: e.y1, False),
    ("y2", "终点 y2", "区间右边界采样点的 y 值", lambda e: e.y2, False),
    ("dx", "区间宽 dx", "x2 - x1", lambda e: e.dx, False),
    ("dy", "变化量 dy", "y2 - y1", lambda e: e.dy, True),
    ("slope", "斜率 slope", "dy / dx（点区间 dx=0 时为 inf）", lambda e: e.slope, True),
    ("y_avg", "均值 y_avg", "区间内有效点的 y 平均值", lambda e: e.y_avg, True),
    ("y_max", "最大值 y_max", "区间内有效点的 y 最大值", lambda e: e.y_max, True),
    ("y_min", "最小值 y_min", "区间内有效点的 y 最小值", lambda e: e.y_min, True),
    ("x_at_y_max", "最大值位置 x@y_max", "y 取最大值时的 x", lambda e: e.x_at_y_max, False),
    ("x_at_y_min", "最小值位置 x@y_min", "y 取最小值时的 x", lambda e: e.x_at_y_min, False),
    ("y_pp", "峰峰值 y_pp", "y_max - y_min", lambda e: e.y_pp, False),
    ("count", "样本数 n", "区间内有效样本点数", lambda e: e.count, False),
]

STAT_COLUMNS: dict[str, StatColumn] = {
    d[0]: StatColumn(*d) for d in _STAT_COLUMN_DEFS
}
DEFAULT_COLUMN_KEYS: list[str] = [k for k, c in STAT_COLUMNS.items() if c.default_on]


@dataclass(frozen=True)
class _ContextColumn:
    key: str
    title: str
    tooltip: str


# 列 0-3 固定上下文列，不参与显隐/顺序配置
CONTEXT_COLUMNS: list[_ContextColumn] = [
    _ContextColumn("page", "页", "曲线所在页"),
    _ContextColumn("plot", "子图", "页内子图序号（1-based）"),
    _ContextColumn("curve", "曲线", "变量名"),
    _ContextColumn("unit", "单位", "变量单位"),
]


def _format_stat_value(value) -> str:
    """自适应精度 4 位有效数字；NaN→"-"，±inf→"inf"/"-inf"，整数原样。"""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    v = float(value)
    if math.isnan(v):
        return "-"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    decimals = 4 - 1 - exp
    if decimals <= 0:
        return f"{v:.0f}"
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") or "0"


# ---------------------------------------------------------------------------
# 表格 Model / Proxy（§5.4 / §5.5）
# ---------------------------------------------------------------------------


@dataclass
class StatRow:
    """一行统计：页 + 子图序号（0-based）+ 该曲线的统计条目。"""

    page: str
    plot_idx: int
    entry: "MarkStatEntry"

    @property
    def key(self) -> tuple[str, int, str]:
        """行 key：双路刷新的选择依据。var_name 缺失时退回 label（防御旧数据）。"""
        return (self.page, self.plot_idx, self.entry.var_name or self.entry.label)


class MarkStatsTableModel(QAbstractTableModel):
    """列 = 固定上下文列(0-3) + 用户已选指标列；行 = StatRow 列表。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[StatRow] = []
        self._column_keys: list[str] = list(DEFAULT_COLUMN_KEYS)

    # ----- 列配置 -----

    def column_keys(self) -> list[str]:
        return list(self._column_keys)

    def set_column_keys(self, keys: list[str]) -> None:
        self.beginResetModel()
        self._column_keys = list(keys)
        self.endResetModel()

    # ----- 行数据 -----

    def rows(self) -> list[StatRow]:
        return self._rows

    def reset_rows(self, rows: list[StatRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def replace_entries(self, new_rows: list[StatRow]) -> bool:
        """行集不变时按 key 原位更新（返回 True），否则返回 False 走 reset。

        两遍式：先验证全部 key 可命中再落写，避免半途失败留下半新半旧的表。
        """
        if len(new_rows) != len(self._rows) or not self._rows:
            return False
        index_by_key: dict[tuple, int] = {}
        for i, r in enumerate(self._rows):
            index_by_key.setdefault(r.key, i)
        positions: list[int] = []
        for r in new_rows:
            i = index_by_key.get(r.key)
            if i is None:
                return False
            positions.append(i)
        for r, i in zip(new_rows, positions):
            self._rows[i].entry = r.entry
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
        if bottom_right.isValid():
            self.dataChanged.emit(top_left, bottom_right)
        return True

    # ----- QAbstractTableModel -----

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(CONTEXT_COLUMNS) + len(self._column_keys)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            col = None
            if 0 <= section < len(CONTEXT_COLUMNS):
                col = CONTEXT_COLUMNS[section]
            else:
                i = section - len(CONTEXT_COLUMNS)
                if 0 <= i < len(self._column_keys):
                    col = STAT_COLUMNS[self._column_keys[i]]
            if col is None:
                return None
            if role == Qt.ItemDataRole.DisplayRole:
                return col.title
            if role == Qt.ItemDataRole.ToolTipRole:
                return col.tooltip
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return section + 1
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if col == 0:  # 页
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                return row.page  # 页名过长时由视图截断，tooltip 全文
            if role == SORT_ROLE:
                return row.page
        elif col == 1:  # 子图
            if role == Qt.ItemDataRole.DisplayRole:
                return str(row.plot_idx + 1)
            if role == SORT_ROLE:
                return row.plot_idx
        elif col == 2:  # 曲线
            name = row.entry.var_name or row.entry.label
            if role in (Qt.ItemDataRole.DisplayRole, SORT_ROLE):
                return name
            if role == Qt.ItemDataRole.ToolTipRole:
                return row.entry.label  # 拼好单位的完整名
        elif col == 3:  # 单位
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole, SORT_ROLE):
                return row.entry.unit
        else:  # 指标列
            i = col - len(CONTEXT_COLUMNS)
            if not 0 <= i < len(self._column_keys):
                return None
            value = STAT_COLUMNS[self._column_keys[i]].getter(row.entry)
            if role == Qt.ItemDataRole.DisplayRole:
                return _format_stat_value(value)
            if role == SORT_ROLE:
                if isinstance(value, (int, np.integer)):
                    return int(value)
                v = float(value)
                # NaN 钉底 -inf（NaN 参与比较行为未定义，隐藏页曲线常态 NaN）；
                # ±inf 原样传递保证 inf 排在数值之后
                return float("-inf") if math.isnan(v) else v
        return None


class MarkStatsProxyModel(QSortFilterProxyModel):
    """曲线名子串筛选（大小写不敏感）+ 页筛选；排序按 SORT_ROLE。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        self.setFilterKeyColumn(2)  # 曲线列（上下文列固定在前，恒为 2）
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._page_filter: str | None = None

    def set_page_filter(self, page: str | None) -> None:
        if page != self._page_filter:
            self._page_filter = page
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if self._page_filter is not None:
            idx = self.sourceModel().index(source_row, 0, source_parent)
            if self.sourceModel().data(idx, Qt.ItemDataRole.DisplayRole) != self._page_filter:
                return False
        return super().filterAcceptsRow(source_row, source_parent)


# ---------------------------------------------------------------------------
# 列配置对话框（§5.6 低频入口）
# ---------------------------------------------------------------------------


class _ColumnConfigDialog(QDialog):
    """双列表：左「已选列」（↑↓调序/移除），右「可用列」（＋加入末尾）。

    至少保留 1 列指标：已选列表为空时禁用「确定」。
    """

    def __init__(self, parent=None, selected: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("自定义统计列")
        self.setMinimumWidth(460)
        self._selected = [k for k in (selected or []) if k in STAT_COLUMNS]

        left = QVBoxLayout()
        left.addWidget(QLabel("已选列（顺序即显示顺序）"))
        self.sel_list = QListWidget(self)
        self.sel_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        left.addWidget(self.sel_list)
        row = QHBoxLayout()
        self.up_btn = QPushButton("↑", self)
        self.down_btn = QPushButton("↓", self)
        self.remove_btn = QPushButton("移除", self)
        row.addWidget(self.up_btn)
        row.addWidget(self.down_btn)
        row.addWidget(self.remove_btn)
        left.addLayout(row)

        right = QVBoxLayout()
        right.addWidget(QLabel("可用列"))
        self.avail_list = QListWidget(self)
        self.avail_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        right.addWidget(self.avail_list)
        self.add_btn = QPushButton("＋ 加入末尾", self)
        right.addWidget(self.add_btn)

        lists = QHBoxLayout()
        lists.addLayout(left, 1)
        lists.addLayout(right, 1)

        self.restore_btn = QPushButton("恢复默认", self)
        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons = QHBoxLayout()
        buttons.addWidget(self.restore_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_box)

        root = QVBoxLayout(self)
        root.addLayout(lists)
        root.addLayout(buttons)

        self.up_btn.clicked.connect(self._move_up)
        self.down_btn.clicked.connect(self._move_down)
        self.remove_btn.clicked.connect(self._remove)
        self.add_btn.clicked.connect(self._add)
        self.restore_btn.clicked.connect(self._restore_default)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)

        self._refresh()

    def selected_keys(self) -> list[str]:
        return list(self._selected)

    def _refresh(self):
        self.sel_list.clear()
        for k in self._selected:
            self.sel_list.addItem(STAT_COLUMNS[k].title)
        self.avail_list.clear()
        for k, c in STAT_COLUMNS.items():
            if k not in self._selected:
                self.avail_list.addItem(c.title)
        ok_btn = self.btn_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setEnabled(bool(self._selected))  # 至少保留 1 列

    def _move_up(self):
        i = self.sel_list.currentRow()
        if i > 0:
            self._selected[i - 1], self._selected[i] = self._selected[i], self._selected[i - 1]
            self._refresh()
            self.sel_list.setCurrentRow(i - 1)

    def _move_down(self):
        i = self.sel_list.currentRow()
        if 0 <= i < len(self._selected) - 1:
            self._selected[i + 1], self._selected[i] = self._selected[i], self._selected[i + 1]
            self._refresh()
            self.sel_list.setCurrentRow(i + 1)

    def _remove(self):
        i = self.sel_list.currentRow()
        if 0 <= i < len(self._selected):
            self._selected.pop(i)
            self._refresh()
            if self._selected:
                self.sel_list.setCurrentRow(min(i, len(self._selected) - 1))

    def _add(self):
        i = self.avail_list.currentRow()
        if i >= 0:
            avail = [k for k in STAT_COLUMNS if k not in self._selected]
            self._selected.append(avail[i])
            self._refresh()
            self.sel_list.setCurrentRow(len(self._selected) - 1)

    def _restore_default(self):
        self._selected = list(DEFAULT_COLUMN_KEYS)
        self._refresh()


# ---------------------------------------------------------------------------
# 统计窗口
# ---------------------------------------------------------------------------


class MarkStatsWindow(QDialog):
    """
    标记统计窗口：扁平表格 + 页/曲线筛选 + 指标排序 + 列客制化。
    使用单例模式确保只有一个统计窗口实例。
    """

    _instance = None  # Singleton instance

    # 激活时不透明
    ACTIVE_OPACITY = 1.0
    # 非激活时 90% 透明
    INACTIVE_OPACITY = 0.9

    @classmethod
    def get_instance(cls, parent=None):
        # 主窗口销毁后子窗口的 C++ 对象已不在，类级单例不能继续复用
        if cls._instance is not None and not isValid(cls._instance):
            cls._instance = None
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("标记区域统计")

        # 窗口标志：Tool窗口 + 仅标题栏和关闭按钮
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        # 汇总信息（update_stats 刷新）
        self._pages: list[str] = []
        self._plot_count = 0
        self._empty_plots = 0
        self._active_page: str | None = None

        settings = AppSettings()
        self._column_keys = settings.get_mark_stats_columns(
            DEFAULT_COLUMN_KEYS, STAT_COLUMNS.keys()
        )

        self._build_ui()
        self._load_header_state()

        parent = self.parent()
        if parent is not None and getattr(parent, "mark_stats_geometry", None):
            self.restoreGeometry(parent.mark_stats_geometry)
        else:
            self.resize(1200, 300)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        # 紧凑布局：压减四周留白与行间距，把高度让给中间表格（可见行数最大化）
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        # 工具栏：页筛选 + 曲线名搜索 + 仅当前页开关
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.page_combo = QComboBox(self)
        self.page_combo.setMinimumWidth(140)
        self.page_combo.setFixedHeight(24)  # 压缩控件高度（默认约 28px）
        self.page_combo.currentIndexChanged.connect(self._on_page_combo_changed)
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("搜索曲线名…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setFixedHeight(24)
        self.search_input.textChanged.connect(self._on_search_changed)
        self.current_page_only = QCheckBox("仅当前页", self)
        self.current_page_only.setChecked(True)
        self.current_page_only.toggled.connect(self._on_current_page_only_toggled)
        bar.addWidget(self.page_combo)
        bar.addWidget(self.search_input, 1)
        bar.addWidget(self.current_page_only)
        layout.addLayout(bar)

        # 表格
        self.model = MarkStatsTableModel(self)
        self.model.set_column_keys(self._column_keys)
        self.proxy = MarkStatsProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setDynamicSortFilter(True)  # dataChanged 后自动重算排序
        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)
        # 默认列宽（header restoreState 失败时的回退）
        header.resizeSection(0, 110)
        header.resizeSection(1, 50)
        header.resizeSection(2, 170)
        header.resizeSection(3, 70)
        layout.addWidget(self.table, 1)

        # 状态栏：汇总计数 + 标记区间 + 导出（单行紧凑条）
        status = QHBoxLayout()
        status.setSpacing(4)
        self.status_label = QLabel("", self)  # 字号与上方控件一致，不缩放
        self.copy_csv_btn = QPushButton("复制 CSV", self)
        self.copy_csv_btn.clicked.connect(self._copy_csv)  # 样式用平台默认（美观优先）
        # 「复制 Markdown」按钮已按需求从界面移除；_copy_markdown 导出
        # 实现保留在下方，供未来重新接入或测试直接调用
        status.addWidget(self.status_label, 1)
        status.addWidget(self.copy_csv_btn)
        layout.addLayout(status)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange:
            self.setWindowOpacity(
                self.ACTIVE_OPACITY if self.isActiveWindow() else self.INACTIVE_OPACITY
            )
        super().changeEvent(event)

    # ------------------------------------------------------------------
    # 数据入口（§5.8 双路刷新）
    # ------------------------------------------------------------------

    def update_stats(self, stats_list, *, active_page: str | None = None):
        """更新统计信息显示。

        Args:
            stats_list: ``[(page_name, idx_in_page, stats), ...]``。
                stats 为该子图的曲线统计列表；``None``/空表示该子图没有可统计
                的曲线（只计数入状态栏，不再生成 "No Curve" 行）。仍兼容旧式
                ``[stats, ...]``（视为单页「图 1」）。
            active_page: 当前活动页名，「仅当前页」跟随目标；None 表示未知
                （不过滤）。
        """
        normalized: list[tuple[str, int, object]] = []
        for position, entry in enumerate(stats_list):
            if isinstance(entry, tuple) and len(entry) == 3:
                normalized.append(entry)
            else:
                normalized.append(("图 1", position, entry))

        rows: list[StatRow] = []
        empty_plots = 0
        pages: list[str] = []
        page_seen: set[str] = set()
        for page_name, idx, stats in normalized:
            if page_name not in page_seen:
                page_seen.add(page_name)
                pages.append(page_name)
            if not stats:
                empty_plots += 1
                continue
            for stat in stats:
                rows.append(StatRow(page=page_name, plot_idx=idx, entry=stat))

        self._active_page = active_page
        self._pages = pages
        self._plot_count = len(normalized)
        self._empty_plots = empty_plots
        self._rebuild_page_controls()

        # 行 key 集不变 → 原位更新（游标拖动高频路径）；否则 reset + 状态恢复
        if not self.model.replace_entries(rows):
            self._reset_model_with_state_restore(rows)

        self._update_status_bar()

    def clear_stats(self):
        """清空统计内容（数据重载/清除绘图时由外部调用，替代旧 tree.clear）。"""
        self.model.reset_rows([])
        self._pages = []
        self._plot_count = 0
        self._empty_plots = 0
        self._active_page = None
        self._rebuild_page_controls()
        self._update_status_bar()

    def _reset_model_with_state_restore(self, rows: list[StatRow]) -> None:
        """reset 路径：手动保存并恢复 selection/滚动/sort（§5.8）。

        Qt 不保证跨 reset 保持 selection；sortColumn/sortOrder 若不按 key
        重映射，列数随配置变化时会漂移到错误列。
        """
        # -- 保存 --
        sel_keys: set[tuple] = set()
        sel_model = self.table.selectionModel()
        if sel_model is not None:
            for proxy_idx in sel_model.selectedRows(0):
                src_row = self.proxy.mapToSource(proxy_idx).row()
                if 0 <= src_row < len(self.model.rows()):
                    sel_keys.add(self.model.rows()[src_row].key)
        scroll_bar = self.table.verticalScrollBar()
        scroll_pos = scroll_bar.value()
        sort_key: str | None = None
        sort_order = Qt.SortOrder.AscendingOrder
        sort_col = self.proxy.sortColumn()
        if sort_col >= 0:
            sort_key = self._column_key_at(sort_col)
            sort_order = self.proxy.sortOrder()

        # -- reset --
        self.model.reset_rows(rows)

        # -- 恢复 --
        if sort_key is not None:
            new_col = self._column_index_for_key(sort_key)
            if new_col is not None:
                self.proxy.sort(new_col, sort_order)
            else:
                self.proxy.sort(-1, Qt.SortOrder.AscendingOrder)  # 列已删除，清排序
        if sel_keys and sel_model is not None:
            command = (
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows
            )
            for i, r in enumerate(self.model.rows()):
                if r.key not in sel_keys:
                    continue
                proxy_idx = self.proxy.mapFromSource(self.model.index(i, 0))
                if proxy_idx.isValid():  # 被筛掉的行不可选，静默跳过
                    sel_model.select(proxy_idx, command)
        scroll_bar.setValue(min(scroll_pos, scroll_bar.maximum()))

    # ------------------------------------------------------------------
    # 列号 ↔ key 映射（proxy 不映射列，两侧列号一致）
    # ------------------------------------------------------------------

    def _column_key_at(self, column: int) -> str | None:
        if 0 <= column < len(CONTEXT_COLUMNS):
            return CONTEXT_COLUMNS[column].key
        keys = self.model.column_keys()
        i = column - len(CONTEXT_COLUMNS)
        return keys[i] if 0 <= i < len(keys) else None

    def _column_index_for_key(self, key: str) -> int | None:
        for i, col in enumerate(CONTEXT_COLUMNS):
            if col.key == key:
                return i
        keys = self.model.column_keys()
        return len(CONTEXT_COLUMNS) + keys.index(key) if key in keys else None

    # ------------------------------------------------------------------
    # 筛选控件（§5.5）
    # ------------------------------------------------------------------

    def _rebuild_page_controls(self):
        multi = len(self._pages) > 1
        self.page_combo.setVisible(multi)
        self.current_page_only.setVisible(multi)
        self.page_combo.setEnabled(not self.current_page_only.isChecked())

        with QSignalBlocker(self.page_combo):
            current = self.page_combo.currentData()
            self.page_combo.clear()
            self.page_combo.addItem("全部页", "")  # "" 哨兵：页名非空
            for p in self._pages:
                self.page_combo.addItem(p, p)
            if current:
                idx = self.page_combo.findData(current)
                self.page_combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.page_combo.setCurrentIndex(0)
        self._apply_page_filter()

    def _apply_page_filter(self):
        if len(self._pages) <= 1:  # 单页退化：不做页过滤
            self.proxy.set_page_filter(None)
            return
        if self.current_page_only.isChecked():
            self.proxy.set_page_filter(self._active_page)
        else:
            data = self.page_combo.currentData()
            self.proxy.set_page_filter(data or None)

    def _on_page_combo_changed(self, _index):
        self._apply_page_filter()
        self._update_status_bar()

    def _on_current_page_only_toggled(self, _checked):
        self.page_combo.setEnabled(not self.current_page_only.isChecked())
        self._apply_page_filter()
        self._update_status_bar()

    def _on_search_changed(self, text):
        self.proxy.setFilterFixedString(text)
        self._update_status_bar()

    # ------------------------------------------------------------------
    # 列配置（§5.6 两级入口）
    # ------------------------------------------------------------------

    def _show_header_menu(self, pos):
        menu = QMenu(self)
        current = self.model.column_keys()
        for key, col in STAT_COLUMNS.items():
            act = menu.addAction(col.title)
            act.setCheckable(True)
            act.setChecked(key in current)
            # 携带载荷：chosen.data() 依赖它识别勾的是哪个指标
            # （漏 setData 会让 data() 返回 None，勾选静默失效）
            act.setData(key)
            # 至少保留 1 列指标：最后一列勾选项禁用取消
            act.setEnabled(not (key in current and len(current) == 1))
        menu.addSeparator()
        order_act = menu.addAction("自定义列顺序…")
        chosen = menu.exec(self.table.horizontalHeader().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is order_act:
            self._open_column_dialog()
            return
        # 防御：非字符串载荷（含 None）直接忽略，不落入 "None" 字面量
        data = chosen.data()
        if not isinstance(data, str):
            return
        key = data
        if key in current:
            new_keys = [k for k in current if k != key]
        else:
            new_keys = current + [key]  # 勾选新指标追加到末尾
        self._apply_column_keys(new_keys)

    def _open_column_dialog(self):
        dlg = _ColumnConfigDialog(self, selected=self.model.column_keys())
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._apply_column_keys(dlg.selected_keys())
        finally:
            dlg.deleteLater()

    def _apply_column_keys(self, keys: list[str]) -> None:
        """应用新列集：持久化 + 列 reset + sortColumn 按 key 重映射。"""
        keys = [k for k in keys if k in STAT_COLUMNS]
        if not keys or keys == self.model.column_keys():
            return
        # 排序先按 key 记忆（列 reset 前取值）
        sort_key: str | None = None
        sort_order = Qt.SortOrder.AscendingOrder
        sort_col = self.proxy.sortColumn()
        if sort_col >= 0:
            sort_key = self._column_key_at(sort_col)
            sort_order = self.proxy.sortOrder()

        self._column_keys = list(keys)
        self.model.set_column_keys(self._column_keys)
        AppSettings().set_mark_stats_columns(self._column_keys)  # 即写即 sync

        if sort_key is not None:
            new_col = self._column_index_for_key(sort_key)
            if new_col is not None:
                self.proxy.sort(new_col, sort_order)
            else:
                self.proxy.sort(-1, Qt.SortOrder.AscendingOrder)
        self._save_header_state()

    # ------------------------------------------------------------------
    # 状态栏与导出（§5.7）
    # ------------------------------------------------------------------

    def _mark_range(self) -> tuple[float, float] | None:
        """区间来源：主窗口 saved_mark_range（各曲线 x 网格不同，不能取 entry.x1）。"""
        parent = self.parent()
        rng = getattr(parent, "saved_mark_range", None) if parent is not None else None
        if not rng:
            return None
        try:
            return (float(rng[0]), float(rng[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def _update_status_bar(self):
        parts = [
            f"{len(self._pages)} 页 · {self._plot_count} 子图 · "
            f"{self.proxy.rowCount()} 条曲线"
        ]
        rng = self._mark_range()
        if rng is not None:
            # 不硬编码单位后缀（x 未必是秒），按 x 轴原样显示数字
            parts.append(
                f"标记区间 [{_format_stat_value(rng[0])}, {_format_stat_value(rng[1])}]"
            )
        if self._empty_plots:
            parts.append(f"{self._empty_plots} 个空子图未统计")
        self.status_label.setText(" · ".join(parts))

    def _visible_columns(self) -> list[int]:
        """导出用列：可见列，按表头视觉顺序（所见即所得）。"""
        header = self.table.horizontalHeader()
        cols = [
            c for c in range(self.model.columnCount()) if not header.isSectionHidden(c)
        ]
        return sorted(cols, key=lambda c: header.visualIndex(c))

    def _export_text(self, as_csv: bool) -> str:
        cols = self._visible_columns()
        headers = [
            str(
                self.model.headerData(
                    c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
                )
            )
            for c in cols
        ]
        lines: list[str] = []
        if as_csv:
            lines.append("\t".join(headers))
        else:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join([" --- "] * len(headers)) + "|")
        for prow in range(self.proxy.rowCount()):
            cells = []
            for c in cols:
                text = self.proxy.data(self.proxy.index(prow, c), Qt.ItemDataRole.DisplayRole)
                cells.append("" if text is None else str(text))
            if as_csv:
                lines.append("\t".join(cells))
            else:
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _copy_csv(self):
        QApplication.clipboard().setText(self._export_text(as_csv=True))

    # 注：「复制 Markdown」按钮已按需求从界面移除（见 _build_ui 状态栏），
    # 本方法与 _export_text(as_csv=False) 的导出实现一并保留，
    # 供未来重新接入 UI 或测试直接调用，勿删。
    def _copy_markdown(self):
        QApplication.clipboard().setText(self._export_text(as_csv=False))

    # ------------------------------------------------------------------
    # 表头状态持久化（§5.3）
    # ------------------------------------------------------------------

    def _load_header_state(self):
        state = AppSettings().get_mark_stats_header_state()
        if state is None:
            return
        try:
            self.table.horizontalHeader().restoreState(state)  # 失败静默回退默认列宽
        except Exception:
            pass

    def _save_header_state(self):
        try:
            AppSettings().set_mark_stats_header_state(
                self.table.horizontalHeader().saveState()
            )
        except Exception:
            pass

    def hideEvent(self, event):
        # toggle_mark_region(False) 走 hide() 不触发 closeEvent，这里兜底保存
        self._save_header_state()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    # 几何与关闭（保持旧接口不变）
    # ------------------------------------------------------------------

    def save_geom(self):
        parent = self.parent()
        if parent is None:
            return
        if self.isMinimized() or self.isMaximized():
            parent.mark_stats_geometry = None  # 不保存，强制下次使用默认
        else:
            parent.mark_stats_geometry = self.saveGeometry()

    def load_geom(self):
        parent = self.parent()
        if parent is not None and parent.mark_stats_geometry is not None:
            geom = parent.mark_stats_geometry
            self.restoreGeometry(geom)
        else:
            # 默认大小和位置：resize并居中
            self.resize(1200, 300)
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                x = (avail.width() - self.width()) // 2
                y = (avail.height() - self.height()) // 2
                self.move(x, y)

        # 既有逻辑保留：防御性重置状态，确保不是 min/max
        if self.isMinimized() or self.isMaximized():
            self.setWindowState(Qt.WindowState.WindowNoState)  # 强制正常状态

    def closeEvent(self, event):
        self._save_header_state()
        self.save_geom()
        parent = self.parent()
        if hasattr(parent, "layout_manager"):
            parent.layout_manager.toggle_mark_region(False)
        super().closeEvent(event)
