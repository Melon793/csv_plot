"""绘图变量编辑器"""

from __future__ import annotations
import math
import re
import sys
from PySide6.QtCore import (
    QEvent,
    QEventLoop,
    QItemSelectionModel,
    QRegularExpression,
    Qt,
    QSignalBlocker,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QKeySequence,
    QRegularExpressionValidator,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMenu,
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
from src.core.config import DEFAULT_CURVE_COLOR, DEFAULT_CURVE_COLORS, widget_alive
from src.core.curve_transform import NON_NUMERIC_FORMATS
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_POST,
    DRAW_MODE_STEP_PRE,
    LINE_PRESET_AUTO,
    LINE_PRESET_BOLD,
    LINE_PRESET_DASH,
    LINE_PRESET_DOT,
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_OFF,
    SYMBOL_POLICY_ON,
)
from src.core.logger import get_logger
from src.ui import theme
from src.ui.drag_drop import parse_var_names_from_mimedata
from src.ui.table_selection import select_table_row
from src.ui.tooltip_text import var_tooltip
from src.ui.widgets.curve_style_panel import exec_curve_panel
from src.ui.widgets.variable_search_bar import VariableSearchBar

logger = get_logger(__name__)

# 绘制方式 / 数据点下拉选项（itemData 存 data_types 常量，变更时取
# currentData() 直接透传给 MultiCurveManager）
_DRAW_MODE_OPTIONS = [
    ("直线", DRAW_MODE_AUTO),
    ("阶梯(后)", DRAW_MODE_STEP_POST),
    ("阶梯(前)", DRAW_MODE_STEP_PRE),
]
_SYMBOL_POLICY_OPTIONS = [
    ("自动", SYMBOL_POLICY_AUTO),
    ("始终显示", SYMBOL_POLICY_ON),
    ("始终隐藏", SYMBOL_POLICY_OFF),
]

# C4（D9=A2）：6 列布局。颜色/绘制方式/数据点/样式收进「样式」摘要列，
# 点击弹 legend 同款快捷面板；新增 Legend 名 / Factor / Offset 行内编辑。
_COL_COUNT = 6
_COL_LABELS = ["显示", "变量名", "Legend 名", "样式", "Factor", "Offset"]
_COL_DEFAULT_WIDTHS = [46, 210, 170, 168, 84, 84]
_ELASTIC_COLS = (1, 2)  # 双弹性列：吸收窗口余量（D10）
_FIXED_MIN_WIDTHS = {0: 44, 3: 96, 4: 70, 5: 70}
# 弹性列最小宽独立常量（评审 P1-7）：_FIXED_MIN_WIDTHS 兼作固定列集合，
# 往里加 1/2 会被 _adjust/_restore 当成固定列迭代，故单独维护。
_ELASTIC_MIN = {1: 140, 2: 120}
# Factor/Offset 换算列：默认宽 84，不按极值文本 sizeHint 自适应（D13）。
# D16 反转 v1 P2-2 的"禁拖"：换算列可拖，拖过即入固定列并持久化，
# 未拖过的列才压回 84（用户偏好可调优先）。
_TRANSFORM_COLS = (4, 5)
_TRANSFORM_COL_WIDTH = 84

# 行高（D20）：单元格里 QLineEdit 的高度由表格决定，不由它自己的 sizeHint 决定 ——
# 实测 框高 = 行高 − 1 − 2×「::item padding 6px」，QSS 的 padding 只吃内容区、
# 不会把框撑高。默认 30px 行高把框压到 17px（字体行高 15），文字上下各只剩 1px；
# 38px 给出 25px 框高（描边内 23px），文字上下各约 4px 且框本身不再低于自身的
# 23px 自然高度。代价：600px 高的窗口可见整行从 11 行降到 9 行（viewport 345px）。
_ROW_HEIGHT = 38

# 行号槽（D1）：空表也占位，避免首行出现时 viewport 抖动 + 尘埃横滚条。
# 最少 2 位（1 行时不至于只给 6px 窘槽），行数跨到 3/4 位数时扩档。
_ORDINAL_MIN_DIGITS = 2
_ORDINAL_GUTTER_PAD = 8

# 列宽存盘 debounce：主窗销毁不走 closeEvent，拖宽后延迟落盘兜底（评审 P1-6）
_COLUMN_SAVE_DELAY_MS = 400

# 样式摘要短标签（只列非默认项，全默认显「默认」）
_PRESET_SHORT = {
    LINE_PRESET_AUTO: "", LINE_PRESET_BOLD: "粗实",
    LINE_PRESET_DASH: "虚线", LINE_PRESET_DOT: "点线",
}
_DRAW_SHORT = {
    DRAW_MODE_AUTO: "", DRAW_MODE_STEP_POST: "阶梯后", DRAW_MODE_STEP_PRE: "阶梯前",
}
_SYMBOL_SHORT = {
    SYMBOL_POLICY_AUTO: "", SYMBOL_POLICY_ON: "显示", SYMBOL_POLICY_OFF: "隐藏",
}


def _style_summary(ci) -> str:
    """样式摘要列文字：非默认项 join，全默认显「默认」（D9 演示口径）。"""
    parts = [
        _PRESET_SHORT.get(getattr(ci, "line_preset", LINE_PRESET_AUTO), ""),
        _DRAW_SHORT.get(getattr(ci, "draw_mode", DRAW_MODE_AUTO), ""),
        _SYMBOL_SHORT.get(getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO), ""),
    ]
    parts = [p for p in parts if p]
    return " · ".join(parts) if parts else "默认"


def _name_badges(ci) -> str:
    """变量名列徽标后缀（D7）：自定义名=名，非默认换算=ƒ。"""
    s = ""
    if getattr(ci, "display_name", ""):
        s += " ·名"
    if getattr(ci, "factor", 1.0) != 1.0 or getattr(ci, "offset", 0.0) != 0.0:
        s += " ·ƒ"
    return s


def _transform_disabled(ci) -> bool:
    """枚举/日期/纯时间通道禁用数值变换（D4）。"""
    return getattr(ci, "y_format", "") in NON_NUMERIC_FORMATS


# Factor/Offset 文本输入（D17）：严格十进制，禁逗号（locale 千分/小数逗号）
# 与科学计数法；空白由 commit 前 strip 处理。
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
_TRANSFORM_MAX_ABS = 1e12


def format_number(value) -> str:
    """换算数值的输入框显示文本：去尾零，-0→0，非有限/异常兜底 "0"。

    已知边界（评审 P2-11）：超过 12 位小数显示损失；|v|<1e-13 显示 "0"。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "0"
    if not math.isfinite(v) or v == 0:
        return "0"
    return f"{v:.12f}".rstrip("0").rstrip(".")


class _NumericLineEdit(QLineEdit):
    """Factor/Offset 行内数值输入（D17）：只打字，无步进/滚轮。

    失焦与 Enter 主动 commit —— Invalid/空文本下 editingFinished 不触发
    （评审 P0-1 实测）；严格正则拒绝逗号，避免 QDoubleValidator 的 locale
    fixup 把 "1,5" 变成 15（评审 P0-2）。
    """

    value_committed = Signal(float)

    def __init__(self, value: float, parent=None):
        super().__init__(parent)
        self._last_valid = float(value)
        self.setValidator(
            QRegularExpressionValidator(QRegularExpression(_NUMBER_RE.pattern), self)
        )
        self._text_snapshot = format_number(value)
        self.setText(self._text_snapshot)

    def set_value(self, value: float):
        """程序化回填；用户正在输入（有焦点）时不覆盖（评审 P1-8）。"""
        self._last_valid = float(value)
        if self.hasFocus():
            return
        self._show(self._last_valid)

    def _show(self, value: float):
        self._text_snapshot = format_number(value)
        with QSignalBlocker(self):
            self.setText(self._text_snapshot)

    def _parse(self) -> float | None:
        text = self.text().strip()
        if text == self._text_snapshot:
            # 文本未被用户改动：用全精度原值，避免经 12 位显示串往返截断
            # （如 1e-13 显示为 "0"，提交另一框不得把它归零）
            return self._last_valid
        if not _NUMBER_RE.match(text):
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return max(-_TRANSFORM_MAX_ABS, min(_TRANSFORM_MAX_ABS, parsed))

    def value(self) -> float:
        parsed = self._parse()
        return self._last_valid if parsed is None else parsed

    def commit(self):
        """提交当前文本：非法/空回退上次合法值；合法值夹 ±1e12 后发信号。"""
        parsed = self._parse()
        if parsed is None:
            self._show(self._last_valid)
            return
        self._last_valid = parsed
        self._show(parsed)
        self.value_committed.emit(parsed)

    def focusOutEvent(self, event):
        self.commit()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        # Enter 必须在此消费：否则冒泡到对话框默认按钮「确定」直接关窗
        # （评审 P2-10）
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.commit()
            event.accept()
            return
        if event.text() == ",":
            # 欧式小数逗号（含数字键盘）：本地译为小数点，防 "1,5" 被过滤
            # 逗号后拼接成 15（评审 P2-4 键盘路径实测）
            self.insert(".")
            return
        super().keyPressEvent(event)


class _StyleCell(QWidget):
    """样式摘要格：色块 + 摘要文字 + ▾，整格点击弹快捷面板（D9）。

    只负责视觉与命中；应用集中在面板 exec 返回后（共享组件红线）。
    R4（D19）：不定高，内容 AlignVCenter；Qt 按 sizeHint 定尺寸并落在
    单元格内容区，固定高度会溢出把内容顶偏（旧版 setFixedHeight(28) 在
    30px 行高下偏低 5px）。
    """

    clicked = Signal(str)

    def __init__(self, var_name: str, parent=None):
        super().__init__(parent)
        self.var_name = var_name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(6)
        vcenter = Qt.AlignmentFlag.AlignVCenter
        self.swatch = QLabel(self)
        self.swatch.setFixedSize(24, 15)
        self.text = QLabel(self)
        self.text.setStyleSheet("color: #444;")
        self.text.setAlignment(Qt.AlignmentFlag.AlignLeft | vcenter)
        carat = QLabel("▾", self)
        carat.setStyleSheet("color: #8a8f94;")
        carat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.swatch, 0, vcenter)
        lay.addWidget(self.text, 1, vcenter)
        lay.addWidget(carat, 0, vcenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pressed = False

    def set_state(self, color: str, summary: str, enabled: bool = True):
        self.swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid #333;"
        )
        self.text.setText(summary)
        self.setEnabled(enabled)
        self.setToolTip(
            "点击修改颜色/绘制方式/数据点/样式" if enabled
            else "变量缺失，恢复后可改样式"
        )

    def mousePressEvent(self, event):
        # 评审 P2-4：press 记录 + release 落点双重判定，防框选/拖拽释放在格上误弹
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._pressed and self.rect().contains(event.position().toPoint())
            self._pressed = False
            if hit and self.isEnabled():
                self.clicked.emit(self.var_name)
                event.accept()
                return
            event.ignore()
            return
        super().mouseReleaseEvent(event)


def _mode_combo(options, current, on_change):
    """构建选项下拉框：itemData 绑常量值，初始值以 QSignalBlocker 静默设置。

    on_change(value, label) 收到常量值与中文选项文案，后者用于状态栏
    回执（与图例菜单路径的中文播报口径一致）。
    """
    combo = QComboBox()
    for text, value in options:
        combo.addItem(text, value)
    with QSignalBlocker(combo):
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.currentIndexChanged.connect(
        lambda _idx, cb=combo: on_change(cb.currentData(), cb.currentText())
    )
    return combo


def natural_sort_key(name: str) -> list:
    """自然排序键：字母段按小写比较，数字段按数值比较

    保证 ``AI05_2 < AI05_10``（纯字典序会把 "10" 排在 "2" 前面）。
    段统一编码为 ``(类型标记, 值)`` 元组：仅同类型段内比值，避免 str/int 混比报错。
    """
    key: list = []
    for part in re.split(r"(\d+)", name):
        if part.isdigit():
            key.append((1, int(part)))
        elif part:
            key.append((0, part.lower()))
    return key


def _visible_grid_dim(page, attr: str, fallback: int) -> int:
    """读页面的可见行列数（rows_current / cols_current），不可用时回退。

    替身页、独立 widget 或字段被写坏（非数值/≤0）时返回 ``fallback``
    （调用方传已建网格的行列数），保证身份行不因这类页面构造失败。
    """
    try:
        value = int(getattr(page, attr, 0) or 0)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


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
        self.setAcceptDrops(True)

        # closeEvent 与 QDialog 默认关闭链触发的 done() 可能先后收口；
        # 两个入口都调用 _teardown，用一次性守卫保证信号只断一次。
        self._teardown_done = False

        # 最近一次排序前的行序快照（单层，供"撤销上次排序"回退）
        self._last_sort_order: list[str] | None = None

        # 固定列（显示/样式/Factor/Offset）是否已完成首次内容自适应：
        # 仅首帧数据填充后执行一次，后续 reload 不重算（尊重用户手动拖拽）
        self._columns_sized = False

        # C4 列宽引擎（D1+D10）：双弹性列 + 定宽列 + 最小宽 + 横滚兜底 +
        # 全局记忆。_width_guard 防程序化 setColumnWidth 与 sectionResized
        # 回写互相触发（照 variable_info_dialog 防自激口径）
        self._width_guard = False
        self._user_fixed_cols: set[int] = set()
        self._col_min: dict[int, int] = {}
        # viewport 重分布的合并闸：一次布局变化只调度一次 singleShot
        self._redistribute_pending = False
        # 最近一次分布时的 viewport 宽度：只有"宽度"变了才重分布。
        # 横滚显隐只改高度（同样发 Resize），不能让拖宽固定列反过来挤压
        # 弹性列（D10 的横滚兜底语义），故按宽度去重。
        self._last_viewport_width = -1

        # 拖宽后 debounce 存盘（两参 singleShot 口径的 QTimer 实例版；
        # 随 self 销毁自动断开，避免主窗销毁时漏存）
        self._width_save_timer = QTimer(self)
        self._width_save_timer.setSingleShot(True)
        self._width_save_timer.setInterval(_COLUMN_SAVE_DELAY_MS)
        self._width_save_timer.timeout.connect(self._save_column_widths)

        # 高DPI支持 - PySide6中不需要WA_UseHighDpiPixmaps
        # PySide6默认支持高DPI，通过样式表控制字体大小

        self.setup_ui()
        self._refresh_identity()
        self.load_current_curves()
        # 几何恢复放在数据填充后：初始宽按实际列宽计算
        self._restore_geometry()

    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout()

        # 顶部身份行：窗口标题已给出名称，这里只回答「我是哪个页、哪个格子」。
        # 12px 次要文字；解析不到时整行隐藏（不留占位、不写"未知绘图区"）。
        self.identity_label = QLabel()
        self.identity_label.setStyleSheet(theme.muted_text())
        self.identity_label.setVisible(False)
        layout.addWidget(self.identity_label)

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

        # 创建表格（C4：6 列，D9=A2 合并样式列）
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(_COL_COUNT)
        self.var_table.setHorizontalHeaderLabels(_COL_LABELS)

        # 设置表格属性
        self.var_table.setDragDropMode(QTableWidget.DragDropMode.DropOnly)
        self.var_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # 扩展选择：Cmd/Ctrl+点击、Shift 范围多选
        # （"仅排序选中行"与"删除选中"的多行场景依赖此模式）
        self.var_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.var_table.setAlternatingRowColors(True)
        # 行高（D20）：行内输入框的高度跟着行高走，见 _ROW_HEIGHT 注释
        self.var_table.verticalHeader().setDefaultSectionSize(_ROW_HEIGHT)
        self.var_table.setStyleSheet(f"""
            QTableWidget {{
                gridline-color: #d0d0d0;
                background-color: white;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 6px;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {theme.TINT_SOFT};
                color: #000000;
            }}
            QCheckBox {{
                font-size: 12px;
            }}
            /* 行内输入框（Legend 名 / Factor / Offset）：裸 QLineEdit 由平台样式
               绘制，Win11 上是"上/左右几乎全白 + 底边一条深线"，白底表格里只看
               得见那一条线；这里统一成与 theme.path_field_style() 同款的描边框，
               跨平台一致。纵向留白交给行高（_ROW_HEIGHT），padding 只调左右：
               Factor/Offset 列窄，4px 会再吃掉 4px 文本区（D13 的窄列取舍）。 */
            QLineEdit {{
                background-color: {theme.BG};
                border: 1px solid #ccc;
                border-radius: {theme.R_FIELD}px;
                padding: 0 2px;
            }}
            QLineEdit:focus {{
                border: 1px solid {theme.ACCENT};
            }}
            QLineEdit:disabled {{
                color: {theme.TEXT_MUTED};
                background-color: #fafafa;
                border-color: #e0e0e0;
            }}
            {theme.item_view_outline_qss("QTableWidget")}
        """)

        # 设置列宽（C4/D10：全列 Interactive 可拖；弹性分配在 resizeEvent）
        header = self.var_table.horizontalHeader()
        for col in range(_COL_COUNT):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setSectionsClickable(True)
        # 每列最小宽 = 表头文字宽 + 余量（跨平台不拍像素）
        fm = header.fontMetrics()
        for col, label in enumerate(_COL_LABELS):
            self._col_min[col] = fm.horizontalAdvance(label) + 14
        for col, minimum in _FIXED_MIN_WIDTHS.items():
            self._col_min[col] = max(self._col_min[col], minimum)
        for col, minimum in _ELASTIC_MIN.items():
            self._col_min[col] = max(self._col_min[col], minimum)
        self._init_column_widths()
        # 行号槽常驻（D1）：空/满两态列位一致；行数变化时按位数重新钉宽。
        # 单一钩子覆盖全部行数变动点（load_current_curves 的清空、两处
        # insertRow、两处 removeRow、clear_all 的清空），无需逐处插桩。
        self._sync_ordinal_gutter()
        self.var_table.model().rowsInserted.connect(self._on_table_rows_changed)
        self.var_table.model().rowsRemoved.connect(self._on_table_rows_changed)
        # viewport 变窄（行号槽扩档 / 竖滚出现）即刻重分布（D3）
        self.var_table.viewport().installEventFilter(self)
        header.sectionResized.connect(self._on_section_resized)
        header.sectionHandleDoubleClicked.connect(self._on_handle_double_clicked)
        # 行右键重置菜单（D8）
        self.var_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.var_table.customContextMenuRequested.connect(self._on_row_context_menu)

        layout.addWidget(self.var_table)

        # 按钮区域
        button_layout = QHBoxLayout()

        # 排序按钮（下拉菜单：升序/降序/仅排序选中行/撤销上次排序）
        self.sort_btn = QPushButton("排序 ▾")
        self.sort_menu = QMenu(self)
        self._add_sort_actions(self.sort_menu)
        # 弹出前刷新动作可用态（选中数/可撤销状态是动态的）
        self.sort_menu.aboutToShow.connect(
            lambda: self._update_sort_menu_actions(self.sort_menu)
        )
        self.sort_btn.setMenu(self.sort_menu)
        self.sort_btn.setEnabled(False)
        button_layout.addWidget(self.sort_btn)

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

        # 底部：提示（左）+ 确定/取消（右）合为一行（D21）。此前提示是单独
        # 一整行、按钮又一行，白白多占 33px（27 行高 + 6 间距）；并排实测只需
        # 476px，比最小窗宽 560 还宽松。提示不再要 margin-top：它在按钮行里
        # 会被布局垂直居中，留着会把文字压偏 10px。
        bottom_layout = QHBoxLayout()
        info_label = QLabel("提示：拖拽变量到此窗口，或用上方搜索框添加")
        info_label.setStyleSheet("color: gray; font-size: 12px;")
        bottom_layout.addWidget(info_label)
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
        # 右键单槽在 setup_ui 前段已连接 _on_row_context_menu（重置组+排序组，
        # 评审 P0-1：不得双连接否则先后弹两个菜单）
        # 不再需要itemChanged信号，因为使用QCheckBox控件
        # 拖拽添加等外部变化时刷新表格（必需，否则拖拽后表格不更新）
        # 性能优化（方案 A）：改走增量更新，仅增删变化行；复杂场景内部回退全量重建
        self.plot_widget.curves_changed.connect(self._on_curves_changed_incremental)

    def _refresh_identity(self) -> None:
        """刷新顶部身份行（无可用身份时整行隐藏，不写占位文字）。

        tooltip 始终给全文：身份行没有省略号，超长页名会被布局硬裁，
        悬停是唯一的完整出口（D6）。
        """
        text = self._plot_identity_text()
        self.identity_label.setText(text)
        self.identity_label.setToolTip(text)
        self.identity_label.setVisible(bool(text))

    def _plot_identity_text(self) -> str:
        """本窗口所属「页名 · 位次」；无法解析时返回空串。

        位次取 QGridLayout 的真实格位（行/列 1 起算），与屏幕上的格子一一
        对应；按**可见**网格形状退化：1×1 只给页名，单行只给列，单列只给行，
        多行多列才写全「第 m 行 第 n 列」（行/列之间只留一个空格，不加标点，
        避免和页名后的「·」形成两级分隔符）。页名用 page.title，与页签同源，
        重命名后自动跟随。

        退化必须用可见行列而不是 layout.rowCount()/columnCount()：缩行列只把
        容器 setVisible(False) 不销毁（plot_page.apply_visibility），已建
        3×2 + 可见 3×1 会多印一个「第 1 列」。目标格本身被隐藏时（编辑器开着
        改布局再重载可达）按已建网格写全并标注隐藏，避免印出超出可见范围的
        位次；替身页/独立 widget 没有可见行列时回退已建网格。

        反查口径与 PlotWidget._plot_badge_label_for 的「页N·图M」一致，只是
        这里用页名 + 行列而非序数。注意：本类自身是 Tool 顶层窗，
        self.window() 返回自己，必须经 plot_widget.window() 取宿主（同
        refresh_data_source 的说明）。
        """
        pw = self.plot_widget
        page = getattr(pw, "page", None)
        if pw is None or page is None:
            return ""
        try:
            host = pw.window()
            page_index_of = getattr(host, "page_index", None)
            if not callable(page_index_of) or page_index_of(page) < 0:
                return ""
            layout = getattr(page, "plot_layout", None)
            position = None
            for container in getattr(page, "plot_widgets", ()):
                if getattr(container, "plot_widget", None) is not pw:
                    continue
                index = layout.indexOf(container) if layout is not None else -1
                if index >= 0:
                    row, col, _row_span, _col_span = layout.getItemPosition(index)
                    built_rows, built_cols = layout.rowCount(), layout.columnCount()
                    rows_visible = _visible_grid_dim(
                        page, "rows_current", built_rows
                    )
                    cols_visible = _visible_grid_dim(
                        page, "cols_current", built_cols
                    )
                    hidden = row >= rows_visible or col >= cols_visible
                    position = (
                        row + 1,
                        col + 1,
                        built_rows if hidden else rows_visible,
                        built_cols if hidden else cols_visible,
                        hidden,
                    )
                break
        except RuntimeError:
            # plot/页 的 C++ 对象已销毁（子图关掉/重建）：静默隐藏
            return ""
        if position is None:
            return ""
        row, col, rows, cols, hidden = position
        if rows <= 1 and cols <= 1:
            place = ""
        elif rows <= 1:
            place = f"第 {col} 列"
        elif cols <= 1:
            place = f"第 {row} 行"
        else:
            place = f"第 {row} 行 第 {col} 列"
        title = str(getattr(page, "title", "") or "").strip()
        text = " · ".join(part for part in (title, place) if part)
        if hidden and place:
            text += "（当前被布局隐藏）"
        return text

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
        # 身份行顺带校正（打开时已算过一次；这里兜住重载期间页名/页序变化，
        # 且不依赖搜索栏是否建成）
        self._refresh_identity()
        if self.search_bar is None:
            return
        pw = self.plot_widget
        if pw is None:
            return
        # 本对话框是顶层 Tool 窗口（见 __init__ 的 setWindowFlag），
        # self.window() 返回自身而非主窗口，必须经 plot_widget 取宿主
        main_window = pw.window()
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
        select_table_row(self.var_table, target_row)
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
        """加载当前绘图中的曲线（统一版：curves + 失效占位）"""
        # 先清空表格
        self.var_table.setRowCount(0)

        for var_name, curve_info in self.plot_widget.curves.items():
            self._add_variable_to_table(var_name, curve_info)

        # 失效占位行置尾部（§6.3）：灰名 +（缺失）+ 占位色块 + visible_preference 复选框
        invalid = getattr(self.plot_widget, "invalid_curves", {}) or {}
        for var_name, ref in invalid.items():
            self._add_invalid_to_table(var_name, ref)

        if not self._columns_sized and self.var_table.rowCount() > 0:
            self._adjust_fixed_column_widths()
            self._columns_sized = True
        self.update_button_states()

    def _adjust_fixed_column_widths(self):
        """定宽列首帧自适应：col0/col3 按内容，换算列默认宽 84（D13/D16）。

        首帧数据填充后执行一次（此时单元格控件已就位，宽度由表头 +
        复选框/摘要格实际内容决定，跨平台字体差异下不手工拍值）；
        换算列例外——未拖过（不在固定列）时固定窄宽，拖过的保留用户宽。
        """
        for col in _FIXED_MIN_WIDTHS:
            self._width_guard = True
            try:
                if col in _TRANSFORM_COLS:
                    # 未拖过的换算列压回默认宽；拖过的保留恢复/默认宽，
                    # 一律不按极值文本 sizeHint 自适应（D13/D16）
                    if col not in self._user_fixed_cols:
                        self.var_table.setColumnWidth(col, _TRANSFORM_COL_WIDTH)
                    continue
                self.var_table.resizeColumnToContents(col)
                self.var_table.setColumnWidth(
                    col, max(self.var_table.columnWidth(col), self._col_min[col])
                )
            finally:
                self._width_guard = False
        self._distribute_column_widths()

    # ------------------------------------------------------------------
    # C4 列宽引擎（D1+D10）
    # ------------------------------------------------------------------
    def _init_column_widths(self):
        """初值：有存档用存档（含用户固定列），无存档用默认宽。

        固定列集合先读出再套用（评审 P2-12）：固定列（含拖过的换算列，
        D16）恢复存档宽；未固定换算列回默认 84（旧档 sizeHint 膨胀值不恢复）。
        """
        from src.core.settings import AppSettings

        settings = AppSettings()
        widths = settings.get_plot_variable_editor_column_widths()
        fixed = settings.get_plot_variable_editor_fixed_cols() or []
        self._user_fixed_cols = {c for c in fixed if 0 <= c < _COL_COUNT}
        self._width_guard = True
        try:
            if widths is not None and len(widths) == _COL_COUNT:
                for col, w in enumerate(widths):
                    if col in _TRANSFORM_COLS and col not in self._user_fixed_cols:
                        # 评审 P0-1：旧档里被 sizeHint 撑大的 4/5 宽一律压回
                        # 定宽；只有用户拖过（固定列）才恢复存档宽
                        self.var_table.setColumnWidth(col, _TRANSFORM_COL_WIDTH)
                        continue
                    self.var_table.setColumnWidth(col, max(w, self._col_min[col]))
                # 评审 P1-2：有存档时跳过首帧内容自适应，否则 _adjust_fixed_
                # column_widths 会覆盖刚恢复的定宽列存档（D10 记忆失效）
                self._columns_sized = True
            else:
                for col, w in enumerate(_COL_DEFAULT_WIDTHS):
                    self.var_table.setColumnWidth(col, w)
        finally:
            self._width_guard = False

    def _save_column_widths(self):
        if not widget_alive(self):
            return
        from src.core.settings import AppSettings

        settings = AppSettings()
        settings.set_plot_variable_editor_column_widths(
            [self.var_table.columnWidth(c) for c in range(_COL_COUNT)]
        )
        settings.set_plot_variable_editor_fixed_cols(sorted(self._user_fixed_cols))

    def _active_elastic(self) -> list[int]:
        return [c for c in _ELASTIC_COLS if c not in self._user_fixed_cols]

    # ------------------------------------------------------------------
    # 行号槽（D1）：常驻宽度 + viewport 变化重分布（D3）
    # ------------------------------------------------------------------
    def _sync_ordinal_gutter(self):
        """按当前行数的位数钉行号槽宽（最少 2 位）。

        不钉宽时行号头在 0 行时为 0px、有行后才有 14px：空表打开后再加
        曲线会让 viewport 变窄，列位右移并挤出 14px 尘埃横滚条（实测
        hmax=13）。钉成常驻槽后空/满两态列位一致；行数跨位（≥100 条）
        时扩一档，宽度变化经 viewport 重分布吸收。

        量测必须用 header 自己的字体：竖表头与表格的 QSS font-size 不是
        同一套（实测两端字号不同），用 var_table.fontMetrics() 会算偏、
        裁掉末位数字。
        """
        if not widget_alive(self):
            return
        header = self.var_table.verticalHeader()
        digits = max(
            _ORDINAL_MIN_DIGITS,
            len(str(max(1, self.var_table.rowCount()))),
        )
        width = (
            header.fontMetrics().horizontalAdvance("9" * digits)
            + _ORDINAL_GUTTER_PAD
        )
        if header.width() != width:
            header.setFixedWidth(width)

    def _on_table_rows_changed(self, *_args):
        """rowsInserted / rowsRemoved：行数变了就重钉槽宽（幂等，仅跨位时生效）。"""
        self._sync_ordinal_gutter()

    def _schedule_viewport_redistribute(self):
        """viewport 尺寸变化的合并调度：一次布局只跑一次，避免拖窗时刷爆。"""
        if self._redistribute_pending:
            return
        self._redistribute_pending = True
        QTimer.singleShot(0, self._run_scheduled_redistribute)

    def _run_scheduled_redistribute(self):
        self._redistribute_pending = False
        self._safe_distribute()

    def _distribute_column_widths(self):
        """窗口余量按默认比例分给双弹性列；缩到最小宽后横滚兜底。"""
        if self._width_guard:
            return
        elastic = self._active_elastic()
        if not elastic:
            return
        viewport_w = self.var_table.viewport().width()
        if viewport_w <= 0:
            return
        self._last_viewport_width = viewport_w
        fixed_sum = sum(
            self.var_table.columnWidth(c)
            for c in range(_COL_COUNT)
            if c not in elastic
        )
        avail = viewport_w - fixed_sum
        min_sum = sum(self._col_min[c] for c in elastic)
        self._width_guard = True
        try:
            if avail >= min_sum:
                def_sum = sum(_COL_DEFAULT_WIDTHS[c] for c in elastic)
                for c in elastic:
                    share = int(avail * _COL_DEFAULT_WIDTHS[c] / def_sum)
                    self.var_table.setColumnWidth(c, max(share, self._col_min[c]))
            else:
                # 余量不足：弹性列落最小宽，总宽超视口 → 横向滚动兜底
                for c in elastic:
                    self.var_table.setColumnWidth(c, self._col_min[c])
        finally:
            self._width_guard = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._distribute_column_widths()

    def _prepare_initial_layout(self):
        """首次显示前落定样式与布局，避免首帧后再收窄弹性列。"""
        self.ensurePolished()
        self.layout().activate()
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        # 事件泵仍可能处理外部信号/销毁事件，之后先确认对象存活。
        if not widget_alive(self):
            return
        self._sync_ordinal_gutter()
        self._distribute_column_widths()

    def show_initial(self):
        """显示并等到首帧布局稳定，供生产打开入口使用。

        隐藏状态下 QAbstractScrollArea 尚不计算竖滚条占位，show 后视口才
        收窄并触发第二次列宽分配。短暂停用绘制，让这段几何变化发生在首帧
        之前；事件泵排除用户输入，完成后同步分布并恢复绘制。
        """
        self._prepare_initial_layout()
        if not widget_alive(self):
            return
        self.setUpdatesEnabled(False)
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        if not widget_alive(self):
            return
        if self.isVisible():
            self._safe_distribute()
        self.setUpdatesEnabled(True)

    def showEvent(self, event):
        """首次显示后补一次列宽分布（R3 / 评审 P1-6）。

        _prepare_initial_layout 已在 show 前落定首帧样式/布局；这里保留
        显示后修正，兜底后续字体、主题或平台延迟送达的尺寸变化。
        构造期视口尺寸与 _restore_geometry 恢复后的视口不一致，构造时分的
        弹性列宽是旧尺寸；不补分布时用户第一次拖窗宽才触发比例分配 →
        弹性列瞬间跳变。singleShot(0) 等布局落定再补一次（两参 + 存活守卫）。
        """
        super().showEvent(event)
        # 行号槽补钉一次：QSS 的 font-size 在 polish 后才落到表头上，
        # 构造期按未 polish 的字体算会宽 2px（实测 24 vs 22），首行出现时
        # 再重钉就会跳一下。
        self._sync_ordinal_gutter()
        self._distribute_column_widths()
        QTimer.singleShot(0, self._safe_distribute)

    def _safe_distribute(self):
        if widget_alive(self):
            # 字体/DPR 变更后槽宽可能已过期；重分布前先重钉（幂等）
            self._sync_ordinal_gutter()
            self._distribute_column_widths()

    def eventFilter(self, watched, event):
        """viewport 宽度变化 → 重分布列宽（D3）。

        覆盖两类不经过对话框 resizeEvent 的宽度变化：行号槽扩档、竖向
        滚动条出现/消失。延后一拍执行，等 Qt 布局落定再按稳定视口重算；
        否则 setColumnWidth 触发滚动条显隐会与本次计算打架。合并闸保证
        拖窗/批量插入时 singleShot 队列不膨胀。

        按**宽度**去重：横滚显隐只改高度却同样发 Resize，不过滤时用户
        把固定列拖宽会连带把弹性列挤到最小宽，破坏 D10 的横滚兜底语义。
        """
        if event.type() == QEvent.Type.Resize and watched is self.var_table.viewport():
            if self.var_table.viewport().width() != self._last_viewport_width:
                self._schedule_viewport_redistribute()
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        """主题字体变更后重钉槽宽（评审 P1-6）。

        槽宽按 header 字体度量，字体变了不重钉会与绘制差 1~2px。构造期
        早于 var_table 建成的 changeEvent 用 getattr 守卫跳过。
        """
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.FontChange
            and getattr(self, "var_table", None) is not None
        ):
            self._sync_ordinal_gutter()

    def _on_section_resized(self, col: int, _old: int, new: int):
        """用户拖拽 → 该列转固定（退出弹性分配）；夹最小宽。

        D16（反转 v1 P2-2）：换算列同样可拖，拖过即入固定列并落盘，
        重开按存档宽恢复（不再压回 84）。
        """
        if self._width_guard:
            return
        if new < self._col_min[col]:
            self._width_guard = True
            try:
                self.var_table.setColumnWidth(col, self._col_min[col])
            finally:
                self._width_guard = False
        self._user_fixed_cols.add(col)
        # 拖宽即时落盘兜底：主窗销毁/进程退出不走 closeEvent，等不到 done()
        self._width_save_timer.start()

    def _on_handle_double_clicked(self, col: int):
        """双击分隔线：普通列恢复弹性 + 内容自适应（D1）；换算列回默认 84。

        换算列双击同样要落盘（评审 P1-5）：早退路径不存盘会让重开闪回
        旧宽（D16）。
        """
        if col in _TRANSFORM_COLS:
            self._user_fixed_cols.discard(col)
            self._width_guard = True
            try:
                self.var_table.setColumnWidth(col, _TRANSFORM_COL_WIDTH)
            finally:
                self._width_guard = False
            self._distribute_column_widths()
            self._width_save_timer.start()
            return
        self._user_fixed_cols.discard(col)
        self._width_guard = True
        try:
            self.var_table.resizeColumnToContents(col)
            self.var_table.setColumnWidth(
                col, max(self.var_table.columnWidth(col), self._col_min[col])
            )
        finally:
            self._width_guard = False
        self._distribute_column_widths()
        self._width_save_timer.start()

    def _geometry_on_screen(self) -> bool:
        """当前窗口几何是否与任一现存屏幕的可用区域相交。

        多显示器拔插守卫：restoreGeometry 可能把窗口放到已拔掉的屏幕
        上（窗口"消失"），此校验失败则丢弃记忆、回退初始几何。
        用 normalGeometry（restoreGeometry 同步写入，对未显示窗口也
        有效）；frameGeometry 对未显示窗口是空矩形，会误判放行。
        """
        from PySide6.QtGui import QGuiApplication

        geo = self.normalGeometry()
        if geo.isNull():
            return True  # 无可判定几何（未布局完成），放行交给用户
        return any(
            geo.intersects(screen.availableGeometry())
            for screen in QGuiApplication.screens()
        )

    def _restore_geometry(self):
        """恢复上次窗口几何；无记忆/记忆损坏/记忆落失效屏幕时回退初始值。

        初始宽按实际固定列宽和 + 变量名列最小宽 + 滚动条/边距余量计算，
        并 clamp 到主窗口宽 × 0.9（Tool 窗宽于主窗观感差）；高度静态 600
        （macOS Tool 窗标题栏高度不同，只自适应宽度更稳）。首次打开
        居中于主窗口。
        """
        from src.core.settings import AppSettings

        self.setMinimumSize(560, 400)
        settings = AppSettings()
        geometry = settings.get_plot_variable_editor_geometry()
        if (
            geometry is not None
            and self.restoreGeometry(geometry)
            and self._geometry_on_screen()
        ):
            # P1-9（第 5 轮）：旧脏存档自救——存档含最小化态时 restore 成功
            # 且 normalGeometry 在屏内会提前 return，窗口保持最小化（丢窗）；
            # 与 mark_stats.load_geom 末尾强制 WindowNoState 同款。
            if self.isMinimized():
                self.setWindowState(Qt.WindowState.WindowNoState)
            return

        # 自适应初始宽：定宽列实际宽 + 双弹性列默认宽 + 表格边框与
        # 滚动条余量 + 窗口边框余量（C4：6 列口径）
        fixed = sum(self.var_table.columnWidth(c) for c in _FIXED_MIN_WIDTHS)
        width = fixed + 380 + 24 + 40
        main_window = self.plot_widget.window()
        if main_window is not None:
            width = min(width, int(main_window.width() * 0.9))
        self.resize(width, 600)
        if main_window is not None:
            geo = self.geometry()
            self.move(
                main_window.x() + (main_window.width() - geo.width()) // 2,
                main_window.y() + (main_window.height() - geo.height()) // 2,
            )
        # 第 6 轮 NB1：回退分支同样自救——脏存档含最小化态且 normalGeometry
        # 落已拔屏幕时走这里，windowState 仍可能为最小化（与屏内分支同款）
        if self.isMinimized():
            self.setWindowState(Qt.WindowState.WindowNoState)

    def _save_geometry(self):
        # P1-9（第 5 轮）：最小化/最大化时 saveGeometry 会把窗口状态一并存盘，
        # 下次开窗直接以最小化出现（等于丢窗）；同 curve_shelf / mark_stats
        # 口径跳过（table_dialog 只 guard 最小化、最大化属用户偏好，编辑器
        # 是 Tool 窗，跟随多数派）。
        if self.isMinimized() or self.isMaximized():
            return
        from src.core.settings import AppSettings

        AppSettings().set_plot_variable_editor_geometry(self.saveGeometry())

    def _on_curves_changed_incremental(self):
        """curves_changed 增量表格更新（性能优化方案 A）

        对表格行与 curves 字典做 diff，按场景分流：
        - 名称+顺序完全一致：轻量就地刷新（可见性/颜色），不重建任何控件；
        - 纯添加且新曲线全部在末尾：仅追加新行（Enter 连续添加的主场景）；
        - 纯删除且剩余顺序一致：仅删除对应行；
        - 其余复杂场景（重排、中间插入、混合增删）：回退 load_current_curves 全量重建。

        回退兜底保证正确性：增量路径出错的最坏结果也只是退回全量重建。

        占位相关一律回退全量重建（§6.3）：diff 口径基于 curves，占位行不在
        curves 中，增量路径会把占位行当作"已删除"误删，或复活后残留灰名
        "（缺失）"文本。检测到 invalid_curves 非空、或表格已有占位行时直接重建。
        """
        table = self.var_table

        # 占位存在（数据侧或表格侧）→ 全量重建，规避 diff 口径不含 invalid_curves
        has_invalid_data = bool(getattr(self.plot_widget, "invalid_curves", {}) or {})
        has_invalid_row = any(
            (table.item(row, 1) is not None
             and table.item(row, 1).data(Qt.ItemDataRole.UserRole + 1))
            for row in range(table.rowCount())
        )
        if has_invalid_data or has_invalid_row:
            self.load_current_curves()
            return

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
        """就地刷新表格行（不重建控件）：可见性/徽标/Legend 名/样式摘要/换算。

        用于 curves 集合与顺序均未变的场景（如颜色修改、外部可见性变更，
        或图例面板切换样式后的回填）。程序化回填一律 QSignalBlocker，
        防 valueChanged/editingFinished 回环（评审 P1-6）。
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
            unit = self.plot_widget.units.get(var_name, "")
            base = f"{var_name} ({unit})" if unit else var_name
            name_item = table.item(row, 1)
            if name_item is not None:
                name_item.setText(base + _name_badges(curve_info))
            name_edit = table.cellWidget(row, 2)
            if name_edit is not None:
                with QSignalBlocker(name_edit):
                    name_edit.setText(getattr(curve_info, "display_name", ""))
            style_cell = table.cellWidget(row, 3)
            if style_cell is not None:
                style_cell.set_state(curve_info.color, _style_summary(curve_info))
            f_edit = table.cellWidget(row, 4)
            o_edit = table.cellWidget(row, 5)
            if f_edit is not None:
                f_edit.set_value(getattr(curve_info, "factor", 1.0))
                f_edit.setEnabled(not _transform_disabled(curve_info))
            if o_edit is not None:
                o_edit.set_value(getattr(curve_info, "offset", 0.0))
                o_edit.setEnabled(not _transform_disabled(curve_info))

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
        # 手动移动后行序已变，"撤销上次排序"快照失效，作废之
        self._last_sort_order = None
        self._apply_curve_order(order)
        self.load_current_curves()
        select_table_row(self.var_table, target_row)

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

    def _add_sort_actions(self, menu: QMenu):
        """向目标菜单注入排序动作组（"排序 ▾"按钮菜单与表格右键菜单共用）"""
        act_asc = menu.addAction("按名称升序")
        act_asc.setData("asc")
        act_asc.triggered.connect(lambda: self._sort_variables(descending=False))
        act_desc = menu.addAction("按名称降序")
        act_desc.setData("desc")
        act_desc.triggered.connect(lambda: self._sort_variables(descending=True))
        menu.addSeparator()
        act_sel = menu.addAction("仅排序选中行")
        act_sel.setData("sel")
        act_sel.triggered.connect(
            lambda: self._sort_variables(descending=False, selected_only=True)
        )
        menu.addSeparator()
        act_undo = menu.addAction("撤销上次排序")
        act_undo.setData("undo")
        act_undo.triggered.connect(self._undo_last_sort)

    def _update_sort_menu_actions(self, menu: QMenu):
        """排序菜单弹出前刷新各动作可用态"""
        selected_rows = {item.row() for item in self.var_table.selectedItems()}
        for action in menu.actions():
            act = action.data()
            if act in ("asc", "desc"):
                action.setEnabled(self.var_table.rowCount() >= 2)
            elif act == "sel":
                action.setEnabled(len(selected_rows) >= 2)
            elif act == "undo":
                action.setEnabled(self._last_sort_order is not None)

    def _show_table_context_menu(self, pos):
        """旧右键入口薄壳（撤入口不撤实现）：委托合并单槽 _on_row_context_menu。"""
        self._on_row_context_menu(pos)

    def _table_order(self) -> list[str]:
        """按表格行序收集变量名（含失效占位行）"""
        order = []
        for row in range(self.var_table.rowCount()):
            name_item = self.var_table.item(row, 1)
            if name_item is not None:
                order.append(name_item.data(Qt.ItemDataRole.UserRole))
        return order

    def _selected_var_names(self) -> set[str]:
        """收集当前选中行对应的变量名集合"""
        names = set()
        for row in {item.row() for item in self.var_table.selectedItems()}:
            name_item = self.var_table.item(row, 1)
            if name_item is not None:
                name = name_item.data(Qt.ItemDataRole.UserRole)
                if name:
                    names.add(name)
        return names

    def _sort_variables(self, descending: bool = False, selected_only: bool = False):
        """按变量名自然排序重排行序（升序/降序/仅排序选中行）

        - 全列表排序：所有行参与；失效占位行由 load_current_curves 置回尾部
        - 仅排序选中行：选中行只在其占用的位置槽内重排，未选中行位置不变
        - 排序前记录单层行序快照供"撤销上次排序"；对已有序列表重复排序时
          提前返回，不覆盖快照
        - 排序后选中跟随行移动（按变量名恢复选中）
        """
        order = self._table_order()
        if len(order) < 2:
            return
        selected_names = self._selected_var_names()
        if selected_only:
            if len(selected_names) < 2:
                return
            sorted_selected = sorted(selected_names, key=natural_sort_key)
            if descending:
                sorted_selected.reverse()
            slot_iter = iter(sorted_selected)
            new_order = [
                next(slot_iter) if name in selected_names else name
                for name in order
            ]
        else:
            new_order = sorted(order, key=natural_sort_key, reverse=descending)
        if new_order == order:
            return
        self._last_sort_order = order
        self._apply_curve_order(new_order)
        self.load_current_curves()
        self._reselect_rows_by_names(selected_names)
        self.update_button_states()

    def _undo_last_sort(self):
        """回退到最近一次排序前的行序（快照用后即清，单层撤销）"""
        if not self._last_sort_order:
            return
        selected_names = self._selected_var_names()
        self._apply_curve_order(list(self._last_sort_order))
        self._last_sort_order = None
        self.load_current_curves()
        self._reselect_rows_by_names(selected_names)
        self.update_button_states()

    def _reselect_rows_by_names(self, names: set[str]):
        """按变量名集合恢复选中行（排序后选中跟随）"""
        table = self.var_table
        table.clearSelection()
        if not names:
            return
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        selection = table.selectionModel()
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            if (
                name_item is not None
                and name_item.data(Qt.ItemDataRole.UserRole) in names
            ):
                selection.select(table.model().index(row, 0), flags)

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

        # 变量名和单位（C4：徽标后缀 名/ƒ，D7）
        unit = self.plot_widget.units.get(var_name, "")
        display_text = f"{var_name} ({unit})" if unit else var_name
        name_item = QTableWidgetItem(display_text + _name_badges(curve_info))
        name_item.setData(Qt.ItemDataRole.UserRole, var_name)
        name_item.setToolTip(var_tooltip(var_name, unit))
        name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.var_table.setItem(row, 1, name_item)

        # Legend 名（C4/D2：空=默认，整串替换；editingFinished 提交）
        default_name = display_text
        name_edit = QLineEdit()
        name_edit.setPlaceholderText(default_name)
        name_edit.setText(getattr(curve_info, "display_name", ""))
        name_edit.setToolTip(f"自定义 Legend 显示名；留空恢复默认：{default_name}")
        name_edit.editingFinished.connect(
            lambda name=var_name, ed=name_edit: self._on_legend_name_changed(name, ed)
        )
        self.var_table.setCellWidget(row, 2, name_edit)

        # 样式摘要格（C4/D9：色块+非默认项摘要，点击弹 legend 同款面板）
        style_cell = _StyleCell(var_name)
        style_cell.set_state(curve_info.color, _style_summary(curve_info))
        style_cell.clicked.connect(self._open_style_panel)
        self.var_table.setCellWidget(row, 3, style_cell)

        # Factor / Offset（C4/D3/D4/D17：行内文本输入，非数值通道置灰）
        factor_edit, offset_edit = self._make_transform_edits(var_name, curve_info)
        self.var_table.setCellWidget(row, 4, factor_edit)
        self.var_table.setCellWidget(row, 5, offset_edit)

    def _make_transform_edits(self, var_name: str, ci, force_disabled: bool = False):
        """Factor/Offset 文本输入对（D17：只打字，无步进/滚轮）。"""
        disabled = force_disabled or _transform_disabled(ci)
        edits = []
        for value in (getattr(ci, "factor", 1.0), getattr(ci, "offset", 0.0)):
            edit = _NumericLineEdit(value)
            edit.setEnabled(not disabled)
            edit.setToolTip(
                "枚举/时间通道不支持数值变换" if disabled
                else "可直接输入数字（支持小数与正负号）"
            )
            edit.value_committed.connect(
                lambda _v, name=var_name: self._on_transform_changed(name)
            )
            edits.append(edit)
        return edits[0], edits[1]

    def _row_of_var(self, var_name: str) -> int | None:
        for row in range(self.var_table.rowCount()):
            item = self.var_table.item(row, 1)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == var_name:
                return row
        return None

    def _on_legend_name_changed(self, var_name: str, edit: QLineEdit):
        """Legend 名提交：空串=恢复默认（共享入口，播报 parity）。"""
        self.plot_widget._multi_curve_manager.apply_curve_display_name(
            var_name, edit.text().strip()
        )

    def _on_transform_changed(self, var_name: str):
        """Factor/Offset 提交：读两框现值走共享入口（同值 no-op）并回写规范文本。"""
        row = self._row_of_var(var_name)
        if row is None:
            return
        f_edit = self.var_table.cellWidget(row, 4)
        o_edit = self.var_table.cellWidget(row, 5)
        if f_edit is None or o_edit is None:
            return
        self.plot_widget._multi_curve_manager.apply_curve_transform(
            var_name, f_edit.value(), o_edit.value()
        )
        ci = self.plot_widget.curves.get(var_name)
        if ci is not None:
            # 统一回写规范文本（去尾零/夹紧）；聚焦中的框由 set_value 自行跳过
            f_edit.set_value(getattr(ci, "factor", 1.0))
            o_edit.set_value(getattr(ci, "offset", 0.0))

    def _open_style_panel(self, var_name: str):
        """样式格点击：singleShot(0) 弹共享面板（存活守卫，frozen 两参口径）。"""
        pw = self.plot_widget
        if var_name not in pw.curves:
            return

        def _show():
            if widget_alive(self) and widget_alive(pw) and var_name in pw.curves:
                exec_curve_panel(pw, var_name, QCursor.pos())

        QTimer.singleShot(0, _show)

    def _on_row_context_menu(self, pos):
        """表格右键单槽（评审 P0-1 合并）：行上=重置组+排序组，空白处仅排序组。

        重置组（D8）走共享入口；排序组复用 _add_sort_actions。一次 exec，
        避免双连接导致先后弹两个菜单。
        """
        menu = QMenu(self.var_table)
        row = self.var_table.rowAt(pos.y())
        var_name = None
        if row >= 0:
            item = self.var_table.item(row, 1)
            if item is not None:
                var_name = item.data(Qt.ItemDataRole.UserRole)
        act_name = act_xf = act_all = None
        if var_name:
            act_name = menu.addAction("重置显示名")
            act_xf = menu.addAction("重置变换（factor=1, offset=0）")
            act_all = menu.addAction("重置该行全部")
            menu.addSeparator()
        self._add_sort_actions(menu)
        self._update_sort_menu_actions(menu)
        chosen = menu.exec(self.var_table.viewport().mapToGlobal(pos))
        menu.deleteLater()
        if var_name is None:
            return
        mcm = self.plot_widget._multi_curve_manager
        if chosen is act_name:
            mcm.apply_curve_display_name(var_name, "")
        elif chosen is act_xf:
            mcm.apply_curve_transform(var_name, 1.0, 0.0)
        elif chosen is act_all:
            mcm.apply_curve_display_name(var_name, "")
            mcm.apply_curve_transform(var_name, 1.0, 0.0)

    @staticmethod
    def _draw_mode_tooltip(curve_info) -> str:
        """绘制方式下拉 tooltip：标注枚举通道自动判定来源（可解释性）。"""
        if (
            getattr(curve_info, "draw_mode", DRAW_MODE_AUTO) == DRAW_MODE_STEP_POST
            and getattr(curve_info, "draw_mode_source", "") == DRAW_MODE_SOURCE_AUTO
        ):
            return (
                "枚举通道自动判定：枚举无中间态，已用阶梯(后)显示；"
                "手动选择其他方式后以手动为准"
            )
        return ""

    def _on_draw_mode_changed(self, var_name: str, draw_mode, label: str):
        """绘制方式下拉变更：委托 MultiCurveManager 应用到曲线。"""
        if draw_mode is None or var_name not in self.plot_widget.curves:
            return
        applied = self.plot_widget._multi_curve_manager.apply_curve_draw_mode(
            var_name, draw_mode
        )
        if applied:
            logger.debug(
                "[EDITOR] 绘制方式变更: var=%s, mode=%s", var_name, draw_mode
            )
            self._notify_style_change(var_name, f"绘制方式（{label}）")

    def _on_symbol_policy_changed(self, var_name: str, policy, label: str):
        """数据点下拉变更：委托 MultiCurveManager 应用到曲线。"""
        if policy is None or var_name not in self.plot_widget.curves:
            return
        applied = self.plot_widget._multi_curve_manager.apply_curve_symbol_policy(
            var_name, policy
        )
        if applied:
            logger.debug(
                "[EDITOR] symbol 策略变更: var=%s, policy=%s", var_name, policy
            )
            self._notify_style_change(var_name, f"数据点显示（{label}）")

    def _notify_style_change(self, var_name: str, what: str):
        """样式类变更的状态栏回执（编辑器是顶层 Tool 窗，经 pw 取宿主）。"""
        pw = self.plot_widget
        if pw is None:
            return
        main_window = pw.window()
        show = getattr(main_window, "_show_status_message", None)
        if callable(show):
            show(f"已更新 {var_name} 的{what}")

    def _add_invalid_to_table(self, var_name, ref):
        """添加失效占位行到表格（§6.3）

        与真实曲线行的差异：变量名灰色 +（缺失）后缀；复选框绑
        ``visible_preference``（复活时生效）；颜色块显示占位色（保留原色）。
        行构造独立于 ``_add_variable_to_table``，避免 CurveInfo/InvalidCurveRef
        字段不适配（§9 R27）。
        """
        row = self.var_table.rowCount()
        self.var_table.insertRow(row)

        checkbox = QCheckBox()
        checkbox.setChecked(ref.visible_preference)
        checkbox.stateChanged.connect(
            lambda state, name=var_name: self._on_checkbox_changed(name, state)
        )
        self.var_table.setCellWidget(row, 0, checkbox)

        unit = self.plot_widget.units.get(var_name, "")
        base = f"{var_name} ({unit})" if unit else var_name
        name_item = QTableWidgetItem(f"{base}（缺失）")
        name_item.setData(Qt.ItemDataRole.UserRole, var_name)
        name_item.setToolTip(var_tooltip(var_name, unit))
        # 占位行标记（UserRole+1）：增量刷新据此识别占位行并回退全量重建，
        # 避免复活后残留灰名"（缺失）"文本（§6.3）
        name_item.setData(Qt.ItemDataRole.UserRole + 1, True)
        name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        name_item.setForeground(QColor("gray"))
        self.var_table.setItem(row, 1, name_item)

        # C4：占位行同样有 Legend 名（可编辑，D6）/样式摘要（置灰）/
        # Factor/Offset（置灰显存档值，复活后生效）
        name_edit = QLineEdit()
        name_edit.setPlaceholderText(base)
        name_edit.setText(getattr(ref, "display_name", ""))
        name_edit.editingFinished.connect(
            lambda name=var_name, ed=name_edit: self._on_legend_name_changed(name, ed)
        )
        self.var_table.setCellWidget(row, 2, name_edit)

        style_cell = _StyleCell(var_name)
        style_cell.set_state(ref.color, "默认", enabled=False)
        self.var_table.setCellWidget(row, 3, style_cell)

        factor_edit, offset_edit = self._make_transform_edits(
            var_name, ref, force_disabled=True
        )
        self.var_table.setCellWidget(row, 4, factor_edit)
        self.var_table.setCellWidget(row, 5, offset_edit)

    def _on_checkbox_changed(self, var_name, state):
        """复选框状态变化处理（统一版）

        真实曲线走 ``apply_curve_visibility`` 共享入口（legend/轴/光标 +
        一次 curves_changed，与图例点名称同口径）；失效占位切
        ``InvalidCurveRef.visible_preference``（复活时生效，当前无曲线可显隐）。
        变量名两侧互斥（不变量），故按存在侧分派。
        """
        is_visible = state == Qt.CheckState.Checked.value
        logger.debug(
            "[EDITOR] _on_checkbox_changed: var=%s, visible=%s", var_name, is_visible
        )

        if var_name in self.plot_widget.curves:
            self.plot_widget._multi_curve_manager.apply_curve_visibility(
                var_name, is_visible
            )
            return

        invalid = getattr(self.plot_widget, "invalid_curves", None)
        if invalid and var_name in invalid:
            invalid[var_name].visible_preference = is_visible

    def on_selection_changed(self):
        """选择改变时的处理"""
        self.update_button_states()

    def on_cell_clicked(self, row, column):
        """单元格点击事件（C4）：样式格 item 区域补开面板。

        色块/文字区域由 _StyleCell.clicked 处理；样式格控件只覆盖单元格内容区
        （行高 _ROW_HEIGHT 减上下 padding 6，约 25px），其余条带只拿到 item
        点击——这里补一条通道，保证整格可点（R4 去定高后评审 P1-2）。颜色列
        入口已撤（D9），set_variable_color 实现保留供程序化调用。
        """
        if column != 3:
            return
        cell = self.var_table.cellWidget(row, 3)
        item = self.var_table.item(row, 1)
        if cell is None or item is None or not cell.isEnabled():
            return
        var_name = item.data(Qt.ItemDataRole.UserRole)
        if var_name:
            self._open_style_panel(var_name)

    def update_button_states(self):
        """更新按钮状态"""
        has_selection = len(self.var_table.selectedItems()) > 0
        has_items = self.var_table.rowCount() > 0

        self.remove_btn.setEnabled(has_selection)
        self.clear_btn.setEnabled(has_items)
        self.reset_color_btn.setEnabled(has_items)
        # 行数 < 2 时排序无意义（与按钮行其他按钮的禁用口径一致）
        self.sort_btn.setEnabled(self.var_table.rowCount() > 1)

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

    def done(self, result: int):
        """统一收口确定/取消（含 Esc）：hide 前落盘并断开信号。

        accept/reject 不经过 closeEvent，此前多数关闭路径根本没有写盘，
        且隐藏实例仍挂在主窗下继续响应 curves_changed（评审 P1-1）。
        """
        self._save_editor_state()
        self._teardown()
        super().done(result)

    def _save_editor_state(self):
        """落盘窗口几何 + 列宽（幂等；done 与 closeEvent 双入口）。

        评审 P2-5：写盘异常不得让 done()/closeEvent 抛错（否则对话框关不掉），
        只记 debug 日志。
        """
        try:
            self._save_geometry()
            self._save_column_widths()
        except Exception:
            logger.debug("编辑器状态落盘失败", exc_info=True)

    def _teardown(self):
        """断开 curves_changed 信号（幂等；done 与 closeEvent 双入口）。

        编辑器是主窗下的非模态 Tool，确定/取消只 hide 不销毁；若不断连，
        隐藏实例会对每次改色/显隐/换算的 emit 继续做表格增量刷新。
        """
        if self._teardown_done:
            return
        self._teardown_done = True
        try:
            self.plot_widget.curves_changed.disconnect(
                self._on_curves_changed_incremental
            )
        except (TypeError, RuntimeError):
            pass
        if self.search_bar is not None:
            try:
                self.plot_widget.curves_changed.disconnect(
                    self.search_bar._on_curves_changed
                )
            except (TypeError, RuntimeError):
                pass

    def closeEvent(self, event):
        """关闭时落盘并断开 curves_changed，避免重复打开导致信号连接累积"""
        self._save_editor_state()
        self._teardown()
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
            select_table_row(self.var_table, next_row)

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
            invalid = getattr(self.plot_widget, "invalid_curves", {}) or {}
            cleared_count = len(self.plot_widget.curves)
            logger.debug(
                "[EDITOR] clear_all_variables: 清除 %d 条曲线 + %d 条占位",
                cleared_count, len(invalid),
            )
            # 批量移除（单次刷新链，不 emit）。占位不经 clear_plot_item，
            # remove_variables_from_plot 两侧都查（§6.3），一并列入即显式清占位。
            self.plot_widget.remove_variables_from_plot(
                list(self.plot_widget.curves) + list(invalid)
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

        pw = self.plot_widget
        changed = 0
        # 批量窗口：逐条改色只刷自身，收尾 emit 一次（评审 P1-10）
        previous_batch = getattr(pw, "_batch_adding", False)
        pw._batch_adding = True
        try:
            if pw.curves:
                for idx, var_name in enumerate(pw.curves.keys()):
                    color_name = color_cycle[idx % len(color_cycle)]
                    if self._apply_color_to_curve(var_name, color_name):
                        changed += 1
        finally:
            pw._batch_adding = previous_batch
        if changed:
            pw.curves_changed.emit()

        # 重新加载表格以更新颜色显示
        selected_row = self._get_selected_row()
        self.load_current_curves()
        if selected_row is not None and selected_row < self.var_table.rowCount():
            select_table_row(self.var_table, selected_row)

    def _apply_color_to_curve(self, var_name: str, color_name: str) -> bool:
        """将指定变量的颜色更新为给定颜色（委托 manager，返回是否实际应用）。

        刷新链已整体下沉到 MultiCurveManager.apply_curve_color（图例
        调色板与编辑器共用入口）；本方法保留薄壳维持既有调用点签名，
        表格色块样式由调用方按返回值决定是否刷新。
        """
        return self.plot_widget._multi_curve_manager.apply_curve_color(
            var_name, color_name
        )

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

        # 打开颜色选择对话框：初值跟随该行现状色（真实曲线取曲线色、占位行
        # 取保留原色），兜底色板首色——不引入色板外纯蓝
        invalid = getattr(self.plot_widget, "invalid_curves", {}) or {}
        if var_name in self.plot_widget.curves:
            current_color = self.plot_widget.curves[var_name].color
        elif var_name in invalid:
            current_color = invalid[var_name].color
        else:
            current_color = DEFAULT_CURVE_COLOR

        # macOS 原生面板不支持 Qt 自定义色，统一用 Qt 自绘对话框，
        # 并把默认色板预填到“自定义颜色”，方便直接点选。
        for index, hex_color in enumerate(DEFAULT_CURVE_COLORS):
            QColorDialog.setCustomColor(index, QColor(hex_color))
        color = QColorDialog.getColor(
            QColor(current_color),
            self,
            "选择颜色",
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )

        if color.isValid():
            updated = self._apply_color_to_curve(var_name, color.name())
            # 更新表格样式格色块（仅当颜色实际应用时；变量已删则不刷）
            if updated:
                style_cell = self.var_table.cellWidget(row, 3)
                ci = self.plot_widget.curves.get(var_name)
                if style_cell is not None and ci is not None:
                    style_cell.set_state(color.name(), _style_summary(ci))

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
