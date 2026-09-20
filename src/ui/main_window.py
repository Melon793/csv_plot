from __future__ import annotations
import sys
import os
import time
from collections import OrderedDict

from src.utils.platform_setup import setup_platform

ico_path = setup_platform()

from src.core.config import (  # noqa: E402
    PLOT_ROW_MAX_DEFAULT, PLOT_COL_MAX_DEFAULT,
    PLOT_ROW_CURRENT_DEFAULT, PLOT_COL_CURRENT_DEFAULT,
    RATIO_RESET_PLOTS,
)
from src.core.settings import AppSettings  # noqa: E402
from src.ui.table_dialog import DropOverlay  # noqa: E402
from src.ui.variable_list import MyTableWidget  # noqa: E402
from src.core.logger import LogManager, get_logger  # noqa: E402
from src.ui.dialogs.log_window import LogWindow  # noqa: E402
from src.ui.widgets.plot_container import PlotContainerWidget  # noqa: E402
from src.ui import theme  # noqa: E402  # 段间竖线色值与抽屉共用同一份色板

from PySide6.QtCore import Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QIcon,
    QPainter,
    QPalette,
    QAction,
    QShortcut,
    QKeySequence,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit,
    QMessageBox, QSplitter, QMenu, QStyle, QStatusBar, QFrame,
)

SCREEN_WIDTH_MARGIN = 0.3
SCREEN_HEIGHT_MARGIN = 0.3
# 播报文本最多占用的窗口宽度比例，超出按 … 截断（完整内容同时进日志）
STATUS_MESSAGE_WIDTH_RATIO = 0.6
# 右区消息停留时长：错误常驻（挡住后续播报直到被新消息替换），其余按时回收
STATUS_MESSAGE_TIMEOUT_MS = {"info": 5000, "warn": 8000}


class _ElideStatusBar(QStatusBar):
    """状态栏不参与窗口最小宽度。

    QLabel 的 minimumSizeHint 等于整行文本宽度，播报与段文案长度不可控；
    若按默认行为，长文案会把 QMainWindow 的最小宽度顶起来，窗口在贴边宽度
    下会被突然拉宽。这里只裁掉宽度维度的最小值，高度维度保持原样，
    文本改由 MainWindow._status_elided 按可用宽度截断。
    """

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint


class _StatusSegment(QLabel):
    """状态栏上的可点击段：整块 hover 底色 + 手型光标，点击发 clicked 信号。

    照 VS Code 状态栏的 item cell：色块占满整行高、文字两侧留内边距，hover 换
    底色而不是加下划线。下划线只有 1 px，实测 hover 前后仅 436 个像素不同
    （作者据此判定"看不出能点"）；整块底色既是反馈，也顺带把命中区从"必须点
    到字"扩成整个格子。

    底色走 ``palette(Highlight)`` 而不是样式表：状态栏坐在系统窗口条上，写死
    浅色会在深色模式里发刺；本仓样式表等 P2-6 统一 token 化，这里不新增色块。

    格子高 = 文字高（实测 17 / 状态栏 22）：VS Code 那种"占满整条"做不到 ——
    ``QStatusBar`` 给每个 item 带对齐标志地塞进布局，纵向 Expanding 实测无效，
    硬撑会把整条状态栏顶高，影响所有平台。
    """

    clicked = Signal()

    PAD_X = 6  # 左右内边距：VS Code 状态栏 item 的留白量级
    # 30%：cocoa 实测 mac 的 Highlight 反推约 #A4CBFD（本身就浅），18% 叠在
    # #ECECEC 上只挪动 R-13/G-6/B+3，肉眼几乎看不出；30% 得到 #D6E2F1
    HOVER_ALPHA = 0x4D

    def __init__(self, text: str = "", tip: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 内边距交给 contentsMargins，sizeHint 与绘制用同一套值：格子比文字宽，
        # 鼠标落在字缝里也点得动
        self.setContentsMargins(self.PAD_X, 0, self.PAD_X, 0)
        self._hovered = False
        if tip:
            self.setToolTip(tip)

    def paintEvent(self, event):
        if self._hovered:
            color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
            color.setAlpha(self.HOVER_ALPHA)
            painter = QPainter(self)
            painter.fillRect(self.rect(), color)
            painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

_widget_logger = get_logger("widget")

class MainWindow(QMainWindow):
    """
    主窗口类
    应用程序的主界面，集成数据加载、图表显示、表格查看等功能
    提供完整的用户交互界面和数据处理流程
    """
    APP_DISPLAY_NAME = "CSV Plot"
    # 加载期间右区文案的秒表步进，只用于"还在干活吗"，不表示百分比
    LOAD_ELAPSED_TICK_MS = 1000

    def __init__(self):
        super().__init__()
        self._init_basic_config()
        self._load_window_context()
        self._init_data_state()
        self._init_central_widget()
        self._init_left_panel()
        self._init_right_panel()
        self._init_status_bar()
        self._init_managers()
        self._handle_cli_args()

    def _init_basic_config(self):
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._drop_event_filter_registered = False
        from src._version import get_version

        self.app_version = get_version()
        self.defaultTitle = f"{self.APP_DISPLAY_NAME} v{self.app_version}"

        if sys.platform == "darwin":
            if os.path.exists(ico_path):
                app_icon = QIcon(str(ico_path))
                app = QApplication.instance()
                app.setWindowIcon(app_icon)
                self.setWindowIcon(app_icon)

        elif sys.platform == "win32":
            if os.path.exists(ico_path):
                self.setWindowIcon(QIcon(str(ico_path)))

        self._update_title("")
        self._factor_default = 1
        self._offset_default = 0
        self.factor = self._factor_default
        self.offset = self._offset_default
        self._active_drag_container: PlotContainerWidget | None = None

        self._baseline_density: float = 0.0
        self._global_max_density: float = 0.0

    def _load_window_context(self):
        _read_status = False
        # 绘图区的隐藏态**刻意不再从配置恢复**（作者定）：「隐藏绘图区」按钮已
        # 从主界面撤下入口，照旧读 hide_plot_area 就会出现"窗口开在绘图区看不见、
        # 界面上又没有按钮能显回来"的死路（只剩拖变量进图那条自动恢复）。
        # 机制本身留着：toggle_plot_btn / toggle_plot_area 全在，入口换地方即可。
        _hide_plot_area = False

        from src.ui.file_loader_manager import FileLoaderManager
        config_path = FileLoaderManager._resolve_config_path("config_dict.json")
        if config_path is not None and os.path.isfile(config_path):
            try:
                config_dict = FileLoaderManager.load_dict(config_path)
                layout_config_dict = config_dict.get("layout_config", {})
                _width = int(layout_config_dict.get("window_width", 0))
                _height = int(layout_config_dict.get("window_height", 0))
                _max_row = int(layout_config_dict.get("max_row", 0))
                _max_col = int(layout_config_dict.get("max_col", 0))
                _default_row = int(layout_config_dict.get("default_row", 0))
                _default_col = int(layout_config_dict.get("default_col", 0))
                _read_status = all(
                    x > 0 for x in (
                        _width, _height, _max_row, _max_col, _default_row, _default_col
                    )
                )
            except Exception as e:
                _widget_logger.warning("配置文件读取失败: %s", e)

        if _read_status:
            self._window_width_default = max(600, _width)
            self._window_height_default = max(400, _height)
            self.resize(self._window_width_default, self._window_height_default)
            self._plot_row_max_default = max(1, _max_row)
            self._plot_col_max_default = max(1, _max_col)
            self._plot_row_current = max(1, min(_default_row, _max_row))
            self._plot_col_current = max(1, min(_default_col, _max_col))
        else:
            CANDIDATES = [
                (1920, 1080),
                (1600, 900),
                (1366, 768),
                (1280, 720),
                (1024, 600),
                (800, 600),
                (640, 480),
            ]

            def best_resolution() -> tuple[int, int]:
                desk = QApplication.primaryScreen().size()
                for w, h in sorted(CANDIDATES, key=lambda t: t[0] * t[1], reverse=True):
                    if w < desk.width() * (1 - SCREEN_WIDTH_MARGIN) and h < desk.height() * (1 - SCREEN_HEIGHT_MARGIN):
                        return w, h
                return desk.width(), desk.height()

            self._window_width_default, self._window_height_default = best_resolution()
            self.resize(self._window_width_default, self._window_height_default)
            self._plot_row_max_default = PLOT_ROW_MAX_DEFAULT
            self._plot_col_max_default = PLOT_COL_MAX_DEFAULT
            self._plot_row_current = PLOT_ROW_CURRENT_DEFAULT
            self._plot_col_current = PLOT_COL_CURRENT_DEFAULT

        self._hide_plot_area = _hide_plot_area

    def _init_data_state(self):
        self.loaded_path = ""
        self._last_open_dir: str | None = None
        self.loader = None
        self.var_names = None
        self.units = None
        self.time_channels_infos = None
        self.data = None
        self.data_validity = None
        self._is_loading_new_data = False
        # file_loader_manager._restore_cursor_state_after_reload 等延迟回调靠它
        # 判断主窗口是否已在退出：缺这行时 getattr(..., False) 恒假，守卫空转
        self._is_being_destroyed = False

        self._data_version = 0
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self._crosshair_update_timer.setSingleShot(True)

        self._filter_debounce_timer = QTimer(self)
        self._filter_debounce_timer.setSingleShot(True)

        self.data_table_geometry = None
        self.mark_stats_geometry = None
        self.time_correction_geometry = None
        self.var_info_geometry = None
        self._mark_stats_dirty = False
        self._mark_stats_timer = QTimer(self)
        self._mark_stats_timer.setSingleShot(True)
        self._is_syncing_crosshair = False
        self._is_syncing_mark_region = False
        self._last_template_name = ""
        self._last_template_desc = ""

        self.value_cache: OrderedDict = OrderedDict()
        self._enum_text_maps: dict = {}
        # 变量信息窗口的统计结果缓存（min/max/mean/std 及 NaN/Inf/有效样本计数）。
        # 只缓存统计不缓存元数据：实测元数据六组全属性仅 98.9 μs（纯内存零 I/O），
        # 而统计需 18.8 ms（776 MB .mf4 的 428k 点通道），相差 190 倍。
        # 单条约 150 字节，上限 512 条共 77 KB，相比 _signal_cache 单条 1.7 MB 可忽略。
        # 失效双保险：_release_old_data 显式清空 + 每条自带 generation 校验。
        self.var_stats_cache: dict = {}

    def _init_central_widget(self):
        central = QWidget()
        central.setAutoFillBackground(True)
        pal = central.palette()
        # pal.setColor(central.backgroundRole(), QColor(255, 255, 255))
        central.setPalette(pal)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.setChildrenCollapsible(False)

        self.var_table_default_width = 280
        self.var_table_user_adjusted = False

        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        self._splitter_ready = False
        self._pending_splitter_adjustment = False

    def _init_left_panel(self):
        left_widget = QWidget(self.main_splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 0, 5, 0)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        left_layout_title = QLabel("变量列表", left_widget)
        font = left_layout_title.font()
        font.setBold(True)
        left_layout_title.setFont(font)
        title_layout.addWidget(left_layout_title)

        title_layout.addStretch(1)

        self.clone_btn = QPushButton("分身", left_widget)
        self.clone_btn.setToolTip("启动独立实例")
        title_layout.addWidget(self.clone_btn)
        self.clone_btn.setVisible(True)

        self.help_btn_small = QPushButton("?", left_widget)
        self.help_btn_small.setToolTip("帮助文档")
        title_layout.addWidget(self.help_btn_small)

        left_layout.addLayout(title_layout)

        self.filter_input = QLineEdit(left_widget)
        self.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.unit_filter_input = QLineEdit(left_widget)
        self.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        self.unit_filter_input.setContentsMargins(60, 0, 0, 0)
        self.unit_filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.unit_filter_input)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QPushButton("导入数据文件", left_widget)

        self.reload_btn = QPushButton("重载", left_widget)

        button_layout.addWidget(self.load_btn, 4)
        button_layout.addWidget(self.reload_btn, 1)

        left_layout.addLayout(button_layout)
        self.list_widget = MyTableWidget(left_widget)
        left_layout.addWidget(self.list_widget)

        # 原先这一行还有「日志」按钮，已撤（作者定）：底部状态栏右端有同名的
        # 日志段，两个入口指向同一个 show_log_window。
        #
        # 「隐藏绘图区」按作者定的口径**只撤入口、留功能**：控件照建、toggled
        # 照接（见 _connect_signals），但不进布局也不显示，将来安到别处只需
        # addWidget + setVisible(True)。刻意不"塞进布局再 hide"：整行即便高度
        # 归零，left_layout 那 2 px 间距仍会为它排一排，变量列表白少一行的位置。
        self.toggle_plot_btn = QPushButton("隐藏绘图区", left_widget)
        self.toggle_plot_btn.setCheckable(True)
        self.toggle_plot_btn.setVisible(False)
        left_layout.setSpacing(2)
        self.left_widget = left_widget

        self._plot_area_visible = True
        self._saved_geometry = None
        self._saved_splitter_sizes = None
        self._was_maximized = False
        self._was_fullscreen = False

    def _init_right_panel(self):
        self.plot_widget = QWidget(self.main_splitter)
        root_layout = QVBoxLayout(self.plot_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 5, 5)

        # 「时间修正」按钮已撤（作者定）：它与状态栏中段的 x 轴抽屉共用
        # layout_manager.apply_time_correction，对绘图的作用完全同源，而抽屉还
        # 多频率输入、预设档位与应用前预览。对话框本体
        # （layout_manager.open_time_correction_dialog）按口径留着，只是暂无入口。
        self.clear_all_plots_btn = QPushButton("清除绘图", self.plot_widget)
        top_bar.addWidget(self.clear_all_plots_btn)

        self._template_menu_btn = QPushButton("模板菜单", self.plot_widget)
        self._template_menu = QMenu(self._template_menu_btn)
        self._template_menu_act_quick = None
        self._template_menu_act_save = self._template_menu.addAction("保存为模板")
        self._template_menu_act_save.triggered.connect(self.save_current_as_template)
        self._template_menu_act_mgr = self._template_menu.addAction("模板管理器")
        self._template_menu_act_mgr.triggered.connect(self.open_template_manager)
        self._template_menu.aboutToShow.connect(self._refresh_template_menu)
        self._template_menu_btn.setMenu(self._template_menu)
        top_bar.addWidget(self._template_menu_btn)

        top_bar.addStretch(1)

        self.auto_range_btn = QPushButton("自动缩放", self.plot_widget)
        self.auto_range_btn.setToolTip("自动缩放XY轴 (Ctrl+Shift+Y)")

        self.auto_y_btn = QPushButton("仅调节y轴", self.plot_widget)
        self.auto_y_btn.setToolTip("自动调节Y轴范围 (Ctrl+Y)")

        self.cursor_btn = QPushButton("显示游标", self.plot_widget)
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setToolTip("切换游标显示 (Ctrl+R)")

        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []

        self.mark_region_btn = QPushButton("标记区域", self.plot_widget)
        self.mark_region_btn.setCheckable(True)
        self.mark_region_btn.setToolTip("切换标记区域 (Ctrl+T)")

        self.grid_layout_btn = QPushButton("修改布局", self.plot_widget)
        self.grid_layout_btn.setToolTip("修改图表布局 (Ctrl+L)")

        top_bar.addWidget(self.grid_layout_btn)
        top_bar.addWidget(self.cursor_btn)
        top_bar.addWidget(self.mark_region_btn)
        top_bar.addWidget(self.auto_y_btn)
        top_bar.addWidget(self.auto_range_btn)

        root_layout.addLayout(top_bar)

        self.plot_layout = QGridLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)
        root_layout.addLayout(self.plot_layout, 1)

        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.plot_widget)

        self.main_splitter.setSizes([self.var_table_default_width, 800])

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        main_layout = self.centralWidget().layout()
        main_layout.addWidget(self.main_splitter)
        QTimer.singleShot(0, self._ensure_splitter_ready)

        self.plot_widgets = []

        self.placeholder_label = QLabel("请导入 CSV 文件以查看数据", self.plot_widget)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.placeholder_label, 0, 0)

        self.drop_overlay = DropOverlay(self.centralWidget())
        self.drop_overlay.lower()
        self.drop_overlay.hide()

        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
            self._drop_event_filter_registered = True

        if self._hide_plot_area:
            self.toggle_plot_btn.setChecked(True)
            self.toggle_plot_btn.setText("显示绘图区")
            self._plot_area_visible = False
            self.plot_widget.hide()

            # 记录 splitter 原始尺寸并压缩右侧
            self._saved_splitter_sizes = self.main_splitter.sizes()
            self.main_splitter.setChildrenCollapsible(True)
            self.main_splitter.setSizes([self.main_splitter.width(), 0])

        self.saved_mark_range = None
        self.mark_stats_window = None

        self.row_height_factors: dict[int, int] = {}

        self._cursor_shortcut = QShortcut(QKeySequence("Ctrl+R"), self.plot_widget)
        self._cursor_shortcut.activated.connect(self._on_cursor_shortcut)

        self._auto_y_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self.plot_widget)
        self._auto_y_shortcut.activated.connect(self._on_auto_y_shortcut)

        self._auto_range_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Y"), self.plot_widget)
        self._auto_range_shortcut.activated.connect(self._on_auto_range_shortcut)

        self._mark_region_shortcut = QShortcut(QKeySequence("Ctrl+T"), self.plot_widget)
        self._mark_region_shortcut.activated.connect(self._on_mark_region_shortcut)

        self._open_file_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self._open_file_shortcut.activated.connect(self._on_open_file_shortcut)

    # ------------------------------------------------------------------
    # 状态栏：常驻四块 —— 文件段 / x 轴段（可点开抽屉）/ 消息区 / 日志入口
    # ------------------------------------------------------------------
    def _edge_spacer(self, width: int = 4) -> QWidget:
        """状态栏两端留白：macOS / Win11 的窗口圆角会切掉贴边的文字。"""
        spacer = QWidget()
        spacer.setFixedWidth(width)
        return spacer

    def _vline_separator(self) -> QFrame:
        """段间竖线：Plain 单笔 + 指定色，只画一条线。

        ``Shadow.Sunken`` 会把这条线交给平台样式画成"暗 + 亮"两笔的 3D 凹槽：
        实测（offscreen/Fusion）单元格 x=[150,153) 里 150=#9f9f9f、151=#ffffff
        两列都有墨迹，且落在单元格左沿 —— 状态栏只有 22 px 高，凹槽两笔加上
        Windows 高 DPI 缩放就是肉眼看到的"细-粗-细"三条、还不居中。
        ``Plain`` 走 ``qDrawPlainLine``，一笔、画在单元格正中（实测间隙
        [85,100) → 线在 x=92，与间隙中心重合）。

        颜色取 ``foregroundRole()``（即 ``QPalette.WindowText``）：QFrame 画
        Plain 线用的就是这一项，实测设 ``Text`` 无效仍是 #000000，设
        ``WindowText`` 才生效。不用样式表是为了不把 ``QFrame`` 的选择器
        波及到内部子控件（见 theme.panel_style 的同类说明）。
        """
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setLineWidth(1)
        palette = line.palette()
        palette.setColor(line.foregroundRole(), QColor(theme.SEP_ON_BAR))
        line.setPalette(palette)
        return line

    def _init_status_bar(self):
        status_bar = _ElideStatusBar(self)
        status_bar.setSizeGripEnabled(False)
        # 关掉样式给状态栏每个 item 画的边框。
        #
        # Windows 样式会在 item 之间的 6 px 间隙里画一对明暗边框（实测：
        # "变量 ↔ x轴" 那 15 px 间隙里有 5 列墨迹、其中 3 条比背景暗，
        # "csv" 左边还有 2 条），于是我们那一根 _vline_separator 被夹在中间，
        # 看起来像三条竖线。``QStatusBar::item { border: none; }`` 是治这个
        # 症状的惯用规则，实测三个区间收敛到 0 / 1 / 1 条。
        #
        # 作用域只有这条状态栏；macOS/Fusion 本来就不画这些边框，实测加与不加
        # 整条栏 0 像素变化。代价是这里从此带一份样式表，P2-6 全局色板
        # token 化时要一并收编。
        status_bar.setStyleSheet("QStatusBar::item { border: none; }")
        self.setStatusBar(status_bar)

        status_bar.addWidget(self._edge_spacer())

        self._file_segment = _StatusSegment(
            "未加载文件", tip="加载数据文件后，点击这里查看文件信息"
        )
        self._file_segment.clicked.connect(self._open_file_info_drawer)
        status_bar.addWidget(self._file_segment)

        self._axis_segment = _StatusSegment("x轴：—", tip="点击设置 x 轴采样频率与偏移")
        self._axis_segment.clicked.connect(self._open_axis_drawer)
        self._axis_segment.hide()

        self._segment_separator = self._vline_separator()
        self._segment_separator.hide()
        status_bar.addWidget(self._segment_separator)

        status_bar.addWidget(self._axis_segment)

        self._message_text = ""
        self._message_level = "info"
        self._message_label = QLabel("")
        status_bar.addPermanentWidget(self._message_label)

        # 日志前的竖线与消息同进退：_message_label 空闲时是常驻空标签
        # （实测仍占一个 7 px 单元格），线一直亮着就成了右边一条左边没内容的
        # 孤线。跟着消息显隐，和 _segment_separator 跟着 x 轴段显隐同一套逻辑。
        self._log_separator = self._vline_separator()
        self._log_separator.hide()
        status_bar.addPermanentWidget(self._log_separator)
        self._log_segment = _StatusSegment("日志", tip="打开日志窗口")
        self._log_segment.clicked.connect(self.show_log_window)
        status_bar.addPermanentWidget(self._log_segment)

        status_bar.addPermanentWidget(self._edge_spacer())

        self._status_busy = False
        self._cursor_overridden = False
        self._load_elapsed_started = 0.0
        self._loading_path = ""
        self._last_load_elapsed = None
        self._file_info_drawer = None
        self._axis_drawer = None
        self._load_elapsed_timer = QTimer(self)
        self._load_elapsed_timer.setInterval(self.LOAD_ELAPSED_TICK_MS)
        self._load_elapsed_timer.timeout.connect(self._tick_load_elapsed)
        self._message_hide_timer = QTimer(self)
        self._message_hide_timer.setSingleShot(True)
        self._message_hide_timer.timeout.connect(self.clear_status_message)

    def _open_file_info_drawer(self):
        """左段抽屉：文件级信息 + 复制 / 打开所在文件夹。

        加载期不开：抽屉里的路径、大小、有效性都来自即将被换掉的 loader，
        弹一个正在失效的快照不如不给。
        """
        if self.reject_when_loading("查看文件信息"):
            return
        if self.loader is None:
            self._broadcast("尚未加载数据文件", level="warn")
            return

        if self._file_info_drawer is None:
            from src.ui.widgets.status_drawer import FileInfoDrawer

            self._file_info_drawer = FileInfoDrawer(self)

        self._close_status_drawers(keep=self._file_info_drawer)
        if not self._file_info_drawer.open_for(
            self, elapsed_s=self._last_load_elapsed
        ):
            self._broadcast("窗口高度不足，放大主窗口后再查看文件信息", level="warn")

    def _open_axis_drawer(self):
        """中段抽屉：按采样频率/系数/偏移设置 x 轴时间基准，带预览行。

        与顶部「时间修正」对话框改的是同一份全局 factor/offset，落地都走
        ``layout_manager.apply_time_correction``，所以这里只管输入与预览。
        """
        if self.reject_when_loading("改 x 轴基准"):
            return
        if self.loader is None:
            self._broadcast("尚未加载数据文件", level="warn")
            return

        if self._axis_drawer is None:
            from src.ui.widgets.status_drawer import XAxisDrawer

            self._axis_drawer = XAxisDrawer(self)

        self._close_status_drawers(keep=self._axis_drawer)
        if not self._axis_drawer.open_for(self):
            self._broadcast("窗口高度不足，放大主窗口后再设置 x 轴基准", level="warn")

    def _status_drawers(self) -> list:
        """已创建的状态栏抽屉清单（尚未创建的不出现在里面）。"""
        return [
            drawer
            for drawer in (
                getattr(self, "_file_info_drawer", None),
                getattr(self, "_axis_drawer", None),
            )
            if drawer is not None
        ]

    def _close_status_drawers(self, keep=None) -> None:
        """收起状态栏抽屉；同一时刻只留一扇。

        两扇抽屉都能改/看 x 轴相关的东西，叠在一起用户分不清刚点的是哪一段。
        """
        for drawer in self._status_drawers():
            if drawer is not keep and drawer.isVisible():
                drawer.hide()

    def _reposition_status_drawers(self):
        """主窗口移动/缩放后把抽屉重新贴回状态栏上方。

        用 getattr 而不是直接取属性：__init__ 里的窗口几何恢复会先发
        resize 事件，那时状态栏和抽屉都还不存在。
        """
        for drawer in self._status_drawers():
            if drawer.isVisible():
                drawer.reposition()

    def _status_elided(self, text: str) -> str:
        """超长文案按 … 截断：状态栏不参与最小宽度，但也不该吃掉整行。"""
        label = self._message_label
        budget = max(120, int(self.width() * STATUS_MESSAGE_WIDTH_RATIO))
        return label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, budget)

    def _reapply_status_elision(self):
        if not hasattr(self, "_message_label"):
            return  # _load_window_context 里的 resize 早于状态栏构建
        self._message_label.setText(self._status_elided(self._message_text))

    def _broadcast(self, message: str, level: str = "info"):
        """右区播报。错误常驻，其余按 STATUS_MESSAGE_TIMEOUT_MS 自动回收。"""
        self._logger.info(message)
        self._message_text = message
        self._message_level = level
        self._message_label.setText(self._status_elided(message))
        self._message_label.setToolTip(message)
        self._log_separator.setVisible(bool(message))
        timeout = STATUS_MESSAGE_TIMEOUT_MS.get(level)
        if timeout:
            self._message_hide_timer.start(timeout)
        else:
            self._message_hide_timer.stop()

    def clear_status_message(self):
        """错误级消息不被自动回收，避免根因还没看清就被抹掉。"""
        if self._message_level == "error" or self._status_busy:
            return
        self._message_text = ""
        self._message_label.setText("")
        self._message_label.setToolTip("")
        self._log_separator.hide()

    def _set_busy_cursor(self, busy: bool):
        """沙漏光标只在加载期覆盖，成对调用避免光标覆盖栈泄漏。"""
        if busy and not self._cursor_overridden:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._cursor_overridden = True
        elif not busy and self._cursor_overridden:
            QApplication.restoreOverrideCursor()
            self._cursor_overridden = False

    def is_data_loading(self) -> bool:
        """后台加载线程是否真在跑 —— 只有这段时间旧 loader 会被换掉。

        刻意不用 _is_loading_new_data：那是 UI 刷新链的锁，在新数据已就位、
        曲线已画完之后仍会短暂持有，用它挡交互会误伤正常的拖拽与点击。
        """
        thread = getattr(self, "_thread", None)
        return thread is not None and thread.isRunning()

    def reject_when_loading(self, action: str) -> bool:
        """加载期间的统一输入闸门。

        模态进度框撤掉后，界面在加载期是可点的，而旧 loader 随时会被释放 ——
        此时改布局、套模板、拖变量都会作用在即将消失的数据上（本仓历史上
        多次修过这类重入崩溃）。返回 True 表示调用方应放弃本次操作。
        """
        if not self.is_data_loading():
            return False
        self._broadcast(f"正在加载数据，请稍候再{action}", level="warn")
        return True

    def _update_title(self, file_path: str = ""):
        """标题只承载三件事：当前文件名、软件名、版本号。"""
        file_name = os.path.basename(file_path) if file_path else ""
        if file_name:
            self.setWindowTitle(f"{file_name} - {self.defaultTitle}")
        else:
            self.setWindowTitle(self.defaultTitle)

    def update_file_status(
        self, file_path: str, row_count: int, channel_count: int, elapsed_s: float
    ):
        """加载收尾：刷新标题与左/中两段。

        行数与耗时不进常驻段（对绘图不可执行），只留在日志与左段抽屉里。
        耗时得存一份（``_last_load_elapsed``）：它是加载链路算出来的量，
        loader 里没有，抽屉打开时才要用的话已经拿不到了。

        MDF 的行数是"最大通道组的声明记录数"（cycles_nr 取 max），口径与
        CSV/Excel 的实际行数不同，因此不在界面上冒充"N 行"。
        """
        self._update_title(file_path)
        suffix = os.path.splitext(file_path)[1].lstrip(".").lower() or "?"
        self._file_segment.setText(f"{suffix}文件 · {channel_count} 个变量")
        self._file_segment.setToolTip(os.path.abspath(file_path))
        self._last_load_elapsed = elapsed_s
        self._update_axis_segment()
        self._logger.info(
            "加载完成: %s（%d 行 / %d 变量 / 耗时 %.2fs）",
            os.path.basename(file_path),
            row_count,
            channel_count,
            elapsed_s,
        )

    def _axis_segment_text(
        self,
        factor: float,
        offset: float,
        always_show_correction: bool = False,
    ) -> str:
        """按**给定**的 factor/offset 算出 x 轴段该显示什么。

        单独一层是给抽屉的预览行用的：预览必须在不动 ``self.factor`` 的前提
        下算出"改完会长什么样"，两处共用一套文案才不会预览一个说法、落地
        另一个说法。

        常态只报轴身份：没修正过时那串系数是恒等的 1 / 反推出来的采样率，
        读它不如去读抽屉。只有真的修正过，才把**生效**的比例系数与偏移量摆
        出来 —— 这一句要回答的是"现在横轴被折成了什么"，所以系数与偏移同时
        给出，只改其一也两句都给，避免同一位置文案长度随改动类型抖动。

        ``always_show_correction`` 给抽屉的预览行用（作者定）：状态栏是常驻段，
        默认值写出来是噪音；抽屉是正在编辑基准的地方，``系数 1 / 偏移 0`` 本身
        就是"我还没改"的确认，空着反而像这里没数据。两处仍共用同一套格式，
        非默认时两句话**逐字相同**。
        """
        loader = getattr(self, "loader", None)
        axis_label = (getattr(loader, "time_axis_label", "") or "Index") if loader else "Index"
        factor = factor or 1.0
        corrected = (
            always_show_correction
            or abs(factor - self._factor_default) > 1e-12
            or abs(offset) > 1e-12
        )
        if not corrected:
            return f"x轴：{axis_label}"
        return f"x轴：{axis_label}（比例系数:{factor:g}, 偏移量:{offset:g}）"

    def _update_axis_segment(self):
        """x 轴段：轴身份取自 loader，被修正过时补上生效的系数与偏移。

        轴标题被 DEFAULT_SHOW_X_AXIS_LABEL=False 关掉了，这一段是全软件唯一
        能看出"横轴是时间还是样本序号、时间基准是多少"的地方。
        """
        loader = getattr(self, "loader", None)
        if loader is None:
            self._axis_segment.hide()
            self._segment_separator.hide()
            return

        self._axis_segment.setText(self._axis_segment_text(self.factor, self.offset))
        self._segment_separator.show()
        self._axis_segment.show()

    def begin_load_feedback(self, file_path: str):
        self._status_busy = True
        self._loading_path = file_path
        self._load_elapsed_started = time.monotonic()
        self._message_hide_timer.stop()
        # 抽屉里的路径/大小/基准属于旧 loader，加载一开始就会被换掉；留着它
        # 等于让用户对着正在失效的快照抄数据
        self._close_status_drawers()
        self._broadcast(f"正在加载 {os.path.basename(file_path)} … 0s")
        self._load_elapsed_timer.start()
        self._set_busy_cursor(True)

    def _tick_load_elapsed(self):
        """秒表只回答"还在干活吗"，不假装知道剩余百分比。"""
        if not self._status_busy:
            self._load_elapsed_timer.stop()
            return
        elapsed = int(time.monotonic() - self._load_elapsed_started)
        name = os.path.basename(self._loading_path or "")
        self._message_text = f"正在加载 {name} … {elapsed}s"
        self._message_label.setText(self._status_elided(self._message_text))

    def end_load_feedback(self, hint: str = "", level: str = "info"):
        self._status_busy = False
        self._load_elapsed_timer.stop()
        self._set_busy_cursor(False)
        if hint:
            self._broadcast(hint, level=level)
        else:
            self._message_hide_timer.stop()
            self.clear_status_message()

    def _init_managers(self):
        from src.ui.file_loader_manager import FileLoaderManager
        from src.ui.cursor_sync_manager import CursorSyncManager
        from src.ui.layout_manager import LayoutManager

        self.file_loader_manager = FileLoaderManager(self)
        self.layout_manager = LayoutManager(self)
        self.cursor_sync_manager = CursorSyncManager(self)

        from src.ui.plot_config_manager import PlotConfigManager
        self.plot_config_manager = PlotConfigManager()

        self._template_menu.addSeparator()
        self._template_menu_act_auto_restore = self._template_menu.addAction("自动恢复")
        self._template_menu_act_auto_restore.setCheckable(True)
        self._template_menu_act_auto_restore.setChecked(
            self.plot_config_manager.auto_save_manager.is_auto_save_enabled()
        )
        self._template_menu_act_auto_restore.triggered.connect(self._on_auto_restore_toggled)
        self._refresh_auto_restore_indicator()

        self._template_settings = AppSettings()
        self._last_template_id = self._template_settings.get_last_template_id()
        self._last_template_name = self._template_settings.get_last_template_name()

        self._log_manager = LogManager.get_instance()
        self._logger = self._log_manager.get_logger("app.main")

        self.file_loader_manager.set_button_status(False)

        # 信号连接（需要 Manager 已初始化）
        self.clone_btn.clicked.connect(self.layout_manager.spawn_clone_window)
        self.help_btn_small.clicked.connect(self.layout_manager.show_help)
        # 按钮不显示，接线照旧：拖变量进图时 variable_actions 靠 setChecked(False)
        # 唤回绘图区，走的就是这一路 toggled
        self.toggle_plot_btn.toggled.connect(self.layout_manager.toggle_plot_area)
        self.grid_layout_btn.clicked.connect(self.layout_manager.open_layout_dialog)
        self._grid_layout_shortcut = QShortcut(QKeySequence("Ctrl+L"), self.plot_widget)
        self._grid_layout_shortcut.activated.connect(self._on_grid_layout_shortcut)
        self.load_btn.clicked.connect(self.file_loader_manager.load_btn_click)
        self.reload_btn.clicked.connect(self.file_loader_manager.reload_data)
        self.clear_all_plots_btn.clicked.connect(self.cursor_sync_manager.clear_all_plots)
        self.auto_range_btn.clicked.connect(self.cursor_sync_manager.auto_range_all_plots)
        self.auto_y_btn.clicked.connect(self.cursor_sync_manager.auto_y_in_x_range)
        self.cursor_btn.clicked.connect(self.cursor_sync_manager.toggle_cursor_all)
        self.mark_region_btn.clicked.connect(self.layout_manager.toggle_mark_region)
        self._crosshair_update_timer.timeout.connect(self.cursor_sync_manager._flush_crosshair_updates)
        self._filter_debounce_timer.timeout.connect(self.cursor_sync_manager.filter_variables)
        self._mark_stats_timer.timeout.connect(self.layout_manager._flush_mark_stats_refresh)

        from src._version import get_version
        self._logger.info(
            "CSV Plot v%s 启动 (Python %s, PySide6 %s)",
            get_version(),
            sys.version.split()[0],
            __import__("PySide6").__version__,
        )

    def _handle_cli_args(self):
        if "--clone-window" in sys.argv:
            return
        positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if positional_args:
            self.file_loader_manager.load_csv_file(positional_args[0])

    def closeEvent(self, event):
        self._logger.info("CSV Plot 应用程序退出")
        # 必须先置位再收尾：下面的清理会开嵌套事件循环（等线程退出、存盘），
        # 期间排着的 singleShot/防抖回调还会打到正在退出的窗口
        self._is_being_destroyed = True
        # 再收抽屉：状态栏抽屉是独立的 Popup 顶层窗口，主窗口销毁时若还开着，
        # 它就会留在屏幕上，鼠标/键盘 grab 也悬在半路
        self._close_status_drawers()
        if self.loader is not None:
            self.plot_config_manager.save_auto_save(self)
        self._shutdown_var_info_worker()
        self.layout_manager._handle_close()
        # 退出前释放 loader 资源（文件句柄、内存映射等）
        try:
            if self.loader is not None:
                if hasattr(self.loader, 'close'):
                    self.loader.close()
        except Exception:
            self._logger.debug("退出时释放 loader 资源失败", exc_info=True)
        super().closeEvent(event)

    def _shutdown_var_info_worker(self):
        """应用退出前终止变量信息窗口的后台统计线程。

        QThread 在运行中被销毁会触发 "QThread: Destroyed while thread is still
        running" 并可能崩溃，因此这里做兜底终止（正常路径由
        VariableInfoDialog.closeEvent 负责）。
        """
        try:
            from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
        except Exception:
            return
        dlg = VariableInfoDialog._live_instance()
        if dlg is None:
            return
        try:
            dlg.shutdown_worker()
        except Exception:
            self._logger.debug("终止变量信息统计线程失败", exc_info=True)

    def show_log_window(self):
        log_window = LogWindow.get_instance(self)
        log_window.show()
        log_window.raise_()
        log_window.activateWindow()
        
    def _on_splitter_moved(self, pos, index):
        self.layout_manager._on_splitter_moved(pos, index)

    def _ensure_splitter_ready(self):
        self.layout_manager._ensure_splitter_ready()

    def _apply_fixed_splitter_width(self):
        self.layout_manager._apply_fixed_splitter_width()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reapply_status_elision()
        self._reposition_status_drawers()
        self.layout_manager._handle_resize(event)
        if sys.platform == "win32" and self.isMaximized():
            if not getattr(self, "_in_sync_resize", False):
                self._in_sync_resize = True
                try:
                    QApplication.processEvents()
                    self.repaint()
                finally:
                    self._in_sync_resize = False

    def moveEvent(self, event):
        super().moveEvent(event)
        # 抽屉是独立顶层窗口，不跟着主窗口走就会浮在原地
        self._reposition_status_drawers()

    def _on_auto_restore_toggled(self, checked):
        self.plot_config_manager.auto_save_manager.set_auto_save_enabled(checked)
        # 开关状态只体现在模板菜单按钮上：标题按 P0-2 的决定只剩
        # 「文件名 - 软件名 版本」，不再挂自动保存状态
        self._refresh_auto_restore_indicator()
        # 那个 ✓ 只在按钮文字上，勾完就收起菜单的人看不见；这里补一句它会
        # 对**下一次加载**做什么，而不是只重复开关名
        if checked:
            self._broadcast("自动恢复已开启：下次加载数据会沿用当前布局")
        else:
            self._broadcast("自动恢复已关闭：加载数据后不再沿用上次布局")

    def _refresh_auto_restore_indicator(self):
        enabled = self.plot_config_manager.auto_save_manager.is_auto_save_enabled()
        if enabled:
            self._template_menu_btn.setText("模板菜单 ✓")
        else:
            self._template_menu_btn.setText("模板菜单")

    def save_current_as_template(self):
        """保存当前配置为模板"""
        if self.loader is None:
            QMessageBox.information(self, "提示", "无可保存的画图配置（请先加载数据并画图）")
            return
        from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog
        config = self.plot_config_manager.export_current_config(self)
        dialog = TemplateEditorDialog(
            self.plot_config_manager.template_manager,
            current_config=config,
            initial_name=self._last_template_name,
            initial_desc=self._last_template_desc,
            parent=self,
        )
        dialog.template_saved.connect(self._on_template_saved)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _on_template_saved(self, template_id: str):
        from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog
        dialog = self.sender()
        summary = "模板已保存"
        if isinstance(dialog, TemplateEditorDialog):
            self._last_template_name = dialog._name_edit.text().strip()
            self._last_template_desc = dialog._desc_edit.text().strip()
            summary = dialog.saved_summary or summary
        self._persist_last_template(template_id, self._last_template_name or "")
        self._show_status_message(
            self._template_saved_text(summary, template_id)
        )

    def _template_saved_text(self, summary: str, template_id: str) -> str:
        """原先播的是 ``template_id``（``uuid4().hex[:8]``，屏上没人读得懂）。

        变量数/子图数与模板管理器列表那两列同一套算法
        （``count_template_variables`` / ``len(config["plots"])``），同一个模板
        不会在两个地方给出两个说法。
        """
        name = self._last_template_name or template_id
        template = self.plot_config_manager.template_manager.get_template(template_id)
        if template is None:
            return f"{summary}：{name}"
        from src.core.template_models import count_template_variables
        plot_count = len(template.config.get("plots", []) or [])
        return (
            f"{summary}：{name} · {count_template_variables(template.config)} 个变量"
            f" / {plot_count} 个子图"
        )

    def _persist_last_template(self, template_id: str, name: str):
        """持久化最后使用的模板信息"""
        self._last_template_id = template_id
        self._last_template_name = name
        self._template_settings.set_last_template_id(template_id)
        self._template_settings.set_last_template_name(name)
        self._template_settings.sync()

    def _refresh_template_menu(self):
        """菜单展开前刷新快速加载项"""
        if self._template_menu_act_quick is not None:
            self._template_menu.removeAction(self._template_menu_act_quick)
            self._template_menu_act_quick.deleteLater()
            self._template_menu_act_quick = None

        tid = self._last_template_id
        name = self._last_template_name
        if tid and name:
            template = self.plot_config_manager.template_manager.get_template(tid)
            if template is not None:
                label = f"应用[{name}]模板"
                self._template_menu_act_quick = QAction(label, self)
                self._template_menu_act_quick.triggered.connect(self._quick_apply_template)
                self._template_menu.insertAction(self._template_menu_act_save, self._template_menu_act_quick)
            else:
                self._last_template_id = None
                self._last_template_name = None
                self._template_settings.set_last_template_id(None)
                self._template_settings.set_last_template_name(None)
                self._template_settings.sync()
    
    def open_template_manager(self):
        """打开模板管理器"""
        from src.ui.dialogs.template_manager_dialog import TemplateManagerDialog
        dialog = TemplateManagerDialog(
            self.plot_config_manager.template_manager,
            parent=self
        )
        dialog.template_applied.connect(self.apply_template)
        try:
            dialog.exec()
        finally:
            # 带 parent 的 exec 对话框在 Python 引用消失后仍作为主窗口子对象
            # 存活，而且与长寿命的 TemplateManager 保持连接：每开一次，之后
            # 一次模板变更就多触发一份 _refresh_template_list
            dialog.deleteLater()
    
    def apply_template(self, template_id: str):
        template = self.plot_config_manager.template_manager.get_template(template_id)
        if not template:
            self._logger.error(f"模板不存在: {template_id}")
            return
        self._persist_last_template(template_id, template.metadata.name)
        self._check_and_apply_template(template, template_id, template.metadata.name)

    def _announce_applied(self, label: str, ratio: float, unmatched) -> None:
        """套用一套通道映射之后的播报，三条路径共用（正常套用 / 强制套用 /
        加载后自动恢复）。

        缺变量就意味着图上有注定空着的格子，所以按 warn（8s）而不是 info
        （5s）：87% 与 100% 在图上看不出差别，只有这一句能告诉人是少了几条
        曲线，而它 5 秒就收走等于没说。
        """
        text = f"{label} · 匹配 {ratio:.0%}"
        if unmatched:
            self._broadcast(f"{text}（{len(unmatched)} 个变量缺失）", level="warn")
        else:
            self._broadcast(text)

    def _announce_cleared_plots(self, label: str, curves: int) -> None:
        """清除绘图之后的播报，三条入口共用（顶部按钮 / 右键菜单 / 双击中键）。

        只在真有曲线被清掉时出声 —— 清一遍本来就没曲线的子图，屏上什么都没变，
        不该占消息区（这条守卫顺带保证文案里的数目 ≥1）。
        """
        if curves <= 0:
            return
        self._broadcast(f"{label} · {curves} 条曲线")

    def _check_and_apply_template(self, template, template_id: str, name: str):
        from src.core.plot_config import PlotSessionConfig
        if self.loader is None:
            QMessageBox.warning(self, "无数据", "请先加载数据文件后再应用模板")
            return
        if self.reject_when_loading("套用模板"):
            return
        config = PlotSessionConfig.from_dict(template.config)
        current_vars = list(self.loader.var_names)
        ratio, matched, unmatched = self.plot_config_manager.check_template_match(
            config, current_vars
        )
        if ratio >= RATIO_RESET_PLOTS:
            success = self.plot_config_manager.apply_config(self, config)
            if success:
                self._logger.info(f"应用模板[{name}]成功，匹配度 {ratio:.0%}")
                self._announce_applied(f"已套用模板[{name}]", ratio, unmatched)
            else:
                QMessageBox.warning(
                    self, "应用失败",
                    f"模板[{name}]应用失败，请检查数据是否已加载。"
                )
        else:
            self._show_match_low_dialog(name, config, ratio, matched, unmatched)

    def _quick_apply_template(self):
        tid = self._last_template_id
        name = self._last_template_name
        if not tid or not name:
            return
        template = self.plot_config_manager.template_manager.get_template(tid)
        if not template:
            self._last_template_id = None
            self._last_template_name = None
            self._template_settings.set_last_template_id(None)
            self._template_settings.set_last_template_name(None)
            self._template_settings.sync()
            # 撤了弹窗改常驻播报：这个分支下面就把快速项从设置里清掉了，
            # 菜单再展开就没有这一项，用户只看到菜单变短 —— 得有个说法
            self._broadcast(f"模板[{name}]已被删除", level="error")
            return
        self._check_and_apply_template(template, tid, name)

    def _show_match_low_dialog(self, name: str, config, ratio: float, matched: set[str], unmatched: set[str]):
        """匹配度过低时弹出详情对话框，允许强制执行"""
        matched_str = ", ".join(sorted(matched)) if matched else "无"
        unmatched_str = ", ".join(sorted(unmatched)) if unmatched else "无"

        msg = (
            f"当前数据的通道与模板[{name}]重合度为 {ratio:.0%}（需 ≥{RATIO_RESET_PLOTS:.0%}）\n\n"
            f"✅ 匹配的变量（{len(matched)} 个）：\n{matched_str}\n\n"
            f"❌ 缺失的变量（{len(unmatched)} 个）：\n{unmatched_str}"
        )

        box = QMessageBox(self)
        box.setWindowTitle("匹配度不足")
        box.setText("模板可能不适用于当前数据文件")
        box.setDetailedText(msg)
        box.setIcon(QMessageBox.Icon.Warning)
        force_btn = box.addButton("仍然加载", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        # 与主窗口同寿命的 QMessageBox 同样是带 parent 的 exec 弹窗，用完即弃
        box.deleteLater()

        if clicked == force_btn:
            self._persist_last_template(self._last_template_id, name)
            self.plot_config_manager.apply_config(self, config)
            self._logger.info(f"强制应用模板[{name}]，匹配度 {ratio:.0%}")
            self._announce_applied(f"已强制套用模板[{name}]", ratio, unmatched)

    def _show_status_message(self, message: str):
        """一次性动作的全局反馈（状态栏右区播报，按级别自动回收）。"""
        self._broadcast(message)
    
    def filter_variables(self):
        """防抖过滤：用户停止输入 180ms 后才真正执行"""
        self._filter_debounce_timer.stop()
        self._filter_debounce_timer.start(180)

    def eventFilter(self, obj, event):
        if not hasattr(self, "layout_manager") or self.layout_manager is None:
            return super().eventFilter(obj, event)
        handled = self.layout_manager._handle_event_filter(obj, event)
        if handled:
            return True
        return super().eventFilter(obj, event)

    def _on_cursor_shortcut(self):
        """Ctrl+R 快捷键：切换光标显示状态"""
        if not self.plot_widgets:
            return
        new_state = not self.cursor_btn.isChecked()
        self.cursor_sync_manager.toggle_cursor_all(new_state)

    def _on_auto_y_shortcut(self):
        """Ctrl+Y 快捷键：自动调节Y轴范围"""
        if not self.plot_widgets:
            return
        self.cursor_sync_manager.auto_y_in_x_range()

    def _on_auto_range_shortcut(self):
        """Ctrl+Shift+Y 快捷键：自动缩放XY轴"""
        if not self.plot_widgets:
            return
        self.cursor_sync_manager.auto_range_all_plots()

    def _on_mark_region_shortcut(self):
        """Ctrl+T 快捷键：切换标记区域"""
        if not self.plot_widgets:
            return
        new_state = not self.mark_region_btn.isChecked()
        self.layout_manager.toggle_mark_region(new_state)

    def _on_open_file_shortcut(self):
        """Ctrl+O 快捷键：加载数据文件"""
        if not self.load_btn.isEnabled():
            return
        self.file_loader_manager.load_btn_click()

    def _on_grid_layout_shortcut(self):
        """Ctrl+L 快捷键：修改图表布局

        守卫条件直接复用 grid_layout_btn.isEnabled()，确保快捷键与
        GUI 按钮激活状态完全一致（未加载数据等场景下按钮被 disabled，
        快捷键同步忽略），避免绕过 set_button_status 的状态控制。
        """
        if not self.grid_layout_btn.isEnabled():
            return
        self.layout_manager.open_layout_dialog()




