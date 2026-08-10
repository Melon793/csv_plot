from __future__ import annotations
import sys
import os

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

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QIcon, QAction, QShortcut, QKeySequence  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit,
    QMessageBox, QSplitter, QMenu, QStyle,
)

SCREEN_WIDTH_MARGIN = 0.3
SCREEN_HEIGHT_MARGIN = 0.3

_widget_logger = get_logger("widget")

class MainWindow(QMainWindow):
    """
    主窗口类
    应用程序的主界面，集成数据加载、图表显示、表格查看等功能
    提供完整的用户交互界面和数据处理流程
    """
    def __init__(self):
        super().__init__()
        self._init_basic_config()
        self._load_window_context()
        self._init_data_state()
        self._init_central_widget()
        self._init_left_panel()
        self._init_right_panel()
        self._init_managers()
        self._handle_cli_args()

    def _init_basic_config(self):
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._drop_event_filter_registered = False
        self.defaultTitle = "数据快速查看器(PySide6), Alpha版本"

        if sys.platform == "darwin":
            if os.path.exists(ico_path):
                app_icon = QIcon(str(ico_path))
                app = QApplication.instance()
                app.setWindowIcon(app_icon)
                self.setWindowIcon(app_icon)

        elif sys.platform == "win32":
            if os.path.exists(ico_path):
                self.setWindowIcon(QIcon(str(ico_path)))

        self.setWindowTitle(self.defaultTitle)
        self._factor_default = 1
        self._offset_default = 0
        self.factor = self._factor_default
        self.offset = self._offset_default
        self._active_drag_container: PlotContainerWidget | None = None

        self._baseline_density: float = 0.0
        self._global_max_density: float = 0.0

    def _load_window_context(self):
        _read_status = False
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
                # hide_plot_area 缺键时回退 False，不纳入 _read_status
                _hide_plot_area = bool(layout_config_dict.get("hide_plot_area", False))
                _read_status = all(
                    x > 0 for x in (
                        _width, _height, _max_row, _max_col, _default_row, _default_col
                    )
                )
                if "hide_plot_area" in layout_config_dict:
                    _widget_logger.debug("layout_config: hide_plot_area=%s", _hide_plot_area)
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
            _hide_plot_area = False

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

        self._data_version = 0
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self._crosshair_update_timer.setSingleShot(True)

        self._filter_debounce_timer = QTimer(self)
        self._filter_debounce_timer.setSingleShot(True)

        self.data_table_geometry = None
        self.mark_stats_geometry = None
        self.time_correction_geometry = None
        self._mark_stats_dirty = False
        self._mark_stats_timer = QTimer(self)
        self._mark_stats_timer.setSingleShot(True)
        self._is_syncing_crosshair = False
        self._is_syncing_mark_region = False
        self._last_template_name = ""
        self._last_template_desc = ""

        self.value_cache = {}
        self._enum_text_maps: dict = {}

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

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(2)
        self.log_btn = QPushButton("日志", left_widget)
        self.log_btn.clicked.connect(self.show_log_window)
        bottom_row.addWidget(self.log_btn)

        self.toggle_plot_btn = QPushButton("隐藏绘图区", left_widget)
        self.toggle_plot_btn.setCheckable(True)
        bottom_row.addWidget(self.toggle_plot_btn)
        left_layout.addLayout(bottom_row)
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

        self.time_correction_btn = QPushButton("时间修正", self.plot_widget)
        top_bar.addWidget(self.time_correction_btn)

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

        self.cursor_btn = QPushButton("显示光标", self.plot_widget)
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setToolTip("切换光标显示 (Ctrl+R)")

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
        self.toggle_plot_btn.toggled.connect(self.layout_manager.toggle_plot_area)
        self.time_correction_btn.clicked.connect(self.layout_manager.open_time_correction_dialog)
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

        self._logger.info(
            "CSV Plot 启动 (Python %s, PySide6 %s)",
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
        if self.loader is not None:
            self.plot_config_manager.save_auto_save(self)
        self.layout_manager._handle_close()
        super().closeEvent(event)

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
        self.layout_manager._handle_resize(event)
        if sys.platform == "win32" and self.isMaximized():
            if not getattr(self, "_in_sync_resize", False):
                self._in_sync_resize = True
                try:
                    QApplication.processEvents()
                    self.repaint()
                finally:
                    self._in_sync_resize = False

    def _on_auto_restore_toggled(self, checked):
        self.plot_config_manager.auto_save_manager.set_auto_save_enabled(checked)
        self._refresh_auto_restore_indicator()

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
        dialog.exec()

    def _on_template_saved(self, template_id: str):
        from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog
        dialog = self.sender()
        if isinstance(dialog, TemplateEditorDialog):
            self._last_template_name = dialog._name_edit.text().strip()
            self._last_template_desc = dialog._desc_edit.text().strip()
        self._persist_last_template(template_id, self._last_template_name or "")
        self._show_status_message(f"模板已保存: {template_id}")

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
        dialog.exec()
    
    def apply_template(self, template_id: str):
        template = self.plot_config_manager.template_manager.get_template(template_id)
        if not template:
            self._logger.error(f"模板不存在: {template_id}")
            return
        self._persist_last_template(template_id, template.metadata.name)
        self._check_and_apply_template(template, template_id, template.metadata.name)

    def _check_and_apply_template(self, template, template_id: str, name: str):
        from src.core.plot_config import PlotSessionConfig
        if self.loader is None:
            QMessageBox.warning(self, "无数据", "请先加载数据文件后再应用模板")
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
            QMessageBox.warning(self, "模板不存在", f"模板[{name}]已被删除")
            self._last_template_id = None
            self._last_template_name = None
            self._template_settings.set_last_template_id(None)
            self._template_settings.set_last_template_name(None)
            self._template_settings.sync()
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

        if box.clickedButton() == force_btn:
            self._persist_last_template(self._last_template_id, name)
            self.plot_config_manager.apply_config(self, config)
            self._logger.info(f"强制应用模板[{name}]，匹配度 {ratio:.0%}")

    def _show_status_message(self, message: str):
        """显示状态消息（在未来可以添加状态栏）"""
        self._logger.info(message)
    
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




