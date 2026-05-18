"""
MainWindowUIManager - 主窗口 UI 初始化与事件过滤管理器

负责 MainWindow 的所有 UI 初始化工作：
- 中央控件、分割器、左右面板创建
- 顶部按钮栏与控件布局
- 全局拖拽事件过滤器安装与业务逻辑
- 窗口缩放、绘图区显示/隐藏、分身窗口等 UI 行为

此模块从 csv_plot_pyqt6.py 迁移而来。
使用 weakref 引用 MainWindow，避免循环引用。
"""

from __future__ import annotations
import weakref
import sys
import os
import subprocess
from typing import Any

from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QLabel, QLineEdit, QPushButton, QMessageBox,
)

from src.core.config import debug_log
from src.ui.table_dialog import DropOverlay
from src.ui.variable_list import MyTableWidget
from src.ui.dialogs.help import HelpDialog
from src.ui.widgets.plot_container import PlotContainerWidget


class MainWindowUIManager:
    """负责 MainWindow UI 初始化和事件过滤器业务逻辑的管理器"""

    def __init__(self, main_window: Any):
        self._mw_ref = weakref.ref(main_window)

    @property
    def _mw(self) -> Any:
        mw = self._mw_ref()
        if mw is None:
            raise RuntimeError("MainWindow has been garbage collected")
        return mw

    # ========================================================================
    # UI 初始化入口
    # ========================================================================

    def setup_ui(self, config_dict: dict, _hide_plot_area: bool) -> None:
        """初始化主窗口所有 UI 组件

        Args:
            config_dict: 布局配置字典
            _hide_plot_area: 是否初始隐藏绘图区
        """
        mw = self._mw

        # ---------------- 中央控件 ----------------
        central = QWidget()
        mw.setCentralWidget(central)

        # ========== 主布局：可调整分界线 ==========
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        mw.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        mw.main_splitter.setHandleWidth(5)
        mw.main_splitter.setChildrenCollapsible(False)

        mw.var_table_default_width = 280
        mw.var_table_user_adjusted = False

        mw.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        mw._splitter_ready = False
        mw._pending_splitter_adjustment = False

        # ---------------- 左侧变量列表 ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 0, 5, 0)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        left_layout_title = QLabel("变量列表")
        font = left_layout_title.font()
        font.setBold(True)
        left_layout_title.setFont(font)
        title_layout.addWidget(left_layout_title)

        title_layout.addStretch(1)

        mw.clone_btn = QPushButton("分身")
        mw.clone_btn.setToolTip("启动独立实例")
        mw.clone_btn.clicked.connect(self.spawn_clone_window)
        title_layout.addWidget(mw.clone_btn)
        mw.clone_btn.setVisible(True)

        mw.help_btn_small = QPushButton("?")
        mw.help_btn_small.setToolTip("帮助文档")
        mw.help_btn_small.clicked.connect(self.show_help)
        title_layout.addWidget(mw.help_btn_small)

        left_layout.addLayout(title_layout)

        mw.filter_input = QLineEdit()
        mw.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        mw.filter_input.textChanged.connect(mw.filter_variables)
        left_layout.addWidget(mw.filter_input)

        mw.unit_filter_input = QLineEdit()
        mw.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        mw.unit_filter_input.setContentsMargins(60, 0, 0, 0)
        mw.unit_filter_input.textChanged.connect(mw.filter_variables)
        left_layout.addWidget(mw.unit_filter_input)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)

        mw.load_btn = QPushButton("导入数据文件")
        mw.load_btn.clicked.connect(mw.load_btn_click)

        mw.reload_btn = QPushButton("重载")
        mw.reload_btn.clicked.connect(mw.reload_data)

        button_layout.addWidget(mw.load_btn, 4)
        button_layout.addWidget(mw.reload_btn, 1)

        left_layout.addLayout(button_layout)
        mw.list_widget = MyTableWidget()
        left_layout.addWidget(mw.list_widget)

        mw.toggle_plot_btn = QPushButton("隐藏绘图区")
        mw.toggle_plot_btn.setCheckable(True)
        mw.toggle_plot_btn.toggled.connect(self.toggle_plot_area)
        left_layout.addWidget(mw.toggle_plot_btn)
        left_layout.setSpacing(2)
        mw.left_widget = left_widget

        mw._plot_area_visible = True
        mw._saved_geometry = None

        # ---------------- 右侧绘图区 ----------------
        mw.plot_widget = QWidget()
        root_layout = QVBoxLayout(mw.plot_widget)

        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 5, 5)

        mw.time_correction_btn = QPushButton("时间修正")
        mw.time_correction_btn.clicked.connect(mw.open_time_correction_dialog)
        top_bar.addWidget(mw.time_correction_btn)

        mw.clear_all_plots_btn = QPushButton("清除绘图")
        mw.clear_all_plots_btn.clicked.connect(mw.clear_all_plots)
        top_bar.addWidget(mw.clear_all_plots_btn)

        top_bar.addStretch(1)

        mw.auto_range_btn = QPushButton("自动缩放")
        mw.auto_range_btn.clicked.connect(mw.auto_range_all_plots)

        mw.auto_y_btn = QPushButton("仅调节y轴")
        mw.auto_y_btn.clicked.connect(mw.auto_y_in_x_range)

        mw.cursor_btn = QPushButton("显示光标")
        mw.cursor_btn.setCheckable(True)
        mw.cursor_btn.clicked.connect(mw.toggle_cursor_all)

        mw.cursor_values_hidden = False
        mw.cursor_mode = "1 free cursor"
        mw.pinned_x_values = []

        mw.mark_region_btn = QPushButton("标记区域")
        mw.mark_region_btn.setCheckable(True)
        mw.mark_region_btn.clicked.connect(mw.toggle_mark_region)

        mw.grid_layout_btn = QPushButton("修改布局")
        mw.grid_layout_btn.clicked.connect(mw.open_layout_dialog)

        self.set_button_status(False)

        top_bar.addWidget(mw.grid_layout_btn)
        top_bar.addWidget(mw.cursor_btn)
        top_bar.addWidget(mw.mark_region_btn)
        top_bar.addWidget(mw.auto_y_btn)
        top_bar.addWidget(mw.auto_range_btn)

        root_layout.addLayout(top_bar)

        mw.plot_layout = QGridLayout()
        mw.plot_layout.setContentsMargins(0, 0, 0, 0)
        mw.plot_layout.setSpacing(0)
        root_layout.addLayout(mw.plot_layout, 1)

        mw.main_splitter.addWidget(left_widget)
        mw.main_splitter.addWidget(mw.plot_widget)

        mw.main_splitter.setSizes([mw.var_table_default_width, 800])

        mw.main_splitter.setStretchFactor(0, 0)
        mw.main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(mw.main_splitter)
        QTimer.singleShot(0, self._ensure_splitter_ready)
        QTimer.singleShot(0, self._ensure_splitter_ready)

        # ---------------- 子图 ----------------
        mw.plot_widgets = []

        mw.placeholder_label = QLabel("请导入 CSV 文件以查看数据", mw.plot_widget)
        mw.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mw.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        mw.plot_layout.addWidget(mw.placeholder_label, 0, 0)

        mw.drop_overlay = DropOverlay(mw.centralWidget())
        mw.drop_overlay.lower()
        mw.drop_overlay.hide()

        app = QApplication.instance()
        if app:
            app.installEventFilter(mw)
            mw._drop_event_filter_registered = True

        if _hide_plot_area:
            mw.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            mw.show()
            _geometry = mw.geometry()

            mw.toggle_plot_btn.setChecked(True)
            mw.toggle_plot_btn.setText("显示绘图区")
            mw._plot_area_visible = False

            mw.plot_widget.hide()

            left_width = mw.left_widget.width()
            main_margin = mw.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            frame_width = mw.frameGeometry().width() - mw.width()
            new_width = left_width + frame_width

            mw.setFixedWidth(new_width)
            mw.move(_geometry.topLeft())
            mw.close()
            mw.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)

            mw._old_max_width = mw._window_width_default

    # ========================================================================
    # 分割器管理
    # ========================================================================

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        mw = self._mw
        mw.var_table_user_adjusted = True
        mw._splitter_ready = True

        sizes = mw.main_splitter.sizes()
        if len(sizes) >= 1:
            mw.var_table_default_width = sizes[0]

    def _ensure_splitter_ready(self) -> None:
        mw = self._mw
        if not hasattr(mw, 'main_splitter'):
            return
        sizes = mw.main_splitter.sizes()
        if len(sizes) >= 2 and all(size > 0 for size in sizes):
            mw._splitter_ready = True
        else:
            QTimer.singleShot(50, self._ensure_splitter_ready)

    def _apply_fixed_splitter_width(self) -> None:
        mw = self._mw
        mw._pending_splitter_adjustment = False
        if (mw.var_table_user_adjusted
                or not getattr(mw, '_splitter_ready', False)
                or not hasattr(mw, 'main_splitter')):
            return

        sizes = mw.main_splitter.sizes()
        if len(sizes) < 2:
            return

        total_width = sum(sizes)
        if total_width <= 0 or total_width <= mw.var_table_default_width:
            return

        right_width = max(total_width - mw.var_table_default_width, 0)
        if right_width <= 0:
            return

        mw.main_splitter.blockSignals(True)
        mw.main_splitter.setSizes([mw.var_table_default_width, right_width])
        mw.main_splitter.blockSignals(False)

    # ========================================================================
    # 窗口大小调整
    # ========================================================================

    def handle_resize(self, event: Any) -> None:
        mw = self._mw
        if (not mw.var_table_user_adjusted
                and getattr(mw, '_splitter_ready', False)
                and hasattr(mw, 'main_splitter')):
            if not getattr(mw, '_pending_splitter_adjustment', False):
                mw._pending_splitter_adjustment = True
                QTimer.singleShot(0, self._apply_fixed_splitter_width)

    # ========================================================================
    # 绘图区显示/隐藏
    # ========================================================================

    def toggle_plot_area(self, checked: bool) -> None:
        mw = self._mw
        if checked:
            mw._saved_geometry = mw.saveGeometry()
            mw.plot_widget.hide()
            mw.toggle_plot_btn.setText("显示绘图区")

            mw._old_max_width = mw.maximumWidth()
            left_width = mw.left_widget.width()
            main_margin = mw.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            frame_width = mw.frameGeometry().width() - mw.width()
            new_width = left_width + frame_width
            mw.setFixedWidth(new_width)
            mw._plot_area_visible = False
        else:
            mw.setMaximumWidth(mw._old_max_width)
            mw.setMinimumWidth(0)
            mw.plot_widget.show()
            mw.toggle_plot_btn.setText("隐藏绘图区")
            if mw._saved_geometry:
                mw.restoreGeometry(mw._saved_geometry)
            mw._plot_area_visible = True

    # ========================================================================
    # 帮助对话框
    # ========================================================================

    def show_help(self) -> None:
        dlg = HelpDialog(self._mw)
        dlg.exec()

    # ========================================================================
    # 拖拽指示器
    # ========================================================================

    def _get_plot_container(self, plot_widget: Any) -> PlotContainerWidget | None:
        parent = plot_widget.parentWidget()
        if isinstance(parent, PlotContainerWidget):
            return parent
        return None

    def _show_drag_indicator_for_plot(
        self,
        plot_widget: Any,
        var_names: list[str],
        text_override: str | None = None,
    ) -> None:
        mw = self._mw
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        if mw._active_drag_container and mw._active_drag_container is not container:
            mw._active_drag_container.hide_drag_indicator()
        container.show_drag_indicator(var_names, text_override)
        mw._active_drag_container = container

    def _hide_drag_indicator_for_plot(self, plot_widget: Any) -> None:
        mw = self._mw
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        container.hide_drag_indicator()
        if mw._active_drag_container is container:
            mw._active_drag_container = None

    # ========================================================================
    # 分身窗口
    # ========================================================================

    def spawn_clone_window(self) -> None:
        try:
            if getattr(sys, "frozen", False):
                args = [sys.executable]
            else:
                script_path = os.path.abspath(sys.argv[0])
                args = [sys.executable, script_path]

            if sys.platform == "win32":
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                    ),
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as e:
            QMessageBox.warning(self._mw, "错误", f"启动独立实例失败: {e}")

    # ========================================================================
    # 按钮状态控制
    # ========================================================================

    def set_button_status(self, status: bool) -> None:
        mw = self._mw
        if status is not None:
            mw.time_correction_btn.setEnabled(status)
            mw.clear_all_plots_btn.setEnabled(status)
            mw.auto_range_btn.setEnabled(status)
            mw.auto_y_btn.setEnabled(status)
            mw.cursor_btn.setEnabled(status)
            mw.mark_region_btn.setEnabled(status)
            mw.grid_layout_btn.setEnabled(status)

    # ========================================================================
    # 全局事件过滤器
    # ========================================================================

    def _unregister_global_event_filter(self) -> None:
        mw = self._mw
        if not getattr(mw, "_drop_event_filter_registered", False):
            return
        app = QApplication.instance()
        if app:
            app.removeEventFilter(mw)
        mw._drop_event_filter_registered = False

    def event_filter(self, obj: Any, event: Any) -> bool:
        mw = self._mw
        if not isinstance(obj, QWidget):
            return False
        if obj.window() is not mw:
            return False
        etype = event.type()
        if etype == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                supported = any(
                    u.toLocalFile().lower().endswith(
                        ('.csv', '.txt', '.mfile', '.t00', '.t01', '.t10', '.t11')
                    )
                    or mw._extract_file_extension(u.toLocalFile()) is not None
                    for u in urls
                )

                if supported:
                    self.show_drop_overlay()
                    mw.drop_overlay.adjust_text(file_type_supported=True)
                    event.acceptProposedAction()
                    return True
                else:
                    self.show_drop_overlay()
                    mw.drop_overlay.adjust_text(file_type_supported=False)
                    event.ignore()
                    return True
        elif etype == QEvent.Type.DragLeave:
            self.hide_drop_overlay()
            return True
        elif etype == QEvent.Type.DragMove:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                supported = any(
                    u.toLocalFile().lower().endswith(
                        ('.csv', '.txt', '.mfile', '.t00', '.t01', '.t10', '.t11')
                    )
                    or mw._extract_file_extension(u.toLocalFile()) is not None
                    for u in urls
                )
                if supported:
                    event.acceptProposedAction()
                    return True
        elif etype == QEvent.Type.Drop:
            self.hide_drop_overlay()
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                for u in urls:
                    path = u.toLocalFile()
                    if (path.lower().endswith(('.csv', '.txt', '.mfile', '.t00', '.t01', '.t10', '.t11'))
                            or mw._extract_file_extension(path) is not None):
                        debug_log("MainWindow.eventFilter drop load path=%s", path)
                        mw.load_csv_file(path)
                        event.accept()
                        return True
        return False

    def show_drop_overlay(self) -> None:
        mw = self._mw
        mw.drop_overlay.setGeometry(mw.centralWidget().rect())
        mw.drop_overlay.raise_()
        mw.drop_overlay.show()
        mw.drop_overlay.activateWindow()

    def hide_drop_overlay(self) -> None:
        self._mw.drop_overlay.hide()
