from __future__ import annotations
import sys
import os
import weakref
import subprocess
from pathlib import Path
from threading import Lock
from typing import Any

if sys.platform == "darwin":  # macOS
    # 屏蔽 macOS ICC 警告
    pass

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

from src.ui.drag_drop import VAR_SEPARATOR, parse_var_names_from_mimedata
from src.ui.widgets.custom_viewbox import CustomViewBox
from src.core.config import (safe_callback, DEFAULT_PADDING_VAL_X, DEFAULT_PADDING_VAL_Y, XRANGE_THRESHOLD_FOR_SYMBOLS, FACTOR_SCROLL_ZOOM, MIN_INDEX_LENGTH, DEFAULT_LINE_WIDTH, THICK_LINE_WIDTH, THIN_LINE_WIDTH, UI_DEBOUNCE_DELAY_MS, PLOT_ROW_MAX_DEFAULT, PLOT_COL_MAX_DEFAULT, PLOT_ROW_CURRENT_DEFAULT, PLOT_COL_CURRENT_DEFAULT, _evaluate_float32_safety, DEFAULT_SHOW_X_AXIS_LABEL, RATIO_RESET_PLOTS)
from src.core.data_types import CurveInfo, MarkStatEntry
from src.core.scheduler import UnifiedUpdateScheduler
from src.ui.table_dialog import DataTableDialog, DropOverlay
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from src.ui.variable_list import MyTableWidget

from src.core.logger import LogManager, get_logger
from src.ui.dialogs.log_window import LogWindow


from PySide6.QtCore import Qt, QMargins, QTimer, QPoint, QPointF, QSize, QRect, QRectF, QItemSelectionModel, QSignalBlocker
from PySide6.QtGui import QFontMetrics, QPen, QColor, QIcon, QFont, QCursor, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QToolButton, QAbstractItemView, QLabel, QLineEdit,
    QMessageBox, QSizePolicy, QGraphicsLinearLayout, QGraphicsProxyWidget, QGraphicsWidget, QRubberBand, QSplitter, QMenu,
)
import pyqtgraph as pg


# PyInstaller 解包目录 — 已迁移至 src/utils/paths.py
from src.utils.paths import resource_path

# 设置应用程序和窗口图标
if sys.platform == "win32": # Windows
    import ctypes
    myappid = 'mycompany.csv_plot.0.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    ico_path = resource_path("assets/icon.ico")  

elif sys.platform == "darwin":  # macOS
    ico_path = resource_path("assets/icon.icns")  

_widget_logger = get_logger("widget")


class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    """
    可拖拽的图形布局控件类
    支持图表区域的拖拽重排和动态布局调整
    提供灵活的图表排列和交互功能
    """
    def __init__(self, units_dict, dataframe, time_channels_info=None, synchronizer=None):
        if time_channels_info is None:
            time_channels_info = {}
        import numpy as np
        import pandas as pd
        globals()['np'] = np
        globals()['pd'] = pd
        super().__init__()
        self.factor = 1.0
        self.offset = 0.0
        self.original_index_x = None
        self.original_y = None
        self.mark_region = None
        self.is_cursor_pinned = False  # 记录cursor是否被固定
        self.pinned_x_value = None  # 记录固定的x值
        self.pinned_index_value = None  # 记录固定的索引值
        self.pinned_x_values = []
        self.pinned_index_values = []
        self._is_updating_data = False  # 标志：正在更新数据，禁止某些操作
        self._is_being_destroyed = False  # 标志：对象正在被销毁
        self._suppress_pin_update = False  # 标志：临时禁止pin状态自动更新
        self._cursor_label_busy = False
        self._cursor_label_dirty = False
        self._cached_data_version = 0  # 【稳定性优化】缓存的数据版本号
        self._pending_delete_items = []  # 【稳定性优化】待删除对象队列
        self._drag_indicator_source = None
        self._drag_indicator_guard = QTimer(self)
        self._drag_indicator_guard.setInterval(120)
        self._drag_indicator_guard.timeout.connect(self._enforce_drag_indicator_visibility)
        # 【稳定性优化】安全删除timer
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.timeout.connect(self._process_pending_deletes)
        self.plot_context = None  # 由 layout_manager 赋值为 PlotContext 实例
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info=None, synchronizer=None):
        if time_channels_info is None:
            time_channels_info = {}
        """
        初始化UI组件和布局
        
        设置图形布局控件的基本配置和数据结构
        初始化绘图相关的属性和同步器
        
        Args:
            units_dict: 单位字典
            dataframe: 数据框
            time_channels_info: 时间通道信息
            synchronizer: 同步器实例
        """
        # 设置大小策略，允许拉伸
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.setAcceptDrops(True)
        self.units = units_dict
        self.data = dataframe
        self.time_channels_info = time_channels_info
        self.synchronizer = synchronizer
        self.curve = None
        self.time_column_name = None
        self.time_axis_label = "Index"
        #self.ci.layout.setContentsMargins(0, 0, 0, 5)
        
        # 多曲线支持
        self.curves = {}  # 存储所有曲线 {var_name: curve_info}
        self.is_multi_curve_mode = False  # 是否处于多曲线模式
        self._batch_adding = False  # 是否正在批量添加变量
        self.curve_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 默认颜色列表
        self.current_color_index = 0  # 当前颜色索引
        self._max_point_density: float = 0.0  # 当前 plot 所有 curve 的最大数据点密度
        
        self.y_name = ''
        self.y_format = ''
        self.x_name = ''
        self.x_format = ''

        self.xMin:int =0 
        self.xMax:int =1 
        # 添加顶部文本区域
        self.setup_header()
        # 主绘图区域设置
        self.setup_plot_area()
        # 坐标轴设置
        self.setup_axes()
        # 交互元素设置
        self.setup_interaction()
        self._init_ui_refresh_coordinator()

        # 布局比例设置 (绘图区占90%)
        self.ci.layout.setContentsMargins(0, 0, 10, 5)  # 消除所有边距
        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowStretchFactor(1, 1)  # 主区域完全拉伸

        # 初始化框选功能
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()

        self._init_manager_chain()

    def _init_manager_chain(self):
        from src.ui.widgets.plot_ui_manager import PlotUIManager
        from src.ui.widgets.axis_manager import AxisManager
        from src.ui.widgets.plot_data_manager import PlotDataManager
        from src.ui.widgets.multi_curve_manager import MultiCurveManager
        from src.ui.widgets.cursor_manager import CursorManager
        from src.ui.widgets.mark_region_manager import MarkRegionManager
        from src.ui.widgets.event_handler import EventHandler

        self._plot_ui_manager = PlotUIManager(self)
        self._axis_manager = AxisManager(self._plot_ui_manager)
        self._plot_data_manager = PlotDataManager(self._axis_manager)
        self._multi_curve_manager = MultiCurveManager(self._plot_data_manager)
        self._cursor_manager = CursorManager(self._multi_curve_manager)
        self._mark_region_manager = MarkRegionManager(self._cursor_manager)
        self._event_handler = EventHandler(self._mark_region_manager)

    @property
    def curve_strategy(self):
        from src.core.curve_strategy import SingleCurveStrategy, MultiCurveStrategy
        if self.is_multi_curve_mode and self.curves:
            return MultiCurveStrategy(self)
        return SingleCurveStrategy(self)

    def setup_header(self):
        """完全修正的顶部文本区域设置方法"""
        header = pg.GraphicsWidget()
        layout = QGraphicsLinearLayout(Qt.Orientation.Horizontal)
        

        # 计算固定Y轴宽度
        font = QApplication.font()
        fm = QFontMetrics(font)
        base_spacing=fm.horizontalAdvance("-10000.01")
        header.setFixedHeight(fm.height() * 2) 

        # 添加左边距（空项）
        left_margin = QGraphicsWidget()        
        layout.addItem(left_margin)
        layout.setItemSpacing(0, base_spacing*0) 
        
        # 左侧文本（使用代理窗口部件）
        self.label_left = QLabel("channel name")
        self.label_left.setStyleSheet("""
            color: #000;
            font-weight: bold;
            background-color: transparent;
        """)
        self.label_left.setSizePolicy(QSizePolicy.Policy.Minimum,
                                      QSizePolicy.Policy.Preferred)
        self.label_left.setTextFormat(Qt.TextFormat.RichText)  # 支持HTML格式
        self.label_left.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)  # 支持交互
        self.label_left.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)  # 禁用右键菜单
        self.label_left.mousePressEvent = self._on_legend_clicked  # 绑定点击事件
        #self.label_left.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(self.label_left)

        # 只添加左侧文本到布局
        layout.addItem(proxy_left)
        layout.setStretchFactor(proxy_left, 1)
        layout.setAlignment(proxy_left, Qt.AlignmentFlag.AlignBottom| Qt.AlignmentFlag.AlignLeft)

        #layout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        header.setLayout(layout)
        #header.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.addItem(header, row=0, col=0, colspan=2)

    def setup_plot_area(self):
        """
        配置绘图区域基本属性
        
        创建和配置主要的绘图区域
        设置视图框、坐标轴和基本绘图属性
        
        性能优化（基于iOS/Android浏览器缩放优化经验）：
        1. 智能降采样（peak模式保留峰值）
        2. 视图裁剪（只渲染可见区域）
        3. 交互期间性能降级（类似iOS快照技术）
        4. 智能防抖延迟（根据数据量动态调整）
        """
        self.plot_item = self.addPlot(row=1, col=0, colspan=2, viewBox=CustomViewBox())
        self.view_box = self.plot_item.vb
        self.view_box.plot_widget = self
        self._connect_viewbox_signals()
        
        # ========== 性能优化 2: 交互状态管理 ==========
        self._is_interacting = False
        self._interaction_timer = QTimer()
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.timeout.connect(self._end_interaction)
        
        # ========== 性能优化 3.1: 同步缩放标志 ==========
        # 防止XLink同步时递归更新导致的性能问题
        self._is_syncing_range = False  # 标记是否正在同步范围（避免递归更新）
        
        # 移除 self._customize_plot_menu()，因为现在用 CustomViewBox 实现菜单定制
        
        self.view_box.setAutoVisible(x=False, y=True)  # 自动适应可视区域
        self.plot_item.setTitle(None)
        self.plot_item.hideButtons()
        
        # ========== 性能优化 3: 视图裁剪和降采样 ==========
        # 类似网页的懒加载和虚拟化技术
        self.plot_item.setClipToView(True)  # 只渲染可见区域
        # 使用peak模式保留峰值，自动降采样支持百万级数据点
        # 当auto=True时，pyqtgraph会根据可见区域自动计算合适的降采样因子
        # 无需指定ds参数，auto模式会自动处理
        self.plot_item.setDownsampling(mode='peak', auto=True)
        
        self.setBackground('w')

        pen = pg.mkPen('#f00',width=1)
        self.plot_item.getAxis('left').setGrid(255) 
        self.plot_item.getAxis('bottom').setGrid(255) 
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

        # 基于点数修改曲线风格
        # 使用防抖机制来优化缩放性能
        self.view_box.sigRangeChanged.connect(self._on_range_changed)

        self.update_x_axis_label()

    def update_x_axis_label(self):
        """更新 X 轴标签文本 → 委托到 AxisManager（初始化阶段 fallback 到内联实现）"""
        if not hasattr(self, '_axis_manager'):
            axis = self.plot_item.getAxis('bottom')
            if DEFAULT_SHOW_X_AXIS_LABEL:
                label = self.time_axis_label if self.time_axis_label else "Index"
                axis.setLabel(label)
                axis.showLabel(True)
            else:
                axis.showLabel(False)
            return
        self._axis_manager.update_x_axis_label()
        
    def jump_to_data_impl(self, x):
        strategy = self.curve_strategy
        if not strategy.has_data():
            return

        var_names = strategy.get_curve_names()

        main_window = self.window()
        if not hasattr(main_window, 'loader') or main_window.loader is None:
            return

        # a. 打开/激活数值变量表，并添加所有变量
        is_mdf_loader = getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
        dlg = None
        for var_name in var_names:
            if is_mdf_loader:
                try:
                    series = main_window.loader.get_series(var_name)
                except KeyError:
                    continue
            elif var_name not in main_window.loader.df.columns:
                continue
            else:
                series = main_window.loader.df[var_name]
            dlg = DataTableDialog.popup(var_name, series, parent=main_window)

        # 如果没有成功打开任何dialog，直接返回
        if dlg is None:
            return

        # 判断"数值变量表"窗口是否被最小化了，如果是，则恢复正常状态
        if dlg.isMinimized():
            dlg.showNormal()
            
        # b. popup 已处理：如果已在（冻结或非冻结），不添加；否则添加到非冻结区域

        # c. 计算行索引（0-based）
        if self.factor == 0:
            return  # 避免除零

        index = (x - self.offset) / self.factor
        index = int(round(index)) - 1  # 转换为 0-based 行索引
        index = max(0, min(index, main_window.loader.datalength - 1))

        # 使用第一个变量来定位和选中
        first_var_name = var_names[0]
        
        # 获取模型和列索引
        model = dlg.model
        col_idx = dlg._df.columns.get_loc(first_var_name)  # 逻辑列索引

        # 确定使用哪个视图（冻结或主视图）
        if first_var_name in dlg.frozen_columns:
            view = dlg.frozen_view
        else:
            view = dlg.main_view

        # 获取视觉列索引（因为列可拖动）
        header = view.horizontalHeader()
        visual_col = header.visualIndex(col_idx)

        # 创建 QModelIndex
        qindex = model.index(index, col_idx)

        # 跳转并居中，使用 QTimer 确保在窗口显示后执行
        QTimer.singleShot(0, lambda: view.scrollTo(qindex, QAbstractItemView.ScrollHint.PositionAtCenter))

        # 选中该单元格
        QTimer.singleShot(0, lambda: view.selectionModel().select(qindex, QItemSelectionModel.SelectionFlag.ClearAndSelect))

    def auto_range(self, external_xmin: float | None = None, external_xmax: float | None = None):
        """自动调整视图范围 → 委托到 AxisManager"""
        return self._axis_manager.auto_range(external_xmin, external_xmax)

    def auto_y_in_x_range(self):
        """在当前 X 范围内自动调整 Y 轴 → 委托到 AxisManager"""
        self._axis_manager.auto_y_in_x_range()

    def update_left_header(self, left_text=None):
        """更新顶部文本内容"""
        if left_text is not None:
            self.label_left.setText(left_text)

    def update_right_header(self, right_text=None):
        """更新顶部文本内容（已移除右侧label）"""
        # 右侧label已被移除，此方法保留以兼容现有代码
        pass

    def _get_safe_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        """确保 X 轴范围非零 → 委托到 AxisManager"""
        return self._axis_manager._get_safe_x_range(min_x, max_x)

    def _get_min_x_range_value(self) -> float:
        """计算最小的可缩放 X 范围 → 委托到 AxisManager"""
        return self._axis_manager._get_min_x_range_value()

    def _set_x_limits_with_min_range(self, limits_xMin: float | None, limits_xMax: float | None):
        """统一设置 X 轴的 limits 和 minXRange → 委托到 AxisManager"""
        self._axis_manager._set_x_limits_with_min_range(limits_xMin, limits_xMax)

    def _set_min_x_range(self, minXRange: float):
        """设置 X 轴的最小范围 → 委托到 AxisManager"""
        self._axis_manager._set_min_x_range(minXRange)

    def _recalc_max_point_density(self):
        """重新计算最大数据点密度 → 委托到 AxisManager"""
        self._axis_manager._recalc_max_point_density()

    def _set_safe_y_range(self, min_y: float, max_y: float, set_limits: bool = True):
        """设置 Y 轴的 viewRange 和 limits → 委托到 AxisManager"""
        self._axis_manager._set_safe_y_range(min_y, max_y, set_limits)

    def reset_plot(self, index_xMin, index_xMax):
        """重置绘图 → 委托到 PlotDataManager"""
        self._plot_data_manager.reset_plot(index_xMin, index_xMax)


    def setup_axes(self):
        """配置坐标轴样式和范围"""
        # X轴配置
        self.axis_x = self.plot_item.getAxis('bottom')
        self.axis_x.setTextPen('black')
        self.axis_x.setPen(QPen(QColor('black'), 1))
        self.axis_x.setRange(0, 10)
        
        # Y轴配置
        self.axis_y = self.plot_item.getAxis('left')
        self.axis_y.enableAutoSIPrefix(False)
        self.axis_y.setTextPen('black')
        self.axis_y.setPen(QPen(QColor('black'), 1))
        

        # 其他边框配置
        for pos in ('top', 'right'):
            ax = self.plot_item.getAxis(pos)
            ax.setVisible(True)
            ax.setTicks([])
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(QPen(QColor('black'), 1))
        
        # 计算固定Y轴宽度
        font = QApplication.font()
        fm = QFontMetrics(font)
        self.axis_y.setWidth(fm.horizontalAdvance("-10000.01") )

        # 基于应用程序基础字体大小，增加2像素作为标签字体大小
        font_family = font.family() 
        # 使用 font.pixelSize() 保证跨平台一致性，并略微增大
        pixel_size = font.pixelSize() + 2

        # Y轴标签
        # self.axis_y.setLabel(
        #     color='black',
        #     angle=-90,
        #     **{'font-family': 'Arial', 'font-size': '12pt', 'font-weight': 'bold'}
        # )
        self.axis_y.setLabel(
            color='black',
            angle=-90,
            # 修正：使用像素大小 'px' 代替点大小 'pt'，并使用系统字体
            **{'font-family': font_family, 'font-size': f'{pixel_size}px', 'font-weight': 'bold'}
        )

    def setup_interaction(self):
        """配置交互元素"""
        # 光标线
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=4) )
        self.vline2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 0, 0, 100), width=4) )
        self.vline.cursor_index = 0
        self.vline2.cursor_index = 1
        self.vline2.setZValue(100)
        self.vline.setZValue(100) 
        self.cursor_label = pg.TextItem("", anchor=(1, 1), color="red")
        self.plot_item.addItem(self.vline, ignoreBounds=True)
        self.plot_item.addItem(self.vline2, ignoreBounds=True)
        self.plot_item.addItem(self.cursor_label, ignoreBounds=True)
        self.vline.setVisible(False)
        self.vline2.setVisible(False)
        self.cursor_label.setVisible(False)
        
        # 多曲线cursor元素
        self.multi_cursor_items = []  # 存储多曲线cursor的可视化元素
        self.show_values_only = True  # 是否只显示x值（不显示圆圈和y值）
        
        # 【内存优化】对象池 - 复用ScatterPlotItem和TextItem，避免重复创建
        self._cursor_item_pool = {
            'circles': [],  # ScatterPlotItem对象池
            'labels': [],   # TextItem对象池（y值标签）
            'x_labels': []
        }
        
        # 信号连接
        # 【性能优化】控制cursor更新频率，减少CPU占用
        # 多曲线时降低频率可显著提升响应速度
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=20, slot=self.mouse_moved)
        self.vline.sigPositionChanged.connect(self.on_vline_position_changed)
        self.vline2.sigPositionChanged.connect(self.on_vline_position_changed)
        self.setAntialiasing(False)
        
        # 【性能优化】cursor更新节流控制
        self._last_cursor_update_time = 0
        self._cursor_update_throttle = 0.016  # 基础节流：16ms（约60fps）
        self._adaptive_throttle_enabled = True  # 启用自适应节流
        self._cursor_refresh_timer = QTimer(self)
        self._cursor_refresh_timer.setSingleShot(True)
        self._cursor_refresh_timer.timeout.connect(self._refresh_cursor_geometry)
        self._pending_cursor_geometry_update = False

    def _init_ui_refresh_coordinator(self):
        self._ui_refresh = UnifiedUpdateScheduler(
            delay_ms=UI_DEBOUNCE_DELAY_MS,
            order=("style", "cursor", "stats"),
            parent=self
        )
        self._ui_refresh.register("style", self._run_style_refresh)
        self._ui_refresh.register("cursor", self._run_cursor_refresh)
        self._ui_refresh.register("stats", self._run_stats_refresh)

    def _queue_ui_refresh(self, *, style=True, cursor=True, stats=True, immediate=False):
        if not hasattr(self, '_ui_refresh'):
            return
        tasks: list[str] = []
        if style:
            tasks.append("style")
        if cursor:
            tasks.append("cursor")
        if stats:
            tasks.append("stats")
        if not tasks:
            return
        if immediate:
            self._ui_refresh.run_immediately(*tasks)
        else:
            self._ui_refresh.schedule(*tasks)

    def _cancel_ui_refresh(self, *tasks):
        if hasattr(self, '_ui_refresh'):
            if tasks:
                self._ui_refresh.cancel(*tasks)
            else:
                self._ui_refresh.cancel()

    def _run_style_refresh(self):
        if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
            return
        if hasattr(self, 'view_box') and hasattr(self, 'plot_item'):
            self.update_plot_style(self.view_box, self.view_box.viewRange(), None)

    def _run_cursor_refresh(self):
        if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
            return
        if getattr(self, '_is_interacting', False):
            return
        if hasattr(self, 'vline') and self.vline.isVisible():
            try:
                self.update_cursor_label()
            except Exception:
                _widget_logger.debug("光标刷新异常（C++对象可能已销毁）")

    def _run_stats_refresh(self):
        main_window = self.window()
        if main_window is not None:
            main_window.request_mark_stats_refresh(immediate=True)

    def _extract_var_names_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for name in text.split(VAR_SEPARATOR):
            name = name.strip()
            if name and name not in seen:
                result.append(name)
                seen.add(name)
        return result

    def _should_hide_drag_indicator(self, main_window) -> bool:
        cursor_pos = QCursor.pos()
        top_left = main_window.mapToGlobal(QPoint(0, 0))
        window_rect = QRect(top_left, main_window.size())
        if not window_rect.contains(cursor_pos):
            return True

        container = None
        if hasattr(main_window, '_get_plot_container'):
            container = main_window._get_plot_container(self)
        if container is None:
            container = getattr(main_window, '_active_drag_container', None)
        if not container or not container.isVisible():
            return True

        container_rect = QRect(container.mapToGlobal(QPoint(0, 0)), container.size())
        if container_rect.contains(cursor_pos):
            return False

        widget_under_cursor = QApplication.widgetAt(cursor_pos)
        if widget_under_cursor:
            current = widget_under_cursor
            while current:
                if current is container:
                    return False
                current = current.parentWidget()

            target_window = widget_under_cursor.window()
            if isinstance(target_window, (DataTableDialog, PlotVariableEditorDialog)):
                return True
            if target_window is not main_window:
                return True

        return True

    def _enforce_drag_indicator_visibility(self):
        main_window = self.window()
        if not main_window:
            self._drag_indicator_guard.stop()
            self._drag_indicator_source = None
            return

        container = getattr(main_window, '_active_drag_container', None)
        if not container or getattr(container, 'plot_widget', None) is not self:
            self._drag_indicator_guard.stop()
            if self._drag_indicator_source is not None:
                self._drag_indicator_source = None
            return

        if self._drag_indicator_source is not None:
            source_widget = self._drag_indicator_source
            if not source_widget or not source_widget.isVisible():
                self._drag_indicator_source = None
            else:
                return

        if self._should_hide_drag_indicator(main_window):
            self._drag_indicator_source = None
            self._drag_indicator_guard.stop()
            main_window._hide_drag_indicator_for_plot(self)

    def _notify_drag_indicator(
        self,
        var_names: list[str] | None = None,
        hide: bool = False,
        source_widget: QWidget | None = None,
        indicator_text: str | None = None,
    ):
        main_window = self.window()

        if not main_window or not hasattr(main_window, '_show_drag_indicator_for_plot'):
            return

        if not hide and source_widget is None and self._should_hide_drag_indicator(main_window):
            hide = True

        if hide:
            self._drag_indicator_source = None
            self._drag_indicator_guard.stop()
            main_window._hide_drag_indicator_for_plot(self)
            return

        self._drag_indicator_source = source_widget
        main_window._show_drag_indicator_for_plot(self, var_names or [], indicator_text)
        if not self._drag_indicator_guard.isActive():
            self._drag_indicator_guard.start()


    def handle_single_point_limits(self, x_values, y_values):
        """处理单点或所有点x坐标相同的特殊情况 → 委托到 PlotDataManager"""
        return self._plot_data_manager.handle_single_point_limits(x_values, y_values)
        
    def wheelEvent(self, ev):
        vb = self.plot_item.getViewBox()
        delta = ev.angleDelta().y()
        # 只在没有按下任何修饰键（Ctrl/Shift/Alt…）时才执行缩放
        if ev.modifiers() == Qt.KeyboardModifier.NoModifier:
            if delta != 0:
                # 获取鼠标位置
                mouse_pos = ev.position().toPoint()
                scene_pos = self.mapToScene(mouse_pos)
                view_pos = vb.mapSceneToView(scene_pos)
                mouse_x = view_pos.x()
                mouse_y = view_pos.y()

                factor = max(0.000001,1-FACTOR_SCROLL_ZOOM)if delta > 0 else (1+FACTOR_SCROLL_ZOOM)
                vb.scaleBy((factor, 1), center=(mouse_x, mouse_y))
                ev.accept()  # 确保事件被处理
            else:
                super().wheelEvent(ev)
        else:
            # 有按键按下，交给父类默认处理（或自己写别的逻辑）
            super().wheelEvent(ev)
        
        #ev.accept()  # 确保事件被处理
    
    @safe_callback
    def mouse_moved(self, evt):
        """鼠标移动事件处理"""
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        if self._is_cursor_update_locked():
            return
        mousePoint = self.plot_item.vb.mapSceneToView(pos)

        # 如果cursor被固定，不跟随鼠标移动
        if self.is_cursor_pinned:
            # 在pin状态下，cursor保持固定位置，不跟随鼠标
            pass
        else:
            # 正常跟随鼠标模式
            if hasattr(self.window(), 'sync_crosshair'):
                self.window().sync_crosshair(mousePoint.x(), self)
            #print(f"mouse in pos {mousePoint.x()}")

    def _is_cursor_update_locked(self) -> bool:
        """判断 cursor 更新是否被锁定 → 委托到 CursorManager"""
        return self._cursor_manager._is_cursor_update_locked()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_cursor_geometry_update()

    @safe_callback
    def on_vline_position_changed(self, line_obj=None):
        """vline 位置变化时更新光标状态"""
        if self._is_cursor_update_locked():
            return
        # 时间修正期间禁止写回，避免固定值被边界夹住后污染
        if self.window() and getattr(self.window(), "_is_time_correction_active", False):
            return

        line = line_obj if line_obj is not None else self.vline
        cursor_index = getattr(line, "cursor_index", 0)

        if self.is_cursor_pinned:
            if getattr(self, "_suppress_pin_update", False):
                return
            x_pos = line.value()
            if len(self.pinned_x_values) <= cursor_index:
                self.pinned_x_values += [x_pos] * (cursor_index + 1 - len(self.pinned_x_values))
            self.pinned_x_values[cursor_index] = x_pos

            if cursor_index == 0:
                self.pinned_x_value = x_pos
                if self.factor != 0:
                    self.pinned_index_value = (x_pos - self.offset) / self.factor
                else:
                    self.pinned_index_value = None

            self.pinned_index_values = []
            for x_val in self.pinned_x_values:
                if self.factor != 0:
                    self.pinned_index_values.append((x_val - self.offset) / self.factor)

            if self.window() and hasattr(self.window(), "pinned_x_values"):
                self.window().pinned_x_values = list(self.pinned_x_values)

            if self.window() and hasattr(self.window(), "plot_widgets"):
                for container in self.window().plot_widgets:
                    widget = container.plot_widget
                    if widget.is_cursor_pinned and widget != self:
                        target_line = widget.vline if cursor_index == 0 else getattr(widget, "vline2", None)
                        if target_line is not None:
                            with QSignalBlocker(target_line):
                                target_line.setPos(x_pos)
                        if len(widget.pinned_x_values) <= cursor_index:
                            widget.pinned_x_values += [x_pos] * (cursor_index + 1 - len(widget.pinned_x_values))
                        widget.pinned_x_values[cursor_index] = x_pos
                        if cursor_index == 0:
                            widget.pinned_x_value = x_pos
                            if widget.factor != 0:
                                widget.pinned_index_value = (x_pos - widget.offset) / widget.factor
                            else:
                                widget.pinned_index_value = None
                        widget.update_cursor_label()

            self.update_cursor_label()
        else:
            if self.show_values_only:
                self._show_x_position_only()
            else:
                self.update_cursor_label()

    def sInt_to_fmtStr(self, value: int):
        """将秒数转换为时间字符串 HH:MM:SS.SS - 优化版避免内存泄漏"""
        # 【优化】直接计算而不创建pandas对象，避免内存累积
        total = value % (24*3600)  # 一天内的秒数
        hh = int(total // 3600)
        mm = int((total % 3600) // 60)
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:05.2f}"
    
    def dateInt_to_fmtStr(self, value: int):
        """将时间戳转换为日期字符串 - 优化版避免内存泄漏"""
        # 【优化】直接使用datetime而不创建pandas Series，避免内存累积
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(value)
            return dt.strftime('%Y/%m/%d')
        except:
            return str(value)
    
    def _significant_decimal_format_str(self,value: float, ref: float, max_dp:int | None = None) -> str:
        """
        根据 ref 的“显示精度”自动决定 value 的字符串格式。
        """
        # check length
        s = format(ref, 'f').rstrip('0').rstrip('.')
        if '.' not in s:
            dp = 0
        else:
            dp = len(s.split('.')[1])

        if max_dp is None or max_dp < 0:
            pass
        else:
            dp = min(max_dp,dp)

        if dp == 0:                       # ref 本身按整数显示
            return str(int(round(value)))
        
        fmt = f'{{:.{dp}f}}'              # 例如保留 2 位 -> "{:.2f}"
        return fmt.format(value).rstrip('0').rstrip('.')  # 去掉无意义的 0
    


    def set_xrange_with_link_handling(self, xmin, xmax, padding: float = 0):
        """设置 X 轴范围并处理联动 → 委托到 AxisManager"""
        self._axis_manager.set_xrange_with_link_handling(xmin, xmax, padding)

    def _get_cursor_mode(self):
        """获取光标模式 → 委托到 CursorManager"""
        return self._cursor_manager._get_cursor_mode()

    def _get_cursor_x_positions(self):
        """获取光标 X 位置列表 → 委托到 CursorManager"""
        return self._cursor_manager._get_cursor_x_positions()

    def _set_vline_visibility_for_mode(self, visible: bool, mode: str):
        """设置 vline 可见性 → 委托到 CursorManager"""
        self._cursor_manager._set_vline_visibility_for_mode(visible, mode)

    def _set_vline_bounds(self, bounds):
        """设置光标线边界 → 委托到 AxisManager"""
        self._axis_manager._set_vline_bounds(bounds)

    def apply_cursor_mode(self, mode, pinned_x_values):
        """应用光标模式 → 委托到 CursorManager"""
        self._cursor_manager.apply_cursor_mode(mode, pinned_x_values)

    def update_cursor_label(self):
        """更新光标标签 → 委托到 CursorManager"""
        self._cursor_manager.update_cursor_label()

    def _update_single_curve_cursor_label(self):
        """更新单曲线光标标签 → 委托到 CursorManager"""
        self._cursor_manager._update_single_curve_cursor_label()
    
    def _get_circle_from_pool(self, index):
        """从对象池获取 ScatterPlotItem → 委托到 CursorManager"""
        return self._cursor_manager._get_circle_from_pool(index)

    def _get_label_from_pool(self, index):
        """从对象池获取 TextItem → 委托到 CursorManager"""
        return self._cursor_manager._get_label_from_pool(index)

    def _get_x_label_from_pool(self, index: int):
        """获取 X 轴标签 TextItem → 委托到 CursorManager"""
        return self._cursor_manager._get_x_label_from_pool(index)

    def _clear_cursor_items(self, hide_only=True):
        """清除或隐藏所有 cursor 可视化元素 → 委托到 CursorManager"""
        self._cursor_manager._clear_cursor_items(hide_only)

    def _queue_item_for_deletion(self, item):
        """将 item 加入待删除队列 → 委托到 CursorManager"""
        self._cursor_manager._queue_item_for_deletion(item)

    def _process_pending_deletes(self):
        """处理待删除队列 → 委托到 CursorManager"""
        self._cursor_manager._process_pending_deletes()

    def _collect_visible_curve_arrays(self, key: str) -> list[np.ndarray]:
        """收集可见曲线的数据数组 → 委托到 MultiCurveManager"""
        return self._multi_curve_manager._collect_visible_curve_arrays(key)

    def _collect_visible_curve_pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """收集可见曲线的 x-y 数据对 → 委托到 MultiCurveManager"""
        return self._multi_curve_manager._collect_visible_curve_pairs()

    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple[float | None, float | None]:
        """获取曲线 X 轴限制 → 委托到 MultiCurveManager"""
        return self._multi_curve_manager.get_curve_x_limits(curves_filter)

    def _safe_clear_plot_items(self):
        """安全地清理所有plot items → 委托到 PlotDataManager"""
        self._plot_data_manager._safe_clear_plot_items()
    
    def _update_multi_curve_cursor_label(self):
        """更新多曲线光标标签 → 委托到 CursorManager"""
        self._cursor_manager._update_multi_curve_cursor_label()

    def _position_labels_avoid_overlap(self, cursor_values: list[dict], x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """标签定位算法 → 委托到 CursorManager"""
        self._cursor_manager._position_labels_avoid_overlap(cursor_values, x_min, x_max, y_min, y_max)

    def toggle_cursor(self, show: bool, hide_values_only: bool = False):
        """切换光标显示状态 → 委托到 CursorManager"""
        self._cursor_manager.toggle_cursor(show, hide_values_only)

    def _show_x_position_only(self, x_positions=None):
        """仅显示 x 位置标签 → 委托到 CursorManager"""
        self._cursor_manager._show_x_position_only(x_positions)

    def _has_visible_curve_data(self) -> bool:
        """判断当前 plot 是否有可见且有数据的曲线 → 委托到 CursorManager"""
        return self._cursor_manager._has_visible_curve_data()

    def pin_cursor(self, x_value):
        """将光标固定到最近的 x 并同步到所有 plot → 委托到 CursorManager"""
        self._cursor_manager.pin_cursor(x_value)

    def free_cursor(self):
        """释放光标固定并恢复自由移动 → 委托到 CursorManager"""
        self._cursor_manager.free_cursor()

    def reset_pin_state(self):
        """重置 pin 状态 → 委托到 CursorManager"""
        self._cursor_manager.reset_pin_state()

    def _update_vline_bounds_from_data(self):
        """根据当前绘制的数据更新vline bounds

        这个函数计算当前所有可见曲线的x范围，并更新vline的移动边界。
        优先使用理论值（基于original_index_x + factor/offset）计算bounds，
        避免因异步更新导致的bounds不一致问题。
        """
        try:
            # 优先策略1：单曲线模式下，使用 original_index_x + factor/offset 计算理论bounds
            if hasattr(self, 'original_index_x') and self.original_index_x is not None and len(self.original_index_x) > 0:
                min_index = np.min(self.original_index_x)
                max_index = np.max(self.original_index_x)
                min_x = self.offset + self.factor * min_index
                max_x = self.offset + self.factor * max_index
                self._set_vline_bounds([min_x, max_x])
                return min_x, max_x

            # 优先策略2：多曲线模式下，使用数据长度 + factor/offset 计算理论bounds
            if self.is_multi_curve_mode and self.curves:
                # 获取任一curve的数据长度
                for ci in self.curves.values():
                    if ci.y_data is not None:
                        datalength = len(ci.y_data)
                        if datalength > 0:
                            min_x = self.offset + self.factor * 1
                            max_x = self.offset + self.factor * datalength
                            self._set_vline_bounds([min_x, max_x])
                            return min_x, max_x
                        break

            # Fallback策略1：从实际curve数据读取（多曲线模式）
            if self.is_multi_curve_mode and self.curves:
                x_arrays = self._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    combined = np.concatenate(x_arrays)
                    min_x, max_x = np.nanmin(combined), np.nanmax(combined)
                    self._set_vline_bounds([min_x, max_x])
                    return min_x, max_x

            # Fallback策略2：从实际curve数据读取（单曲线模式）
            if self.curve is not None:
                x_data, _ = self.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    min_x, max_x = np.min(x_data), np.max(x_data)
                    self._set_vline_bounds([min_x, max_x])
                    return min_x, max_x

            # Fallback策略3：使用xMin/xMax
            if hasattr(self, 'xMin') and hasattr(self, 'xMax'):
                self._set_vline_bounds([self.xMin, self.xMax])
                return self.xMin, self.xMax
            else:
                self._set_vline_bounds([None, None])
                return None, None
        except Exception as e:
            logger.warning("Error updating vline bounds: %s", e)
            self._set_vline_bounds([None, None])
            return None, None
    
    def _update_cursor_after_plot(self, min_x_bound: float, max_x_bound: float):
        """绘图后更新光标边界和可见性 → 委托到 CursorManager"""
        self._cursor_manager._update_cursor_after_plot(min_x_bound, max_x_bound)

    def clear_value_cache(self):
        """清除值缓存 → 委托到 PlotDataManager"""
        self._plot_data_manager.clear_value_cache()

    def datetime_to_unix_seconds(self, series: pd.Series) -> pd.Series:
        """将datetime Series转换为Unix时间戳 → 委托到 PlotDataManager"""
        return self._plot_data_manager.datetime_to_unix_seconds(series)
        
    def get_value_from_name(self, var_name) -> tuple | None:
        """根据变量名获取值和格式 → 委托到 PlotDataManager"""
        return self._plot_data_manager.get_value_from_name(var_name)
    
    def update_time_correction(self, new_factor, new_offset):
        """更新时间修正参数 → 委托到 PlotDataManager"""
        self._plot_data_manager.update_time_correction(new_factor, new_offset)

    # ---------------- 拖拽相关 ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            var_names = self._extract_var_names_from_text(event.mimeData().text())
            self._notify_drag_indicator(var_names, hide=False)
            event.acceptProposedAction()
        else:
            self._notify_drag_indicator(hide=True)
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            var_names = self._extract_var_names_from_text(event.mimeData().text())
            self._notify_drag_indicator(var_names, hide=False)
            event.acceptProposedAction()
        else:
            self._notify_drag_indicator(hide=True)
            event.ignore()

    def dragLeaveEvent(self, event):
        self._notify_drag_indicator(hide=True)
        event.accept()

    def dropEvent(self, event):
        self._notify_drag_indicator(hide=True)
        var_names = parse_var_names_from_mimedata(event.mimeData())
        self.add_variables_to_plot(var_names)
        event.acceptProposedAction()
        if self.window():
            self.window().request_mark_stats_refresh()

    def add_variables_to_plot(self, var_names: list[str]):
        """批量添加变量到当前绘图区，供拖拽或右键操作复用"""
        names = [name.strip() for name in (var_names or []) if isinstance(name, str) and name.strip()]
        if not names:
            return

        if len(names) > 1:
            failed_vars = []
            success_vars = []
            variables_data = []

            for var_name in names:
                is_valid, _ = self._validate_plot_data(var_name)
                if not is_valid:
                    failed_vars.append(var_name)
                    continue

                success, _, x_array, y_array, y_format = self._prepare_plot_data(var_name)
                if not success:
                    failed_vars.append(var_name)
                    continue

                if (self.is_multi_curve_mode and var_name in self.curves) or \
                   (not self.is_multi_curve_mode and var_name == self.y_name):
                    failed_vars.append(var_name)
                    continue

                variables_data.append((var_name, x_array, y_array, y_format))

            self._batch_adding = True

            if variables_data:
                if not self.is_multi_curve_mode and self.curve and self.y_name:
                    current_color = 'blue'
                    if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                        current_pen = self.curve.opts['pen']
                        if hasattr(current_pen, 'color'):
                            current_color = current_pen.color().name()

                    x_data_val = self.offset + self.factor * self.original_index_x if self.original_index_x is not None else None

                    self.curves[self.y_name] = CurveInfo(
                        var_name=self.y_name,
                        curve=self.curve,
                        x_data=x_data_val,
                        y_data=self.original_y if self.original_y is not None else None,
                        color=current_color,
                        y_format=self.y_format,
                        visible=True
                    )
                    self.current_color_index = 1

                for var_name, x_array, y_array, y_format in variables_data:
                    x_values = self.offset + self.factor * x_array
                    color = self.curve_colors[self.current_color_index % len(self.curve_colors)]
                    self.current_color_index += 1

                    pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
                    curve = self.plot_item.plot(x_values, y_array, pen=pen, name=var_name, skipFiniteCheck=True)

                    self.curves[var_name] = CurveInfo(
                        var_name=var_name,
                        curve=curve,
                        x_data=x_values,
                        y_data=y_array,
                        color=color,
                        y_format=y_format or '',
                        visible=True
                    )

                    success_vars.append(var_name)

                self.is_multi_curve_mode = len(self.curves) > 1

            self._batch_adding = False

            if success_vars:
                if self.is_multi_curve_mode:
                    self.update_legend()

                self._update_axes_for_multi_curve(update_x_range=False)

                x_arrays = self._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    combined = np.concatenate(x_arrays)
                    min_x, max_x = np.nanmin(combined), np.nanmax(combined)
                    self._set_vline_bounds([min_x, max_x])
                    self._update_cursor_after_plot(min_x, max_x)

                if self.vline.isVisible():
                    self.update_cursor_label()

                self._recalc_max_point_density()
                main_window = self.window()
                if main_window is not None and hasattr(main_window, '_sync_min_xrange'):
                    main_window._sync_min_xrange()

            if failed_vars:
                QMessageBox.information(self, "提示", f"以下变量已在绘图中:\n" + "\n".join(failed_vars))
        else:
            self.plot_variable(names[0])

    def _validate_plot_data(self, var_name: str) -> tuple[bool, str]:
        """验证绘图数据的有效性 → 委托到 PlotDataManager"""
        return self._plot_data_manager._validate_plot_data(var_name)

    def _get_x_data_for_variable(self, y_len: int) -> np.ndarray:
        return np.arange(1, y_len + 1, dtype=np.float32)

    def _prepare_plot_data(self, var_name: str) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
        """准备绘图数据 → 委托到 PlotDataManager"""
        return self._plot_data_manager._prepare_plot_data(var_name)

    def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
        """绘制变量到图表 → 委托到 PlotDataManager"""
        return self._plot_data_manager.plot_variable(var_name, show_duplicate_warning)

    def _compute_valid_min_max(self, values) -> tuple[float | None, float | None]:
        """Safely compute min/max ignoring NaN/INF values → 委托到 PlotDataManager"""
        return self._plot_data_manager._compute_valid_min_max(values)

    def _get_y_range_in_x_window(self, x_values: np.ndarray, y_values: np.ndarray, x_min: float, x_max: float):
        """计算在指定x轴范围内的y值范围 → 委托到 PlotDataManager"""
        return self._plot_data_manager._get_y_range_in_x_window(x_values, y_values, x_min, x_max)
    
    def _setup_plot_axes(self, x_values: np.ndarray, y_values: np.ndarray, update_x_range: bool = True):
        """设置绘图坐标轴 → 委托到 AxisManager"""
        self._axis_manager._setup_plot_axes(x_values, y_values, update_x_range)

    def _reset_plot_limits(self):
        """重置绘图限制 → 委托到 AxisManager"""
        self._axis_manager._reset_plot_limits()

    def _clear_plot_data(self):
        """清除绘图数据 → 委托到 PlotDataManager"""
        self._plot_data_manager._clear_plot_data()

    def clear_plot_item(self):
        """清除绘图项 → 委托到 PlotDataManager"""
        self._plot_data_manager.clear_plot_item()
        
    def add_variable_to_plot(self, var_name: str, x_values: np.ndarray = None, y_values: np.ndarray = None,
                             y_format: str = None, skip_existence_check: bool = False,
                             show_duplicate_warning: bool = True, preferred_color: str | None = None) -> bool:
        """添加变量到多曲线绘图
        
        这是多曲线绘图的核心方法，支持以下功能：
        1. 自动处理单曲线到多曲线模式的转换
        2. 防止重复添加相同变量
        3. 支持批量添加模式（抑制中间坐标轴更新）
        4. 自动颜色分配和曲线样式优化
        
        工作流程：
        - 检查变量是否已存在（可选）
        - 如果是从单曲线模式转换，将现有单曲线迁移到curves字典
        - 创建新曲线并设置性能优化选项
        - 更新坐标轴范围（非批量模式）
        - 更新cursor显示
        
        Args:
            var_name: 变量名称
            x_values: X轴数据（可选，如果为None则从dataframe准备）
            y_values: Y轴数据（可选，如果为None则从dataframe准备）
            y_format: Y轴格式（可选，如's'时间格式、'date'日期格式等）
            skip_existence_check: 是否跳过存在性检查（内部使用）
            show_duplicate_warning: 是否显示重复变量警告（批量添加时设为False）
            preferred_color: 恢复曲线时指定的颜色（可选）
            
        Returns:
            bool: 添加是否成功。失败原因可能是：变量已存在、数据无效等
        """
        try:
            # 如果数据未提供，则准备数据
            if x_values is None or y_values is None:
                success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
                if not success:
                    QMessageBox.warning(self, "错误", error_msg)
                    return False
                x_values = self.offset + self.factor * x_array
                y_values = y_array
            
            # 检查变量是否已存在（除非跳过检查）
            if not skip_existence_check:
                if (self.is_multi_curve_mode and var_name in self.curves) or \
                   (not self.is_multi_curve_mode and var_name == self.y_name):
                    if show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
            
            # 特殊情况：多曲线模式但curves为空，需要迁移单曲线
            # 说明正在从单曲线过渡到多曲线，需要先将self.curve迁移到curves字典
            if self.is_multi_curve_mode and len(self.curves) == 0 and self.curve and self.y_name:
                # 将当前单曲线添加到curves字典
                current_color = 'blue'
                if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                    current_pen = self.curve.opts['pen']
                    if hasattr(current_pen, 'color'):
                        current_color = current_pen.color().name()
                
                x_data_val_1 = self.offset + self.factor * self.original_index_x if self.original_index_x is not None else None

                self.curves[self.y_name] = CurveInfo(
                    var_name=self.y_name,
                    curve=self.curve,
                    x_data=x_data_val_1,
                    y_data=self.original_y if self.original_y is not None else None,
                    color=current_color,
                    y_format=self.y_format,
                    visible=True
                )
                self.current_color_index = 1  # 从第二个颜色开始
                
                # 如果要添加的变量与已迁移的相同，直接返回
                if var_name == self.y_name and not skip_existence_check:
                    if show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
            
            # 如果当前是单曲线模式，需要先转换到多曲线模式
            if not self.is_multi_curve_mode and self.curve and self.y_name:
                # 检查要添加的变量是否与当前单曲线相同
                if var_name == self.y_name and not skip_existence_check:
                    # 相同变量，不需要转换模式，直接返回
                    if show_duplicate_warning:
                        QMessageBox.information(self, "提示", f"变量 {var_name} 已在绘图中")
                    return False
                
                # 将当前单曲线添加到curves字典
                current_color = 'blue'  # 默认颜色
                if hasattr(self.curve, 'opts') and 'pen' in self.curve.opts:
                    current_pen = self.curve.opts['pen']
                    if hasattr(current_pen, 'color'):
                        current_color = current_pen.color().name()
                
                x_data_val_2 = self.offset + self.factor * self.original_index_x if self.original_index_x is not None else x_values

                self.curves[self.y_name] = CurveInfo(
                    var_name=self.y_name,
                    curve=self.curve,
                    x_data=x_data_val_2,
                    y_data=self.original_y if self.original_y is not None else y_values,
                    color=current_color,
                    y_format=self.y_format,
                    visible=True
                )
                self.current_color_index = 1  # 从第二个颜色开始
            
            # 选择颜色
            default_color = self.curve_colors[self.current_color_index % len(self.curve_colors)]
            self.current_color_index += 1
            color = preferred_color or default_color
            
            # ========== 性能优化：创建曲线并配置渲染选项 ==========
            pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)

            # 创建曲线（保持简单参数以确保兼容性）
            curve = self.plot_item.plot(
                x_values, y_values, 
                pen=pen, 
                name=var_name,
                skipFiniteCheck=True
            )
            
            # 性能优化说明：
            # - 自动降采样：plot_item.setDownsampling(mode='peak', auto=True) 已在setup_plot_area中配置
            # - 视图裁剪：plot_item.setClipToView(True) 已在setup_plot_area中配置
            # - 智能防抖：根据数据量和曲线数动态调整延迟
            # 这些设置会自动应用到所有曲线，无需OpenGL也能获得良好性能
            
            # 存储曲线信息到curves字典
            self.curves[var_name] = CurveInfo(
                var_name=var_name,
                curve=curve,
                x_data=x_values,
                y_data=y_values,
                color=color,
                y_format=y_format or '',
                visible=True
            )
            
            # 更新多曲线模式
            self.update_multi_curve_mode()

            if len(self.curves) == 1 and not self.y_name:
                single_name = next(iter(self.curves.keys()))
                single_ci = self.curves[single_name]
                self.y_name = single_name
                self.y_format = single_ci.y_format or ''
            
            # 更新坐标轴范围（批量添加时跳过，避免重复更新）
            batch_adding = getattr(self, '_batch_adding', False)
            if not batch_adding:
                main_window = self.window()
                is_mdf = (
                    main_window is not None
                    and hasattr(main_window, 'loader')
                    and main_window.loader is not None
                    and getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
                )
                # 始终保持x轴范围不变，只更新y轴范围
                # 因为所有plot的x轴都是linked的，改变x轴会影响其他plot
                
                # 1. 先计算所有曲线的全范围y值，用于设置y轴limits
                y_arrays = self._collect_visible_curve_arrays('y_data')
                if y_arrays:
                    combined_y = np.concatenate(y_arrays)
                    if combined_y.size:
                        all_data_min_y = np.nanmin(combined_y)
                        all_data_max_y = np.nanmax(combined_y)
                        # 设置y轴limits为所有数据的范围
                        self._set_safe_y_range(all_data_min_y, all_data_max_y, set_limits=True)

                # 2. 再根据当前x范围设置y轴viewRange
                # 检查是否是单点数据
                special_limits = self.handle_single_point_limits(x_values, y_values)
                if special_limits:
                    # 单点数据：使用特殊处理
                    min_x, max_x, min_y, max_y = special_limits
                    
                    # 检查是否是第一个曲线
                    has_other_curves = len(self.curves) > 1
                    
                    if not has_other_curves:
                        # 第一次添加曲线：直接设置y轴viewRange
                        self._set_safe_y_range(min_y, max_y, set_limits=False)
                    else:
                        # 已有曲线：根据新曲线扩展y轴viewRange
                        current_y_range = self.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        final_min_y = min(current_min_y, min_y)
                        final_max_y = max(current_max_y, max_y)
                        self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                    
                    # self._update_x_limits_for_plot(x_values, y_values, is_mdf)
                else:
                    # 正常数据
                    current_x_range = self.view_box.viewRange()[0]
                    x_min, x_max = current_x_range
                    
                    # 计算新曲线在当前x轴范围内的y值范围
                    new_min_y, new_max_y = self._get_y_range_in_x_window(x_values, y_values, x_min, x_max)
                    
                    # 检查是否是第一个曲线
                    has_other_curves = len(self.curves) > 1
                    
                    if not has_other_curves:
                        # 第一次添加曲线：直接设置y轴viewRange为新曲线在当前x范围内的范围
                        self._set_safe_y_range(new_min_y, new_max_y, set_limits=False)
                    else:
                        # 已有曲线：根据新曲线扩展y轴viewRange
                        current_y_range = self.view_box.viewRange()[1]
                        current_min_y, current_max_y = current_y_range
                        
                        # 扩展y轴viewRange（只考虑新曲线的min/max）
                        final_min_y = min(current_min_y, new_min_y)
                        final_max_y = max(current_max_y, new_max_y)
                        
                        # 更新y轴viewRange
                        self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)
                    
                    # 3. 更新x轴limits（合并本 plot 和全局所有可见 plot 的范围）
                    # self._update_x_limits_for_plot(x_values, y_values, is_mdf)
            
            # 更新cursor边界 - 使用所有曲线的x范围（而不仅仅是当前添加的变量）
            x_arrays = self._collect_visible_curve_arrays('x_data')
            if x_arrays:
                combined_x = np.concatenate(x_arrays)
                min_x, max_x = np.nanmin(combined_x), np.nanmax(combined_x)
            else:
                # 如果没有其他曲线，使用当前变量的范围
                min_x, max_x = np.min(x_values), np.max(x_values)
            self._set_vline_bounds([min_x, max_x])
            
            # 应用全局cursor值显示状态
            self._update_cursor_after_plot(min_x, max_x)
            
            # 如果cursor可见，立即更新cursor标签以显示新添加的曲线
            if self.vline.isVisible():
                self.update_cursor_label()

            if not batch_adding:
                self._recalc_max_point_density()
                main_window = self.window()
                if main_window is not None and hasattr(main_window, '_sync_min_xrange'):
                    main_window._sync_min_xrange()

            return True
            
        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"添加变量时发生错误: {str(e)}")
            return False
    
    def update_multi_curve_mode(self):
        """更新多曲线模式状态 → 委托到 MultiCurveManager"""
        self._multi_curve_manager.update_multi_curve_mode()

    def update_legend(self):
        """更新图例显示 → 委托到 MultiCurveManager"""
        self._multi_curve_manager.update_legend()
    
    def toggle_curve_visibility_by_name(self, var_name):
        """通过变量名切换曲线可见性 → 委托到 MultiCurveManager"""
        self._multi_curve_manager.toggle_curve_visibility_by_name(var_name)

    def _recreate_curve(self, var_name):
        """重新创建失效的曲线 → 委托到 MultiCurveManager"""
        self._multi_curve_manager._recreate_curve(var_name)
    
    def _on_legend_clicked(self, event):
        """Legend点击事件处理 → 委托到 MultiCurveManager"""
        self._multi_curve_manager._on_legend_clicked(event)
    
    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        """为多曲线更新坐标轴范围 → 委托到 MultiCurveManager"""
        self._multi_curve_manager._update_axes_for_multi_curve(update_x_range)

    def _update_x_limits_for_plot(self, x_values: np.ndarray, y_values: np.ndarray, is_mdf: bool):
        """
        统一更新 X 轴 limits，合并本 plot 的可见曲线范围和全局所有可见 plot 的范围
        """
        x_arrays = self._collect_visible_curve_arrays('x_data')
        if x_arrays:
            combined_x = np.concatenate(x_arrays)
            data_min_x = float(np.nanmin(combined_x))
            data_max_x = float(np.nanmax(combined_x))
        else:
            data_min_x = float(np.min(x_values))
            data_max_x = float(np.max(x_values))

        main_window = self.window()
        if main_window is not None and hasattr(main_window, 'collect_global_x_range'):
            global_min, global_max = main_window.collect_global_x_range(curves_filter="all")
            if global_min is not None:
                data_min_x = min(data_min_x, global_min)
                data_max_x = max(data_max_x, global_max)

        padding_x = DEFAULT_PADDING_VAL_X
        limits_xMin = data_min_x - padding_x * (data_max_x - data_min_x)
        limits_xMax = data_max_x + padding_x * (data_max_x - data_min_x)
        self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mouseDoubleClickEvent(event)
            return
        from src.ui.dialogs.axis import AxisDialog
        from src.ui.plot_variable_editor import PlotVariableEditorDialog
        
        if event.button() == Qt.MouseButton.MiddleButton:
            self.clear_plot_item()
            self.window().request_mark_stats_refresh(immediate=True)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            
            # 获取坐标轴区域
            y_axis_rect_scene = self.axis_y.mapToScene(self.axis_y.boundingRect()).boundingRect()
            x_axis_rect_scene = self.axis_x.mapToScene(self.axis_x.boundingRect()).boundingRect()
            
            # 获取绘图区域的实际范围（排除坐标轴区域）- 使用view_box而不是plot_item
            view_box_rect = self.view_box.boundingRect()
            view_box_rect_scene = self.view_box.mapToScene(view_box_rect).boundingRect()
            
            # 缩小X轴检测区域，只检测X轴标签区域（底部部分）
            x_axis_label_rect = QRectF(x_axis_rect_scene.left(), x_axis_rect_scene.bottom() - 30, x_axis_rect_scene.width(), 30)
            
            # 优先检测X轴标签区域（最具体）
            if x_axis_label_rect.contains(scene_pos):
                dialog = AxisDialog(self.axis_x, self.view_box, "X", self)
                if dialog.exec():
                    min_val, max_val = self.view_box.viewRange()[0]
                    for view in self.window().findChildren(DraggableGraphicsLayoutWidget):
                        #view.view_box.setXRange(min_val, max_val, padding=0.00)
                        #view.plot_item.setXRange(min_val, max_val, padding=0.00)
                        self.set_xrange_with_link_handling(xmin=min_val,xmax=max_val,padding=DEFAULT_PADDING_VAL_X)
                        view.plot_item.update()
                return
            # 然后检测绘图区域（在检测Y轴之前）
            elif view_box_rect_scene.contains(scene_pos):
                # 双击绘图区域（网格内部），弹出变量编辑器
                dialog = PlotVariableEditorDialog(self, self.window())
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            # 最后检测Y轴区域（最后兜底）
            elif y_axis_rect_scene.contains(scene_pos):
                dialog = AxisDialog(self.axis_y, self.view_box, "Y", self)
                if dialog.exec():
                    self.plot_item.update()
                return
        return super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubberBand.isVisible():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rubberBand.isVisible() and event.button() == Qt.MouseButton.LeftButton:
            self.rubberBand.hide()
            rect = self.rubberBand.geometry()
            if rect.width() > 10 and rect.height() > 10:  # 避免误触
                topLeft = self.mapToScene(rect.topLeft())
                bottomRight = self.mapToScene(rect.bottomRight())

                p1 = self.view_box.mapSceneToView(topLeft)
                p2 = self.view_box.mapSceneToView(bottomRight)

                x_min = min(p1.x(), p2.x())
                x_max = max(p1.x(), p2.x())
                y_min = min(p1.y(), p2.y())
                y_max = max(p1.y(), p2.y())

                # 添加10% margin
                dx = x_max - x_min
                dy = y_max - y_min
                margin = 0.1
                x_min -= margin * dx
                x_max += margin * dx
                y_min -= margin * dy
                y_max += margin * dy

                self.view_box.setXRange(x_min, x_max, padding=0)
                self.view_box.setYRange(y_min, y_max, padding=0)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def add_mark_region(self, min_x, max_x):
        """添加标记区域 → 委托到 MarkRegionManager"""
        self._mark_region_manager.add_mark_region(min_x, max_x)

    def remove_mark_region(self):
        """移除标记区域 → 委托到 MarkRegionManager"""
        self._mark_region_manager.remove_mark_region()

    def update_mark_region(self):
        """更新标记区域 → 委托到 MarkRegionManager"""
        self._mark_region_manager.update_mark_region()

    def get_mark_stats(self):
        """获取标记区域统计 → 委托到 MarkRegionManager"""
        return self._mark_region_manager.get_mark_stats()

    def _apply_plot_style(self, show_symbols: bool):
        """应用绘图样式 - 基于xrange只有两种搭配：细线+symbol 或 粗线无symbol

        【内存优化】缓存pen对象，避免zoom时重复创建导致内存泄漏
        """
        try:
            # 优先检查curves字典（多曲线模式或从多曲线删到单曲线的情况）
            if self.curves:
                # 有curves字典：遍历所有曲线应用样式
                for var_name, ci in self.curves.items():
                    if ci.curve is None:
                        continue

                    curve = ci.curve
                    color = ci.color

                    # 缓存pen对象：检查当前样式是否匹配，避免重复创建
                    if show_symbols:
                        # xrange小：细线 + 符号
                        cache_key = f'thin_{color}'
                        if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                            pen = pg.mkPen(color=color, width=THIN_LINE_WIDTH)
                            curve.setPen(pen)
                            curve._cached_pen_key = cache_key

                        if not hasattr(curve, '_has_symbols') or not curve._has_symbols:
                            curve.setSymbol('s')
                            curve.setSymbolSize(3)
                            curve.setSymbolPen(color)
                            curve.setSymbolBrush(color)
                            curve._has_symbols = True
                    else:
                        # xrange大：粗线无符号
                        cache_key = f'thick_{color}'
                        if not hasattr(curve, '_cached_pen_key') or curve._cached_pen_key != cache_key:
                            pen = pg.mkPen(color=color, width=THICK_LINE_WIDTH)
                            curve.setPen(pen)
                            curve._cached_pen_key = cache_key

                        if not hasattr(curve, '_has_symbols') or curve._has_symbols:
                            curve.setSymbol(None)
                            curve._has_symbols = False
            elif self.curve:
                # 没有curves字典但有单曲线：使用self.curve
                # 获取当前曲线的颜色
                current_pen = self.curve.opts.get('pen', pg.mkPen('blue'))
                color = current_pen.color().name() if hasattr(current_pen, 'color') else 'blue'

                # 缓存pen对象：检查当前样式是否匹配，避免重复创建
                if show_symbols:
                    # xrange小：细线 + 符号
                    cache_key = f'thin_{color}'
                    if not hasattr(self.curve, '_cached_pen_key') or self.curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THIN_LINE_WIDTH)
                        self.curve.setPen(pen)
                        self.curve._cached_pen_key = cache_key

                    if not hasattr(self.curve, '_has_symbols') or not self.curve._has_symbols:
                        self.curve.setSymbol('s')
                        self.curve.setSymbolSize(3)
                        self.curve.setSymbolPen(color)
                        self.curve.setSymbolBrush(color)
                        self.curve._has_symbols = True
                else:
                    # xrange大：粗线无符号
                    cache_key = f'thick_{color}'
                    if not hasattr(self.curve, '_cached_pen_key') or self.curve._cached_pen_key != cache_key:
                        pen = pg.mkPen(color=color, width=THICK_LINE_WIDTH)
                        self.curve.setPen(pen)
                        self.curve._cached_pen_key = cache_key

                    if not hasattr(self.curve, '_has_symbols') or self.curve._has_symbols:
                        self.curve.setSymbol(None)
                        self.curve._has_symbols = False
        except Exception as e:
            print(f"应用绘图样式时出错: {e}")

    def _calculate_visible_points(self, range):
        """计算当前可见范围的点数估算
        
        Args:
            range: 视图范围 [[x_min, x_max], [y_min, y_max]]
            
        Returns:
            tuple: (index_range_width, visible_points)
                - index_range_width: 索引范围宽度（考虑factor）
                - visible_points: 可见点数估算（考虑曲线数量）
        """
        # 获取当前视图的xRange
        x_min, x_max = range[0]
        x_range_width = x_max - x_min
        
        # 考虑factor的影响 - 将xRange转换为索引范围
        # x = offset + factor * index，所以 index_range = x_range / factor
        if hasattr(self, 'factor') and self.factor != 0:
            index_range_width = x_range_width / abs(self.factor)
        else:
            index_range_width = x_range_width
        
        # 计算曲线数量（考虑单曲线和多曲线两种模式）
        if hasattr(self, 'is_multi_curve_mode') and self.is_multi_curve_mode:
            # 多曲线模式：使用 self.curves 字典的长度
            curve_count = len(self.curves) if hasattr(self, 'curves') and self.curves else 0
        else:
            # 单曲线模式：检查是否有曲线
            curve_count = 1 if hasattr(self, 'curve') and self.curve is not None else 0
        
        # 至少按1条曲线计算（避免除0或无意义的计算）
        curve_count = max(curve_count, 1)
        
        # 计算可见点数：索引范围 × 曲线数量
        visible_points = index_range_width * curve_count
        
        return index_range_width, visible_points
    
    def update_plot_style(self, view_box, range, rect=None):
        """更新绘图样式 - 基于xRange宽度判断，只有两种搭配：细线+symbol 或 粗线无symbol
        
        【性能优化】交互期间降低样式更新频率，支持百万级数据点流畅绘制
        """
        try:
            # 【安全检查】如果正在更新数据或对象被销毁，跳过样式更新
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                return
            
            # 【安全检查】确保关键对象存在
            if not hasattr(self, 'factor') or not hasattr(self, 'plot_item'):
                return
            
            # 【性能优化】交互期间：完全跳过样式更新，避免遍历所有曲线导致卡顿
            # 样式更新只在交互结束后执行一次，保证缩放时的流畅性
            is_interacting = getattr(self, '_is_interacting', False)
            if is_interacting:
                return  # 交互期间完全跳过样式更新，避免卡顿

            # 使用共用方法计算索引范围
            index_range_width, visible_points = self._calculate_visible_points(range)

            # 基于索引范围宽度判断样式：阈值根据全局最大密度动态调整
            main_window = self.window()
            density = getattr(main_window, '_global_max_density', 0.0) if main_window else 0.0
            if density > 0:
                effective_threshold = XRANGE_THRESHOLD_FOR_SYMBOLS / density
            else:
                effective_threshold = XRANGE_THRESHOLD_FOR_SYMBOLS
            show_symbols = index_range_width < effective_threshold

            # 应用样式到所有曲线
            self._apply_plot_style(show_symbols)
            
        except Exception as e:
            print(f"更新绘图样式时出错: {e}")


    @safe_callback
    def _on_range_changed(self, view_box, range, changed=None):
        """ViewBox范围变化回调处理"""
        try:
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                self._cancel_ui_refresh()
                return

            if getattr(self, '_is_syncing_range', False):
                return

            if not self._is_interacting:
                self._is_interacting = True
                self._start_interaction()

            if hasattr(self, '_interaction_timer'):
                self._interaction_timer.stop()
                self._interaction_timer.start(UI_DEBOUNCE_DELAY_MS)

            if self._is_interacting:
                self._cancel_ui_refresh('style', 'cursor')
                return

            self._queue_ui_refresh()
        except Exception as e:
            print(f"范围变化处理出错: {e}")

    def _start_interaction(self):
        """开始交互时的优化处理
        
        类似iOS的快照策略：在交互期间临时降低渲染质量
        """
        try:
            # 【性能优化】交互期间临时提高降采样阈值，减少渲染的点数
            # 这样可以显著提升缩放时的流畅度
            if hasattr(self, 'plot_item'):
                # 保存原始降采样设置
                if not hasattr(self, '_original_downsample_ds'):
                    # 获取当前降采样设置（如果有）
                    self._original_downsample_ds = getattr(self.plot_item, '_downsample', None)
                
                # 临时提高降采样阈值：交互期间使用更激进的降采样
                # 通过设置更大的ds值来减少渲染的点数
                # 注意：pyqtgraph的auto模式会自动处理，这里主要是确保降采样更激进
                # 实际上，pyqtgraph的auto模式已经会根据可见区域自动调整
                # 但我们可以通过临时禁用某些昂贵的操作来提升性能
                pass  # pyqtgraph的auto模式已经足够智能，无需手动调整
            
            # 【性能优化】交互期间禁用样式更新（已在update_plot_style中实现）
            # 这样可以避免在缩放时遍历所有曲线并更新样式
        except Exception as e:
            print(f"开始交互优化时出错: {e}")
    
    def _end_interaction(self):
        """结束交互时的处理"""
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            if getattr(self, '_pending_cursor_geometry_update', False):
                self._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception as e:
            print(f"结束交互出错: {e}")

    def _schedule_cursor_geometry_update(self):
        if not hasattr(self, 'vline') or not self.vline.isVisible():
            return
        if getattr(self, '_cursor_refresh_timer', None) is None:
            return
        if getattr(self, '_is_interacting', False):
            self._pending_cursor_geometry_update = True
            return
        self._pending_cursor_geometry_update = False
        # 重启单次定时器，合并短时间内的多次请求
        self._cursor_refresh_timer.start(max(15, UI_DEBOUNCE_DELAY_MS ))

    def _refresh_cursor_geometry(self):
        if not hasattr(self, 'vline') or not self.vline.isVisible():
            return
        if getattr(self, '_is_interacting', False):
            self._pending_cursor_geometry_update = True
            return
        if self.show_values_only:
            self._show_x_position_only()
        else:
            self.update_cursor_label()

    def _connect_viewbox_signals(self):
        vb = self.view_box
        vb.plot_widget = self
        vb.signals.request_jump_to_data.connect(self._on_vb_jump)
        vb.signals.request_clear_plot.connect(self._on_vb_clear)
        vb.signals.request_auto_y.connect(self._on_vb_auto_y)
        vb.signals.request_set_cursor_mode.connect(self._on_vb_set_cursor_mode)
        vb.signals.request_show_cursor_value.connect(self._on_vb_show_cursor)
        vb.signals.request_hide_cursor_value.connect(self._on_vb_hide_cursor)
        vb.signals.request_set_row_height.connect(self._on_vb_set_row_height)
        vb.signals.request_set_all_row_height.connect(self._on_vb_set_all_row_height)
        vb.signals.request_copy_name.connect(self._on_vb_copy_name)
        vb.signals.request_variable_editor.connect(self._on_vb_var_editor)

    def _on_vb_jump(self, pw, ctx_x):
        if pw:
            pw.jump_to_data_impl(ctx_x)

    def _on_vb_clear(self, pw):
        if pw:
            pw.clear_plot_item()
            if pw.window():
                pw.window().request_mark_stats_refresh(immediate=True)

    def _on_vb_auto_y(self, pw):
        if pw and pw.window() and hasattr(pw.window(), "auto_y_in_x_range"):
            pw.window().auto_y_in_x_range()

    def _on_vb_set_cursor_mode(self, mode, pw, ctx_x):
        if pw and pw.window() and hasattr(pw.window(), "set_cursor_mode"):
            pw.window().set_cursor_mode(mode, source_plot=pw, context_x=ctx_x)

    def _on_vb_show_cursor(self, pw):
        if pw and pw.window() and hasattr(pw.window(), "cursor_values_hidden"):
            pw.window().cursor_values_hidden = False
            if pw.window().cursor_btn.isChecked():
                for c in pw.window().plot_widgets:
                    c.plot_widget.toggle_cursor(True)

    def _on_vb_hide_cursor(self, pw):
        if pw and pw.window() and hasattr(pw.window(), "cursor_values_hidden"):
            pw.window().cursor_values_hidden = True
            if pw.window().cursor_btn.isChecked():
                for c in pw.window().plot_widgets:
                    c.plot_widget.toggle_cursor(False, hide_values_only=True)

    def _on_vb_set_row_height(self, pct, pw):
        if pw and pw.window() and hasattr(pw.window(), "plot_widgets"):
            w = pw.window()
            for idx, c in enumerate(w.plot_widgets):
                if c.plot_widget is pw:
                    row, _ = divmod(idx, w._plot_col_max_default)
                    w.set_row_height(row, pct)
                    break

    def _on_vb_set_all_row_height(self, pct):
        w = self.window()
        if w and hasattr(w, "set_all_row_height"):
            w.set_all_row_height(pct)

    def _on_vb_copy_name(self, pw):
        if not pw:
            return
        var_names = []
        if getattr(pw, "is_multi_curve_mode", False) and pw.curves:
            var_names = list(pw.curves.keys())
        elif getattr(pw, "y_name", ""):
            var_names = [pw.y_name]
        if var_names:
            from PySide6.QtWidgets import QApplication as _QA
            _QA.clipboard().setText(" ".join(var_names))

    def _on_vb_var_editor(self, pw):
        if pw:
            from src.ui.plot_variable_editor import PlotVariableEditorDialog
            dialog = PlotVariableEditorDialog(pw, pw.window() if pw.window() and hasattr(pw.window(), "loader") else None)
            dialog.show()
            dialog.raise_()


