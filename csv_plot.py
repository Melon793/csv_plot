from __future__ import annotations
import sys
import os
import weakref
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from threading import Lock
from typing import Any

if sys.platform == "darwin":  # macOS
    # 屏蔽 macOS ICC 警告
    os.environ["QT_LOGGING_RULES"] = (
        "qt6ct.debug=false; "      # 原来想关的 qt6ct 日志
        "qt.gui.icc=false"         # 关闭 ICC 解析相关日志
    )

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

from src.ui.drag_drop import VAR_SEPARATOR, parse_var_names_from_mimedata
from src.ui.widgets.custom_viewbox import CustomViewBox
from src.core.config import (safe_callback, DEFAULT_PADDING_VAL_X, DEFAULT_PADDING_VAL_Y, XRANGE_THRESHOLD_FOR_SYMBOLS, FACTOR_SCROLL_ZOOM, MIN_INDEX_LENGTH, DEFAULT_LINE_WIDTH, THICK_LINE_WIDTH, THIN_LINE_WIDTH, UI_DEBOUNCE_DELAY_MS, PLOT_ROW_MAX_DEFAULT, PLOT_COL_MAX_DEFAULT, PLOT_ROW_CURRENT_DEFAULT, PLOT_COL_CURRENT_DEFAULT, _evaluate_float32_safety, DEFAULT_SHOW_X_AXIS_LABEL)
from src.core.data_types import CurveInfo
from src.core.scheduler import UnifiedUpdateScheduler
from src.ui.table_dialog import DataTableDialog, DropOverlay
from src.ui.variable_list import MyTableWidget

from src.core.logger import LogManager, get_logger
from src.ui.dialogs.log_window import LogWindow


from PySide6.QtCore import Qt, QMargins, QTimer, QPoint, QPointF, QSize, QRect, QRectF, QItemSelectionModel, QSignalBlocker
from PySide6.QtGui import QFontMetrics, QPen, QColor, QIcon, QFont, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QToolButton, QAbstractItemView, QLabel, QLineEdit,
    QMessageBox, QSizePolicy, QGraphicsLinearLayout, QGraphicsProxyWidget, QGraphicsWidget, QRubberBand, QSplitter, QMenu,
)
import pyqtgraph as pg

# 主界面
# 屏幕边距系数（用于自动选择窗口大小时，窗口不超过屏幕尺寸的比例）
SCREEN_WITDH_MARGIN = 0.3
SCREEN_HEIGHT_MARGIN = 0.3

# PyInstaller 解包目录
def resource_path(relative_path: str) -> Path:
    """
    获取打包后的资源文件路径
    
    用于处理PyInstaller打包后的资源文件路径问题
    在开发环境中返回相对路径，在打包环境中返回临时解包路径
    (兼容 PyInstaller/Nuitka/PyOxidizer Standalone)
    Args:
        relative_path: 资源文件的相对路径
        
    Returns:
        Path: 正确的资源文件路径
    """
    if hasattr(sys, "_MEIPASS"):
        # 模式 1: PyInstaller OneFile 模式
        return Path(os.path.join(sys._MEIPASS, relative_path))
    
    elif getattr(sys, "frozen", False):
        # 模式 2: 其他 Standalone 模式 (PyOxidizer/Nuitka)
        # 资源文件通常位于可执行文件所在目录
        return Path(os.path.dirname(sys.executable)) / relative_path
        
    else:
        # 模式 3: 开发环境
        return Path(relative_path)

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
    def __init__(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
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
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info={},synchronizer=None):
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
        """更新 X 轴标签文本"""
        axis = self.plot_item.getAxis('bottom')
        if DEFAULT_SHOW_X_AXIS_LABEL:
            label = self.time_axis_label if self.time_axis_label else "Index"
            axis.setLabel(label)
            axis.showLabel(True)
        else:
            axis.showLabel(False)
        
    def jump_to_data_impl(self, x):
        # 检查是否有数据
        has_data = False
        var_names = []
        
        # 收集所有要显示的变量名
        if self.is_multi_curve_mode and self.curves:
            # 多曲线模式：使用curves字典中的所有变量
            var_names = list(self.curves.keys())
            has_data = len(var_names) > 0
        elif self.curve and self.y_name:
            # 单曲线模式
            var_names = [self.y_name]
            has_data = True
        
        if not has_data:
            # 没有曲线，直接返回
            return

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
        main_window = self.window()
        is_mdf = (
            main_window is not None
            and hasattr(main_window, 'loader')
            and main_window.loader is not None
            and getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
        )

        has_own_data = bool(self.curve or self.curves)

        # 获取 x_values（仅当有自身数据时）
        if has_own_data:
            self.axis_x.setTicks(None)
            self.axis_y.setTicks(None)

            if self.is_multi_curve_mode and self.curves:
                x_arrays = self._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    x_values = np.concatenate(x_arrays)
                else:
                    x_values = None
            else:
                if self.original_index_x is not None:
                    x_values = self.offset + self.factor * self.original_index_x
                elif self.curve:
                    x_data, _ = self.curve.getData()
                    x_values = x_data if x_data is not None else None
                else:
                    x_values = None

            if x_values is not None:
                own_min_x = np.min(x_values)
                own_max_x = np.max(x_values)
            else:
                own_min_x = None
                own_max_x = None
        else:
            x_values = None
            own_min_x = None
            own_max_x = None

        # 合并内部和外部范围
        if external_xmin is not None:
            min_x = min(own_min_x, external_xmin) if own_min_x is not None else external_xmin
        else:
            min_x = own_min_x

        if external_xmax is not None:
            max_x = max(own_max_x, external_xmax) if own_max_x is not None else external_xmax
        else:
            max_x = own_max_x

        if min_x is None or max_x is None:
            return False

        if not has_own_data:
            self.axis_x.setTicks(None)
            self.axis_y.setTicks(None)

        padding_xVal = DEFAULT_PADDING_VAL_X  
        padding_yVal = 0.5
        
        if has_own_data:
            if self.is_multi_curve_mode and self.curves:
                y_arrays = self._collect_visible_curve_arrays('y_data')
                if y_arrays:
                    combined = np.concatenate(y_arrays)
                    if combined.size:
                        min_y = np.nanmin(combined)
                        max_y = np.nanmax(combined)
                    else:
                        min_y, max_y = 0, 1
                else:
                    min_y, max_y = 0, 1
            else:
                if self.original_y is not None:
                    special_limits = self.handle_single_point_limits(x_values, self.original_y)
                    if special_limits:
                        min_x, max_x, min_y, max_y = special_limits
                    else:
                        min_y = np.nanmin(self.original_y)
                        max_y = np.nanmax(self.original_y)
                elif self.curve:
                    _, y_data = self.curve.getData()
                    if y_data is not None:
                        min_y = np.nanmin(y_data)
                        max_y = np.nanmax(y_data)
                    else:
                        min_y, max_y = 0, 1
                else:
                    min_y, max_y = 0, 1
        else:
            min_y, max_y = 0, 1

        limits_xMin = min_x - padding_xVal * (max_x - min_x)
        limits_xMax = max_x + padding_xVal * (max_x - min_x)

        self.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)
        self._set_safe_y_range(min_y, max_y)

        minXRange_val = self._get_min_x_range_value()
        if is_mdf:
            self.plot_item.setLimits(minXRange=minXRange_val)
        else:
            self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val)

        # self.window().sync_all_x_limits(limits_xMin, limits_xMax, min(3,len(x_values))*self.factor)
        self._set_vline_bounds([min_x, max_x])

        # 在设置完新范围后，立即直接调用样式更新函数。
        self._queue_ui_refresh(immediate=True)
        self.plot_item.update()
        self._update_cursor_after_plot(min_x, max_x)

        return True

    def auto_y_in_x_range(self):
        vb=self.view_box
        vb.enableAutoRange(axis=vb.YAxis, enable=True)
        vb.plot_widget.axis_y.setTicks(None)

    def update_left_header(self, left_text=None):
        """更新顶部文本内容"""
        if left_text is not None:
            self.label_left.setText(left_text)

    def update_right_header(self, right_text=None):
        """更新顶部文本内容（已移除右侧label）"""
        # 右侧label已被移除，此方法保留以兼容现有代码
        pass

    def _get_safe_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        """
        确保X轴范围非零，如果 min_x == max_x，则基于 factor 扩展。
        """
        if min_x == max_x:
            min_x_safe = min_x - 0.5 * self.factor
            max_x_safe = max_x + 0.5 * self.factor
            return min_x_safe, max_x_safe
        return min_x, max_x
    
    def _get_min_x_range_value(self) -> float:
        """
        根据全局最大数据点密度计算最小的可缩放 X 范围 (minXRange)。
        优先从 MainWindow 读取 _global_max_density，fallback 为 MIN_INDEX_LENGTH。
        """

        main_window = self.window()
        if main_window is not None and hasattr(main_window, '_global_max_density'):
            density = main_window._global_max_density
        else:
            density = 0.0

        if density > 0:
            result = MIN_INDEX_LENGTH / density
        else:
            # 单点数据时，使用较小的 minXRange，避免自动扩展范围
            result = 1.0  # 与单点扩展范围 (0.5, 1.5) 匹配
        
        return result

    def _set_x_limits_with_min_range(self, limits_xMin: float | None, limits_xMax: float | None):
        """
        统一设置 X 轴的 limits 和 minXRange。
        """
        minXRange_val = self._get_min_x_range_value()
        self.plot_item.setLimits(xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val)

    def _set_min_x_range(self, minXRange: float):
        self.plot_item.setLimits(minXRange=minXRange)

    def _recalc_max_point_density(self):
        densities: list[float] = []
        for ci in self.curves.values():
            if ci.point_density > 0:
                densities.append(ci.point_density)
        if not densities and self.curve is not None and self.original_y is not None:
            n = len(self.original_y)
            if n > 1 and self.original_index_x is not None:
                x_span = self.offset + self.factor * float(np.max(self.original_index_x)) - (
                    self.offset + self.factor * float(np.min(self.original_index_x)))
                if x_span > 0:
                    densities.append(n / x_span)
        self._max_point_density = max(densities) if densities else 0.0

    def _set_safe_y_range(self, min_y: float, max_y: float, set_limits: bool = True):
        """
        设置 Y 轴的 viewRange 和 limits，自动处理 NaN 或恒定值。
        
        Args:
            min_y: Y轴最小值
            max_y: Y轴最大值
            set_limits: 是否同时设置y轴limits，默认为True。当为False时只设置viewRange。
        """
        
        # Y 轴 limit 的内外边距 (0.5 表示上下各扩展 50%)
        padding_yVal_limit = 0.5 

        if np.isnan(min_y) or np.isnan(max_y) or min_y == max_y:
            # 如果是 NaN 或恒定值
            y_center = min_y if not np.isnan(min_y) else 0
            # 保证最小范围为 1.0，或者为中心值的 20%
            y_range_half = (1.0 if y_center == 0 else abs(y_center) * 0.2)
            
            y_min_view = y_center - y_range_half
            y_max_view = y_center + y_range_half
            
            # Limits 也使用这个扩展后的范围
            y_min_limit = y_min_view
            y_max_limit = y_max_view
        else:
            # 如果是正常范围
            y_min_view = min_y
            y_max_view = max_y
            
            # Limits 应用 50% 的外边距
            y_range = max_y - min_y
            y_min_limit = min_y - padding_yVal_limit * y_range
            y_max_limit = max_y + padding_yVal_limit * y_range

        # 只在需要时设置limits
        if set_limits:
            self.plot_item.setLimits(yMin=y_min_limit, yMax=y_max_limit)
        # ViewRange 使用 PADDING_Y (默认0.1) 的内边距
        self.view_box.setYRange(y_min_view, y_max_view, padding=DEFAULT_PADDING_VAL_Y)

    def reset_plot(self,index_xMin,index_xMax):

        self.plot_item.setLimits(xMin=None, xMax=None)  # 解除X轴限制
        self.plot_item.setLimits(yMin=None, yMax=None)  # 解除Y轴限制
        
        xMin = self.offset + self.factor * index_xMin
        xMax = self.offset + self.factor * index_xMax

        if not (np.isnan(xMax) or np.isinf(xMax)):
            xMin, xMax = self._get_safe_x_range(xMin, xMax)

            self.view_box.setXRange(xMin, xMax, padding=DEFAULT_PADDING_VAL_X)
            padding_xVal=DEFAULT_PADDING_VAL_X
            limits_xMin = xMin - padding_xVal * (xMax - xMin)
            limits_xMax = xMax + padding_xVal * (xMax - xMin)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

        self.view_box.setYRange(0,1,padding=DEFAULT_PADDING_VAL_Y) 
        self._set_vline_bounds([None, None]) 

        self.xMin = xMin
        self.xMax = xMax
        self.y_name = ''
        self.y_format = ''
        #self.plot_item.update()
        # 先清除cursor items（包括scene中的items）
        # 重置plot时完全清除对象池，避免复用异常状态的items
        self._clear_cursor_items(hide_only=False)
        self._safe_clear_plot_items() 
        self.axis_y.setLabel(text="")
        self.update_left_header("channel name")
        self.update_right_header("")

        self.curve = None
        self.original_index_x = None
        self.original_y = None


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
        """处理单点或所有点x坐标相同的特殊情况，避免x轴范围为0
        
        Args:
            x_values: x坐标数组
            y_values: y坐标数组
            
        Returns:
            tuple: (min_x, max_x, min_y, max_y) 或 None（正常情况不需要特殊处理）
        """
        if len(x_values) == 1:
            # 单点情况：扩展x轴范围
            x = x_values[0]
            min_x, max_x = self._get_safe_x_range(x, x)
            if len(y_values) == 1:
                y = y_values[0]
                min_y = y - 0.5 if y != 0 else -0.5
                max_y = y + 0.5 if y != 0 else 0.5
            else:
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
            return min_x, max_x, min_y, max_y
        else:
            # 检查是否所有x值都相同（多点但x坐标相同的情况）
            unique_x = set(x_values)
            if len(unique_x) == 1:
                # 所有点的x坐标相同，扩展x轴范围
                x = list(unique_x)[0]
                min_x, max_x = self._get_safe_x_range(x, x)
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
                return min_x, max_x, min_y, max_y
            else:
                # 正常情况：有多个不同的x值
                return None
        
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
        """
        判断cursor相关回调是否需要被暂时禁用

        当plot正在更新数据或主窗口处于新数据加载流程中时，所有cursor相关的信号都会跳过，
        以避免访问不完整的数据结构。
        【稳定性优化】添加版本号检查，确保数据一致性。
        """
        if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
            return True

        window = self.window()
        if window:
            # 检查是否正在加载新数据
            if getattr(window, '_is_loading_new_data', False):
                return True

            # 【版本号检查】确保数据版本一致
            current_version = getattr(window, '_data_version', 0)
            my_version = getattr(self, '_cached_data_version', 0)
            if my_version != 0 and my_version != current_version:
                return True  # 版本不匹配，说明正在加载中

        return False

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
    


    def set_xrange_with_link_handling(self, xmin, xmax,padding:float = 0):
        plot=self.plot_item
        # 1. 记录当前联动对象
        linked = plot.getViewBox().linkedView(0)
        
        # 2. 临时断开联动
        if linked is not None:
            plot.setXLink(None)
        
        # 3. 安全设置范围
        plot.getViewBox().enableAutoRange(x=False)
        plot.setXRange(xmin, xmax, padding=max(0,padding))
        
        # 4. 恢复联动
        if linked is not None:
            plot.setXLink(linked)

    def _get_cursor_mode(self):
        window = self.window()
        if window and hasattr(window, "cursor_mode"):
            return window.cursor_mode
        return "1 free cursor"

    def _get_cursor_x_positions(self):
        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            if self.pinned_x_values and len(self.pinned_x_values) >= 2:
                return list(self.pinned_x_values[:2])
            positions = []
            if hasattr(self, "vline") and self.vline.isVisible():
                positions.append(self.vline.value())
            if hasattr(self, "vline2") and self.vline2.isVisible():
                positions.append(self.vline2.value())
            return positions
        if mode == "1 anchored cursor":
            if self.pinned_x_values:
                return [self.pinned_x_values[0]]
            if self.pinned_x_value is not None:
                return [self.pinned_x_value]
        if hasattr(self, "vline"):
            return [self.vline.value()]
        return []

    def _set_vline_visibility_for_mode(self, visible: bool, mode: str):
        if not hasattr(self, "vline"):
            return
        if mode == "2 anchored cursor":
            self.vline.setVisible(visible)
            if hasattr(self, "vline2"):
                self.vline2.setVisible(visible)
        else:
            self.vline.setVisible(visible)
            if hasattr(self, "vline2"):
                self.vline2.setVisible(False)

    def _set_vline_bounds(self, bounds):
        if hasattr(self, "vline"):
            self.vline.setBounds(bounds)
        if hasattr(self, "vline2"):
            self.vline2.setBounds(bounds)

    def apply_cursor_mode(self, mode, pinned_x_values):
        if mode == "1 free cursor":
            self.is_cursor_pinned = False
            self.pinned_x_value = None
            self.pinned_index_value = None
            self.pinned_x_values = []
            self.pinned_index_values = []
            if hasattr(self, "vline"):
                self.vline.setMovable(False)
            if hasattr(self, "vline2"):
                self.vline2.setMovable(False)
            if hasattr(self.view_box, "is_cursor_pinned"):
                self.view_box.is_cursor_pinned = False
            self._set_vline_visibility_for_mode(True, mode)
            return

        if mode == "1 anchored cursor":
            self.is_cursor_pinned = True
            self.pinned_x_values = list(pinned_x_values[:1]) if pinned_x_values else self.pinned_x_values[:1]
            if self.pinned_x_values:
                self.pinned_x_value = self.pinned_x_values[0]
            if self.factor != 0 and self.pinned_x_value is not None:
                self.pinned_index_value = (self.pinned_x_value - self.offset) / self.factor
            else:
                self.pinned_index_value = None
            self.pinned_index_values = [self.pinned_index_value] if self.pinned_index_value is not None else []
            if hasattr(self, "vline") and self.pinned_x_value is not None:
                self.vline.setMovable(True)
                with QSignalBlocker(self.vline):
                    self.vline.setPos(self.pinned_x_value)
            if hasattr(self, "vline2"):
                self.vline2.setMovable(False)
            if hasattr(self.view_box, "is_cursor_pinned"):
                self.view_box.is_cursor_pinned = True
            self._set_vline_visibility_for_mode(True, mode)
            return

        if mode == "2 anchored cursor":
            self.is_cursor_pinned = True
            if pinned_x_values and len(pinned_x_values) >= 2:
                self.pinned_x_values = list(pinned_x_values[:2])
            elif len(self.pinned_x_values) >= 2:
                self.pinned_x_values = list(self.pinned_x_values[:2])
            elif len(self.pinned_x_values) == 1:
                self.pinned_x_values = [self.pinned_x_values[0], self.pinned_x_values[0]]
            else:
                view_min, view_max = self.view_box.viewRange()[0]
                if view_min is not None and view_max is not None:
                    x1 = view_min + (view_max - view_min) / 3
                    x2 = view_min + 2 * (view_max - view_min) / 3
                    self.pinned_x_values = [x1, x2]
                else:
                    self.pinned_x_values = [0.0, 0.0]
            self.pinned_x_value = self.pinned_x_values[0]
            self.pinned_index_values = []
            for x_val in self.pinned_x_values:
                if self.factor != 0:
                    self.pinned_index_values.append((x_val - self.offset) / self.factor)
            if hasattr(self, "vline"):
                self.vline.setMovable(True)
            if hasattr(self, "vline2"):
                self.vline2.setMovable(True)
            if hasattr(self, "vline") and self.pinned_x_values:
                with QSignalBlocker(self.vline):
                    self.vline.setPos(self.pinned_x_values[0])
            if hasattr(self, "vline2") and len(self.pinned_x_values) > 1:
                with QSignalBlocker(self.vline2):
                    self.vline2.setPos(self.pinned_x_values[1])
            if hasattr(self.view_box, "is_cursor_pinned"):
                self.view_box.is_cursor_pinned = True
            self._set_vline_visibility_for_mode(True, mode)
            return

    def update_cursor_label(self):
        """
        更新光标标签位置和内容

        【稳定性优化】使用循环替代递归，限制最大重试次数，防止栈溢出。
        """
        MAX_RETRIES = 3  # 最大重试次数
        retry_count = 0

        while retry_count < MAX_RETRIES:

            if self._is_cursor_update_locked():
                return

            if self._cursor_label_busy:
                self._cursor_label_dirty = True
                return

            self._cursor_label_busy = True
            self._cursor_label_dirty = False  # 进入时清除dirty

            try:
                # 统一使用多曲线样式的cursor显示
                self._update_multi_curve_cursor_label()
            except (RuntimeError, AttributeError) as e:
                # 对象可能已被销毁
                pass
            finally:
                self._cursor_label_busy = False

            # 检查是否需要重试
            if self._cursor_label_dirty:
                self._cursor_label_dirty = False
                retry_count += 1
                continue  # 循环重试，而非递归
            else:
                break  # 无需重试，退出

    
    def _update_single_curve_cursor_label(self):
        """更新单曲线模式的光标标签"""
        if len(self.plot_item.listDataItems()) == 0:
            self.update_right_header("")
            return
        
        try:
            x = self.vline.value()           
            curve = self.plot_item.listDataItems()[0]
            x_data, y_data = curve.getData()
            if x_data is None or len(x_data) == 0:
                self.update_right_header("")
                return
            x = np.clip(x, x_data.min(), x_data.max())
            idx = np.argmin(np.abs(x_data - x))
            y_val = y_data[idx]
            x_str = self._significant_decimal_format_str(value=float(x),ref=self.factor)
            if self.y_format == 'enum':
                window = self.window()
                enum_map = getattr(window, '_enum_text_maps', {}).get(self.y_name, {})
                y_str = enum_map.get(int(y_val), str(y_val))
                self.update_right_header(f"x={x_str}, y={y_str}")
            elif self.y_format == 's':
                time_str=self.sInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x_str}, y={time_str}")
            elif self.y_format == 'date':
                date_str=self.dateInt_to_fmtStr(y_val)
                self.update_right_header(f"x={x_str}, y={date_str}")
            else:
                self.update_right_header(f"x={x_str}, y={y_val:.5g}")

        except Exception as e:
            print(f"Cursor update error: {e}")
            self.update_right_header("")
    
    def _get_circle_from_pool(self, index):
        """从对象池获取ScatterPlotItem，如果不存在则创建
        
        使用对象池复用ScatterPlotItem，避免重复创建导致内存泄漏。
        每个索引位置对应一个ScatterPlotItem实例，用于在cursor交点处显示圆圈标记。
        
        Args:
            index: 对象池索引位置
            
        Returns:
            ScatterPlotItem: 从池中获取或新创建的圆圈标记对象
        """
        pool = self._cursor_item_pool['circles']
        
        # 如果池中已有该索引的对象，直接复用
        if index < len(pool):
            return pool[index]
        
        # 否则创建新对象并加入池
        circle = pg.ScatterPlotItem(
            symbol='o',
            size=8,
            brush=None
        )
        pool.append(circle)
        return circle
    
    def _get_label_from_pool(self, index):
        """从对象池获取TextItem，如果不存在则创建
        
        使用对象池复用TextItem，避免重复创建导致内存泄漏。
        每个索引位置对应一个TextItem实例，用于显示cursor交点处的y值标签。
        
        Args:
            index: 对象池索引位置
            
        Returns:
            TextItem: 从池中获取或新创建的文本标签对象
        """
        pool = self._cursor_item_pool['labels']
        
        # 如果池中已有该索引的对象，直接复用
        if index < len(pool):
            return pool[index]
        
        # 否则创建新对象并加入池
        label = pg.TextItem(
            color=(0, 0, 0),
            fill=pg.mkBrush(255, 255, 255, 220),
            anchor=(0.5, 0.5)
        )
        # label.setFont(QFont('Arial', 8))

        font = QApplication.font()  # 获取App的默认字体
        font.setPixelSize(11)     # 设置一个跨平台一致的逻辑像素大小 (11px)
        label.setFont(font)

        pool.append(label)
        return label
    
    def _get_x_label_from_pool(self, index: int):
        """获取 X 轴标签 TextItem（用于光标显示）"""
        pool = self._cursor_item_pool["x_labels"]
        if index < len(pool):
            return pool[index]

        x_label = pg.TextItem(
            color=(255, 255, 255),
            fill=pg.mkBrush(64, 64, 64, 230),
            border=pg.mkPen(128, 128, 128, width=1),
            anchor=(0.5, 0)
        )

        font = QApplication.font()
        font.setPixelSize(12)
        x_label.setFont(font)

        pool.append(x_label)
        return x_label

    def _clear_cursor_items(self, hide_only=True):
        """清除或隐藏所有cursor可视化元素

        默认模式下只隐藏元素（供下次复用），完全清除模式下才会删除对象。
        这种策略通过对象池复用机制避免频繁创建/销毁对象导致的内存泄漏。
        【稳定性优化】使用延迟删除队列，避免deleteLater与即时访问冲突。

        Args:
            hide_only: 如果为True，只隐藏所有元素（默认，用于复用）
                      如果为False，完全删除所有元素和对象池（用于切换数据文件等场景）
        """
        # 【安全检查】确保关键对象存在
        if not hasattr(self, 'multi_cursor_items') or not hasattr(self, 'plot_item'):
            return

        # 【修复QPainter错误】先隐藏所有item，避免清理过程中触发绘制
        for item in self.multi_cursor_items:
            try:
                if item is not None:
                    item.setVisible(False)
            except (RuntimeError, AttributeError):
                pass  # 对象可能已被销毁

        # 分类处理：对象池中的元素清除数据，非池对象删除
        for item in self.multi_cursor_items:
            try:
                item_type = type(item).__name__
                if item_type == 'ScatterPlotItem':
                    # 对象池中的圆圈标记：清除数据
                    try:
                        item.clear()  # 清除ScatterPlotItem的数据，释放内存
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self._cursor_item_pool.get('x_labels', []):
                    # X轴标签：清空文本
                    try:
                        item.setText("")  # 清空文本，释放字符串占用的内存
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self._cursor_item_pool.get('labels', []):
                    # 对象池中的y值标签：清空文本
                    try:
                        item.setText("")  # 清空文本，释放字符串占用的内存
                    except (RuntimeError, AttributeError):
                        pass
                else:
                    # 不在对象池中的项（理论上不应该存在）：加入待删除队列
                    self._queue_item_for_deletion(item)
            except Exception:
                pass  # 忽略 queue_item 清理过程中的错误

        # 清空当前使用列表
        self.multi_cursor_items.clear()

        if not hide_only:
            # 完全清除模式（仅在真正需要清理时使用，如切换数据文件）
            # 将对象池中的对象加入待删除队列
            for circle in self._cursor_item_pool.get('circles', []):
                self._queue_item_for_deletion(circle)

            for label in self._cursor_item_pool.get('labels', []):
                self._queue_item_for_deletion(label)

            for x_label in self._cursor_item_pool.get('x_labels', []):
                self._queue_item_for_deletion(x_label)

            # 重置对象池
            self._cursor_item_pool = {
                'circles': [],
                'labels': [],
                'x_labels': []
            }

            # 延迟执行实际删除（等待当前事件循环完成）
            if self._pending_delete_items and not self._cleanup_timer.isActive():
                self._cleanup_timer.start(100)  # 100ms后执行

    def _queue_item_for_deletion(self, item):
        """将item加入待删除队列"""
        if item is not None and item not in self._pending_delete_items:
            try:
                item.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            self._pending_delete_items.append(item)

    def _process_pending_deletes(self):
        """安全地处理待删除队列 - 延迟删除回调"""
        if self._is_updating_data or self._is_being_destroyed:
            # 数据正在更新，延迟处理
            if self._pending_delete_items:
                self._cleanup_timer.start(100)
            return

        items_to_delete = self._pending_delete_items.copy()
        self._pending_delete_items.clear()

        for item in items_to_delete:
            try:
                if item is None:
                    continue

                # 安全地从scene移除
                try:
                    scene = item.scene()
                    if scene is not None:
                        scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass  # scene可能已被销毁

                # 安全地删除
                try:
                    if hasattr(item, 'deleteLater'):
                        item.deleteLater()
                except (RuntimeError, AttributeError):
                    pass

            except Exception as e:
                pass

    def _collect_visible_curve_arrays(self, key: str) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        if not getattr(self, 'curves', None):
            return arrays
        for ci in self.curves.values():
            if not ci.visible:
                continue
            data = getattr(ci, key, None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.size == 0:
                continue
            arrays.append(arr)
        return arrays

    def _collect_visible_curve_pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        if not getattr(self, 'curves', None):
            return pairs
        for ci in self.curves.values():
            if not ci.visible:
                continue
            x_data = ci.x_data
            y_data = ci.y_data
            if x_data is None or y_data is None:
                continue
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            if x_arr.size == 0 or y_arr.size == 0:
                continue
            pairs.append((x_arr, y_arr))
        return pairs

    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple[float | None, float | None]:
        """
        返回当前 plot 中曲线的 X 轴范围

        Args:
            curves_filter: "visible" — 仅可见曲线；"all" — 所有曲线（含隐藏）

        Returns:
            (min_x, max_x) 或 (None, None) 当无数据时
        """
        mins: list[float] = []
        maxs: list[float] = []

        if self.curves:
            for ci in self.curves.values():
                if curves_filter == "visible" and not ci.visible:
                    continue
                mins.append(ci.x_min)
                maxs.append(ci.x_max)
        elif self.curve and self.y_name:
            if self.original_index_x is not None:
                x_data = self.offset + self.factor * self.original_index_x
            else:
                x_data, _ = self.curve.getData()
                if x_data is None:
                    return (None, None)
            mins.append(float(np.min(x_data)))
            maxs.append(float(np.max(x_data)))

        if not mins:
            return (None, None)
        return (min(mins), max(maxs))

    def _safe_clear_plot_items(self):
        """安全地清理所有plot items，避免scene不匹配问题
        
        【内存优化】清除曲线时释放其数据和样式缓存
        """
        try:
            # 【安全检查】确保plot_item存在且有效
            if not hasattr(self, 'plot_item') or self.plot_item is None:
                return
            
            current_scene = self.plot_item.scene()
            
            if current_scene is not None:
                # 获取所有items
                all_items = current_scene.items()
                
                # 手动清理所有items，避免使用clearPlots()
                items_removed = 0
                for i, item in enumerate(all_items):
                    try:
                        # 检查item是否仍然有效
                        item_scene = item.scene()
                        if item_scene == current_scene:
                            # 只移除数据曲线，不移除cursor相关items（由_clear_cursor_items管理）
                            should_remove = False
                            item_type = type(item).__name__
                            
                            # 检查是否是数据曲线（PlotDataItem）
                            if hasattr(item, 'getData') and hasattr(item, 'opts'):
                                # 确保不是坐标轴
                                if not hasattr(item, 'setLabel'):
                                    should_remove = True
                                    
                                    # 清除曲线的缓存数据，释放内存
                                    if hasattr(item, '_cached_pen_key'):
                                        delattr(item, '_cached_pen_key')
                                    if hasattr(item, '_has_symbols'):
                                        delattr(item, '_has_symbols')
                                    
                                    # 清除曲线数据
                                    try:
                                        item.clear()
                                    except Exception:
                                        _widget_logger.debug("clear_all_plots: item.clear() 异常")
                            
                            # 注意：不在这里清理TextItem和ScatterPlotItem
                            # 这些cursor相关的items由_clear_cursor_items()管理
                            
                            if should_remove:
                                current_scene.removeItem(item)
                                items_removed += 1
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass  # 单个 scene item 清理异常，继续处理其他
                
            
            
        except Exception as e:
            _widget_logger.debug("clear_all_plots_manager 异常: %s", e)
    
    def _update_multi_curve_cursor_label(self):
        """更新多曲线光标标签（多光标模式）"""
        if getattr(self, "_is_interacting", False):
            return

        import time
        current_time = time.time()

        if self._adaptive_throttle_enabled and hasattr(self, "curves"):
            curve_count = len(self.curves)
            adaptive_throttle = min(0.016 + curve_count * 0.002, 0.1)
        else:
            adaptive_throttle = self._cursor_update_throttle

        if hasattr(self, "_last_cursor_update_time"):
            time_since_last = current_time - self._last_cursor_update_time
            if time_since_last < adaptive_throttle:
                return
        self._last_cursor_update_time = current_time

        self._clear_cursor_items()

        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            vline_visible = bool(self.vline.isVisible() or self.vline2.isVisible())
        else:
            vline_visible = self.vline.isVisible()
        if not vline_visible:
            self.update_right_header("")
            return

        has_attr = hasattr(self, "show_values_only")
        show_values_only = self.show_values_only if has_attr else False
        if has_attr and show_values_only:
            self._show_x_position_only()
            return

        if not self.curves and not self.curve:
            self.update_right_header("")
            return

        x_positions = self._get_cursor_x_positions()
        if not x_positions:
            self.update_right_header("")
            return

        try:
            cursor_values = []
            (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()

            curves_to_process = []
            if self.curves:
                for var_name, ci in self.curves.items():
                    if not ci.visible:
                        continue
                    window = self.window()
                    curves_to_process.append({
                        "var_name": var_name,
                        "x_data": ci.x_data,
                        "y_data": ci.y_data,
                        "color": ci.color,
                        "y_format": ci.y_format,
                        "unit": self.units.get(var_name, ""),
                        "enum_map": getattr(window, '_enum_text_maps', {}).get(var_name, {})
                    })
            elif not self.is_multi_curve_mode and self.curve and self.y_name:
                x_data, y_data = self.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    curve_color = "blue"
                    try:
                        if hasattr(self.curve, "opts") and "pen" in self.curve.opts:
                            pen = self.curve.opts["pen"]
                            if hasattr(pen, "color"):
                                curve_color = pen.color().name()
                    except Exception:
                        pass  # pen.color 可能不可用
                    window = self.window()
                    curves_to_process.append({
                        "var_name": self.y_name,
                        "x_data": x_data,
                        "y_data": y_data,
                        "color": curve_color,
                        "y_format": self.y_format,
                        "unit": self.units.get(self.y_name, ""),
                        "enum_map": getattr(window, '_enum_text_maps', {}).get(self.y_name, {})
                    })

            for x in x_positions:
                if x < x_min or x > x_max:
                    continue
                for curve_data in curves_to_process:
                    var_name = curve_data["var_name"]
                    x_data = curve_data["x_data"]
                    y_data = curve_data["y_data"]
                    color = curve_data["color"]
                    y_format = curve_data["y_format"]

                    if x_data is None or len(x_data) == 0:
                        continue
                    if x < x_data.min() or x > x_data.max():
                        continue

                    try:
                        idx = np.searchsorted(x_data, x, side="left")
                        if idx >= len(x_data):
                            idx = len(x_data) - 1
                        elif idx > 0:
                            if abs(x_data[idx - 1] - x) < abs(x_data[idx] - x):
                                idx = idx - 1
                    except (ValueError, TypeError):
                        idx = np.argmin(np.abs(x_data - x))

                    y_val = y_data[idx]
                    x_actual = x_data[idx]
                    if np.isnan(x_actual) or np.isnan(y_val):
                        continue
                    if y_val < y_min or y_val > y_max:
                        continue

                    if y_format == "enum":
                        enum_map = curve_data.get("enum_map", {})
                        y_str = enum_map.get(int(y_val), str(y_val))
                    elif y_format == "s":
                        y_str = self.sInt_to_fmtStr(y_val)
                    elif y_format == "date":
                        y_str = self.dateInt_to_fmtStr(y_val)
                    else:
                        y_str = f"{y_val:.5g}"

                    cursor_values.append({
                        "var_name": var_name,
                        "x_pos": x_actual,
                        "y_pos": y_val,
                        "y_value": y_str,
                        "color": color
                    })

                    circle = self._get_circle_from_pool(len(cursor_values) - 1)
                    circle.clear()
                    circle.setData([x_actual], [y_val])
                    if not hasattr(circle, "_cached_color") or circle._cached_color != color:
                        pen = pg.mkPen(color, width=1.5)
                        circle.setPen(pen)
                        circle._cached_color = color
                    circle.setVisible(True)
                    circle.setZValue(200)
                    circle_scene = circle.scene()
                    plot_scene = self.plot_item.scene()
                    if circle_scene != plot_scene:
                        if circle_scene is not None:
                            circle_scene.removeItem(circle)
                        self.plot_item.addItem(circle, ignoreBounds=True)
                    self.multi_cursor_items.append(circle)

            self._position_labels_avoid_overlap(cursor_values, x_min, x_max, y_min, y_max)

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)
                view_rect = self.plot_item.vb.sceneBoundingRect()
                scene_point = self.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)
                scene = self.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)
                self.multi_cursor_items.append(x_info_item)

        except Exception as e:
            print(f"Multi-curve cursor update error: {e}")
            self.update_right_header("")

    def _position_labels_avoid_overlap(self, cursor_values: list[dict], x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """优化的标签定位算法，使用对角线位置避免遮挡曲线。
        
        使用4个候选位置策略（右上、左上、右下、左下）依次尝试，
        选择第一个在边界内的位置。如果都超出边界，则约束在右上位置。
        
        【内存优化】使用对象池复用TextItem，避免频繁创建销毁
        
        Args:
            cursor_values: 光标值列表，每个元素为包含var_name, x_pos, y_pos, y_value, color的字典
            x_min: 视图x轴最小值（数据坐标）
            x_max: 视图x轴最大值（数据坐标）
            y_min: 视图y轴最小值（数据坐标）
            y_max: 视图y轴最大值（数据坐标）
        """
        if not cursor_values:
            return
        
        # 计算视图范围，用于边界检查
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 获取实际视图尺寸
        view_box = self.plot_item.getViewBox()
        view_width_pixels = max(1, view_box.width())  # 防止除零
        view_height_pixels = max(1, view_box.height())
        
        # 预计算转换比例（像素 -> 数据坐标）
        pixel_to_data_x = x_range / view_width_pixels
        pixel_to_data_y = y_range / view_height_pixels
        
        # 设置固定的屏幕像素偏移
        gap_pixels = 5  # 文本框左边缘距离cursor的水平像素间隔
        vertical_gap_pixels = 10  # 垂直像素间隔
        
        # 获取TextItem的字体来动态计算标签尺寸（缓存font_metrics避免重复创建）
        if not hasattr(self, '_cached_font_metrics'):
            sample_text_item = self._get_label_from_pool(0)
            text_font = sample_text_item.textItem.font()
            self._cached_font_metrics = QFontMetrics(text_font)
            self._cached_label_height_pixels = self._cached_font_metrics.height() + 6
        
        font_metrics = self._cached_font_metrics
        label_height_pixels = self._cached_label_height_pixels
        label_height_data = label_height_pixels * pixel_to_data_y  # 在循环外计算
        
        for idx, item in enumerate(cursor_values):
            x_pos = item['x_pos']
            y_pos = item['y_pos']
            y_value = item['y_value']
            color = item['color']
            
            # 从对象池获取TextItem并更新其属性
            text_item = self._get_label_from_pool(idx)
            text_item.setText(y_value)
            
            # 复用pen对象或只在颜色变化时创建
            if not hasattr(text_item, '_cached_border_color') or text_item._cached_border_color != color:
                border_pen = pg.mkPen(color, width=1.5)
                text_item.border = border_pen
                text_item._cached_border_color = color
            text_item.setVisible(True)
            
            # 根据实际文本内容动态计算标签宽度
            text_width = font_metrics.horizontalAdvance(y_value)
            label_width_pixels = text_width + 12
            label_width_data = label_width_pixels * pixel_to_data_x  # 在循环内计算宽度（动态的）
            
            # 将数据坐标转换为场景坐标（屏幕像素）
            cursor_scene_pos = view_box.mapViewToScene(QPointF(x_pos, y_pos))
            cursor_scene_x = cursor_scene_pos.x()
            cursor_scene_y = cursor_scene_pos.y()
            
            # 计算文本框中心的偏移（TextItem的anchor=(0.5, 0.5)）
            offset_x_right = gap_pixels + label_width_pixels / 2
            offset_x_left = -(gap_pixels + label_width_pixels / 2)
            offset_y_up = -(vertical_gap_pixels + label_height_pixels / 2)
            offset_y_down = vertical_gap_pixels + label_height_pixels / 2
            
            # 尝试4个候选位置（右上、左上、右下、左下）
            strategies = [
                (offset_x_right, offset_y_up, "右上"),
                (offset_x_left, offset_y_up, "左上"),
                (offset_x_right, offset_y_down, "右下"),
                (offset_x_left, offset_y_down, "左下"),
            ]
            
            label_scene_x, label_scene_y = None, None
            
            for strategy_idx, (dx_pixels, dy_pixels, name) in enumerate(strategies):
                candidate_scene_x = cursor_scene_x + dx_pixels
                candidate_scene_y = cursor_scene_y + dy_pixels
                
                # 转换回数据坐标检查边界
                candidate_data_pos = view_box.mapSceneToView(QPointF(candidate_scene_x, candidate_scene_y))
                candidate_x = candidate_data_pos.x()
                candidate_y = candidate_data_pos.y()
                
                # 检查是否在数据范围内
                left_ok = candidate_x - label_width_data * 0.5 >= x_min
                right_ok = candidate_x + label_width_data * 0.5 <= x_max
                bottom_ok = candidate_y - label_height_data * 0.5 >= y_min
                top_ok = candidate_y + label_height_data * 0.5 <= y_max
                
                in_bounds = left_ok and right_ok and bottom_ok and top_ok
                
                if in_bounds:
                    label_scene_x = candidate_scene_x
                    label_scene_y = candidate_scene_y
                    label_x = candidate_x
                    label_y = candidate_y
                    break
            
            # 如果所有策略都超界，默认使用右上并约束在边界内
            if label_scene_x is None:
                label_scene_x = cursor_scene_x + offset_x_right
                label_scene_y = cursor_scene_y + offset_y_up
                
                label_data_pos = view_box.mapSceneToView(QPointF(label_scene_x, label_scene_y))
                label_x = label_data_pos.x()
                label_y = label_data_pos.y()
                
                label_x = max(x_min + label_width_data * 0.5, 
                             min(x_max - label_width_data * 0.5, label_x))
                label_y = max(y_min + label_height_data * 0.5, 
                             min(y_max - label_height_data * 0.5, label_y))
            
            # 边缘避让逻辑：防止标签在边缘抖动
            edge_margin_strict = label_height_data * 1.5
            y_center = (y_min + y_max) / 2
            y_quarter_upper = y_min + (y_max - y_min) * 0.25
            y_quarter_lower = y_max - (y_max - y_min) * 0.25
            
            data_point_near_bottom = (y_pos - y_min) < edge_margin_strict
            data_point_near_top = (y_max - y_pos) < edge_margin_strict
            
            if data_point_near_bottom:
                label_y = max(y_quarter_upper, label_y)
                label_y = min(label_y, y_center)
            elif data_point_near_top:
                label_y = min(y_quarter_lower, label_y)
                label_y = max(label_y, y_center)
            else:
                edge_margin_soft = label_height_data * 2.0
                if label_y - y_min < edge_margin_soft:
                    label_y = y_min + edge_margin_soft
                elif y_max - label_y < edge_margin_soft:
                    label_y = y_max - edge_margin_soft
            
            text_item.setPos(label_x, label_y)
            text_item.setZValue(201)
            
            text_scene = text_item.scene()
            plot_scene = self.plot_item.scene()
            if text_scene != plot_scene:
                if text_scene is not None:
                    text_scene.removeItem(text_item)
                self.plot_item.addItem(text_item, ignoreBounds=True)
            
            self.multi_cursor_items.append(text_item)

    def toggle_cursor(self, show: bool, hide_values_only: bool = False):
        """切换光标显示状态"""
        mode = self._get_cursor_mode()

        if hide_values_only:
            self._set_vline_visibility_for_mode(True, mode)
            self.cursor_label.setVisible(False)
            self.show_values_only = True
            self._clear_cursor_items()
            self._show_x_position_only()
        else:
            self._set_vline_visibility_for_mode(show, mode)
            self.cursor_label.setVisible(show)
            self.show_values_only = not show

            if not show:
                self._clear_cursor_items()
                self.update_right_header("")
                self.is_cursor_pinned = False
                self.pinned_x_value = None
                self.pinned_index_value = None
                self.pinned_x_values = []
                self.pinned_index_values = []
            else:
                self.update_cursor_label()

    def _show_x_position_only(self, x_positions=None):
        """仅显示 x 位置标签（隐藏光标数值）"""
        try:
            if not self._has_visible_curve_data():
                self._clear_cursor_items()
                self.update_right_header("")
                return
            x_positions = x_positions if x_positions is not None else self._get_cursor_x_positions()
            if not x_positions:
                return

            (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
            self._clear_cursor_items()

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)

                view_rect = self.plot_item.vb.sceneBoundingRect()
                scene_point = self.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)

                scene = self.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)

                self.multi_cursor_items.append(x_info_item)

        except Exception as e:
            print(f"x_position_only error: {e}")

    def _has_visible_curve_data(self) -> bool:
        """判断当前 plot 是否有可见且有数据的曲线"""
        try:
            if self.curves:
                for ci in self.curves.values():
                    if not ci.visible:
                        continue
                    x_data = ci.x_data
                    if x_data is not None and len(x_data) > 0:
                        return True
                return False
            if self.curve:
                x_data, _ = self.curve.getData()
                return x_data is not None and len(x_data) > 0
            return False
        except Exception:
            return False

    def pin_cursor(self, x_value):
        """将光标固定到最近的 x 并同步到所有 plot"""
        if getattr(self, "_is_pinning_cursor", False):
            return
        self._is_pinning_cursor = True
        try:
            if not self.window() or not hasattr(self.window(), "cursor_btn") or not self.window().cursor_btn.isChecked():
                self.window().cursor_btn.setChecked(True)
                self.window().toggle_cursor_all(True)

            curves_x_data = []
            if self.is_multi_curve_mode and self.curves:
                for _, ci in self.curves.items():
                    if ci.visible and ci.x_data is not None:
                        curves_x_data.append(ci.x_data)
            elif self.curve:
                x_data, _ = self.curve.getData()
                if x_data is not None:
                    curves_x_data.append(x_data)

            if not curves_x_data:
                return

            globally_closest_x = None
            min_distance = float("inf")
            for x_data in curves_x_data:
                if x_data is None or len(x_data) == 0:
                    continue
                try:
                    idx = np.searchsorted(x_data, x_value, side="left")
                    if idx > 0 and idx < len(x_data):
                        dist_left = abs(x_data[idx - 1] - x_value)
                        dist_right = abs(x_data[idx] - x_value)
                        if dist_left < dist_right:
                            idx = idx - 1
                    elif idx == len(x_data):
                        idx = len(x_data) - 1
                except (ValueError, TypeError):
                    idx = np.argmin(np.abs(x_data - x_value))
                nearest_x_in_curve = x_data[idx]
                distance = abs(nearest_x_in_curve - x_value)
                if distance < min_distance:
                    min_distance = distance
                    globally_closest_x = nearest_x_in_curve

            if globally_closest_x is not None:
                display_x = globally_closest_x
                if self.window() and hasattr(self.window(), "plot_widgets"):
                    main_window = self.window()
                    if hasattr(main_window, "cursor_mode"):
                        main_window.cursor_mode = "1 anchored cursor"
                    if hasattr(main_window, "pinned_x_values"):
                        main_window.pinned_x_values = [display_x]
                    widgets_to_update = []
                    for container in main_window.plot_widgets:
                        widget = container.plot_widget
                        widget.apply_cursor_mode("1 anchored cursor", [display_x])
                        if hasattr(widget, "_last_cursor_update_time"):
                            widget._last_cursor_update_time = 0
                        if hasattr(widget.view_box, "is_cursor_pinned"):
                            widget.view_box.is_cursor_pinned = True
                        widgets_to_update.append(widget)

                    def _delayed_label_update(widgets=widgets_to_update):
                        for widget in widgets:
                            if not getattr(widget, "_is_being_destroyed", False):
                                try:
                                    widget.update_cursor_label()
                                except (RuntimeError, AttributeError):
                                    pass

                    QTimer.singleShot(0, _delayed_label_update)
        finally:
            self._is_pinning_cursor = False

    def free_cursor(self):
        """释放光标固定并恢复自由移动"""
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []

        if hasattr(self, "vline"):
            self.vline.setMovable(False)
        if hasattr(self, "vline2"):
            self.vline2.setMovable(False)

        if hasattr(self.view_box, "is_cursor_pinned"):
            self.view_box.is_cursor_pinned = False

        if self.window() and hasattr(self.window(), "plot_widgets"):
            main_window = self.window()
            if hasattr(main_window, "cursor_mode"):
                main_window.cursor_mode = "1 free cursor"
            if hasattr(main_window, "pinned_x_values"):
                main_window.pinned_x_values = []
            for container in main_window.plot_widgets:
                widget = container.plot_widget
                widget.apply_cursor_mode("1 free cursor", [])
                if hasattr(widget.view_box, "is_cursor_pinned"):
                    widget.view_box.is_cursor_pinned = False

    def reset_pin_state(self):
        """重置 pin 状态"""
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []
        if hasattr(self, "vline"):
            self.vline.setMovable(False)
        if hasattr(self, "vline2"):
            self.vline2.setMovable(False)
        if hasattr(self, "vline2"):
            self.vline2.setVisible(False)
        if hasattr(self.view_box, "is_cursor_pinned"):
            self.view_box.is_cursor_pinned = False

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
            print(f"Warning: Error updating vline bounds: {e}")
            self._set_vline_bounds([None, None])
            return None, None
    
    def _update_cursor_after_plot(self, min_x_bound: float, max_x_bound: float):
        """在绘图或自动缩放后，更新光标线的边界和可见性
        
        根据数据范围更新cursor的移动边界，并根据主窗口的cursor状态决定显示模式。
        
        Args:
            min_x_bound: cursor允许的最小x值
            max_x_bound: cursor允许的最大x值
        """
        main_window = self.window()
        if main_window and hasattr(main_window, 'cursor_btn'):
            # 设置cursor的移动边界
            self._set_vline_bounds([min_x_bound, max_x_bound])
            cursor_enabled = main_window.cursor_btn.isChecked()
            cursor_values_hidden = getattr(main_window, 'cursor_values_hidden', False)
            
            # 根据全局cursor状态决定显示模式
            if cursor_enabled and cursor_values_hidden:
                # cursor启用但只显示vline和x值
                self.toggle_cursor(False, hide_values_only=True)
            else:
                # cursor完全启用或禁用
                self.toggle_cursor(cursor_enabled)
        else:
            # 无主窗口或cursor按钮，禁用cursor
            self._set_vline_bounds([None, None])
            self.toggle_cursor(False)

    def clear_value_cache(self):
        #self._value_cache: dict[str, tuple] = {}
        pass
    def datetime_to_unix_seconds(self, series: pd.Series) -> pd.Series:
        """将datetime Series转换为Unix时间戳（秒，float64精度）"""
        if "ns" in str(series.dtype):
            return (series.astype("int64") / 10**9).astype("float64")
        elif "us" in str(series.dtype):
            return (series.astype("int64") / 10**6).astype("float64")
        elif "ms" in str(series.dtype):
            return (series.astype("int64") / 10**3).astype("float64")
        else:
            raise ValueError(f"Unsupported datetime dtype: {series.dtype}")
        
    def get_value_from_name(self,var_name)-> tuple | None:
        main_window = self.window()
        if var_name in main_window.value_cache:
            return main_window.value_cache[var_name]

        if main_window and hasattr(main_window, 'loader') and main_window.loader is not None:
            loader = main_window.loader
            if getattr(loader, 'LOADER_TYPE', '') == 'mdf':
                raw_values = loader.get_series(var_name)
            else:
                raw_values = self.data[var_name]
        else:
            raw_values = self.data[var_name]

        if (
            main_window is not None
            and hasattr(main_window, 'loader')
            and main_window.loader is not None
            and hasattr(main_window.loader, 'get_value_from_name')
            and getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
        ):
            try:
                _, _, _, text_map = main_window.loader.get_value_from_name(var_name)
                if text_map:
                    y_values = raw_values
                    y_format = 'enum'
                    if not hasattr(main_window, '_enum_text_maps'):
                        main_window._enum_text_maps = {}
                    main_window._enum_text_maps[var_name] = text_map
                    main_window.value_cache[var_name] = (y_values, y_format)
                    return y_values, y_format
            except KeyError:
                pass

        dtype_kind = raw_values.dtype.kind
        y_values = None
        y_format = 'number'

        if dtype_kind in "iuf":
            y_values = raw_values
        elif dtype_kind == "b":
            y_values = raw_values.astype(np.int32)
        elif var_name in self.time_channels_info:
            fmt = self.time_channels_info[var_name]
            try:
                if "%H:%M:%S" in fmt:
                    # 时间格式：提取时间部分并转换为Unix时间戳
                    times = pd.to_datetime(raw_values, format=fmt, errors="coerce")
                    today = pd.Timestamp.today().normalize()
                    # 提取从午夜开始的时间差（保留毫秒/微秒精度）
                    time_deltas = times - times.dt.normalize()
                    dt_values = today + time_deltas
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 's'
                else:
                    # 日期格式：直接转换为Unix时间戳
                    dt_values = pd.to_datetime(raw_values, format=fmt, errors='coerce')
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = 'date'
            except (ValueError, TypeError):
                # 无法解析时间格式
                return None, None
        else:
            # 非时间通道：尝试将object等类型转换为数字，只要存在至少一个有效值就接受
            try:
                numeric_values = pd.to_numeric(raw_values, errors='coerce')
            except Exception:
                numeric_values = None

            if numeric_values is not None:
                finite_mask = np.isfinite(numeric_values.to_numpy(dtype=np.float64))
                if finite_mask.any():
                    y_values = numeric_values
                else:
                    return None, None
            else:
                return None, None
        
        if y_values is None:
            return None, None

        main_window.value_cache[var_name] = (y_values, y_format)
        return y_values, y_format
    
    def update_time_correction(self, new_factor, new_offset):
        self._suppress_pin_update = True
        try:
            old_factor = self.factor
            old_offset = self.offset
            self.factor = new_factor
            self.offset = new_offset

            main_window = self.window()
            is_mdf = (
                main_window is not None
                and hasattr(main_window, 'loader')
                and main_window.loader is not None
                and getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
            )

            if self.is_multi_curve_mode:
                for var_name, ci in self.curves.items():
                    if ci.curve is not None and ci.y_data is not None:
                        curve = ci.curve
                        y_data = ci.y_data
                        if is_mdf:
                            old_x_data = ci.x_data
                            if old_x_data is not None and old_factor != 0:
                                original_time = (old_x_data - old_offset) / old_factor
                            else:
                                original_time = np.arange(1, len(y_data) + 1)
                        else:
                            original_time = np.arange(1, len(y_data) + 1)
                        new_x = self.offset + self.factor * original_time
                        curve.setData(new_x, y_data)
                        ci.x_data = new_x
                        ci.update_x_range()
            else:
                if self.original_index_x is not None:
                    new_x = self.offset + self.factor * self.original_index_x
                    self.curve.setData(new_x, self.original_y)
            if self.is_multi_curve_mode and self.curves:
                first_curve_info = next(iter(self.curves.values()))
                datalength = len(first_curve_info.y_data) if first_curve_info.y_data is not None else 0
            elif self.original_index_x is not None:
                datalength = len(self.original_index_x)
            else:
                datalength = self.window().loader.datalength if hasattr(self.window(), 'loader') else 0
            padding_xVal = DEFAULT_PADDING_VAL_X
            if is_mdf and main_window is not None and hasattr(main_window.loader, 'global_time_range'):
                x_min, x_max = main_window.loader.global_time_range
                data_min_x = self.offset + self.factor * x_min
                data_max_x = self.offset + self.factor * x_max
            else:
                index_min = 1 - padding_xVal * datalength
                index_max = datalength + padding_xVal * datalength
                data_min_x = self.offset + self.factor * index_min
                data_max_x = self.offset + self.factor * index_max
            limits_xMin = data_min_x - padding_xVal * (data_max_x - data_min_x)
            limits_xMax = data_max_x + padding_xVal * (data_max_x - data_min_x)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            self._update_vline_bounds_from_data()
            if self.mark_region is not None and self is self.window().plot_widgets[0].plot_widget:
                old_min, old_max = self.mark_region.getRegion()
                if old_factor != 0:
                    index_min = (old_min - old_offset) / old_factor
                    index_max = (old_max - old_offset) / old_factor
                    new_min = new_offset + new_factor * index_min
                    new_max = new_offset + new_factor * index_max
                    blocker = QSignalBlocker(self.mark_region)
                    self.mark_region.setRegion([new_min, new_max])
                    self.window().sync_mark_regions(self.mark_region)
        finally:
            if hasattr(self, 'window') and self.window() is not None:
                if not getattr(self, '_is_being_destroyed', False):
                    self.window().request_mark_stats_refresh()
            self._suppress_pin_update = False

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
        """
        验证绘图数据的有效性
        
        检查变量名和数据源的有效性
        确保数据可以安全地用于绘图
        
        Args:
            var_name: 要验证的变量名称
            
        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not isinstance(var_name, str) or not var_name.strip():
            return False, "变量名无效"

        main_window = self.window()
        if main_window and hasattr(main_window, 'loader') and main_window.loader is not None:
            loader = main_window.loader
            if getattr(loader, 'LOADER_TYPE', '') == 'mdf':
                return True, ""

        if not hasattr(self, 'data') or self.data is None:
            return False, "没有可用的数据"
            
        if not hasattr(self.data, 'columns'):
            return False, "数据格式无效"
            
        if var_name not in self.data.columns:
            return False, f"变量 {var_name} 不存在"
            
        return True, ""

    def _get_x_data_for_variable(self, y_len: int) -> np.ndarray:
        return np.arange(1, y_len + 1, dtype=np.float32)

    def _prepare_plot_data(self, var_name: str) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
        """
        准备绘图数据
        
        从数据源中提取指定变量的数据，进行格式化和预处理
        生成用于绘图的x和y数组
        
        Args:
            var_name: 变量名称
            
        Returns:
            tuple: (是否成功, 错误信息, x数组, y数组, y格式)
        """
        try:
            y_values, y_format = self.get_value_from_name(var_name=var_name)
            
            if y_values is None or len(y_values) == 0:
                return False, f"变量 {var_name} 没有有效数据", None, None, ""
            
            # 转换为numpy数组，根据数据类型选择合适的精度
            # 时间数据（Unix时间戳）使用float64以保留毫秒精度
            # 其他数据使用float32以减少内存
            if isinstance(y_values, pd.Series):
                array_source = y_values.to_numpy()
                safety_source = y_values
            else:
                array_source = np.asarray(y_values)
                safety_source = array_source

            float32_safe, abs_max = _evaluate_float32_safety(safety_source)
            # 检查时间数据：Unix时间戳通常 > 1e8
            is_time_data = bool(abs_max is not None and abs_max > 1e8)
            prefer_float64 = is_time_data or not float32_safe
            target_dtype = np.float64 if prefer_float64 else np.float32

            try:
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=target_dtype)
                else:
                    y_array = np.asarray(array_source, dtype=target_dtype)
            except (OverflowError, ValueError, TypeError):
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=np.float64)
                else:
                    y_array = np.asarray(array_source, dtype=np.float64)

            if target_dtype == np.float32 and np.any(np.isinf(y_array)):
                if isinstance(y_values, pd.Series):
                    y_array = y_values.to_numpy(dtype=np.float64)
                else:
                    y_array = np.asarray(array_source, dtype=np.float64)

            # 检查数据是否全为NaN
            if np.all(np.isnan(y_array)):
                return False, f"变量 {var_name} 的数据全为无效值", None, None, ""
                
            # 使用统一的时间轴数据作为X轴
            main_window = self.window()
            if main_window and hasattr(main_window, 'loader') and hasattr(main_window.loader, 'get_value_from_name'):
                try:
                    x_array, _, _, _ = main_window.loader.get_value_from_name(var_name)
                    x_array = x_array[:len(y_array)]
                except KeyError:
                    x_array = self._get_x_data_for_variable(len(y_array))
            else:
                x_array = self._get_x_data_for_variable(len(y_array))

            return True, "", x_array, y_array, y_format
            
        except Exception as e:
            return False, f"处理数据时出错: {str(e)}", None, None, ""

    def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
        """
        绘制变量到图表
        
        将指定的数据变量绘制到当前图表中
        包括数据验证、格式化和图形渲染
        
        Args:
            var_name: 要绘制的变量名称
            show_duplicate_warning: 是否显示重复变量警告
            
        Returns:
            bool: 绘制是否成功
        """
        # 验证输入
        is_valid, error_msg = self._validate_plot_data(var_name)
        if not is_valid:
            QMessageBox.warning(self, "错误", error_msg)
            return False
        
        # 准备数据
        success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
        if not success:
            QMessageBox.warning(self, "错误", error_msg)
            return False
        
        try:
            main_window = self.window()
            is_mdf = (
                main_window is not None
                and hasattr(main_window, 'loader')
                and main_window.loader is not None
                and getattr(main_window.loader, 'LOADER_TYPE', '') == 'mdf'
            )

            if self.is_multi_curve_mode:
                # 多曲线模式：直接添加新曲线
                x_values = self.offset + self.factor * x_array
                return self.add_variable_to_plot(var_name, x_values, y_array, y_format, show_duplicate_warning=show_duplicate_warning)
            
            # 单曲线模式：设置绘图数据
            self.y_format = y_format
            self.y_name = var_name
            # x_array是索引数组，使用float32足够
            self.original_index_x = np.asarray(x_array, dtype=np.float32)
            safe_for_float32, abs_max_plot = _evaluate_float32_safety(y_array)
            keep_float64 = (
                y_format in ['s', 'date']
                or not safe_for_float32
                or (abs_max_plot is not None and abs_max_plot > 1e8)
            )
            target_y_dtype = np.float64 if keep_float64 else np.float32
            self.original_y = np.asarray(y_array, dtype=target_y_dtype)
            x_values = self.offset + self.factor * self.original_index_x
            
            # 单曲线模式：清除旧图并绘制新图
            # 先清除cursor items（包括scene中的items）
            # 绘制新变量时完全清除对象池，避免复用异常状态的items
            self._clear_cursor_items(hide_only=False)
            
            # 手动清理所有图形项，避免PyQtGraph的clearPlots scene不匹配问题
            self._safe_clear_plot_items()
            self.curves.clear()  # 清空多曲线数据
            
            # ========== 性能优化：创建单曲线 ==========
            _pen = pg.mkPen(color='blue', width=DEFAULT_LINE_WIDTH)
            self.curve = self.plot_item.plot(
                x_values, self.original_y, 
                pen=_pen, 
                name=var_name,
                skipFiniteCheck=True
            )
            
            # 性能优化说明：
            # - 自动降采样：plot_item.setDownsampling(mode='peak', auto=True) 已配置
            # - 视图裁剪：plot_item.setClipToView(True) 已配置
            # - 智能防抖：根据数据量动态调整延迟
            # 这些设置会自动应用到曲线，无需OpenGL也能获得良好性能
            
            # 延迟更新样式（带安全检查）
            self._queue_ui_refresh()

            # 更新标题
            full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
            self.update_left_header(full_title)
            
            # 设置坐标轴范围
            # 始终保持x轴范围不变，只更新y轴范围
            # 因为所有plot的x轴都是linked的，改变x轴会影响其他plot
            
            # 处理单点或所有点x坐标相同的特殊情况
            special_limits = self.handle_single_point_limits(x_values, self.original_y)
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
                self._set_safe_y_range(min_y, max_y)
                # if not is_mdf:
                #     self._set_x_limits_with_min_range(min_x, max_x)
            else:
                # 1. 基于数据的全范围设置y轴limits
                data_min_y = np.nanmin(self.original_y)
                data_max_y = np.nanmax(self.original_y)
                self._set_safe_y_range(data_min_y, data_max_y, set_limits=True)
                
                # 2. 基于当前x轴范围内的数据设置y轴viewRange
                current_x_range = self.view_box.viewRange()[0]
                x_min, x_max = current_x_range
                min_y, max_y = self._get_y_range_in_x_window(x_values, self.original_y, x_min, x_max)
                self._set_safe_y_range(min_y, max_y, set_limits=False)
                
                # 3. 更新x轴limits（合并本 plot 和全局所有可见 plot 的范围）
                # self._update_x_limits_for_plot(x_values, self.original_y, is_mdf)
            
            # 更新光标 - 在单曲线模式下使用当前数据范围即可
            min_x, max_x = np.min(x_values), np.max(x_values)
            self._set_vline_bounds([min_x, max_x])
            self.plot_item.update()
            self._update_cursor_after_plot(min_x, max_x)

            self._recalc_max_point_density()
            main_window = self.window()
            if main_window is not None and hasattr(main_window, '_sync_min_xrange'):
                main_window._sync_min_xrange()

            return True
            
        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘制变量时发生错误: {str(e)}")
            return False

    def _compute_valid_min_max(self, values) -> tuple[float | None, float | None]:
        """Safely compute min/max ignoring NaN/INF values."""
        if values is None:
            return None, None

        try:
            if isinstance(values, pd.Series):
                arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            try:
                arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64)
            except Exception:
                return None, None

        if arr.size == 0:
            return None, None

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return None, None

        finite_values = arr[finite_mask]
        return float(np.min(finite_values)), float(np.max(finite_values))

    def _get_y_range_in_x_window(self, x_values: np.ndarray, y_values: np.ndarray, x_min: float, x_max: float):
        """计算在指定x轴范围内的y值范围
        
        Args:
            x_values: X轴数据数组
            y_values: Y轴数据数组
            x_min: X轴范围最小值
            x_max: X轴范围最大值
            
        Returns:
            tuple: (min_y, max_y) 在x范围内的y值最小值和最大值
        """
        try:
            # 找到在x范围内的数据点
            mask = (x_values >= x_min) & (x_values <= x_max)
            if not np.any(mask):
                # 如果没有数据点在范围内，返回全部数据的范围
                bounds = self._compute_valid_min_max(y_values)
            else:
                y_in_range = y_values[mask]
                bounds = self._compute_valid_min_max(y_in_range)
                if bounds[0] is None or bounds[1] is None:
                    bounds = self._compute_valid_min_max(y_values)

            if bounds[0] is None or bounds[1] is None:
                return 0.0, 1.0
            return bounds
        except Exception:
            # 出错时返回全部数据范围
            bounds = self._compute_valid_min_max(y_values)
            if bounds[0] is None or bounds[1] is None:
                return 0.0, 1.0
            return bounds
    
    def _setup_plot_axes(self, x_values: np.ndarray, y_values: np.ndarray, update_x_range: bool = True):
        """设置绘图坐标轴
        
        根据数据范围设置X和Y轴的显示范围和限制范围。
        对于单点或所有点x坐标相同的特殊情况，会自动扩展x轴范围。
        
        Args:
            x_values: X轴数据数组
            y_values: Y轴数据数组
            update_x_range: 是否更新X轴范围，默认为True
        """
        try:
            # 处理特殊情况（单点或所有点x坐标相同）
            special_limits = self.handle_single_point_limits(x_values, y_values)
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
            else:
                min_x = np.min(x_values)
                max_x = np.max(x_values)
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
                
            # 计算X轴的限制范围（允许的最大范围）
            padding_x = DEFAULT_PADDING_VAL_X
            limits_xMin = min_x - padding_x * (max_x - min_x)
            limits_xMax = max_x + padding_x * (max_x - min_x)
            
            # 只在update_x_range为True时设置X轴的viewRange（显示范围）
            if update_x_range:
                self.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)
            
            # 设置Y轴范围和X轴limits
            self._set_safe_y_range(min_y, max_y)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            
        except Exception as e:
            # 出错时使用默认范围
            self._set_safe_y_range(0, 1)
            self._set_x_limits_with_min_range(0, 1)

    def _reset_plot_limits(self):
        """重置绘图限制"""
        try:
            self.plot_item.setLimits(yMin=None, yMax=None)
            self.view_box.setYRange(0, 1, padding=DEFAULT_PADDING_VAL_Y)
            self._set_vline_bounds([None, None])
        except Exception as e:
            print(f"重置绘图限制时出错: {e}")

    def _clear_plot_data(self):
        """清除绘图数据"""
        try:
            # 先清除cursor items（包括scene中的items）
            # 重要：清除plot时需要完全清除对象池（hide_only=False）
            # 这样可以避免对象池中的items处于异常状态（scene=None但PyQtGraph仍认为它属于PlotItem）
            self._clear_cursor_items(hide_only=False)
            
            # 清除所有plot items
            self._safe_clear_plot_items()
            self.axis_y.setLabel(text="")
            self.y_name = ''
            self.y_format = ''
            self.update_left_header("channel name")
            self.update_right_header("")
            
            # 清除单曲线的缓存数据
            if self.curve:
                if hasattr(self.curve, '_cached_pen_key'):
                    delattr(self.curve, '_cached_pen_key')
                if hasattr(self.curve, '_has_symbols'):
                    delattr(self.curve, '_has_symbols')
                try:
                    self.curve.clear()
                except:
                    pass
            
            self.curve = None
            self.original_index_x = None
            self.original_y = None
            
            # 清除多曲线数据
            for var_name, ci in self.curves.items():
                if ci.curve is not None:
                    curve = ci.curve
                    # 清除样式缓存
                    if hasattr(curve, '_cached_pen_key'):
                        delattr(curve, '_cached_pen_key')
                    if hasattr(curve, '_has_symbols'):
                        delattr(curve, '_has_symbols')
                    # 清除数据
                    try:
                        curve.clear()
                    except:
                        pass
            
            self.curves.clear()
            self.is_multi_curve_mode = False
            self.current_color_index = 0

            import gc
            gc.collect()

            self._recalc_max_point_density()
            main_window = self.window()
            if main_window is not None and hasattr(main_window, '_sync_min_xrange'):
                main_window._sync_min_xrange()
        except Exception as e:
            print(f"清除绘图数据时出错: {e}")

    def clear_plot_item(self):
        """清除绘图项"""
        self._reset_plot_limits()
        self._clear_plot_data()
        
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
        """更新多曲线模式状态"""
        curve_count = len(self.curves)
        
        # 如果正在批量添加，不要自动切换模式
        if not hasattr(self, '_batch_adding'):
            self._batch_adding = False
            
        if not self._batch_adding:
            self.is_multi_curve_mode = curve_count > 1
        
        if self.is_multi_curve_mode:
            # 多曲线模式：显示legend
            self.update_legend()
        else:
            # 单曲线模式：显示传统标题
            if curve_count == 1:
                var_name = list(self.curves.keys())[0]
                full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
                self.update_left_header(full_title)
            else:
                self.update_left_header("channel name")
                self.update_right_header("")
    
    def update_legend(self):
        """更新图例显示
        
        在多曲线模式下，在左上角显示所有曲线的图例。
        图例样式：
        - 可见曲线：实心方块(■) + 曲线颜色 + 变量名(单位)
        - 隐藏曲线：空心方块(□) + 半透明颜色 + 灰色文字
        
        点击图例中的曲线名可以切换该曲线的显示/隐藏状态。
        """
        if not self.is_multi_curve_mode:
            return
            
        # 构建图例文本（包含所有曲线，不管是否可见）
        legend_items = []
        for var_name, ci in self.curves.items():
            color = ci.color
            unit = self.units.get(var_name, '')
            legend_text = f"{var_name} ({unit})" if unit else var_name
            
            # 根据可见性调整显示样式
            if ci.visible:
                # 可见：实心方块 + 加粗文字
                legend_items.append(f"<span style='color: {color}; font-weight: bold;'>■</span> {legend_text}")
            else:
                # 隐藏：空心方块 + 灰色文字
                legend_items.append(f"<span style='color: {color}; opacity: 0.5;'>□</span> <span style='color: gray;'>{legend_text}</span>")
        
        if legend_items:
            legend_text = " | ".join(legend_items)
            self.update_left_header(legend_text)
        else:
            self.update_left_header("channel name")
    
    def toggle_curve_visibility_by_name(self, var_name):
        """通过变量名切换曲线可见性

        点击图例中的曲线名时调用，切换该曲线的显示/隐藏状态。
        如果曲线对象失效（不在scene中），会尝试重新创建。

        Args:
            var_name: 要切换可见性的变量名
        """
        if var_name not in self.curves:
            return

        ci = self.curves[var_name]
        # 切换可见性状态
        ci.visible = not ci.visible
        new_visible = ci.visible

        # 更新曲线对象的可见性
        if ci.curve is not None:
            curve_obj = ci.curve

            try:
                # 检查曲线对象是否仍然有效
                if curve_obj.scene() is not None:
                    curve_obj.setVisible(new_visible)
                else:
                    # 曲线对象已经不在scene中，重新创建
                    self._recreate_curve(var_name)
            except Exception as e:
                # 尝试重新创建曲线
                self._recreate_curve(var_name)
        # 更新图例显示
        self.update_legend()

        # 更新 Y轴范围以适应所有可见曲线
        # 当切换曲线可见性时，需要重新计算y轴范围，确保所有可见曲线都能完整显示
        if self.is_multi_curve_mode:
            self._update_axes_for_multi_curve(update_x_range=False)
        # 更新cursor显示（如果cursor可见）
        if self.vline.isVisible():
            self.update_cursor_label()
    
    def _recreate_curve(self, var_name):
        """重新创建失效的曲线"""
        try:
            if var_name in self.curves:
                ci = self.curves[var_name]
                success = self.add_variable_to_plot(
                    var_name,
                    skip_existence_check=True,
                    preferred_color=ci.color
                )
                if success:
                    pass
                else:
                    pass
            else:
                pass
        except Exception as e:
            pass
    
    def _on_legend_clicked(self, event):
        """Legend点击事件处理
        
        使用QTextDocument进行精确的hitTest，定位用户点击的是哪条曲线，
        然后切换该曲线的显示/隐藏状态。
        
        处理流程：
        1. 将legend HTML文本解析为QTextDocument
        2. 使用hitTest找到点击位置对应的文本位置
        3. 根据文本位置确定对应的曲线索引
        4. 调用toggle_curve_visibility_by_name切换曲线可见性
        
        Args:
            event: 鼠标点击事件
        """
        if not self.is_multi_curve_mode:
            return
        
        # 获取点击位置
        pos = event.pos()
        click_x = pos.x()
        
        # 改进的点击检测：基于实际legend文本内容进行更精确的匹配
        if not self.curves:
            return
            
        # 获取当前曲线列表（按legend显示顺序）
        curve_list = list(self.curves.items())
        
        if not curve_list:
            return
        
        # 使用QTextDocument + hitTest精确定位点击位置
        from PySide6.QtGui import QTextDocument, QTextCursor
        from PySide6.QtCore import QPointF
        
        # 构建完整的legend HTML（与update_legend完全一致）
        legend_parts = []
        for var_name, ci in curve_list:
            color = ci.color
            unit = self.units.get(var_name, '')
            legend_text = f"{var_name} ({unit})" if unit else var_name
            
            if ci.visible:
                symbol = f"<span style='color: {color}; font-weight: bold;'>■</span>"
                legend_parts.append(f"{symbol} {legend_text}")
            else:
                # 隐藏时：空心方格 + 灰色文字（与update_legend一致）
                symbol = f"<span style='color: {color}; opacity: 0.5;'>□</span>"
                legend_parts.append(f"{symbol} <span style='color: gray;'>{legend_text}</span>")
        
        full_html = " | ".join(legend_parts)
        
        # 创建QTextDocument来进行hitTest
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(self.label_left.font())
        doc.setHtml(full_html)
        
        # 使用hitTest找到点击位置对应的字符位置
        layout = doc.documentLayout()
        hit_pos = layout.hitTest(QPointF(click_x, pos.y()), Qt.HitTestAccuracy.ExactHit)
        
        # 计算每个legend部分在HTML中的字符位置范围
        clicked_index = -1
        char_pos = 0
        item_ranges = []
        
        for i, part in enumerate(legend_parts):
            if i > 0:
                # 加上分隔符" | "的长度（注意：纯文本长度，不是HTML长度）
                char_pos += 3  # " | " = 3个字符
            
            part_start = char_pos
            # 计算这个part的纯文本长度（去除HTML标签）
            part_doc = QTextDocument()
            part_doc.setHtml(part)
            part_text_length = len(part_doc.toPlainText())
            part_end = part_start + part_text_length
            
            item_ranges.append({
                'index': i,
                'start': part_start,
                'end': part_end,
                'var_name': curve_list[i][0]
            })
            
            if hit_pos >= part_start and hit_pos < part_end:
                clicked_index = i
                break
            
            char_pos = part_end
        
        # 如果hitTest没有精确匹配（点击在分隔符区域或文本范围外），找距离最近的item
        if clicked_index == -1:
            # 如果hitTest失败（返回-1），说明点击在文本范围外
            if hit_pos < 0:
                # 根据实际点击像素位置判断：左侧选第一个，右侧选最后一个
                total_text_width = doc.size().width()
                if click_x < total_text_width / 2:
                    clicked_index = 0
                else:
                    clicked_index = len(curve_list) - 1
            else:
                # 计算到每个item的距离，选择最近的
                min_distance = float('inf')
                for item in item_ranges:
                    if hit_pos < item['start']:
                        distance = item['start'] - hit_pos
                    elif hit_pos >= item['end']:
                        distance = hit_pos - item['end']
                    else:
                        distance = 0
                    
                    if distance < min_distance:
                        min_distance = distance
                        clicked_index = item['index']
        
        # 确保索引在有效范围内
        clicked_index = max(0, min(clicked_index, len(curve_list) - 1))
        
        # 切换对应曲线的可见性
        var_name = curve_list[clicked_index][0]
        self.toggle_curve_visibility_by_name(var_name)
    
    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        """为多曲线更新坐标轴范围
        
        计算所有可见曲线的数据范围，并更新坐标轴显示范围。
        只考虑visible=True的曲线，忽略隐藏的曲线。
        
        Args:
            update_x_range: 是否更新X轴范围。默认为False，保持当前x轴范围不变。
                           当为True时（通常是第一次添加曲线或批量添加完成），会设置x轴范围为数据的全范围。
        """
        if not self.curves:
            return

        pairs = self._collect_visible_curve_pairs()
        if not pairs:
            return
        x_values = np.concatenate([p[0] for p in pairs])
        y_values = np.concatenate([p[1] for p in pairs])
        if x_values.size == 0 or y_values.size == 0:
            return

        if update_x_range:
            # 更新x和y轴范围（第一次添加曲线或批量添加完成）
            self._setup_plot_axes(x_values, y_values, update_x_range=True)
        else:
            # 保持x轴范围不变，只更新y轴范围

            # 1. 先基于所有数据的全范围设置y轴limits
            all_data_min_y = np.nanmin(y_values)
            all_data_max_y = np.nanmax(y_values)
            self._set_safe_y_range(all_data_min_y, all_data_max_y, set_limits=True)

            # 2. 再根据当前x范围设置y轴viewRange
            # 检查是否是单点数据
            special_limits = self.handle_single_point_limits(x_values, y_values)
            if special_limits:
                # 单点数据：使用特殊处理
                # handle_single_point_limits已经返回了扩展后的x范围，直接使用
                min_x, max_x, min_y, max_y = special_limits
                self._set_safe_y_range(min_y, max_y, set_limits=False)
                # 更新x轴limits（不再额外扩展，因为handle_single_point_limits已经扩展过了）
                # self._set_x_limits_with_min_range(min_x, max_x)
            else:
                # 正常数据
                current_x_range = self.view_box.viewRange()[0]
                x_min, x_max = current_x_range

                # 计算所有曲线在当前x轴范围内的y值范围
                all_y_in_range = []
                for x_arr, y_arr in pairs:
                    min_y, max_y = self._get_y_range_in_x_window(
                        x_arr,
                        y_arr,
                        x_min,
                        x_max
                    )
                    all_y_in_range.extend([min_y, max_y])

                if all_y_in_range:
                    final_min_y = np.nanmin(all_y_in_range)
                    final_max_y = np.nanmax(all_y_in_range)
                    self._set_safe_y_range(final_min_y, final_max_y, set_limits=False)

                # 3. 更新x轴limits — 已关闭
                # data_min_x = np.min(x_values)
                # data_max_x = np.max(x_values)
                # 
                # main_window = self.window()
                # if main_window is not None and hasattr(main_window, 'collect_global_x_range'):
                #     global_min, global_max = main_window.collect_global_x_range(curves_filter="all")
                #     if global_min is not None:
                #         data_min_x = min(data_min_x, global_min)
                #         data_max_x = max(data_max_x, global_max)

                # padding_x = DEFAULT_PADDING_VAL_X
                # limits_xMin = data_min_x - padding_x * (data_max_x - data_min_x)
                # limits_xMax = data_max_x + padding_x * (data_max_x - data_min_x)
                # self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

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
        self.mark_region = pg.LinearRegionItem([min_x, max_x], movable=True)
        #self.mark_region.setBrush(pg.mkBrush(181,196,177, 80))
        for line in self.mark_region.lines:
            line.setHoverPen(pg.mkPen(color='r', width=10)) 
            
        self.plot_item.addItem(self.mark_region)
        self.mark_region.sigRegionChanged.connect(self.window().sync_mark_regions)

    def remove_mark_region(self):
        if self.mark_region and self.mark_region.scene() is not None:
            self.plot_item.removeItem(self.mark_region)
        self.mark_region = None

    def update_mark_region(self):
        if self.mark_region:
            old_min, old_max = self.mark_region.getRegion()
            # 更新基于新factor/offset，但由于x是scaled的，不需要额外缩放
            blocker = QSignalBlocker(self.mark_region)
            self.mark_region.setRegion([old_min, old_max])  # 实际不需要变，因为x已scale

    def get_mark_stats(self):
        """获取标记区域的统计信息
        
        【NumPy优化】使用NumPy掩码数组批量计算统计值，避免循环过滤
        """
        if not self.mark_region:
            return None
        
        min_x, max_x = self.mark_region.getRegion()
        
        if self.is_multi_curve_mode:
            # 多曲线模式：返回每个曲线的统计信息
            stats_list = []
            for var_name, ci in self.curves.items():
                if not ci.visible:
                    continue
                
                if ci.curve is None:
                    continue
                
                # 【NumPy优化】优先使用缓存的x_data和y_data，如果没有则从curve获取
                if ci.x_data is not None and ci.y_data is not None:
                    x_data = ci.x_data
                    y_data = ci.y_data
                elif ci.y_data is not None:
                    x_data = self.offset + self.factor * np.arange(1, len(ci.y_data) + 1, dtype=np.float32)
                    y_data = ci.y_data
                else:
                    curve = ci.curve
                    x_data, y_data = curve.getData()
                    if x_data is None or len(x_data) == 0:
                        continue
                
                # 确保是NumPy数组（保持原有精度，不强制转换）
                x_data = np.asarray(x_data)
                y_data = np.asarray(y_data)
                # 如果是整数类型且幅值适合，则转换为float32以减少内存
                if x_data.dtype.kind in 'iu':
                    safe_x, _ = _evaluate_float32_safety(x_data)
                    x_dtype = np.float32 if safe_x else np.float64
                    x_data = x_data.astype(x_dtype)
                if y_data.dtype.kind in 'iu':
                    safe_y, _ = _evaluate_float32_safety(y_data)
                    y_dtype = np.float32 if safe_y else np.float64
                    y_data = y_data.astype(y_dtype)
                
                # 计算边界点
                idx_left = np.argmin(np.abs(x_data - min_x))
                idx_right = np.argmin(np.abs(x_data - max_x))
                x1 = x_data[idx_left]
                y1 = y_data[idx_left]
                x2 = x_data[idx_right]
                y2 = y_data[idx_right]
                dx = x2 - x1
                dy = y2 - y1
                slope = float('inf') if dx == 0 else dy / dx
                
                # 【NumPy优化】使用掩码数组批量计算统计值
                mask = (x_data >= min_x) & (x_data <= max_x)
                if not np.any(mask):
                    y_avg = y_max = y_min = np.nan
                else:
                    y_masked = y_data[mask]
                    valid_y = y_masked[~np.isnan(y_masked)]
                    if len(valid_y) > 0:
                        y_avg = np.mean(valid_y)
                        y_max = np.max(valid_y)
                        y_min = np.min(valid_y)
                    else:
                        y_avg = y_max = y_min = np.nan
                
                # 添加变量名到标签
                unit = self.units.get(var_name, '')
                label = f"{var_name} ({unit})" if unit else var_name
                
                stats_list.append((x1, x2, y1, y2, dx, dy, slope, label, y_avg, y_max, y_min))
            
            return stats_list if stats_list else None
        else:
            # 单曲线模式：优先使用original_index_x和original_y
            if not self.curve:
                return None
            
            # 【NumPy优化】优先使用original_index_x和original_y
            if hasattr(self, 'original_index_x') and hasattr(self, 'original_y') and self.original_index_x is not None:
                x_data = self.offset + self.factor * self.original_index_x
                y_data = self.original_y
            else:
                x_data, y_data = self.curve.getData()
                if x_data is None or len(x_data) == 0:
                    return None
            
            # 确保是NumPy数组（保持原有精度，不强制转换）
            x_data = np.asarray(x_data)
            y_data = np.asarray(y_data)
            # 如果是整数类型且幅值适合，则转换为float32以减少内存
            if x_data.dtype.kind in 'iu':
                safe_x, _ = _evaluate_float32_safety(x_data)
                x_dtype = np.float32 if safe_x else np.float64
                x_data = x_data.astype(x_dtype)
            if y_data.dtype.kind in 'iu':
                safe_y, _ = _evaluate_float32_safety(y_data)
                y_dtype = np.float32 if safe_y else np.float64
                y_data = y_data.astype(y_dtype)
            
            idx_left = np.argmin(np.abs(x_data - min_x))
            idx_right = np.argmin(np.abs(x_data - max_x))
            x1 = x_data[idx_left]
            y1 = y_data[idx_left]
            x2 = x_data[idx_right]
            y2 = y_data[idx_right]
            dx = x2 - x1
            dy = y2 - y1
            slope = float('inf') if dx == 0 else dy / dx
            
            # 【NumPy优化】使用掩码数组批量计算统计值
            mask = (x_data >= min_x) & (x_data <= max_x)
            if not np.any(mask):
                y_avg = y_max = y_min = np.nan
            else:
                y_masked = y_data[mask]
                valid_y = y_masked[~np.isnan(y_masked)]
                if len(valid_y) > 0:
                    y_avg = np.mean(valid_y)
                    y_max = np.max(valid_y)
                    y_min = np.min(valid_y)
                else:
                    y_avg = y_max = y_min = np.nan
            
            return [(x1, x2, y1, y2, dx, dy, slope, self.label_left.text(), y_avg, y_max, y_min)]

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


class PlotContainerWidget(QWidget):
    """包装单个 Plot, 负责显示拖拽提示"""
    def __init__(self, plot_widget: DraggableGraphicsLayoutWidget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        # 设置容器的大小策略，允许拉伸
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(QMargins(0, 0, 5, 5))
        layout.setSpacing(0)
        layout.addWidget(plot_widget, 1)  # 拉伸因子1，让plot占用所有空间
        self._init_indicator()

    def _init_indicator(self):
        self._indicator = QWidget(self)
        self._indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._indicator.hide()
        self._indicator.setStyleSheet(
            "background-color: rgba(0, 120, 215, 40);"
            "border: 2px dashed #0078d7;"
            "border-radius: 12px;"
        )
        layout = QVBoxLayout(self._indicator)
        layout.setContentsMargins(16, 16, 16, 16)
        self._indicator_label = QLabel("", self._indicator)
        self._indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._indicator_label.setWordWrap(True)
        self._indicator_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self._indicator_label.setStyleSheet(
            "color: #0b365a; font-size: 16px; font-weight: bold; background: transparent; border: none;"
        )
        layout.addWidget(self._indicator_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _build_indicator_text(self, var_names: list[str]) -> str:
        has_curve = bool(getattr(self.plot_widget, "curve", None))
        has_multi_curves = bool(getattr(self.plot_widget, "curves", None))
        multi_mode = bool(getattr(self.plot_widget, "is_multi_curve_mode", False) or len(var_names) > 1 or has_multi_curves)

        if multi_mode:
            return "释放以添加"

        if has_curve:
            return "释放以替换"

        return "释放以添加"

    def show_drag_indicator(self, var_names: list[str] | None = None, text_override: str | None = None):
        text = text_override or self._build_indicator_text(var_names or [])
        self._indicator_label.setText(text)
        self._indicator.setGeometry(self.rect())
        self._indicator.raise_()
        self._indicator.show()

    def hide_drag_indicator(self):
        self._indicator.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._indicator.isVisible():
            self._indicator.setGeometry(self.rect())

class MainWindow(QMainWindow):
    """
    主窗口类
    应用程序的主界面，集成数据加载、图表显示、表格查看等功能
    提供完整的用户交互界面和数据处理流程
    """
    def __init__(self):
        super().__init__()
        from src.ui.file_loader_manager import FileLoaderManager
        from src.ui.cursor_sync_manager import CursorSyncManager
        from src.ui.layout_manager import LayoutManager

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._drop_event_filter_registered = False
        self.defaultTitle = "数据快速查看器(PySide6), Alpha版本"

        # 设置应用程序图标（影响Dock图标）
        if sys.platform == "darwin":  # macOS
            if os.path.exists(ico_path):
                app_icon = QIcon(str(ico_path))
                app = QApplication.instance()
                app.setWindowIcon(app_icon)
                self.setWindowIcon(app_icon)

        elif sys.platform == "win32":  # Windows
            if os.path.exists(ico_path):
                self.setWindowIcon(QIcon(str(ico_path))) 
       
        self.setWindowTitle(self.defaultTitle)
        self._factor_default  = 1
        self._offset_default = 0
        self.factor = self._factor_default
        self.offset = self._offset_default
        self._active_drag_container: PlotContainerWidget | None = None

        # 数据点密度相关
        self._baseline_density: float = 0.0
        self._global_max_density: float = 0.0

        # try to load config json files
        _read_status = False
        if os.path.isfile("config_dict.json"):            
            try:
                config_dict=self.load_dict("config_dict.json")
                layout_config_dict=config_dict.get("layout_config",{})
                _width = int(layout_config_dict.get('window_width',0))
                _height = int(layout_config_dict.get('window_height',0))
                _max_row = int(layout_config_dict.get('max_row',0))
                _max_col = int(layout_config_dict.get('max_col',0))
                _default_row = int(layout_config_dict.get('default_row',0))
                _default_col = int(layout_config_dict.get('default_col',0))
                _hide_plot_area = bool(layout_config_dict.get('hide_plot_area',None))
                _read_status = all(x > 0 for x in (_width, _height, _max_row, _max_col,_default_row,_default_col)) and _hide_plot_area is not None
            except Exception as e:     
                print(f"配置文件读取失败: {e}")

        if _read_status == True:
            self._window_width_default = max(600,_width)
            self._window_height_default = max(400,_height)
            self.resize(self._window_width_default, self._window_height_default)
            self._plot_row_max_default = max(1,_max_row)
            self._plot_col_max_default = max(1,_max_col)
            self._plot_row_current = max(1,min(_default_row,_max_row))
            self._plot_col_current = max(1,min(_default_col,_max_col))
        else:
            CANDIDATES = [
                (1920,1080),
                (1600, 900),
                (1366, 768),
                (1280, 720),
                (1024, 600),
                ( 800, 600),
                ( 640, 480),
                ]

            def best_resolution() -> tuple[int, int]:
                desk = QApplication.primaryScreen().size()
                for w, h in sorted(CANDIDATES, key=lambda t: t[0]*t[1], reverse=True):
                    if w < desk.width()*(1-SCREEN_WITDH_MARGIN) and h < desk.height()*(1-SCREEN_HEIGHT_MARGIN):
                        return w, h
                return desk.width(), desk.height()

            self._window_width_default, self._window_height_default = best_resolution()
            self.resize(self._window_width_default, self._window_height_default)
            # put default plots into the window
            self._plot_row_max_default = PLOT_ROW_MAX_DEFAULT
            self._plot_col_max_default = PLOT_COL_MAX_DEFAULT
            self._plot_row_current = PLOT_ROW_CURRENT_DEFAULT
            self._plot_col_current = PLOT_COL_CURRENT_DEFAULT
            _hide_plot_area = False

        self.loaded_path = ''
        self._last_open_dir: str | None = None
        self.var_names = None
        self.units = None
        self.time_channels_infos = None
        self.data = None
        self.data_validity = None
        self._is_loading_new_data = False  # 标志：是否正在加载新数据，用于屏蔽交互信号

        # 【稳定性优化】数据版本号机制，用于检测竞态条件
        self._data_version = 0  # 每次加载新数据时递增
        self._pending_crosshair_x = None  # 待更新的crosshair位置
        self._crosshair_update_timer = QTimer(self)
        self._crosshair_update_timer.setSingleShot(True)
        self._crosshair_update_timer.timeout.connect(self._flush_crosshair_updates)

        # 窗口几何信息
        self.data_table_geometry = None
        self.mark_stats_geometry = None
        self.time_correction_geometry = None
        self._mark_stats_dirty = False
        self._mark_stats_timer = QTimer(self)
        self._mark_stats_timer.setSingleShot(True)
        self._mark_stats_timer.timeout.connect(self._flush_mark_stats_refresh)
        self._is_syncing_crosshair = False
        self._is_syncing_mark_region = False

        # value cache
        self.value_cache = {}

        # ---------------- 中央控件 ----------------
        central = QWidget()
        self.setCentralWidget(central)

        # ========== 主布局：可调整分界线 ==========
        # 使用QSplitter实现变量表和绘图区之间的可拖动分界线
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建水平分隔器
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(5)             # 分界线宽度（可拖动区域）
        self.main_splitter.setChildrenCollapsible(False)  # 禁止折叠（确保两侧始终可见）
        
        # 变量表宽度管理
        # - 默认宽度：280px（首次启动时）
        # - 用户调整标记：记录用户是否手动拖动过分界线
        # - 行为逻辑：
        #   * 窗口缩放时，如果用户未手动调整过，保持变量表宽度不变，只改变绘图区宽度
        #   * 一旦用户手动拖动分界线，后续窗口缩放会按比例调整两侧宽度
        self.var_table_default_width = 280
        self.var_table_user_adjusted = False
        
        # 监听分界线拖动事件
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        self._splitter_ready = False  # 用于防止首个 resize 时重复 setSizes
        self._pending_splitter_adjustment = False

        # ---------------- 左侧变量列表 ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 0, 5, 0)

        # 左侧大标题
        # 创建一个水平布局来放置标题和帮助按钮
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧大标题
        left_layout_title = QLabel("变量列表")
        font = left_layout_title.font()
        font.setBold(True)
        left_layout_title.setFont(font)
        title_layout.addWidget(left_layout_title)

        # 添加弹簧，将按钮推到右侧
        title_layout.addStretch(1)

        self.clone_btn = QPushButton("分身")
        self.clone_btn.setToolTip("启动独立实例")
        self.clone_btn.clicked.connect(self.spawn_clone_window)
        title_layout.addWidget(self.clone_btn)
        self.clone_btn.setVisible(True)

        # 帮助按钮（使用小图标按钮样式）
        self.help_btn_small = QPushButton("?")
        #self.help_btn_small.setFixedSize(25, 25)  # 设置固定大小
        self.help_btn_small.setToolTip("帮助文档")  # 添加提示
        self.help_btn_small.clicked.connect(self.show_help)
        title_layout.addWidget(self.help_btn_small)

        # 将标题布局添加到左侧布局
        left_layout.addLayout(title_layout)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入变量名关键词（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.unit_filter_input = QLineEdit()
        self.unit_filter_input.setPlaceholderText("输入单位关键词（空格分隔）")
        self.unit_filter_input.setContentsMargins(60,0,0,0)

        self.unit_filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.unit_filter_input)

        # 创建一个水平布局来放置这两个按钮
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距

        # 创建按钮
        self.load_btn = QPushButton("导入数据文件")
        self.load_btn.clicked.connect(self.load_btn_click)

        self.reload_btn = QPushButton("重载")
        self.reload_btn.clicked.connect(self.reload_data)

        # 设置按钮的拉伸比例（4:1）
        button_layout.addWidget(self.load_btn, 4)  # 导入按钮占4份
        button_layout.addWidget(self.reload_btn, 1)  # 重新加载按钮占1份

        # 将按钮布局添加到左侧布局
        left_layout.addLayout(button_layout)
        self.list_widget = MyTableWidget()
        left_layout.addWidget(self.list_widget)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(2)
        self.log_btn = QPushButton("日志")
        self.log_btn.clicked.connect(self.show_log_window)
        bottom_row.addWidget(self.log_btn)

        self.toggle_plot_btn = QPushButton("隐藏绘图区")
        self.toggle_plot_btn.setCheckable(True)
        self.toggle_plot_btn.toggled.connect(self.toggle_plot_area)
        bottom_row.addWidget(self.toggle_plot_btn)
        left_layout.addLayout(bottom_row)
        left_layout.setSpacing(2)
        self.left_widget=left_widget

        # 添加成员变量来保存窗口状态
        self._plot_area_visible = True
        self._saved_geometry = None

        # ---------------- 右侧绘图区 ----------------
        self.plot_widget = QWidget()
        root_layout = QVBoxLayout(self.plot_widget)

        root_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        root_layout.setSpacing(0)  # Remove spacing

        # 顶部按钮栏：弹簧 + 光标按钮（右对齐）
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 5, 5)
        
        # 左侧按钮
        self.time_correction_btn = QPushButton("时间修正")
        
        self.time_correction_btn.clicked.connect(self.open_time_correction_dialog)
        top_bar.addWidget(self.time_correction_btn)


        self.clear_all_plots_btn = QPushButton("清除绘图")
        self.clear_all_plots_btn.clicked.connect(self.clear_all_plots)
        top_bar.addWidget(self.clear_all_plots_btn)
        
        # 模板按钮（二级菜单）
        self._template_menu_btn = QToolButton()
        self._template_menu_btn.setText("模板")
        self._template_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._template_menu = QMenu(self._template_menu_btn)
        self._template_menu_act_save = self._template_menu.addAction("保存为模板")
        self._template_menu_act_save.triggered.connect(self.save_current_as_template)
        self._template_menu_act_mgr = self._template_menu.addAction("模板管理器")
        self._template_menu_act_mgr.triggered.connect(self.open_template_manager)
        self._template_menu_btn.setMenu(self._template_menu)
        top_bar.addWidget(self._template_menu_btn)

        # 中键占位
        top_bar.addStretch(1)

        # 右侧按钮        
        
        self.auto_range_btn = QPushButton("自动缩放")
        self.auto_range_btn.clicked.connect(self.auto_range_all_plots)
        

        self.auto_y_btn = QPushButton("仅调节y轴")
        self.auto_y_btn.clicked.connect(self.auto_y_in_x_range)
        
        
        self.cursor_btn = QPushButton("显示光标")
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.clicked.connect(self.toggle_cursor_all)
        
        # 全局cursor值显示状态：False表示显示所有值，True表示只显示x值
        self.cursor_values_hidden = False  # 默认显示完整cursor（包括圆圈和y值）
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []

        self.mark_region_btn = QPushButton("标记区域")
        self.mark_region_btn.setCheckable(True)
        self.mark_region_btn.clicked.connect(self.toggle_mark_region)
        
        self.grid_layout_btn = QPushButton("修改布局")
        self.grid_layout_btn.clicked.connect(self.open_layout_dialog)
        
        top_bar.addWidget(self.grid_layout_btn)
        top_bar.addWidget(self.cursor_btn)
        top_bar.addWidget(self.mark_region_btn)
        top_bar.addWidget(self.auto_y_btn)
        top_bar.addWidget(self.auto_range_btn)
        
        # 添加布局
        root_layout.addLayout(top_bar)

        # 真正容纳子图的布局
        self.plot_layout=QGridLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.plot_layout.setSpacing(0)  # No spacing
        root_layout.addLayout(self.plot_layout, 1)    # 1 表示可伸缩
        
        # 将左右两个widget添加到splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(self.plot_widget)
        
        # 设置初始分割比例（左侧固定宽度，右侧自适应）
        self.main_splitter.setSizes([self.var_table_default_width, 800])
        
        # 设置拉伸因子：左侧0（不拉伸），右侧1（可拉伸）
        self.main_splitter.setStretchFactor(0, 0)  # 变量表不拉伸
        self.main_splitter.setStretchFactor(1, 1)  # 绘图区可拉伸
        
        # 将splitter添加到主布局
        main_layout.addWidget(self.main_splitter)
        QTimer.singleShot(0, self._ensure_splitter_ready)
        QTimer.singleShot(0, self._ensure_splitter_ready)

        # ---------------- 子图 ----------------
        self.plot_widgets = []

        self.placeholder_label = QLabel("请导入 CSV 文件以查看数据", self.plot_widget)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.placeholder_label, 0, 0)

        self.drop_overlay = DropOverlay(self.centralWidget())
        self.drop_overlay.lower()          # 初始在最下层
        self.drop_overlay.hide()

        # 全局拖拽过滤器（按需安装，便于多窗口独立卸载）
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
            self._drop_event_filter_registered = True
   
        if _hide_plot_area:
            # 临时设置离屏属性，模拟显示以计算布局尺寸（无闪烁）
            self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            self.show()  # 此处仅计算布局，不实际显示在屏幕上
            _geometry=self.geometry()

            # 更新按钮文本、checked状态和可见性标志
            self.toggle_plot_btn.setChecked(True)
            self.toggle_plot_btn.setText("显示绘图区")
            self._plot_area_visible = False

            # 隐藏右侧绘图区域
            self.plot_widget.hide()

            # 计算左侧部件的实际宽度（包括边距和框架）
            left_width = self.left_widget.width()
            main_margin = self.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            frame_width = self.frameGeometry().width() - self.width()
            new_width = left_width + frame_width

            # 设置固定宽度，并关闭离屏模拟
            self.setFixedWidth(new_width)
            self.move(_geometry.topLeft())
            self.close()  # 关闭模拟窗口
            self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)

            # 保存原最大宽度（用于后续显示时恢复）
            self._old_max_width = self._window_width_default  # 或实际原宽

        # 标记区域相关
        self.saved_mark_range = None
        self.mark_stats_window = None

        # 行高度百分比跟踪 {row_index: percentage}
        self.row_height_factors: dict[int, int] = {}

        self.file_loader_manager = FileLoaderManager(self)
        self.layout_manager = LayoutManager(self)
        self.cursor_sync_manager = CursorSyncManager(self)

        # 初始化模板配置管理器（必须在 layout_manager 之后，避免 eventFilter 回调时 layout_manager 未初始化）
        from src.ui.plot_config_manager import PlotConfigManager
        self.plot_config_manager = PlotConfigManager()

        self._log_manager = LogManager.get_instance()
        self._logger = self._log_manager.get_logger("app.main")

        self.set_button_status(False)

        self._logger.info(
            "CSV Plot 启动 (Python %s, PySide6 %s)",
            sys.version.split()[0],
            __import__("PySide6").__version__,
        )

        # ---------------- 命令行直接加载文件 ----------------
        positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if positional_args:
            self.load_csv_file(positional_args[0])

    def closeEvent(self, event):
        self._logger.info("CSV Plot 应用程序退出")
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

    def toggle_plot_area(self, checked):
        self.layout_manager.toggle_plot_area(checked)
            
    def show_help(self):
        self.layout_manager.show_help()

    def _get_plot_container(self, plot_widget) -> PlotContainerWidget | None:
        return self.layout_manager._get_plot_container(plot_widget)

    def _show_drag_indicator_for_plot(self, plot_widget, var_names: list[str], text_override: str | None = None):
        self.layout_manager._show_drag_indicator_for_plot(plot_widget, var_names, text_override=text_override)

    def _hide_drag_indicator_for_plot(self, plot_widget):
        self.layout_manager._hide_drag_indicator_for_plot(plot_widget)

    def spawn_clone_window(self):
        return self.layout_manager.spawn_clone_window()

    def load_btn_click(self):
        self.file_loader_manager.load_btn_click()

    def _validate_file_path(self, file_path: str) -> bool:
        return self.file_loader_manager._validate_file_path(file_path)
    
    def _check_file_size(self, file_path: str) -> bool:
        return self.file_loader_manager._check_file_size(file_path)

    def _begin_data_reload(self):
        self.file_loader_manager._begin_data_reload()

    def _end_data_reload(self):
        self.file_loader_manager._end_data_reload()

    def _post_reload_ui_refresh(self):
        self.file_loader_manager._post_reload_ui_refresh()

    def load_csv_file(self, file_path: str):
        self.file_loader_manager.load_csv_file(file_path)

    def set_button_status(self,status:bool):
        self.file_loader_manager.set_button_status(status)

    def reload_data(self):
        self.file_loader_manager.reload_data()

    def _load_file(self, file_path: str, is_reload: bool = False):
        self.file_loader_manager._load_file(file_path, is_reload=is_reload)

    @property
    def _has_valid_loader(self) -> bool:
        return self.file_loader_manager._has_valid_loader
    
    @property
    def _has_valid_data(self) -> bool:
        return self.file_loader_manager._has_valid_data
    
    @property
    def _current_data_length(self) -> int:
        return self.file_loader_manager._current_data_length

    def _cleanup_old_data(self):
        self.file_loader_manager._cleanup_old_data()


    def _post_load_actions(self, file_path: str):
        self.file_loader_manager._post_load_actions(file_path)

    def _remember_last_open_dir(self, file_path: str):
        self.file_loader_manager._remember_last_open_dir(file_path)

    def _get_dialog_initial_directory(self) -> str:
        return self.file_loader_manager._get_dialog_initial_directory()

    def _default_system_directory(self) -> str:
        return self.file_loader_manager._default_system_directory()

    @staticmethod
    def load_dict(path: str, *, default=None) -> dict:
        from src.ui.file_loader_manager import FileLoaderManager
        return FileLoaderManager.load_dict(path, default=default)
        
    def _extract_file_extension(self, file_path: str) -> str:
        return self.file_loader_manager._extract_file_extension(file_path)
    
    def save_current_as_template(self):
        """保存当前配置为模板"""
        from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog
        config = self.plot_config_manager.export_current_config(self)
        if not hasattr(self, "_last_template_name"):
            self._last_template_name = ""
            self._last_template_desc = ""
        dialog = TemplateEditorDialog(
            self.plot_config_manager.template_manager,
            current_config=config,
            initial_name=getattr(self, "_last_template_name", ""),
            initial_desc=getattr(self, "_last_template_desc", ""),
            parent=self,
        )
        dialog.template_saved.connect(self._on_template_saved)
        dialog.exec()

    def _on_template_saved(self, template_id: str):
        """模板保存后记录名称/描述"""
        dialog = self.sender()
        if dialog and hasattr(dialog, "_name_edit"):
            self._last_template_name = dialog._name_edit.text().strip()
            self._last_template_desc = dialog._desc_edit.text().strip()
        self._show_status_message(f"模板已保存: {template_id}")
    
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
        """应用指定的模板"""
        success = self.plot_config_manager.apply_template(self, template_id)
        if success:
            self._show_status_message("模板已应用")
        else:
            self._show_status_message("应用模板失败")
    
    def _show_status_message(self, message: str):
        """显示状态消息（在未来可以添加状态栏）"""
        self._logger.info(message)
        # 目前可以使用 QMessageBox 提示，或者之后添加状态栏
        # QMessageBox.information(self, "信息", message)
    
    def _validate_load_parameters(self, file_path: str, desc_rows, sep, has_unit) -> tuple[bool, str]:
        return self.file_loader_manager._validate_load_parameters(file_path, desc_rows, sep, has_unit)

    def _load_sync(self, 
                   file_path: str,
                   desc_rows: int = 0,
                   sep: str = ',',
                   has_unit: bool = True,
                   encoding: str | None = None):
        return self.file_loader_manager._load_sync(file_path, desc_rows=desc_rows, sep=sep, has_unit=has_unit, encoding=encoding)

    def _on_load_done(self,loader, file_path: str):
        self.file_loader_manager._on_load_done(loader, file_path)

    def _on_load_error(self, msg):
        self.file_loader_manager._on_load_error(msg)

    def _apply_loader(self):
        self.file_loader_manager._apply_loader()

    def filter_variables(self):
        self.cursor_sync_manager.filter_variables()

    def toggle_mark_region(self, checked):
        self.layout_manager.toggle_mark_region(checked)

    def sync_mark_regions(self, region_item):
        self.layout_manager.sync_mark_regions(region_item)

    def request_mark_stats_refresh(self, *, immediate: bool = False):
        self.layout_manager.request_mark_stats_refresh(immediate=immediate)

    def _flush_mark_stats_refresh(self):
        self.layout_manager._flush_mark_stats_refresh()

    def update_mark_stats(self):
        self.layout_manager.update_mark_stats()

    def open_layout_dialog(self):
        self.layout_manager.open_layout_dialog()

    def open_time_correction_dialog(self):
        self.layout_manager.open_time_correction_dialog()

    def update_mark_regions_on_layout_change(self):
        self.layout_manager.update_mark_regions_on_layout_change()

    def _unregister_global_event_filter(self):
        self.layout_manager._unregister_global_event_filter()

    def eventFilter(self, obj, event):
        if not hasattr(self, "layout_manager") or self.layout_manager is None:
            return super().eventFilter(obj, event)
        handled = self.layout_manager._handle_event_filter(obj, event)
        if handled:
            return True
        return super().eventFilter(obj, event)

    def show_drop_overlay(self):
        self.layout_manager.show_drop_overlay()

    def hide_drop_overlay(self):
        self.layout_manager.hide_drop_overlay()


    def reset_plots_after_loading(self,index_xMin,index_xMax, *, reason: str | None = None):
        self.cursor_sync_manager.reset_plots_after_loading(index_xMin, index_xMax, reason=reason)


    def _get_cursor_source_plot(self, source_plot=None):
        return self.cursor_sync_manager._get_cursor_source_plot(source_plot)

    def _get_cursor_view_range(self, source_plot=None):
        return self.cursor_sync_manager._get_cursor_view_range(source_plot)

    @staticmethod
    def _clamp_value(value, min_val, max_val):
        from src.ui.cursor_sync_manager import CursorSyncManager
        return CursorSyncManager._clamp_value(value, min_val, max_val)

    def _calc_second_cursor_position(self, pinned_x, view_min, view_max):
        return self.cursor_sync_manager._calc_second_cursor_position(pinned_x, view_min, view_max)

    def _select_farthest_cursor_index(self, context_x):
        return self.cursor_sync_manager._select_farthest_cursor_index(context_x)

    def _apply_cursor_mode_to_plots(self):
        return self.cursor_sync_manager._apply_cursor_mode_to_plots()

    def set_cursor_mode(self, mode, *, source_plot=None, context_x=None):
        self.cursor_sync_manager.set_cursor_mode(mode, source_plot=source_plot, context_x=context_x)

    def set_cursor_enabled(self, enabled: bool) -> None:
        return self.cursor_sync_manager.set_cursor_enabled(enabled)

    def is_cursor_enabled(self) -> bool:
        return self.cursor_sync_manager.is_cursor_enabled()

    def toggle_cursor_all(self, checked):
        self.cursor_sync_manager.toggle_cursor_all(checked)

    def _realign_pinned_cursor_after_time_correction(self, old_factor, old_offset, new_factor, new_offset):
        self.cursor_sync_manager._realign_pinned_cursor_after_time_correction(old_factor, old_offset, new_factor, new_offset)

    def sync_crosshair(self, x, sender_widget):
        self.cursor_sync_manager.sync_crosshair(x, sender_widget)

    def _flush_crosshair_updates(self):
        self.cursor_sync_manager._flush_crosshair_updates()

    def reset_all_pin_states(self):
        self.cursor_sync_manager.reset_all_pin_states()

    def clear_all_plots(self):
        self.cursor_sync_manager.clear_all_plots()

    def collect_global_x_range(self, curves_filter: str = "visible") -> tuple[float | None, float | None]:
        return self.cursor_sync_manager.collect_global_x_range(curves_filter)

    def _compute_baseline_density(self):
        self.cursor_sync_manager._compute_baseline_density()

    def _sync_min_xrange(self):
        self.cursor_sync_manager._sync_min_xrange()

    def auto_range_all_plots(self):
        self.cursor_sync_manager.auto_range_all_plots()
            
    def auto_y_in_x_range(self):
        self.cursor_sync_manager.auto_y_in_x_range()

    def create_subplots_matrix(self, m: int, n: int):
        self.layout_manager.create_subplots_matrix(m, n)

    def set_row_height(self, row: int, percentage: int) -> None:
        self.layout_manager.set_row_height(row, percentage)

    def set_all_row_height(self, percentage: int) -> None:
        self.layout_manager.set_all_row_height(percentage)
    
    def get_row_height(self, row: int) -> int:
        return self.layout_manager.get_row_height(row)

    def set_plots_visible(self, row_set: int = 1, col_set: int = 1):
        self.layout_manager.set_plots_visible(row_set, col_set)

    def replots_after_loading(self):
        self.cursor_sync_manager.replots_after_loading()


if __name__ == "__main__":

    # 启用 OpenGL (极大提升大数据的渲染性能)
    # pyqtgraph 0.14.0 以后不需要 enableExperimental=True 了
    # pg.setConfigOptions(useOpenGL=True) 
    
    # 禁用抗锯齿 (大数据量下抗锯齿非常消耗资源且视觉收益低)
    pg.setConfigOptions(antialias=False)

    # print(f"Qt 绑定: {pg.Qt.QT_LIB}")
    # print(f"Qt 版本: {pg.Qt.VERSION_INFO}")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    
    if sys.platform == "win32":
        from src.core.font_cache import get_windows_chinese_font_cached
        font_name = get_windows_chinese_font_cached()
        if font_name:
            font = QFont(font_name)
        else:
            font = QApplication.font()
        pixel_size = 12
        font.setPixelSize(pixel_size)
        app.setFont(font)
        
    elif sys.platform == "darwin":
        font = QApplication.font()
        font.setPixelSize(13) # macOS 默认字体稍大一点可能观感更好
        app.setFont(font)

    # 提前设置应用图标，让 macOS Dock 在 splash 阶段就能显示图标
    if 'ico_path' in globals() and os.path.exists(ico_path):
        app.setWindowIcon(QIcon(str(ico_path)))

    # QSplashScreen 不算真正的窗口, 必须关闭此选项否则 exec() 立即退出
    app.setQuitOnLastWindowClosed(False)

    skip_splash = "--no-splash" in sys.argv

    if skip_splash:
        window = MainWindow()
        window.show()
        app.setQuitOnLastWindowClosed(True)
        app._main_window_ref = window
    else:
        from src.ui.splash_screen import SplashScreen
        splash = SplashScreen()
        splash.show()
        app.processEvents()

        # 延迟显示主窗口, 便于观察 splash 效果
        # 可通过命令行参数 --splash-delay=N 控制延迟秒数 (默认 1 秒)

        MIN_SPLASH_MS = 1000
        delay_arg = next((a for a in sys.argv if a.startswith("--splash-delay=")), None)
        splash_delay = int(delay_arg.split("=")[1]) * 1000 if delay_arg else MIN_SPLASH_MS

        effective_delay = max(splash_delay, MIN_SPLASH_MS)

        def _show_main():
            window = MainWindow()
            splash.finish(window)
            window.show()
            app.setQuitOnLastWindowClosed(True)
            app._main_window_ref = window

        QTimer.singleShot(effective_delay, _show_main)

    sys.exit(app.exec())

# pyinstaller
#     - one file
# pyinstaller csv_plot.py --onefile --name csv_plot --icon icon.ico --add-data "icon.ico;." --add-data "README.md;." --noconsole --noupx --clean --noconfirm --hidden-import PySide6
#     - one dir
# pyinstaller csv_plot.py --onedir --name csv_plot --icon icon.ico --add-data "icon.ico;." --add-data "README.md;." --noconsole --clean --noconfirm --hidden-import PySide6


# nuitka
# nuitka --onefile --standalone --output-filename=csv_plot --windows-console-mode=disable --windows-icon-from-ico=icon.ico --enable-plugin=pyside6 --include-data-file=icon.ico=data --include-data-file=README.md=data csv_plot.py
# nuitka --standalone --output-filename=csv_plot --windows-console-mode=disable --windows-icon-from-ico=icon.ico --enable-plugin=pyside6 --include-data-file=icon.ico=data --include-data-file=README.md=data csv_plot.py
