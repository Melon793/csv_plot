from __future__ import annotations
import sys
import os
from typing import Any

from src.utils.platform_setup import setup_platform
setup_platform()

import numpy as np
import pandas as pd

from src.ui.drag_drop import VAR_SEPARATOR, parse_var_names_from_mimedata
from src.core.config import (safe_callback, DEFAULT_PADDING_VAL_X, XRANGE_THRESHOLD_FOR_SYMBOLS, FACTOR_SCROLL_ZOOM, DEFAULT_LINE_WIDTH, THICK_LINE_WIDTH, THIN_LINE_WIDTH, UI_DEBOUNCE_DELAY_MS, PLOT_ROW_MAX_DEFAULT, PLOT_ROW_CURRENT_DEFAULT, DEFAULT_SHOW_X_AXIS_LABEL)
from src.core.data_types import CurveInfo
from src.ui.table_dialog import DataTableDialog
from src.ui.plot_variable_editor import PlotVariableEditorDialog


from PySide6.QtCore import Qt, QTimer, QPoint, QSize, QRect, QRectF, QItemSelectionModel
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView,
    QMessageBox,
)
import pyqtgraph as pg


class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    """
    可拖拽的图形布局控件类
    支持图表区域的拖拽重排和动态布局调整
    提供灵活的图表排列和交互功能
    """
    def __init__(self, units_dict, dataframe, time_channels_info=None, synchronizer=None):
        if time_channels_info is None:
            time_channels_info = {}
        super().__init__()
        self.factor = 1.0
        self.offset = 0.0
        self.original_index_x = None
        self.original_y = None
        self.mark_region = None
        self.is_cursor_pinned = False  # 记录cursor是否被固定
        self.last_valid_cursor_mode = "1 free cursor"
        self.pinned_x_value = None  # 记录固定的x值
        self.pinned_index_value = None  # 记录固定的索引值
        self.pinned_x_values = []
        self.pinned_index_values = []
        self._is_updating_data = False  # 标志：正在更新数据，禁止某些操作
        self._is_being_destroyed = False  # 标志：对象正在被销毁
        self._suppress_pin_update = False  # 标志：临时禁止pin状态自动更新
        self._cursor_label_busy = False
        self._cached_data_version = 0  # 【稳定性优化】缓存的数据版本号
        self._original_downsample_ds = None
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
        self._init_manager_chain()
        self.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
    def setup_ui(self, units_dict, dataframe, time_channels_info=None, synchronizer=None):
        if time_channels_info is None:
            time_channels_info = {}
        self._plot_ui_manager.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)

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
        """配置顶部 header → 委托到 PlotUIManager"""
        self._plot_ui_manager._setup_header(self)

    def setup_plot_area(self):
        """配置绘图区域 → 委托到 PlotUIManager"""
        self._plot_ui_manager._setup_plot_area(self)

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
        """配置坐标轴样式 → 委托到 PlotUIManager"""
        self._plot_ui_manager._setup_axes(self)

    def setup_interaction(self):
        """配置交互元素 → 委托到 PlotUIManager"""
        self._plot_ui_manager._setup_interaction(self)

    def _init_ui_refresh_coordinator(self):
        """初始化 UI 刷新调度器 → 委托到 PlotUIManager"""
        self._plot_ui_manager._init_ui_refresh_coordinator(self)

    def _queue_ui_refresh(self, *, style=True, cursor=True, stats=True, immediate=False):
        """调度 UI 更新 → 委托到 PlotUIManager"""
        self._plot_ui_manager._queue_ui_refresh(self, style=style, cursor=cursor, stats=stats, immediate=immediate)

    def _cancel_ui_refresh(self, *tasks):
        """取消 UI 刷新 → 委托到 PlotUIManager"""
        self._plot_ui_manager._cancel_ui_refresh(self, *tasks)

    def _run_style_refresh(self):
        """执行样式刷新 → 委托到 PlotUIManager"""
        self._plot_ui_manager._run_style_refresh(self)

    def _run_cursor_refresh(self):
        """执行光标刷新 → 委托到 PlotUIManager"""
        self._plot_ui_manager._run_cursor_refresh(self)

    def _run_stats_refresh(self):
        """执行统计刷新 → 委托到 PlotUIManager"""
        self._plot_ui_manager._run_stats_refresh(self)

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
        if hasattr(main_window, 'layout_manager'):
            container = main_window.layout_manager._get_plot_container(self)
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
            main_window.layout_manager._hide_drag_indicator_for_plot(self)

    def _notify_drag_indicator(
        self,
        var_names: list[str] | None = None,
        hide: bool = False,
        source_widget: QWidget | None = None,
        indicator_text: str | None = None,
    ):
        main_window = self.window()

        if not main_window or not hasattr(main_window, 'layout_manager'):
            return

        if not hide and source_widget is None and self._should_hide_drag_indicator(main_window):
            hide = True

        if hide:
            self._drag_indicator_source = None
            self._drag_indicator_guard.stop()
            main_window.layout_manager._hide_drag_indicator_for_plot(self)
            return

        self._drag_indicator_source = source_widget
        main_window.layout_manager._show_drag_indicator_for_plot(self, var_names or [], indicator_text)
        if not self._drag_indicator_guard.isActive():
            self._drag_indicator_guard.start()


    def handle_single_point_limits(self, x_values, y_values):
        """处理单点或所有点x坐标相同的特殊情况 → 委托到 PlotDataManager"""
        return self._plot_data_manager.handle_single_point_limits(x_values, y_values)
        
    def paintEvent(self, event):
        """重写 paintEvent：数据重载期间跳过绘制，防止 SIGSEGV。

        虽然 _begin_data_reload 会调用 setUpdatesEnabled(False)，
        但 QWidgetRepaintManager::sync() 可能绕过该标志强制刷新，
        导致 QGraphicsView 在 scene items 被销毁/重建期间尝试绘制。
        """
        if getattr(self, '_is_updating_data', False):
            return
        if getattr(self, '_is_cursor_modifying_scene', False):
            return
        main_window = self.window()
        if main_window is None:
            return
        if getattr(main_window, '_is_loading_new_data', False):
            return
        if hasattr(self, 'plot_item') and self.plot_item is not None:
            try:
                if self.plot_item.scene() is None:
                    return
            except RuntimeError:
                return
        if hasattr(self, 'vline'):
            try:
                _ = self.vline.scene()
            except RuntimeError:
                return
        try:
            super().paintEvent(event)
        except RuntimeError:
            pass
        except Exception:
            pass

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
            if hasattr(self.window(), 'cursor_sync_manager'):
                self.window().cursor_sync_manager.sync_crosshair(mousePoint.x(), self)
            #print(f"mouse in pos {mousePoint.x()}")

    def _is_cursor_update_locked(self) -> bool:
        """判断 cursor 更新是否被锁定 → 委托到 CursorManager"""
        return self._cursor_manager._is_cursor_update_locked()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_event_handler'):
            self._schedule_cursor_geometry_update()

    @safe_callback
    def on_vline_position_changed(self, line_obj=None):
        """vline 位置变化时更新光标状态 → 委托到 CursorManager"""
        self._cursor_manager.on_vline_position_changed(line_obj)

    def sInt_to_fmtStr(self, value: int):
        """将秒数转换为时间字符串 → 委托到 CursorManager"""
        return self._cursor_manager.sInt_to_fmtStr(value)
    
    def dateInt_to_fmtStr(self, value: int):
        """将时间戳转换为日期字符串 → 委托到 CursorManager"""
        return self._cursor_manager.dateInt_to_fmtStr(value)
    
    def _significant_decimal_format_str(self, value: float, ref: float, max_dp: int | None = None) -> str:
        """根据 ref 的显示精度自动决定 value 的字符串格式 → 委托到 CursorManager"""
        return self._cursor_manager._significant_decimal_format_str(value, ref, max_dp)
    


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
        """根据当前绘制的数据更新vline bounds → 委托到 CursorManager"""
        return self._cursor_manager._update_vline_bounds_from_data()
    
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
            self.window().layout_manager.request_mark_stats_refresh()

    def add_variables_to_plot(self, var_names: list[str]):
        """批量添加变量到当前绘图区 → 委托到 MultiCurveManager"""
        self._multi_curve_manager.add_variables_to_plot(var_names)

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
        """添加变量到多曲线绘图 → 委托到 MultiCurveManager"""
        return self._multi_curve_manager.add_variable_to_plot(
            var_name, x_values, y_values, y_format,
            skip_existence_check, show_duplicate_warning, preferred_color
        )
    
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
        """统一更新 X 轴 limits → 委托到 AxisManager"""
        self._axis_manager._update_x_limits_for_plot(x_values, y_values, is_mdf)

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mouseDoubleClickEvent(event)
            return
        from src.ui.dialogs.axis import AxisDialog
        from src.ui.plot_variable_editor import PlotVariableEditorDialog
        
        if event.button() == Qt.MouseButton.MiddleButton:
            self.clear_plot_item()
            self.window().layout_manager.request_mark_stats_refresh(immediate=True)
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
        """应用绘图样式 → 委托到 MultiCurveManager"""
        self._multi_curve_manager._apply_plot_style(show_symbols)

    def _calculate_visible_points(self, range):
        """计算当前可见范围的点数估算 → 委托到 PlotUIManager"""
        return self._plot_ui_manager._calculate_visible_points(self, range)

    def update_plot_style(self, view_box, range, rect=None):
        """更新绘图样式 → 委托到 PlotUIManager"""
        self._plot_ui_manager.update_plot_style(self, view_box, range, rect)


    @safe_callback
    def _on_range_changed(self, view_box, range, changed=None):
        """ViewBox范围变化回调 → 委托到 EventHandler"""
        self._event_handler._on_range_changed(view_box, range, changed)

    def _start_interaction(self):
        """开始交互优化 → 委托到 EventHandler"""
        self._event_handler._start_interaction()

    def _end_interaction(self):
        """结束交互处理 → 委托到 EventHandler"""
        self._event_handler._end_interaction()

    def _schedule_cursor_geometry_update(self):
        """调度光标几何更新 → 委托到 EventHandler"""
        if not hasattr(self, '_event_handler'):
            return
        self._event_handler._schedule_cursor_geometry_update()

    def _refresh_cursor_geometry(self):
        """刷新光标几何 → 委托到 EventHandler"""
        self._event_handler._refresh_cursor_geometry()

    def _connect_viewbox_signals(self):
        """连接 ViewBox 信号 → 委托到 EventHandler"""
        self._event_handler._connect_viewbox_signals()

    def _on_vb_jump(self, pw, ctx_x):
        self._event_handler._on_vb_jump(pw, ctx_x)

    def _on_vb_clear(self, pw):
        self._event_handler._on_vb_clear(pw)

    def _on_vb_auto_y(self, pw):
        self._event_handler._on_vb_auto_y(pw)

    def _on_vb_set_cursor_mode(self, mode, pw, ctx_x):
        self._event_handler._on_vb_set_cursor_mode(mode, pw, ctx_x)

    def _on_vb_show_cursor(self, pw):
        self._event_handler._on_vb_show_cursor(pw)

    def _on_vb_hide_cursor(self, pw):
        self._event_handler._on_vb_hide_cursor(pw)

    def _on_vb_set_row_height(self, pct, pw):
        self._event_handler._on_vb_set_row_height(pct, pw)

    def _on_vb_set_all_row_height(self, pct):
        self._event_handler._on_vb_set_all_row_height(pct)

    def _on_vb_copy_name(self, pw):
        self._event_handler._on_vb_copy_name(pw)

    def _on_vb_var_editor(self, pw):
        self._event_handler._on_vb_var_editor(pw)


