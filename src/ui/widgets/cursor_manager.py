"""
CursorManager - 光标管理

负责 DraggableGraphicsLayoutWidget 的所有光标相关功能：
- 光标模式管理（自由光标、固定光标、双固定光标、off 模式）
- 光标位置和标签更新
- 光标可视化元素管理（vline、圆圈、文本标签）
- 光标对象池管理
- 光标相关 ViewBox 信号处理

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np

from src.core.config import (
    DEBUG_LOG_ENABLED,
    debug_log,
)

if TYPE_CHECKING:
    from src.ui.widgets.multi_curve_manager import MultiCurveManager


class CursorManager:
    """负责光标位置、标签、模式、对象池管理和 ViewBox 信号处理"""

    def __init__(self, multi_curve_manager: MultiCurveManager):
        if multi_curve_manager is None:
            raise ValueError("CursorManager requires a valid MultiCurveManager instance")
        self._data_manager = multi_curve_manager

    @property
    def pw(self) -> Any:
        return self._data_manager.pw

    @property
    def _is_interacting(self) -> bool:
        return getattr(self.pw, '_is_interacting', False)
    
    @_is_interacting.setter
    def _is_interacting(self, value: bool):
        self.pw._is_interacting = value

    @property
    def _cursor_label_busy(self) -> bool:
        return getattr(self.pw, '_cursor_label_busy', False)
    
    @_cursor_label_busy.setter
    def _cursor_label_busy(self, value: bool):
        self.pw._cursor_label_busy = value

    @property
    def _cursor_label_dirty(self) -> bool:
        return getattr(self.pw, '_cursor_label_dirty', False)
    
    @_cursor_label_dirty.setter
    def _cursor_label_dirty(self, value: bool):
        self.pw._cursor_label_dirty = value

    @property
    def show_values_only(self) -> bool:
        return getattr(self.pw, 'show_values_only', False)
    
    @show_values_only.setter
    def show_values_only(self, value: bool):
        self.pw.show_values_only = value

    @property
    def last_valid_cursor_mode(self) -> str:
        return getattr(self.pw, 'last_valid_cursor_mode', "1 free cursor")
    
    @last_valid_cursor_mode.setter
    def last_valid_cursor_mode(self, value: str):
        self.pw.last_valid_cursor_mode = value

    @property
    def is_cursor_pinned(self) -> bool:
        return getattr(self.pw, 'is_cursor_pinned', False)
    
    @is_cursor_pinned.setter
    def is_cursor_pinned(self, value: bool):
        self.pw.is_cursor_pinned = value

    @property
    def pinned_x_value(self) -> float | None:
        return getattr(self.pw, 'pinned_x_value', None)
    
    @pinned_x_value.setter
    def pinned_x_value(self, value: float | None):
        self.pw.pinned_x_value = value

    @property
    def pinned_x_values(self) -> list:
        return getattr(self.pw, 'pinned_x_values', [])
    
    @pinned_x_values.setter
    def pinned_x_values(self, value: list):
        self.pw.pinned_x_values = value

    @property
    def pinned_index_value(self) -> float | None:
        return getattr(self.pw, 'pinned_index_value', None)
    
    @pinned_index_value.setter
    def pinned_index_value(self, value: float | None):
        self.pw.pinned_index_value = value

    @property
    def pinned_index_values(self) -> list:
        return getattr(self.pw, 'pinned_index_values', [])
    
    @pinned_index_values.setter
    def pinned_index_values(self, value: list):
        self.pw.pinned_index_values = value

    @property
    def factor(self) -> float:
        return getattr(self.pw, 'factor', 1.0)
    
    @property
    def offset(self) -> float:
        return getattr(self.pw, 'offset', 0.0)

    @property
    def y_format(self) -> str:
        return getattr(self.pw, 'y_format', '')
    
    @property
    def y_name(self) -> str:
        return getattr(self.pw, 'y_name', '')

    def _get_cursor_mode(self) -> str:
        """获取当前光标模式"""
        if self.pw.plot_context and hasattr(self.pw.plot_context, "cursor_mode"):
            return self.pw.plot_context.cursor_mode
        return "1 free cursor"

    def _get_cursor_x_positions(self) -> list:
        """获取光标的 x 位置列表"""
        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            if self.pinned_x_values and len(self.pinned_x_values) >= 2:
                return list(self.pinned_x_values[:2])
            positions = []
            if hasattr(self.pw, "vline") and self.pw.vline.isVisible():
                positions.append(self.pw.vline.value())
            if hasattr(self.pw, "vline2") and self.pw.vline2.isVisible():
                positions.append(self.pw.vline2.value())
            return positions
        if mode == "1 anchored cursor":
            if self.pinned_x_values:
                return [self.pinned_x_values[0]]
            if self.pinned_x_value is not None:
                return [self.pinned_x_value]
        if hasattr(self.pw, "vline"):
            return [self.pw.vline.value()]
        return []

    def _set_vline_visibility_for_mode(self, visible: bool, mode: str):
        """根据模式设置 vline 可见性"""
        if not hasattr(self.pw, "vline"):
            return
        if mode == "2 anchored cursor":
            self.pw.vline.setVisible(visible)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setVisible(visible)
        else:
            self.pw.vline.setVisible(visible)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setVisible(False)

    def apply_cursor_mode(self, mode: str, pinned_x_values: list = None):
        """应用光标模式"""
        if pinned_x_values is None:
            pinned_x_values = []

        if mode == "off":
            if self._get_cursor_mode() != "off":
                self.last_valid_cursor_mode = self._get_cursor_mode()
            if hasattr(self.pw, "vline"):
                self.pw.vline.setVisible(False)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setVisible(False)
            self.pw.update_right_header("")
            self._clear_cursor_items(hide_only=True)
            return

        if mode == "1 free cursor":
            self.is_cursor_pinned = False
            self.pinned_x_value = None
            self.pinned_index_value = None
            self.pinned_x_values = []
            self.pinned_index_values = []
            if hasattr(self.pw, "vline"):
                self.pw.vline.setMovable(False)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setMovable(False)
            if hasattr(self.pw.view_box, "is_cursor_pinned"):
                self.pw.view_box.is_cursor_pinned = False
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
            if hasattr(self.pw, "vline") and self.pinned_x_value is not None:
                self.pw.vline.setMovable(True)
                from PyQt6.QtCore import QSignalBlocker
                with QSignalBlocker(self.pw.vline):
                    self.pw.vline.setPos(self.pinned_x_value)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setMovable(False)
            if hasattr(self.pw.view_box, "is_cursor_pinned"):
                self.pw.view_box.is_cursor_pinned = True
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
                view_min, view_max = self.pw.view_box.viewRange()[0]
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
            if hasattr(self.pw, "vline"):
                self.pw.vline.setMovable(True)
            if hasattr(self.pw, "vline2"):
                self.pw.vline2.setMovable(True)
            if hasattr(self.pw, "vline") and self.pinned_x_values:
                from PyQt6.QtCore import QSignalBlocker
                with QSignalBlocker(self.pw.vline):
                    self.pw.vline.setPos(self.pinned_x_values[0])
            if hasattr(self.pw, "vline2") and len(self.pinned_x_values) > 1:
                with QSignalBlocker(self.pw.vline2):
                    self.pw.vline2.setPos(self.pinned_x_values[1])
            if hasattr(self.pw.view_box, "is_cursor_pinned"):
                self.pw.view_box.is_cursor_pinned = True
            self._set_vline_visibility_for_mode(True, mode)
            return

    def update_cursor_label(self):
        """更新光标标签位置和内容"""
        MAX_RETRIES = 3
        retry_count = 0

        while retry_count < MAX_RETRIES:
            if DEBUG_LOG_ENABLED:
                debug_log(
                    "Plot.update_cursor_label start y=%s locked=%s busy=%s dirty=%s retry=%s",
                    getattr(self.pw, "y_name", None),
                    self._is_cursor_update_locked(),
                    self._cursor_label_busy,
                    self._cursor_label_dirty,
                    retry_count,
                )

            if self._is_cursor_update_locked():
                return

            if self._cursor_label_busy:
                self._cursor_label_dirty = True
                return

            self._cursor_label_busy = True
            self._cursor_label_dirty = False

            try:
                self._update_multi_curve_cursor_label()
            except (RuntimeError, AttributeError) as e:
                debug_log("update_cursor_label error: %s", e)
            finally:
                self._cursor_label_busy = False

            if self._cursor_label_dirty:
                self._cursor_label_dirty = False
                retry_count += 1
                continue
            else:
                break

        if retry_count >= MAX_RETRIES:
            debug_log("update_cursor_label exceeded max retries for y=%s", getattr(self.pw, "y_name", None))

    def _is_cursor_update_locked(self) -> bool:
        """判断 cursor 相关回调是否需要被暂时禁用"""
        if getattr(self.pw, '_is_updating_data', False) or getattr(self.pw, '_is_being_destroyed', False):
            return True

        if self.pw.plot_context:
            if getattr(self.pw.plot_context, '_is_loading_new_data', False):
                return True
            current_version = getattr(self.pw.plot_context, '_data_version', 0)
            my_version = getattr(self.pw, '_cached_data_version', 0)
            if my_version != 0 and my_version != current_version:
                return True

        return False

    def _update_single_curve_cursor_label(self):
        """更新单曲线模式的光标标签"""
        if len(self.pw.plot_item.listDataItems()) == 0:
            self.pw.update_right_header("")
            return

        try:
            x = self.pw.vline.value()
            curve = self.pw.plot_item.listDataItems()[0]
            x_data, y_data = curve.getData()
            if x_data is None or len(x_data) == 0:
                self.pw.update_right_header("")
                return
            x = np.clip(x, x_data.min(), x_data.max())
            idx = np.argmin(np.abs(x_data - x))
            y_val = y_data[idx]
            x_str = self.pw._significant_decimal_format_str(value=float(x), ref=self.factor)
            if self.y_format == 'enum':
                enum_map = getattr(self.pw.plot_context, '_enum_text_maps', {}).get(self.y_name, {})
                y_str = enum_map.get(int(y_val), str(y_val))
                self.pw.update_right_header(f"x={x_str}, y={y_str}")
            elif self.y_format == 's':
                time_str = self.pw.sInt_to_fmtStr(y_val)
                self.pw.update_right_header(f"x={x_str}, y={time_str}")
            elif self.y_format == 'date':
                date_str = self.pw.dateInt_to_fmtStr(y_val)
                self.pw.update_right_header(f"x={x_str}, y={date_str}")
            else:
                self.pw.update_right_header(f"x={x_str}, y={y_val:.5g}")

        except Exception as e:
            print(f"Cursor update error: {e}")
            self.pw.update_right_header("")

    def _get_circle_from_pool(self, index: int):
        """从对象池获取 ScatterPlotItem"""
        pool = self.pw._cursor_item_pool['circles']
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        circle = pg.ScatterPlotItem(symbol='o', size=8, brush=None)
        pool.append(circle)
        return circle

    def _get_label_from_pool(self, index: int):
        """从对象池获取 TextItem"""
        pool = self.pw._cursor_item_pool['labels']
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        from PyQt6.QtWidgets import QApplication
        
        label = pg.TextItem(
            color=(0, 0, 0),
            fill=pg.mkBrush(255, 255, 255, 220),
            anchor=(0.5, 0.5)
        )
        font = QApplication.font()
        font.setPixelSize(11)
        label.setFont(font)
        pool.append(label)
        return label

    def _get_x_label_from_pool(self, index: int):
        """获取 X 轴标签 TextItem"""
        pool = self.pw._cursor_item_pool["x_labels"]
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        from PyQt6.QtWidgets import QApplication
        
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

    def _clear_cursor_items(self, hide_only: bool = True):
        """清除或隐藏所有 cursor 可视化元素"""
        if not hasattr(self.pw, 'multi_cursor_items') or not hasattr(self.pw, 'plot_item'):
            return

        for item in self.pw.multi_cursor_items:
            try:
                if item is not None:
                    item.setVisible(False)
            except (RuntimeError, AttributeError):
                pass

        for item in self.pw.multi_cursor_items:
            try:
                item_type = type(item).__name__
                if item_type == 'ScatterPlotItem':
                    try:
                        item.clear()
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self.pw._cursor_item_pool.get('x_labels', []):
                    try:
                        item.setText("")
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self.pw._cursor_item_pool.get('labels', []):
                    try:
                        item.setText("")
                    except (RuntimeError, AttributeError):
                        pass
                else:
                    self._queue_item_for_deletion(item)
            except Exception:
                pass

        self.pw.multi_cursor_items.clear()

        if not hide_only:
            for circle in self.pw._cursor_item_pool.get('circles', []):
                self._queue_item_for_deletion(circle)
            for label in self.pw._cursor_item_pool.get('labels', []):
                self._queue_item_for_deletion(label)
            for x_label in self.pw._cursor_item_pool.get('x_labels', []):
                self._queue_item_for_deletion(x_label)

            self.pw._cursor_item_pool = {
                'circles': [],
                'labels': [],
                'x_labels': []
            }

            if self.pw._pending_delete_items and not self.pw._cleanup_timer.isActive():
                self.pw._cleanup_timer.start(100)

    def _queue_item_for_deletion(self, item):
        """将 item 加入待删除队列"""
        if item is not None and item not in self.pw._pending_delete_items:
            try:
                item.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            self.pw._pending_delete_items.append(item)

    def _process_pending_deletes(self):
        """安全地处理待删除队列"""
        if self.pw._is_updating_data or self.pw._is_being_destroyed:
            if self.pw._pending_delete_items:
                self.pw._cleanup_timer.start(100)
            return

        items_to_delete = self.pw._pending_delete_items.copy()
        self.pw._pending_delete_items.clear()

        for item in items_to_delete:
            try:
                if item is None:
                    continue
                try:
                    scene = item.scene()
                    if scene is not None:
                        scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
                try:
                    if hasattr(item, 'deleteLater'):
                        item.deleteLater()
                except (RuntimeError, AttributeError):
                    pass
            except Exception as e:
                debug_log("_process_pending_deletes error: %s", e)

    def _update_multi_curve_cursor_label(self):
        """更新多曲线光标标签"""
        if self._is_interacting:
            return

        import time
        current_time = time.time()

        adaptive_throttle = 0.05
        if hasattr(self.pw, '_adaptive_throttle_enabled') and self.pw._adaptive_throttle_enabled and hasattr(self.pw, "curves"):
            curve_count = len(self.pw.curves) if self.pw.curves else 0
            adaptive_throttle = min(0.016 + curve_count * 0.002, 0.1)
        elif hasattr(self.pw, '_cursor_update_throttle'):
            adaptive_throttle = self.pw._cursor_update_throttle

        if hasattr(self.pw, "_last_cursor_update_time"):
            time_since_last = current_time - self.pw._last_cursor_update_time
            if time_since_last < adaptive_throttle:
                return
        self.pw._last_cursor_update_time = current_time

        self._clear_cursor_items()

        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            vline_visible = bool(self.pw.vline.isVisible() or self.pw.vline2.isVisible())
        else:
            vline_visible = self.pw.vline.isVisible()
        if not vline_visible:
            self.pw.update_right_header("")
            return

        if self.show_values_only:
            self._show_x_position_only()
            return

        if not self.pw.curves and not self.pw.curve:
            self.pw.update_right_header("")
            return

        x_positions = self._get_cursor_x_positions()
        if not x_positions:
            self.pw.update_right_header("")
            return

        try:
            cursor_values = []
            (x_min, x_max), (y_min, y_max) = self.pw.view_box.viewRange()

            curves_to_process = []
            if self.pw.curves:
                for var_name, ci in self.pw.curves.items():
                    if not ci.visible:
                        continue
                    curves_to_process.append({
                        "var_name": var_name,
                        "x_data": ci.x_data,
                        "y_data": ci.y_data,
                        "color": ci.color,
                        "y_format": ci.y_format,
                        "unit": self.pw.units.get(var_name, ""),
                        "enum_map": getattr(self.pw.plot_context, '_enum_text_maps', {}).get(var_name, {})
                    })
            elif not self.pw.is_multi_curve_mode and self.pw.curve and self.y_name:
                x_data, y_data = self.pw.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    curve_color = "blue"
                    try:
                        if hasattr(self.pw.curve, "opts") and "pen" in self.pw.curve.opts:
                            pen = self.pw.curve.opts["pen"]
                            if hasattr(pen, "color"):
                                curve_color = pen.color().name()
                    except Exception:
                        pass
                    curves_to_process.append({
                        "var_name": self.y_name,
                        "x_data": x_data,
                        "y_data": y_data,
                        "color": curve_color,
                        "y_format": self.y_format,
                        "unit": self.pw.units.get(self.y_name, ""),
                        "enum_map": getattr(self.pw.plot_context, '_enum_text_maps', {}).get(self.y_name, {})
                    })

            import pyqtgraph as pg

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
                        y_str = self.pw.sInt_to_fmtStr(y_val)
                    elif y_format == "date":
                        y_str = self.pw.dateInt_to_fmtStr(y_val)
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
                    plot_scene = self.pw.plot_item.scene()
                    if circle_scene != plot_scene:
                        if circle_scene is not None:
                            circle_scene.removeItem(circle)
                        self.pw.plot_item.addItem(circle, ignoreBounds=True)
                    self.pw.multi_cursor_items.append(circle)

            self._position_labels_avoid_overlap(cursor_values, x_min, x_max, y_min, y_max)

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self.pw._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)
                view_rect = self.pw.plot_item.vb.sceneBoundingRect()
                scene_point = self.pw.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)
                scene = self.pw.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)
                self.pw.multi_cursor_items.append(x_info_item)

        except Exception as e:
            print(f"Multi-curve cursor update error: {e}")
            self.pw.update_right_header("")

    def _position_labels_avoid_overlap(self, cursor_values: list, x_min: float, x_max: float, y_min: float, y_max: float):
        """优化的标签定位算法"""
        if not cursor_values:
            return

        import pyqtgraph as pg

        x_range = x_max - x_min
        y_range = y_max - y_min
        view_box = self.pw.plot_item.getViewBox()
        view_width_pixels = max(1, view_box.width())
        view_height_pixels = max(1, view_box.height())

        pixel_to_data_x = x_range / view_width_pixels
        pixel_to_data_y = y_range / view_height_pixels

        gap_pixels = 5
        vertical_gap_pixels = 10

        for idx, cursor_val in enumerate(cursor_values):
            var_name = cursor_val["var_name"]
            x_pos = cursor_val["x_pos"]
            y_pos = cursor_val["y_pos"]
            y_value = cursor_val["y_value"]
            color = cursor_val["color"]

            unit = ""
            if hasattr(self.pw, 'units'):
                unit = self.pw.units.get(var_name, "")
            label_text = f"{y_value}"
            if unit:
                label_text += f" {unit}"

            label = self._get_label_from_pool(idx)
            label.setText(label_text)

            if not hasattr(label, "_cached_color") or label._cached_color != color:
                label.setColor(color)
                label._cached_color = color

            view_rect = self.pw.plot_item.vb.sceneBoundingRect()

            positions = [
                (x_pos + gap_pixels * pixel_to_data_x, y_max - vertical_gap_pixels * pixel_to_data_y),
                (x_pos - gap_pixels * pixel_to_data_x, y_max - vertical_gap_pixels * pixel_to_data_y),
                (x_pos + gap_pixels * pixel_to_data_x, y_min + vertical_gap_pixels * pixel_to_data_y),
                (x_pos - gap_pixels * pixel_to_data_x, y_min + vertical_gap_pixels * pixel_to_data_y),
            ]

            selected_pos = positions[0]
            for pos in positions:
                candidate_x, candidate_y = pos
                if x_min <= candidate_x <= x_max and y_min <= candidate_y <= y_max:
                    selected_pos = pos
                    break

            scene_point = self.pw.plot_item.vb.mapViewToScene(pg.Point(selected_pos[0], selected_pos[1]))
            label.setPos(scene_point.x(), scene_point.y())
            label.setVisible(True)
            label.setZValue(1000 + idx)

            scene = self.pw.plot_item.scene()
            label_scene = label.scene()
            if label_scene != scene:
                if label_scene is not None:
                    label_scene.removeItem(label)
                scene.addItem(label)
            self.pw.multi_cursor_items.append(label)

    def _show_x_position_only(self, x_positions=None):
        """仅显示 x 位置标签（隐藏光标数值），同时在图上绘制 x_label 元素"""
        try:
            if not self._has_visible_curve_data():
                self._clear_cursor_items()
                self.pw.update_right_header("")
                return
            x_positions = x_positions if x_positions is not None else self._get_cursor_x_positions()
            if not x_positions:
                self.pw.update_right_header("")
                return

            (x_min, x_max), (y_min, y_max) = self.pw.view_box.viewRange()
            self._clear_cursor_items()

            import pyqtgraph as pg

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self.pw._significant_decimal_format_str(value=float(x), ref=self.factor)
                x_info_item = self._get_x_label_from_pool(idx)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)

                view_rect = self.pw.plot_item.vb.sceneBoundingRect()
                scene_point = self.pw.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)

                scene = self.pw.plot_item.scene()
                x_scene = x_info_item.scene()
                if x_scene != scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    scene.addItem(x_info_item)

                self.pw.multi_cursor_items.append(x_info_item)

            parts = []
            for x in x_positions:
                x_str = self.pw._significant_decimal_format_str(value=float(x), ref=self.factor)
                parts.append(f"x={x_str}")
            header_text = " | ".join(parts)
            self.pw.update_right_header(header_text)

        except Exception as e:
            print(f"x_position_only error: {e}")

    def _has_visible_curve_data(self) -> bool:
        """判断当前 plot 是否有可见且有数据的曲线"""
        try:
            if self.pw.curves:
                for ci in self.pw.curves.values():
                    if not ci.visible:
                        continue
                    x_data = ci.x_data
                    if x_data is not None and len(x_data) > 0:
                        return True
                return False
            if self.pw.curve:
                x_data, _ = self.pw.curve.getData()
                return x_data is not None and len(x_data) > 0
            return False
        except Exception:
            return False

    def toggle_cursor(self, show: bool, hide_values_only: bool = False):
        """切换光标显示状态"""
        if not hasattr(self.pw, "vline"):
            return

        mode = self._get_cursor_mode()

        if hide_values_only:
            self._set_vline_visibility_for_mode(True, mode)
            self.pw.cursor_label.setVisible(False)
            self.show_values_only = True
            self._clear_cursor_items()
            self._show_x_position_only()
        else:
            self._set_vline_visibility_for_mode(show, mode)
            self.pw.cursor_label.setVisible(show)
            self.show_values_only = not show

            if not show:
                self._clear_cursor_items()
                self.pw.update_right_header("")
                self.is_cursor_pinned = False
                self.pinned_x_value = None
                self.pinned_index_value = None
                self.pinned_x_values = []
                self.pinned_index_values = []
            else:
                self.update_cursor_label()

    def pin_cursor(self, x_value: float):
        """固定光标到指定位置"""
        self.is_cursor_pinned = True
        self.pinned_x_value = x_value
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []

        self.pw.vline.setMovable(False)
        if hasattr(self.pw, "vline2"):
            self.pw.vline2.setMovable(False)

        if hasattr(self.pw.view_box, "is_cursor_pinned"):
            self.pw.view_box.is_cursor_pinned = False

    def free_cursor(self):
        """释放固定光标"""
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []

        self.pw.vline.setMovable(False)
        if hasattr(self.pw, "vline2"):
            self.pw.vline2.setMovable(False)

        if hasattr(self.pw.view_box, "is_cursor_pinned"):
            self.pw.view_box.is_cursor_pinned = False

    def reset_pin_state(self):
        """重置固定状态"""
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_index_value = None
        self.pinned_x_values = []
        self.pinned_index_values = []

        if hasattr(self.pw, "vline"):
            self.pw.vline.setMovable(False)
        if hasattr(self.pw, "vline2"):
            self.pw.vline2.setMovable(False)
        if hasattr(self.pw, "vline2"):
            self.pw.vline2.setVisible(False)

    def _update_vline_bounds_from_data(self):
        """从数据更新 vline 边界"""
        x_min, x_max = self.pw.get_curve_x_limits()
        if x_min is not None and x_max is not None:
            self.pw.view_box.setXRange(x_min, x_max, padding=0.02)

    def _update_cursor_after_plot(self, min_x_bound: float, max_x_bound: float):
        """绘图后更新光标"""
        if self.is_cursor_pinned and self.pinned_x_value is not None:
            if self.pinned_x_value < min_x_bound or self.pinned_x_value > max_x_bound:
                self.pinned_x_value = (min_x_bound + max_x_bound) / 2
                self.pw.vline.setPos(self.pinned_x_value)
                if hasattr(self.pw, "vline2") and len(self.pinned_x_values) > 1:
                    self.pw.vline2.setPos(self.pinned_x_values[1])

    def _start_interaction(self):
        """开始交互"""
        try:
            if not hasattr(self.pw, '_is_interacting'):
                self.pw._is_interacting = False
            if not self.pw._is_interacting:
                self.pw._is_interacting = True
            if not hasattr(self.pw, '_cursor_refresh_timer'):
                return
            if self.pw._cursor_refresh_timer.isActive():
                self.pw._cursor_refresh_timer.stop()
        except Exception as e:
            print(f"开始交互出错: {e}")

    def _end_interaction(self):
        """结束交互"""
        try:
            if hasattr(self.pw, '_is_interacting'):
                self.pw._is_interacting = False
            if hasattr(self.pw, '_cursor_refresh_timer'):
                self.pw._cursor_refresh_timer.start(50)
        except Exception as e:
            print(f"结束交互出错: {e}")

    def _schedule_cursor_geometry_update(self):
        """调度光标几何更新"""
        if not hasattr(self.pw, 'vline') or not self.pw.vline.isVisible():
            return
        if getattr(self.pw, '_cursor_refresh_timer', None) is None:
            return
        if not self.pw._cursor_refresh_timer.isActive():
            self.pw._cursor_refresh_timer.start(50)

    def _refresh_cursor_geometry(self):
        """刷新光标几何"""
        if not hasattr(self.pw, 'vline') or not self.pw.vline.isVisible():
            return
        if self._is_interacting:
            return
        self.update_cursor_label()

    def on_vline_position_changed(self, line_obj=None):
        """vline 位置变化时更新光标状态"""
        if self._is_cursor_update_locked():
            return
        if self.pw.plot_context and getattr(self.pw.plot_context, "_is_time_correction_active", False):
            return

        line = line_obj if line_obj is not None else self.pw.vline
        cursor_index = getattr(line, "cursor_index", 0)

        if self.is_cursor_pinned:
            if getattr(self.pw, "_suppress_pin_update", False):
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

            if self.pw.plot_context:
                self.pw.plot_context.pinned_x_values = list(self.pinned_x_values)

            if self.pw.plot_context:
                for container in self.pw.plot_context.plot_widgets:
                    widget = container
                    if widget.is_cursor_pinned and widget != self.pw:
                        target_line = widget.vline if cursor_index == 0 else getattr(widget, "vline2", None)
                        if target_line is not None:
                            from PyQt6.QtCore import QSignalBlocker
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

    def _on_vb_set_cursor_mode(self, mode: str, pw, ctx_x: float):
        """ViewBox 信号：设置光标模式"""
        if pw != self.pw:
            return
        if self.pw.plot_context:
            self.pw.plot_context.cursor_mode = mode
        self.apply_cursor_mode(mode)

    def _on_vb_show_cursor(self, pw):
        """ViewBox 信号：显示光标"""
        if pw != self.pw:
            return
        self.toggle_cursor(True)

    def _on_vb_hide_cursor(self, pw):
        """ViewBox 信号：隐藏光标"""
        if pw != self.pw:
            return
        self.toggle_cursor(False)
