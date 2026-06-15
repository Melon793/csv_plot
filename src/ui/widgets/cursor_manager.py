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

import pyqtgraph as pg
from PySide6.QtGui import QFontMetrics
from PySide6.QtCore import QPointF, QSignalBlocker

if TYPE_CHECKING:
    from src.ui.widgets.multi_curve_manager import MultiCurveManager


class CursorManager:
    """负责光标位置、标签、模式、对象池管理和 ViewBox 信号处理"""

    def __init__(self, multi_curve_manager: MultiCurveManager):
        """初始化光标管理器，绑定到 MultiCurveManager 以获取依赖链"""
        import numpy as np
        globals()['np'] = np
        if multi_curve_manager is None:
            raise ValueError(
                "CursorManager requires a valid MultiCurveManager instance"
            )
        self._data_manager = multi_curve_manager

    @property
    def pw(self) -> Any:
        """关联的 DraggableGraphicsLayoutWidget 实例"""
        return self._data_manager.pw

    @property
    def _is_interacting(self) -> bool:
        """用户是否正在交互（拖拽/缩放中）

        状态所有者：widget（self.pw._is_interacting）。
        所有 Manager 通过 self.pw 读写，不独立存储。
        """
        return getattr(self.pw, "_is_interacting", False)

    @_is_interacting.setter
    def _is_interacting(self, value: bool):
        self.pw._is_interacting = value

    @property
    def _cursor_label_busy(self) -> bool:
        """光标标签是否正在更新（防抖标志）"""
        return getattr(self.pw, "_cursor_label_busy", False)

    @_cursor_label_busy.setter
    def _cursor_label_busy(self, value: bool):
        self.pw._cursor_label_busy = value

    @property
    def _cursor_label_dirty(self) -> bool:
        """光标标签数据是否过期需要刷新"""
        return getattr(self.pw, "_cursor_label_dirty", False)

    @_cursor_label_dirty.setter
    def _cursor_label_dirty(self, value: bool):
        self.pw._cursor_label_dirty = value

    @property
    def show_values_only(self) -> bool:
        """是否仅显示坐标值（隐藏曲线数值标签）"""
        return getattr(self.pw, "show_values_only", False)

    @show_values_only.setter
    def show_values_only(self, value: bool):
        self.pw.show_values_only = value

    @property
    def last_valid_cursor_mode(self) -> str:
        """上一次有效的光标模式（用于恢复）"""
        return getattr(self.pw, "last_valid_cursor_mode", "1 free cursor")

    @last_valid_cursor_mode.setter
    def last_valid_cursor_mode(self, value: str):
        self.pw.last_valid_cursor_mode = value

    @property
    def is_cursor_pinned(self) -> bool:
        """光标当前是否被固定"""
        return getattr(self.pw, "is_cursor_pinned", False)

    @is_cursor_pinned.setter
    def is_cursor_pinned(self, value: bool):
        self.pw.is_cursor_pinned = value

    @property
    def pinned_x_value(self) -> float | None:
        """固定光标的当前 x 值（单光标模式）"""
        return getattr(self.pw, "pinned_x_value", None)

    @pinned_x_value.setter
    def pinned_x_value(self, value: float | None):
        self.pw.pinned_x_value = value

    @property
    def pinned_x_values(self) -> list:
        """固定光标的 x 值列表（多光标模式）"""
        return getattr(self.pw, "pinned_x_values", [])

    @pinned_x_values.setter
    def pinned_x_values(self, value: list):
        self.pw.pinned_x_values = value

    @property
    def pinned_index_value(self) -> float | None:
        """固定光标对应的数据索引值（单光标模式）"""
        return getattr(self.pw, "pinned_index_value", None)

    @pinned_index_value.setter
    def pinned_index_value(self, value: float | None):
        self.pw.pinned_index_value = value

    @property
    def pinned_index_values(self) -> list:
        """固定光标对应的数据索引值列表（多光标模式）"""
        return getattr(self.pw, "pinned_index_values", [])

    @pinned_index_values.setter
    def pinned_index_values(self, value: list):
        self.pw.pinned_index_values = value

    @property
    def factor(self) -> float:
        """线性变换缩放因子 (x = index * factor + offset)"""
        return getattr(self.pw, "factor", 1.0)

    @property
    def offset(self) -> float:
        """线性变换偏移量 (x = index * factor + offset)"""
        return getattr(self.pw, "offset", 0.0)

    @property
    def y_format(self) -> str:
        return getattr(self.pw, "y_format", "")

    @property
    def y_name(self) -> str:
        return getattr(self.pw, "y_name", "")

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
            self.pinned_x_values = (
                list(pinned_x_values[:1])
                if pinned_x_values
                else self.pinned_x_values[:1]
            )
            if self.pinned_x_values:
                self.pinned_x_value = self.pinned_x_values[0]
            if self.factor != 0 and self.pinned_x_value is not None:
                self.pinned_index_value = (
                    self.pinned_x_value - self.offset
                ) / self.factor
            else:
                self.pinned_index_value = None
            self.pinned_index_values = (
                [self.pinned_index_value] if self.pinned_index_value is not None else []
            )
            if hasattr(self.pw, "vline") and self.pinned_x_value is not None:
                self.pw.vline.setMovable(True)
                from PySide6.QtCore import QSignalBlocker

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
                self.pinned_x_values = [
                    self.pinned_x_values[0],
                    self.pinned_x_values[0],
                ]
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
                from PySide6.QtCore import QSignalBlocker

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

            if self._is_cursor_update_locked():
                return

            if self._cursor_label_busy:
                self._cursor_label_dirty = True
                return

            self._cursor_label_busy = True
            self._cursor_label_dirty = False

            try:
                self._update_multi_curve_cursor_label()
            except (RuntimeError, AttributeError):
                pass
            finally:
                self._cursor_label_busy = False

            if self._cursor_label_dirty:
                self._cursor_label_dirty = False
                retry_count += 1
                continue
            else:
                break

        if retry_count >= MAX_RETRIES:
            pass

    def _is_cursor_update_locked(self) -> bool:
        """判断 cursor 相关回调是否需要被暂时禁用"""
        if getattr(self.pw, "_is_updating_data", False) or getattr(
            self.pw, "_is_being_destroyed", False
        ):
            return True

        if self.pw.plot_context:
            if getattr(self.pw.plot_context, "_is_loading_new_data", False):
                return True
            current_version = getattr(self.pw.plot_context, "_data_version", 0)
            my_version = getattr(self.pw, "_cached_data_version", 0)
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
            x_str = self._significant_decimal_format_str(
                value=float(x), ref=self.factor
            )
            if self.y_format == "enum":
                enum_map = getattr(self.pw.plot_context, "_enum_text_maps", {}).get(
                    self.y_name, {}
                )
                y_str = enum_map.get(int(y_val), str(y_val))
                self.pw.update_right_header(f"x={x_str}, y={y_str}")
            elif self.y_format == "s":
                time_str = self.sInt_to_fmtStr(y_val)
                self.pw.update_right_header(f"x={x_str}, y={time_str}")
            elif self.y_format == "date":
                date_str = self.dateInt_to_fmtStr(y_val)
                self.pw.update_right_header(f"x={x_str}, y={date_str}")
            else:
                self.pw.update_right_header(f"x={x_str}, y={y_val:.5g}")

        except Exception:
            self.pw.update_right_header("")

    def _get_circle_from_pool(self, index: int):
        """从对象池获取 ScatterPlotItem"""
        pool = self.pw._cursor_item_pool["circles"]
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg

        circle = pg.ScatterPlotItem(symbol="o", size=8, brush=None)
        pool.append(circle)
        return circle

    def _get_label_from_pool(self, index: int):
        """从对象池获取 TextItem"""
        pool = self.pw._cursor_item_pool["labels"]
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication

        label = pg.TextItem(
            color=(0, 0, 0), fill=pg.mkBrush(255, 255, 255, 220), anchor=(0.5, 0.5)
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
        from PySide6.QtWidgets import QApplication

        x_label = pg.TextItem(
            color=(255, 255, 255),
            fill=pg.mkBrush(64, 64, 64, 230),
            border=pg.mkPen(128, 128, 128, width=1),
            anchor=(0.5, 0),
        )
        font = QApplication.font()
        font.setPixelSize(12)
        x_label.setFont(font)
        pool.append(x_label)
        return x_label

    def _clear_cursor_items(self, hide_only: bool = True):
        """清除或隐藏所有 cursor 可视化元素"""
        if not hasattr(self.pw, "multi_cursor_items") or not hasattr(
            self.pw, "plot_item"
        ):
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
                if item_type == "ScatterPlotItem":
                    try:
                        item.clear()
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self.pw._cursor_item_pool.get("x_labels", []):
                    try:
                        item.setText("")
                    except (RuntimeError, AttributeError):
                        pass
                elif item in self.pw._cursor_item_pool.get("labels", []):
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
            # 修复2：立即从 scene 移除旧 cursor items，避免延迟清理窗口内的状态不一致
            old_circles = list(self.pw._cursor_item_pool.get("circles", []))
            old_labels = list(self.pw._cursor_item_pool.get("labels", []))
            old_x_labels = list(self.pw._cursor_item_pool.get("x_labels", []))

            self.pw._cursor_item_pool = {"circles": [], "labels": [], "x_labels": []}

            scene = None
            if hasattr(self.pw, 'plot_item') and self.pw.plot_item is not None:
                scene = self.pw.plot_item.scene()

            for item in (old_circles + old_labels + old_x_labels):
                if item is None:
                    continue
                try:
                    item.setVisible(False)
                except (RuntimeError, AttributeError):
                    pass
                try:
                    if scene is not None and item.scene() == scene:
                        scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
                try:
                    item.deleteLater()
                except (RuntimeError, AttributeError):
                    pass

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
                    if hasattr(item, "deleteLater"):
                        item.deleteLater()
                except (RuntimeError, AttributeError):
                    pass
            except (RuntimeError, AttributeError):
                pass

    def _update_multi_curve_cursor_label(self):
        """更新多曲线光标标签"""
        if self._is_interacting:
            return

        import time

        current_time = time.time()

        adaptive_throttle = 0.05
        if (
            hasattr(self.pw, "_adaptive_throttle_enabled")
            and self.pw._adaptive_throttle_enabled
            and hasattr(self.pw, "curves")
        ):
            curve_count = len(self.pw.curves) if self.pw.curves else 0
            adaptive_throttle = min(0.016 + curve_count * 0.002, 0.1)
        elif hasattr(self.pw, "_cursor_update_throttle"):
            adaptive_throttle = self.pw._cursor_update_throttle

        if hasattr(self.pw, "_last_cursor_update_time"):
            time_since_last = current_time - self.pw._last_cursor_update_time
            if time_since_last < adaptive_throttle:
                return
        self.pw._last_cursor_update_time = current_time

        self._clear_cursor_items()

        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            vline_visible = bool(
                self.pw.vline.isVisible() or self.pw.vline2.isVisible()
            )
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
                    curves_to_process.append(
                        {
                            "var_name": var_name,
                            "x_data": ci.x_data,
                            "y_data": ci.y_data,
                            "color": ci.color,
                            "y_format": ci.y_format,
                            "unit": self.pw.units.get(var_name, ""),
                            "enum_map": getattr(
                                self.pw.plot_context, "_enum_text_maps", {}
                            ).get(var_name, {}),
                        }
                    )
            elif not self.pw.is_multi_curve_mode and self.pw.curve and self.y_name:
                x_data, y_data = self.pw.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    curve_color = "blue"
                    try:
                        if (
                            hasattr(self.pw.curve, "opts")
                            and "pen" in self.pw.curve.opts
                        ):
                            pen = self.pw.curve.opts["pen"]
                            if hasattr(pen, "color"):
                                curve_color = pen.color().name()
                    except Exception:
                        pass
                    curves_to_process.append(
                        {
                            "var_name": self.y_name,
                            "x_data": x_data,
                            "y_data": y_data,
                            "color": curve_color,
                            "y_format": self.y_format,
                            "unit": self.pw.units.get(self.y_name, ""),
                            "enum_map": getattr(
                                self.pw.plot_context, "_enum_text_maps", {}
                            ).get(self.y_name, {}),
                        }
                    )

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
                        y_str = self.sInt_to_fmtStr(y_val)
                    elif y_format == "date":
                        y_str = self.dateInt_to_fmtStr(y_val)
                    else:
                        y_str = f"{y_val:.5g}"

                    cursor_values.append(
                        {
                            "var_name": var_name,
                            "x_pos": x_actual,
                            "y_pos": y_val,
                            "y_value": y_str,
                            "color": color,
                        }
                    )

                    circle = self._get_circle_from_pool(len(cursor_values) - 1)
                    circle.clear()
                    circle.setData([x_actual], [y_val])
                    if (
                        not hasattr(circle, "_cached_color")
                        or circle._cached_color != color
                    ):
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

            self._position_labels_avoid_overlap(
                cursor_values, x_min, x_max, y_min, y_max
            )

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
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

        except Exception:
            self.pw.update_right_header("")

    def _position_labels_avoid_overlap(
        self,
        cursor_values: list,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
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

        pw = self.pw

        # 计算视图范围，用于边界检查
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 获取实际视图尺寸
        view_box = pw.plot_item.getViewBox()
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
            plot_scene = pw.plot_item.scene()
            if text_scene != plot_scene:
                if text_scene is not None:
                    text_scene.removeItem(text_item)
                pw.plot_item.addItem(text_item, ignoreBounds=True)

            pw.multi_cursor_items.append(text_item)

    def _show_x_position_only(self, x_positions=None):
        """仅显示 x 位置标签（隐藏光标数值），同时在图上绘制 x_label 元素"""
        try:
            if not self._has_visible_curve_data():
                self._clear_cursor_items()
                self.pw.update_right_header("")
                return
            x_positions = (
                x_positions
                if x_positions is not None
                else self._get_cursor_x_positions()
            )
            if not x_positions:
                self.pw.update_right_header("")
                return

            (x_min, x_max), (y_min, y_max) = self.pw.view_box.viewRange()
            self._clear_cursor_items()

            import pyqtgraph as pg

            for idx, x in enumerate(x_positions):
                if x < x_min or x > x_max:
                    continue
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
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
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
                parts.append(f"x={x_str}")
            header_text = " | ".join(parts)
            self.pw.update_right_header(header_text)

        except Exception:
            pass

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

        self.pw.vline.setMovable(True)
        if hasattr(self.pw, "vline2"):
            self.pw.vline2.setMovable(False)

        if hasattr(self.pw.view_box, "is_cursor_pinned"):
            self.pw.view_box.is_cursor_pinned = True

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

    def _update_view_range_from_data(self):
        """从数据更新视图范围"""
        x_min, x_max = self.pw.get_curve_x_limits()
        if x_min is not None and x_max is not None:
            self.pw.view_box.setXRange(x_min, x_max, padding=0.02)

    def _update_cursor_after_plot(self, min_x_bound: float, max_x_bound: float):
        """绘图后更新光标边界和可见性"""
        pw = self.pw
        main_window = pw.window()
        if main_window and hasattr(main_window, 'cursor_btn'):
            # 设置 cursor 的移动边界
            pw._set_vline_bounds([min_x_bound, max_x_bound])
            cursor_enabled = main_window.cursor_btn.isChecked()
            cursor_values_hidden = getattr(main_window, 'cursor_values_hidden', False)

            # 根据全局 cursor 状态决定显示模式
            if cursor_enabled and cursor_values_hidden:
                # cursor 启用但只显示 vline 和 x 值
                pw.toggle_cursor(False, hide_values_only=True)
            else:
                # cursor 完全启用或禁用
                pw.toggle_cursor(cursor_enabled)
        else:
            # 无主窗口或 cursor 按钮，禁用 cursor
            pw._set_vline_bounds([None, None])
            pw.toggle_cursor(False)

    def on_vline_position_changed(self, line_obj=None):
        """vline 位置变化时更新光标状态"""
        if self._is_cursor_update_locked():
            return
        if self.pw.plot_context and getattr(
            self.pw.plot_context, "_is_time_correction_active", False
        ):
            return

        line = line_obj if line_obj is not None else self.pw.vline
        cursor_index = getattr(line, "cursor_index", 0)

        if self.is_cursor_pinned:
            if getattr(self.pw, "_suppress_pin_update", False):
                return
            x_pos = line.value()
            if len(self.pinned_x_values) <= cursor_index:
                self.pinned_x_values += [x_pos] * (
                    cursor_index + 1 - len(self.pinned_x_values)
                )
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
                    widget = container.plot_widget
                    if widget.is_cursor_pinned and widget != self.pw:
                        target_line = (
                            widget.vline
                            if cursor_index == 0
                            else getattr(widget, "vline2", None)
                        )
                        if target_line is not None:
                            from PySide6.QtCore import QSignalBlocker

                            with QSignalBlocker(target_line):
                                target_line.setPos(x_pos)
                        if len(widget.pinned_x_values) <= cursor_index:
                            widget.pinned_x_values += [x_pos] * (
                                cursor_index + 1 - len(widget.pinned_x_values)
                            )
                        widget.pinned_x_values[cursor_index] = x_pos
                        if cursor_index == 0:
                            widget.pinned_x_value = x_pos
                            if widget.factor != 0:
                                widget.pinned_index_value = (
                                    x_pos - widget.offset
                                ) / widget.factor
                            else:
                                widget.pinned_index_value = None
                        widget.update_cursor_label()

            self.update_cursor_label()
        else:
            if self.show_values_only:
                self._show_x_position_only()
            else:
                self.update_cursor_label()

    def _update_vline_bounds_from_data(self):
        """根据当前绘制的数据更新vline bounds"""
        pw = self.pw
        try:
            if hasattr(pw, 'original_index_x') and pw.original_index_x is not None and len(pw.original_index_x) > 0:
                min_index = np.min(pw.original_index_x)
                max_index = np.max(pw.original_index_x)
                min_x = pw.offset + pw.factor * min_index
                max_x = pw.offset + pw.factor * max_index
                pw._set_vline_bounds([min_x, max_x])
                return min_x, max_x

            if pw.is_multi_curve_mode and pw.curves:
                for ci in pw.curves.values():
                    if ci.y_data is not None:
                        datalength = len(ci.y_data)
                        if datalength > 0:
                            min_x = pw.offset + pw.factor * 1
                            max_x = pw.offset + pw.factor * datalength
                            pw._set_vline_bounds([min_x, max_x])
                            return min_x, max_x
                        break

            if pw.is_multi_curve_mode and pw.curves:
                x_arrays = pw._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    combined = np.concatenate(x_arrays)
                    min_x, max_x = np.nanmin(combined), np.nanmax(combined)
                    pw._set_vline_bounds([min_x, max_x])
                    return min_x, max_x

            if pw.curve is not None:
                x_data, _ = pw.curve.getData()
                if x_data is not None and len(x_data) > 0:
                    min_x, max_x = np.min(x_data), np.max(x_data)
                    pw._set_vline_bounds([min_x, max_x])
                    return min_x, max_x

            if hasattr(pw, 'xMin') and hasattr(pw, 'xMax'):
                pw._set_vline_bounds([pw.xMin, pw.xMax])
                return pw.xMin, pw.xMax
            else:
                pw._set_vline_bounds([None, None])
                return None, None
        except Exception as e:
            print(f"Error updating vline bounds: {e}")
            pw._set_vline_bounds([None, None])
            return None, None

    # ── 格式化工助方法 ──

    def sInt_to_fmtStr(self, value: int) -> str:
        """将秒数转换为时间字符串 HH:MM:SS.SS"""
        total = value % (24 * 3600)
        hh = int(total // 3600)
        mm = int((total % 3600) // 60)
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:05.2f}"

    def dateInt_to_fmtStr(self, value: int) -> str:
        """将时间戳转换为日期字符串"""
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(value)
            return dt.strftime('%Y/%m/%d')
        except Exception:
            return str(value)

    def _significant_decimal_format_str(self, value: float, ref: float, max_dp: int | None = None) -> str:
        """根据 ref 的显示精度自动决定 value 的字符串格式"""
        s = format(ref, 'f').rstrip('0').rstrip('.')
        if '.' not in s:
            dp = 0
        else:
            dp = len(s.split('.')[1])

        if max_dp is not None and max_dp >= 0:
            dp = min(max_dp, dp)

        if dp == 0:
            return str(int(round(value)))

        fmt = f'{{:.{dp}f}}'
        return fmt.format(value).rstrip('0').rstrip('.')
