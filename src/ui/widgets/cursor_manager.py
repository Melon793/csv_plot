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
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QFontMetrics
from PySide6.QtCore import QPointF, QSignalBlocker, Qt

from src.core.logger import get_logger

logger = get_logger("widget.cursor")

if TYPE_CHECKING:
    from src.ui.widgets.multi_curve_manager import MultiCurveManager


@dataclass
class LabelLayout:
    """标签布局结果数据结构（Phase 2 输出，纯数据，不操作 UI）"""
    layout_x: float   # 标签的 scene x 坐标
    layout_y: float   # 标签的 scene y 坐标
    index: int        # cursor_values 中的原始索引
    x_pos: float      # 数据交点 x 坐标
    y_pos: float      # 数据交点 y 坐标
    y_value: str      # 标签文本
    color: str        # 标签颜色


class CursorManager:
    """负责光标位置、标签、模式、对象池管理和 ViewBox 信号处理"""

    # 标签间距参数（像素），新旧算法共用
    LABEL_GAP_TO_CURSOR = 10       # 标签左边缘到光标的水平间距
    LABEL_COL_GAP = 15            # 多列时列与列之间的间距
    LABEL_VERTICAL_GAP = 4        # 同一列内标签之间的垂直间距
    LABEL_Y_OFFSET = 10           # 标签底部到交点的垂直距离（向上偏移）

    def __init__(self, multi_curve_manager: MultiCurveManager):
        """初始化光标管理器，绑定到 MultiCurveManager 以获取依赖链"""
        if multi_curve_manager is None:
            raise ValueError(
                "CursorManager requires a valid MultiCurveManager instance"
            )
        self._data_manager = multi_curve_manager

        # 新标签布局算法的状态
        self._use_new_label_layout = True   # 使用新标签布局算法（Phase 1/2/3）
        self._prev_layout: list[LabelLayout] = []  # 上一帧的布局结果
        self._column_count: int = 1          # 当前列数
        self._column_hysteresis_counter: int = 0  # 列数切换滞回计数器
        self._prev_cursor_scene_pos: QPointF | None = None  # 上一帧光标 scene 位置

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
        if self._is_cursor_update_locked():
            return

        if self._cursor_label_busy:
            return

        self._cursor_label_busy = True
        try:
            self._update_multi_curve_cursor_label()
        except (RuntimeError, AttributeError):
            pass
        finally:
            self._cursor_label_busy = False

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

        # 安全网：vline bounds 为 [None, None] 时禁止 cursor 更新，
        # 防止 vline 处于无效状态（如 reload 中间态）时触发 cursor 回调导致 SIGSEGV
        if hasattr(self.pw, "vline"):
            try:
                bounds = self.pw.vline.bounds
                if bounds == [None, None] or bounds == (None, None):
                    return True
            except (RuntimeError, AttributeError):
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

        while len(pool) <= index:
            circle = pg.ScatterPlotItem(symbol="o", size=8, brush=None)
            pool.append(circle)
        return pool[index]

    def _get_label_from_pool(self, index: int):
        """从对象池获取 TextItem"""
        pool = self.pw._cursor_item_pool["labels"]
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication

        while len(pool) <= index:
            label = pg.TextItem(
                color=(0, 0, 0), fill=pg.mkBrush(255, 255, 255, 220), anchor=(0, 0.5)
            )
            font = QApplication.font()
            font.setPixelSize(11)
            label.setFont(font)
            pool.append(label)
        return pool[index]

    def _get_x_label_from_pool(self, index: int):
        """获取 X 轴标签 TextItem"""
        pool = self.pw._cursor_item_pool["x_labels"]
        if index < len(pool):
            return pool[index]

        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication

        while len(pool) <= index:
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
        return pool[index]

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

        # 预构建 O(1) 查找集合，避免 list 的 O(n) 线性搜索
        x_labels_set = set(self.pw._cursor_item_pool.get("x_labels", []))
        labels_set = set(self.pw._cursor_item_pool.get("labels", []))

        for item in self.pw.multi_cursor_items:
            try:
                item_type = type(item).__name__
                if item_type == "ScatterPlotItem":
                    try:
                        item.clear()
                    except (RuntimeError, AttributeError):
                        pass
                elif item in x_labels_set:
                    try:
                        item.setText("")
                    except (RuntimeError, AttributeError):
                        pass
                elif item in labels_set:
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
            # 先将上一轮 trash 清空（上一轮的 item 已安全，可以释放）
            if hasattr(self.pw, '_cursor_trash_bin'):
                self.pw._cursor_trash_bin.clear()
            else:
                self.pw._cursor_trash_bin = []

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
                # 移入 trash bin 保持引用，延迟到下一次 _clear_cursor_items 时释放。
                # scene.removeItem 已从场景移除，paint event 不会访问该 item。
                # trash bin 在下一轮清理时清空，确保至少经历一个完整事件循环。
                self.pw._cursor_trash_bin.append(item)


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
            pw = self.pw
            view_box = pw.view_box
            plot_scene = pw.plot_item.scene()

            cursor_values = []
            (x_min, x_max), (y_min, y_max) = view_box.viewRange()

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

            for cursor_id, x in enumerate(x_positions):
                # anchored cursor 的 x 位置是用户固定的，不应因 view range
                # 变化而被过滤（例如添加 item 后触发的 auto-range 会改变 view）。
                # 数据范围的有效性已由下方的 x_data.min()/max() 检查保证。
                if mode not in ("1 anchored cursor", "2 anchored cursor"):
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
                            "cursor_id": cursor_id,  # 标识属于哪个 cursor
                        }
                    )

                    circle = self._get_circle_from_pool(len(cursor_values) - 1)
                    # 先确保 circle 在正确的 scene 里，再设置属性
                    # （Qt 的 prepareGeometryChange 需要 item 在 scene 中才能正确通知 view）
                    circle_scene = circle.scene()
                    if circle_scene != plot_scene:
                        if circle_scene is not None:
                            circle_scene.removeItem(circle)
                        pw.plot_item.addItem(circle, ignoreBounds=True)
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
                    pw.multi_cursor_items.append(circle)

            self._position_labels_avoid_overlap(
                cursor_values, x_min, x_max, y_min, y_max
            )

            for idx, x in enumerate(x_positions):
                # anchored cursor 的 x 位置是用户固定的，不因为 view range 变化而被过滤
                if mode not in ("1 anchored cursor", "2 anchored cursor"):
                    if x < x_min or x > x_max:
                        continue
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
                x_info_item = self._get_x_label_from_pool(idx)
                # 先加入 scene，再设置属性（Qt 规范顺序）
                x_scene = x_info_item.scene()
                if x_scene != plot_scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    plot_scene.addItem(x_info_item)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)
                view_rect = pw.plot_item.vb.sceneBoundingRect()
                scene_point = pw.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)
                pw.multi_cursor_items.append(x_info_item)

            # 所有 cursor 可视化元素（labels + circles + x-labels）设置完毕后，
            # 强制触发 ViewBox 重绘。Qt GraphicsView 框架在某些情况下不会
            # 自动调度 paint event（尤其在 reload/批量创建 item 后），
            # 导致部分 TextItem / ScatterPlotItem 不可见，
            # 直到用户交互（拖动 cursor / 缩放）才恢复。

            view_box.update()

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
        """标签定位算法入口，通过 feature flag 分发到新旧算法"""
        if getattr(self, '_use_new_label_layout', False):
            self._position_labels_new(cursor_values, x_min, x_max, y_min, y_max)
        else:
            self._position_labels_legacy(cursor_values, x_min, x_max, y_min, y_max)

    def _position_labels_legacy(
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
        gap_pixels = self.LABEL_GAP_TO_CURSOR  # 文本框左边缘距离cursor的水平像素间隔
        vertical_gap_pixels = self.LABEL_Y_OFFSET  # 垂直像素间隔（标签底部到交点的距离）

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

            # 计算文本框偏移（TextItem的anchor=(0, 0.5)，位置为左边缘）
            offset_x_right = gap_pixels
            offset_x_left = -(gap_pixels + label_width_pixels)
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

                # 检查是否在数据范围内（anchor=(0,0.5)，candidate_x 是左边缘）
                left_ok = candidate_x >= x_min
                right_ok = candidate_x + label_width_data <= x_max
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

                label_x = max(x_min, min(x_max - label_width_data, label_x))
                label_y = max(y_min + label_height_data * 0.5,
                             min(y_max - label_height_data * 0.5, label_y))

            # 边缘避让逻辑：防止标签在边缘抖动
            edge_margin_strict = label_height_data * 0.1
            y_center = (y_min + y_max) / 2
            # y_quarter_upper = y_min + (y_max - y_min) * 0.25
            # y_quarter_lower = y_max - (y_max - y_min) * 0.25

            data_point_near_bottom = (y_pos - y_min) < edge_margin_strict
            data_point_near_top = (y_max - y_pos) < edge_margin_strict

            if data_point_near_bottom:
                # label_y = max(y_quarter_upper, label_y)
                label_y = min(label_y, y_center)
            elif data_point_near_top:
                # label_y = min(y_quarter_lower, label_y)
                label_y = max(label_y, y_center)
            else:
                edge_margin_soft = label_height_data * 0.25
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

    # ========================================================================
    # 新标签布局算法 (Phase 1 → Phase 2 → Phase 3)
    # ========================================================================

    def _position_labels_new(
        self,
        cursor_values: list,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        """新标签定位算法：全局排布 + 多列自适应

        Phase 1: 数据过滤 → Phase 2: 布局计算 → Phase 3: 渲染

        支持多 cursor 模式：按 cursor_id 分组，每个 cursor 的标签独立布局。
        """
        if not cursor_values:
            return

        pw = self.pw
        view_box = pw.plot_item.getViewBox()
        view_width_pixels = max(1, view_box.width())
        view_height_pixels = max(1, view_box.height())

        x_range = x_max - x_min
        y_range = y_max - y_min
        pixel_to_data_x = x_range / view_width_pixels
        pixel_to_data_y = y_range / view_height_pixels

        # 缓存 font metrics
        if not hasattr(self, '_cached_font_metrics'):
            sample_text_item = self._get_label_from_pool(0)
            text_font = sample_text_item.textItem.font()
            self._cached_font_metrics = QFontMetrics(text_font)
            self._cached_label_height_pixels = self._cached_font_metrics.height() + 6

        font_metrics = self._cached_font_metrics
        label_height_pixels = self._cached_label_height_pixels
        label_height_data = label_height_pixels * pixel_to_data_y

        # 按 cursor_id 分组（保留原始索引）
        cursor_groups = {}
        for orig_idx, item in enumerate(cursor_values):
            cid = item.get('cursor_id', 0)
            if cid not in cursor_groups:
                cursor_groups[cid] = []
            cursor_groups[cid].append({**item, '_orig_idx': orig_idx})

        # 对每个 cursor 独立处理
        all_layouts = []
        all_filtered_out_indices = set()
        for cid, group_values in sorted(cursor_groups.items()):
            # --- Phase 1: 数据过滤 ---
            visible_labels, filtered_out_indices = self._filter_labels_phase1(
                group_values, x_min, x_max, y_min, y_max,
                label_height_data, font_metrics,
            )
            all_filtered_out_indices.update(filtered_out_indices)
            if not visible_labels:
                continue

            # --- Phase 2: 布局计算 ---
            # 获取该 cursor 的 scene 位置
            cursor_scene_pos = view_box.mapViewToScene(
                QPointF(group_values[0]['x_pos'], group_values[0]['y_pos'])
            )

            layouts, layout_hide_indices = self._compute_label_layout(
                visible_labels, cursor_scene_pos, view_box,
                x_min, x_max, y_min, y_max,
                label_height_pixels, font_metrics,
            )
            all_layouts.extend(layouts)
            all_filtered_out_indices.update(layout_hide_indices)

        # --- Phase 3: 渲染 ---
        self._render_label_layout(all_layouts)

        # --- Phase 4: 统一隐藏未使用的标签 ---
        # 在所有 cursor group 处理完毕后，隐藏被过滤掉的标签。
        # 排除已被本次渲染使用的 pool index，避免误隐藏其他 cursor 的标签。
        used_indices = {layout.index for layout in all_layouts}
        indices_to_hide = all_filtered_out_indices - used_indices

        pool_labels = self.pw._cursor_item_pool.get("labels", [])
        for idx in indices_to_hide:
            if idx >= len(pool_labels):
                continue
            text_item = self._get_label_from_pool(idx)
            text_item.setVisible(False)

    def _filter_labels_phase1(
        self,
        cursor_values: list,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        label_height_data: float,
        font_metrics,
    ) -> tuple:
        """Phase 1: 在上游过滤基础上增加边界余量二次过滤

        返回 (visible_labels, filtered_out_indices)。
        visible_labels: 按 y_pos 降序排列的可见标签列表（高 y 值在上，保留原始索引）。
        filtered_out_indices: 被过滤掉的标签 pool index 集合（延迟到所有 cursor group
        处理完毕后统一隐藏，避免多 cursor 共享 pool 时相互干扰）。
        注意：cursor_values 中的 item 应已包含 '_orig_idx' 字段。
        """
        margin = label_height_data * 0.5

        visible = [
            v for v in cursor_values
            if y_min + margin < v['y_pos'] < y_max - margin
            and x_min < v['x_pos'] < x_max
        ]

        # 收集被过滤掉的 pool index（不立即隐藏，由调用方统一处理）
        visible_indices = {item['_orig_idx'] for item in visible}
        filtered_out_indices = {
            item['_orig_idx']
            for item in cursor_values
            if item['_orig_idx'] not in visible_indices
        }

        # 按 y 降序排列（高 y 值在上），与 scene 坐标系方向一致
        visible.sort(key=lambda v: v['y_pos'], reverse=True)
        return visible, filtered_out_indices

    def _compute_label_layout(
        self,
        visible_labels: list,
        cursor_scene_pos,
        view_box,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        label_height_pixels: float,
        font_metrics,
    ) -> tuple:
        """Phase 2: 计算所有标签的布局位置（纯数学，不操作 UI）

        返回 (layouts, hide_indices)。
        hide_indices: 因窗口极小无法放置而需要隐藏的标签 pool index 集合。
        """
        if not visible_labels:
            return [], set()

        # 计算标签宽度
        max_label_width_pixels = 200  # 最大标签宽度（像素）
        label_widths_pixels = []
        for item in visible_labels:
            text_width = font_metrics.horizontalAdvance(item['y_value'])
            w = min(text_width + 12, max_label_width_pixels)
            label_widths_pixels.append(w)

        max_label_width_px = max(label_widths_pixels)

        # 获取 y 轴占用宽度
        y_axis_width = 0
        left_axis = self.pw.plot_item.getAxis('left')
        if left_axis:
            y_axis_width = left_axis.width()

        view_width_pixels = max(1, view_box.width())
        view_height_pixels = max(1, view_box.height())

        # Scene 坐标边界
        x_min_scene = view_box.mapViewToScene(QPointF(x_min, 0)).x() + y_axis_width
        x_max_scene = view_box.mapViewToScene(QPointF(x_max, 0)).x()
        y_min_scene = view_box.mapViewToScene(QPointF(0, y_min)).y()
        y_max_scene = view_box.mapViewToScene(QPointF(0, y_max)).y()

        # 注意：scene y 坐标通常与数据 y 坐标方向相反（scene 中上方为负或小值）
        # 确保 y_min_scene < y_max_scene（从上到下）
        if y_min_scene > y_max_scene:
            y_min_scene, y_max_scene = y_max_scene, y_min_scene
        scene_height = y_max_scene - y_min_scene

        # 间距参数（像素），统一由类常量管理
        gap_pixels = self.LABEL_GAP_TO_CURSOR       # 标签到光标水平间距
        col_gap_pixels = self.LABEL_COL_GAP         # 列间距
        vertical_gap_pixels = self.LABEL_VERTICAL_GAP  # 列内标签垂直间距

        gap_to_cursor = gap_pixels
        label_height_total = label_height_pixels + vertical_gap_pixels

        # 计算列容量
        margin_pixels_scene = label_height_pixels * 0.1
        available_height = scene_height - 2 * margin_pixels_scene
        per_column_capacity = max(1, int(available_height / label_height_total))

        # 列数计算（带滞回）
        n_labels = len(visible_labels)
        new_column_count = max(1, int(np.ceil(n_labels / per_column_capacity)))
        column_count = self._apply_column_hysteresis(new_column_count)

        # 总标签高度超 plot 高度时的极限保护
        if per_column_capacity < 1:
            # 窗口极小，返回所有标签的 pool index 供调用方统一隐藏
            hide_indices = {item['_orig_idx'] for item in visible_labels}
            return [], hide_indices

        # 计算列放置方向
        cursor_scene_x = cursor_scene_pos.x() if cursor_scene_pos else 0
        right_available = x_max_scene - cursor_scene_x - gap_to_cursor
        left_available = cursor_scene_x - gap_to_cursor - x_min_scene

        # 分配列到两侧
        column_sides = self._compute_column_positions(
            column_count, right_available, left_available,
            max_label_width_px, col_gap_pixels,
        )
        # column_sides: list of ('right', col_idx) or ('left', col_idx)

        # 分配标签到各列（均匀分配）
        labels_per_column = self._distribute_labels(visible_labels, column_count)

        # 计算每列的场景 x 坐标（anchor=(0,0.5)，cx 即标签左边缘）
        column_scene_x = []
        for side, col_idx in column_sides:
            offset = gap_to_cursor + col_idx * (max_label_width_px + col_gap_pixels)
            if side == 'right':
                cx = cursor_scene_x + offset
            else:
                cx = cursor_scene_x - offset - max_label_width_px
            column_scene_x.append(cx)

        # 列内垂直堆叠
        column_scene_y_min = y_min_scene + margin_pixels_scene
        column_scene_y_max = y_max_scene - margin_pixels_scene

        all_layouts: list[LabelLayout] = []

        for col_idx, col_labels in enumerate(labels_per_column):
            if not col_labels:
                continue

            col_layouts = self._layout_column(
                col_labels, column_scene_x[col_idx],
                column_scene_y_min, column_scene_y_max,
                label_height_pixels, vertical_gap_pixels,
            )
            all_layouts.extend(col_layouts)

        return all_layouts, set()

    def _apply_column_hysteresis(self, new_column_count: int) -> int:
        """列数切换滞回：连续 3 次调用才切换，防止临界抖动"""
        HYSTERESIS_THRESHOLD = 3
        if new_column_count != self._column_count:
            self._column_hysteresis_counter += 1
            if self._column_hysteresis_counter >= HYSTERESIS_THRESHOLD:
                self._column_count = new_column_count
                self._column_hysteresis_counter = 0
        else:
            self._column_hysteresis_counter = 0
        return self._column_count

    def _compute_column_positions(
        self,
        column_count: int,
        right_available: float,
        left_available: float,
        max_label_width: float,
        col_gap: float,
    ) -> list:
        """决定每列的放置方向（右侧优先，空间不足时启用左侧）

        返回 list of (side, col_idx)
        """
        sides = []
        col_width = max_label_width + col_gap
        right_cols_possible = max(0, int(right_available / col_width)) if right_available > col_width else 0
        left_cols_possible = max(0, int(left_available / col_width)) if left_available > col_width else 0

        # 优先填充右侧
        right_used = min(column_count, right_cols_possible)
        for i in range(right_used):
            sides.append(('right', i))

        # 剩余列放左侧
        remaining = column_count - right_used
        left_used = min(remaining, left_cols_possible)
        for i in range(left_used):
            sides.append(('left', i))

        # 如果列数超出可用空间，全部压缩到右侧
        if len(sides) < column_count:
            for i in range(len(sides), column_count):
                sides.append(('right', i - right_used))

        return sides

    def _distribute_labels(
        self, visible_labels: list, column_count: int
    ) -> list[list]:
        """将标签均匀分配到各列"""
        n = len(visible_labels)
        if n == 0:
            return [[] for _ in range(column_count)]

        base = n // column_count
        remainder = n % column_count

        result = []
        start = 0
        for col in range(column_count):
            size = base + (1 if col < remainder else 0)
            result.append(visible_labels[start:start + size])
            start += size
        return result

    def _layout_column(
        self,
        labels: list,
        col_scene_x: float,
        col_y_min: float,
        col_y_max: float,
        label_height_pixels: float,
        gap_pixels: float,
    ) -> list[LabelLayout]:
        """列内垂直堆叠：自适应权重的约束垂直堆叠"""
        n = len(labels)
        if n == 0:
            return []

        half_height = label_height_pixels / 2
        view_box = self.pw.plot_item.getViewBox()

        # 单标签：放在交点上方（统一向上偏移 LABEL_Y_OFFSET）
        if n == 1:
            label = labels[0]
            scene_pos = view_box.mapViewToScene(QPointF(label['x_pos'], label['y_pos']))
            target_y = scene_pos.y() - self.LABEL_Y_OFFSET  # 向上偏移
            target_y = max(col_y_min + half_height,
                           min(col_y_max - half_height, target_y))
            return [LabelLayout(
                layout_x=col_scene_x, layout_y=target_y,
                index=label['_orig_idx'],
                x_pos=label['x_pos'], y_pos=label['y_pos'],
                y_value=label['y_value'], color=label['color'],
            )]

        # 多标签：自适应权重堆叠
        ideal_spacing = (col_y_max - col_y_min) / n
        total_height = label_height_pixels + gap_pixels
        results: list[LabelLayout] = []

        # 虚拟"上一个标签底部"：从列顶部开始，保证第一个标签也有最小位置约束
        prev_bottom = col_y_min + half_height

        for i, label in enumerate(labels):
            scene_pos = view_box.mapViewToScene(QPointF(label['x_pos'], label['y_pos']))
            # 所有标签统一向上偏移 LABEL_Y_OFFSET，保持间距一致
            ideal_y = scene_pos.y() - self.LABEL_Y_OFFSET
            uniform_y = col_y_min + ideal_spacing * (i + 0.5)  # 均匀分布

            # 自适应权重：密集区域偏向均匀分布，稀疏区域偏向交点位置
            local_density = self._count_nearby(labels, label['y_pos'],
                                                radius=label_height_pixels * 3)
            w_ideal = max(0.2, 1.0 - local_density * 0.15)
            target_y = w_ideal * ideal_y + (1 - w_ideal) * uniform_y

            # 动态最大偏移量
            max_offset = label_height_pixels * (0.5 + n * 0.1)
            max_offset = min(max_offset, label_height_pixels * 2)
            target_y = max(ideal_y - max_offset, min(ideal_y + max_offset, target_y))

            # 与上一个标签防重叠（包括第一个标签与列顶部的约束）
            target_y = max(target_y, prev_bottom + gap_pixels + half_height)

            # 边界 clamp
            target_y = max(col_y_min + half_height,
                           min(col_y_max - half_height, target_y))

            results.append(LabelLayout(
                layout_x=col_scene_x, layout_y=target_y,
                index=label['_orig_idx'],
                x_pos=label['x_pos'], y_pos=label['y_pos'],
                y_value=label['y_value'], color=label['color'],
            ))

            # 更新 prev_bottom 供下一个标签使用
            prev_bottom = target_y + half_height

        # Pass 2: 正向扫描，修正边界 clamp 导致的重叠（自顶向下）
        for i in range(1, len(results)):
            prev_bottom = results[i - 1].layout_y + half_height
            curr_top = results[i].layout_y - half_height
            if curr_top < prev_bottom + gap_pixels:
                results[i].layout_y = prev_bottom + gap_pixels + half_height
                # 检查是否超出底部边界
                results[i].layout_y = min(results[i].layout_y,
                                           col_y_max - half_height)

        # Pass 3: 反向扫描，修正被推挤后产生的新重叠（自底向上）
        for i in range(len(results) - 2, -1, -1):
            next_top = results[i + 1].layout_y - half_height
            curr_bottom = results[i].layout_y + half_height
            if curr_bottom > next_top - gap_pixels:
                results[i].layout_y = next_top - gap_pixels - half_height
                # 检查是否超出顶部边界，超出则截断
                results[i].layout_y = max(results[i].layout_y,
                                           col_y_min + half_height)

        return results

    def _count_nearby(self, labels: list, y_pos: float, radius: float) -> int:
        """计算 y_pos 附近 radius 范围内的标签数量"""
        return sum(1 for l in labels if abs(l['y_pos'] - y_pos) <= radius)

    def _render_label_layout(self, layouts: list[LabelLayout]):
        """Phase 3: 按计算结果渲染标签"""
        pw = self.pw
        plot_scene = pw.plot_item.scene()
        view_box = pw.plot_item.getViewBox()

        MAX_LABEL_WIDTH_PIXELS = 200  # 最大标签宽度
        if not hasattr(self, '_cached_font_metrics'):
            sample_text_item = self._get_label_from_pool(0)
            text_font = sample_text_item.textItem.font()
            self._cached_font_metrics = QFontMetrics(text_font)

        font_metrics = self._cached_font_metrics

        for layout in layouts:
            text_item = self._get_label_from_pool(layout.index)
            # 先加入 scene，再设置属性（Qt 规范：prepareGeometryChange 需要 item 在 scene 中）
            text_scene = text_item.scene()
            if text_scene != plot_scene:
                if text_scene is not None:
                    text_scene.removeItem(text_item)
                pw.plot_item.addItem(text_item, ignoreBounds=True)

            # 标签文本截断（防止过长文本导致列宽计算失准）
            display_text = layout.y_value
            text_width = font_metrics.horizontalAdvance(display_text) + 12
            if text_width > MAX_LABEL_WIDTH_PIXELS:
                display_text = font_metrics.elidedText(
                    display_text, Qt.ElideRight, MAX_LABEL_WIDTH_PIXELS - 12
                )
                text_width = MAX_LABEL_WIDTH_PIXELS

            text_item.setText(display_text)

            # 边框颜色
            if (not hasattr(text_item, '_cached_border_color')
                    or text_item._cached_border_color != layout.color):
                border_pen = pg.mkPen(layout.color, width=1.5)
                text_item.border = border_pen
                text_item._cached_border_color = layout.color

            text_item.setVisible(True)

            # 将 scene 坐标转回数据坐标（TextItem 添加到 plot_item 时使用数据坐标）
            data_pos = view_box.mapSceneToView(
                QPointF(layout.layout_x, layout.layout_y)
            )
            text_item.setPos(data_pos.x(), data_pos.y())
            text_item.setZValue(201)
            pw.multi_cursor_items.append(text_item)

    # ========================================================================

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

            pw = self.pw
            view_box = pw.view_box
            plot_scene = pw.plot_item.scene()

            (x_min, x_max), (y_min, y_max) = view_box.viewRange()
            self._clear_cursor_items()

            import pyqtgraph as pg

            show_x_mode = self._get_cursor_mode()
            for idx, x in enumerate(x_positions):
                # anchored cursor 的 x 位置是用户固定的，不因为 view range 变化而被过滤
                if show_x_mode not in ("1 anchored cursor", "2 anchored cursor"):
                    if x < x_min or x > x_max:
                        continue
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
                x_info_item = self._get_x_label_from_pool(idx)
                # 先加入 scene，再设置属性（Qt 规范顺序）
                x_scene = x_info_item.scene()
                if x_scene != plot_scene:
                    if x_scene is not None:
                        x_scene.removeItem(x_info_item)
                    plot_scene.addItem(x_info_item)
                x_info_item.setText(x_str)
                x_info_item.setVisible(True)

                view_rect = pw.plot_item.vb.sceneBoundingRect()
                scene_point = pw.plot_item.vb.mapViewToScene(pg.Point(x, y_min))
                scene_x = scene_point.x()
                scene_y = view_rect.bottom()
                x_info_item.setPos(scene_x, scene_y)
                x_info_item.setZValue(100000)

                pw.multi_cursor_items.append(x_info_item)

            parts = []
            for x in x_positions:
                x_str = self._significant_decimal_format_str(
                    value=float(x), ref=self.factor
                )
                parts.append(f"x={x_str}")
            header_text = " | ".join(parts)
            self.pw.update_right_header(header_text)

            # 所有 cursor 可视化元素设置完毕后，
            # 强制触发 ViewBox 重绘，避免部分 item 不可见
            view_box.update()

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

        # 使用 QSignalBlocker 阻断 setMovable(False) 期间的 sigPositionChanged 信号，
        # 避免 handle items 销毁时意外触发 on_vline_position_changed 回调导致 SIGSEGV
        if hasattr(self.pw, "vline"):
            with QSignalBlocker(self.pw.vline):
                self.pw.vline.setMovable(False)
        if hasattr(self.pw, "vline2"):
            with QSignalBlocker(self.pw.vline2):
                self.pw.vline2.setMovable(False)
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
            logger.warning("Error updating vline bounds: %s", e)
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
        if ref == 0.0 or not np.isfinite(ref):
            dp = 2  # 零值/非有限值默认 2 位小数
        else:
            s = format(ref, 'f').rstrip('0').rstrip('.')
            dp = 0 if '.' not in s else len(s.split('.')[1])

        if max_dp is not None and max_dp >= 0:
            dp = min(max_dp, dp)

        if dp == 0:
            return str(int(round(value)))

        fmt = f'{{:.{dp}f}}'
        return fmt.format(value).rstrip('0').rstrip('.')
