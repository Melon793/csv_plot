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
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QBrush, QFontMetrics, QLinearGradient
from PySide6.QtCore import QPointF, QSignalBlocker, Qt
from PySide6.QtWidgets import QApplication

from src.core.config import is_grouped_loader, safe_qt_op
from src.core.logger import get_logger
from src.core.value_format import decimals_for_step, format_fixed_trim

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
    pinned: bool = False  # 交点 y 出界：标签吸附到视口边缘
    up: bool = False      # pinned 方向：交点在上方/下方视口外
    w: float = 0.0        # 标签渲染宽度（像素，含内边距与 ▲/▼ 前缀）
    first_col: bool = False  # 是否「该侧最内列」（左右各一列，用直线引导线）


class CursorManager:
    """负责光标位置、标签、模式、对象池管理和 ViewBox 信号处理"""

    # 标签间距参数（像素），新旧算法共用
    LABEL_GAP_TO_CURSOR = 10       # 标签左边缘到光标的水平间距
    LABEL_COL_GAP = 15            # 多列时列与列之间的间距
    LABEL_VERTICAL_GAP = 2        # 同一列内标签之间的垂直间距（可见间隙 = +2px 布局补偿）
    LABEL_BOTTOM_GAP = 6          # 标签底边到交点的净空（像素；视觉净空 = +1px 布局补偿）
    # 列预算余量（像素）：单侧可用宽 ≥ 列宽 + margin 才算一列。
    # max_cols 截断与方向分配共用同一口径，避免「截断后分配数不足 →
    # 外侧追加 → 出视口被裁」的消失路径。
    COLUMN_BUDGET_MARGIN = 2

    # 形态开关：改常量后需重新构建/分发，非运行期开关
    CURSOR_EDGE_PIN = True         # Y 出界交点的标签吸附到视口边缘（▲/▼ 虚线）
    CURSOR_LEADER_ELBOW = True     # 引导线改 elbow 正交折线
    CURSOR_CROSS_AVOID = True      # 双游标跨组碰撞时向外分裂

    def __init__(self, multi_curve_manager: MultiCurveManager):
        """初始化光标管理器，绑定到 MultiCurveManager 以获取依赖链"""
        if multi_curve_manager is None:
            raise ValueError(
                "CursorManager requires a valid MultiCurveManager instance"
            )
        self._data_manager = multi_curve_manager

        # 标签布局算法的状态
        # 列数滞回状态按 cursor_id 分键：(当前列数, 切换计数)。
        # 单游标固定 key=0，初值与旧单份状态一致（1 列、0 计数）。
        self._column_state: dict[int, tuple[int, int]] = {}
        self._prev_cursor_scene_pos: QPointF | None = None  # 上一帧光标 scene 位置

    @property
    def pw(self) -> Any:
        """关联的 DraggableGraphicsLayoutWidget 实例"""
        dm = self._data_manager
        if dm is None:
            raise RuntimeError(
                "CursorManager: dependency chain broken (_data_manager is None)"
            )
        return dm.pw

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
            safe_qt_op(self._update_multi_curve_cursor_label)
        finally:
            self._cursor_label_busy = False

    def _is_cursor_update_locked(self) -> bool:
        """判断 cursor 相关回调是否需要被暂时禁用"""
        lock_reason = None

        if getattr(self.pw, "_is_updating_data", False) or getattr(
            self.pw, "_is_being_destroyed", False
        ):
            lock_reason = (
                f"_is_updating_data={getattr(self.pw, '_is_updating_data', False)} "
                f"_is_being_destroyed={getattr(self.pw, '_is_being_destroyed', False)}"
            )
        elif self.pw.plot_context:
            if getattr(self.pw.plot_context, "_is_loading_new_data", False):
                lock_reason = "_is_loading_new_data=True"
            else:
                current_version = getattr(self.pw.plot_context, "_data_version", 0)
                my_version = getattr(self.pw, "_cached_data_version", 0)
                if my_version != 0 and my_version != current_version:
                    lock_reason = f"version mismatch my={my_version} current={current_version}"

        # 注意：不要用 vline.bounds 表达「reload 中间态 / 无数据」。bounds 的语义是
        # pyqtgraph 的「无界」（[None, None] = 不限），全局数据源不可用时它本身就是
        # [None, None]，拿它当状态哨兵会在未加载数据时锁死整条游标更新链。
        # （v0.3.15 及之前这里读 self.pw.vline.bounds 这个「方法」而非 bounds()，
        # 条件恒不成立，实际从未生效；删掉死代码而不是补括号。）
        # 中间态防护由上面的 _is_updating_data / _is_being_destroyed /
        # _is_loading_new_data / 数据版本号四道门承担。

        # 诊断日志：仅在锁定状态变化时记录，避免拖动期间日志洪泛
        if logger.isEnabledFor(logging.DEBUG):
            was_locked = getattr(self.pw, "_cursor_lock_logged", False)
            is_locked = lock_reason is not None
            if is_locked and not was_locked:
                logger.debug(
                    "[CURSOR_LOCK] LOCKED: %s plot=%s",
                    lock_reason, getattr(self.pw, "y_name", "?"),
                )
            elif not is_locked and was_locked:
                logger.debug(
                    "[CURSOR_LOCK] UNLOCKED: plot=%s",
                    getattr(self.pw, "y_name", "?"),
                )
            self.pw._cursor_lock_logged = is_locked

        return lock_reason is not None

    def _get_circle_from_pool(self, index: int):
        """从对象池获取 ScatterPlotItem"""
        pool = self.pw._cursor_item_pool["circles"]
        if index < len(pool):
            return pool[index]

        while len(pool) <= index:
            circle = pg.ScatterPlotItem(symbol="o", size=8, brush=None)
            pool.append(circle)
        return pool[index]

    def _get_label_from_pool(self, index: int):
        """从对象池获取 TextItem"""
        pool = self.pw._cursor_item_pool["labels"]
        if index < len(pool):
            return pool[index]

        while len(pool) <= index:
            label = pg.TextItem(
                color=(0, 0, 0), fill=pg.mkBrush(255, 255, 255, 255), anchor=(0, 0.5)
            )
            # 白底默认不透明（渲染时降为 90α 以透出曲线）；
            # 身份色由 _render_label_layout 设置的左侧全色条承担。
            # 边框在渲染时按曲线色设 1.5px：浅灰对白底对比仅 1.49 几乎不可见，
            # 曲线色 5.0+ 才能把文本框从白底里勾出来（main 原色）。
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

    def _get_leader_from_pool(self, index: int):
        """获取交点→标签引导线（PlotCurveItem，数据坐标，随视图缩放）"""
        pool = self.pw._cursor_item_pool["leaders"]
        if index < len(pool):
            return pool[index]

        while len(pool) <= index:
            leader = pg.PlotCurveItem(pen=pg.mkPen(120, 120, 120, 150, width=1))
            leader.setZValue(200.5)  # 圆点 200 < 引导线 < 标签 201
            pool.append(leader)
        return pool[index]

    def _clear_cursor_items(self, hide_only: bool = True):
        """清除或隐藏所有 cursor 可视化元素"""
        if not hasattr(self.pw, "multi_cursor_items") or not hasattr(
            self.pw, "plot_item"
        ):
            return

        for item in self.pw.multi_cursor_items:
            if item is not None:
                safe_qt_op(item.setVisible, False)

        # 预构建 O(1) 查找集合，避免 list 的 O(n) 线性搜索
        x_labels_set = set(self.pw._cursor_item_pool.get("x_labels", []))
        labels_set = set(self.pw._cursor_item_pool.get("labels", []))
        leaders_set = set(self.pw._cursor_item_pool.get("leaders", []))

        for item in self.pw.multi_cursor_items:
            try:
                item_type = type(item).__name__
                if item_type == "ScatterPlotItem":
                    safe_qt_op(item.clear)
                elif item in x_labels_set:
                    safe_qt_op(item.setText, "")
                elif item in labels_set:
                    safe_qt_op(item.setText, "")
                elif item in leaders_set:
                    pass  # 池化复用：顶部已 setVisible(False)，下一帧重设数据
                else:
                    self._queue_item_for_deletion(item)
            except Exception:
                logger.debug("清理 cursor item 异常", exc_info=True)

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
            old_leaders = list(self.pw._cursor_item_pool.get("leaders", []))

            self.pw._cursor_item_pool = {
                "circles": [], "labels": [], "x_labels": [], "leaders": [],
            }

            scene = None
            plot_item = None
            if hasattr(self.pw, 'plot_item') and self.pw.plot_item is not None:
                plot_item = self.pw.plot_item
                scene = plot_item.scene()

            for item in (old_circles + old_labels + old_x_labels + old_leaders):
                if item is None:
                    continue
                safe_qt_op(item.setVisible, False)
                try:
                    in_scene = item.scene() == scene and scene is not None
                except RuntimeError:
                    in_scene = False
                if in_scene:
                    # 【崩溃修复】优先用 PlotItem.removeItem 同时清理 scene 与
                    # PlotItem.items：仅用 scene.removeItem 会在 PlotItem.items
                    # 残留引用，导致后续 addItem 被 pyqtgraph 判重跳过（坐标失效）。
                    removed = False
                    if plot_item is not None:
                        try:
                            plot_item.removeItem(item)
                            removed = True
                        except Exception:
                            logger.debug(
                                "PlotItem.removeItem 失败，回退 scene.removeItem",
                                exc_info=True,
                            )
                    if not removed:
                        safe_qt_op(lambda it=item: scene.removeItem(it))
                # 移入 trash bin 保持引用，延迟到下一次 _clear_cursor_items 时释放。
                # scene.removeItem 已从场景移除，paint event 不会访问该 item。
                # trash bin 在下一轮清理时清空，确保至少经历一个完整事件循环。
                self.pw._cursor_trash_bin.append(item)


    def _queue_item_for_deletion(self, item):
        """将 item 加入待删除队列"""
        if item is not None and item not in self.pw._pending_delete_items:
            safe_qt_op(item.setVisible, False)
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
            if item is None:
                continue
            safe_qt_op(lambda: self._safe_delete_item(item))

    def _safe_delete_item(self, item):
        """安全地从场景移除并调度删除 item"""
        scene = safe_qt_op(item.scene)
        if scene is not None:
            safe_qt_op(scene.removeItem, item)
        if hasattr(item, "deleteLater"):
            safe_qt_op(item.deleteLater)

    def _update_multi_curve_cursor_label(self):
        """更新多曲线光标标签"""
        if self._is_interacting:
            return

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
            return

        if self.show_values_only:
            self._show_x_position_only()
            return

        if not self.pw.curves:
            return

        x_positions = self._get_cursor_x_positions()
        if not x_positions:
            return

        try:
            pw = self.pw
            view_box = pw.view_box
            plot_scene = pw.plot_item.scene()

            cursor_values = []
            circle_index = 0  # 圆点池独立计数器（pinned 项不建圆点，不占索引）
            (x_min, x_max), (y_min, y_max) = view_box.viewRange()

            curves_to_process = []
            # 统一路径：始终从 curves 字典获取
            for var_name, ci in self.pw.curves.items():
                if not ci.visible:
                    continue
                curves_to_process.append(
                    {
                        "var_name": var_name,
                        "x_data": ci.x_data,
                        "y_data": ci.y_data,
                        "x_min": ci.x_min,
                        "x_max": ci.x_max,
                        "color": ci.color,
                        "y_format": ci.y_format,
                        "unit": self.pw.units.get(var_name, ""),
                        "enum_map": getattr(
                            self.pw.plot_context, "_enum_text_maps", {}
                        ).get(var_name, {}),
                    }
                )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[CURSOR] update_cursor_label: 处理曲线数=%d, x_positions=%s",
                    len(curves_to_process),
                    [f"{x:.4f}" for x in x_positions[:3]],
                )

            for cursor_id, x in enumerate(x_positions):
                # anchored cursor 的 x 位置是用户固定的，不应因 view range
                # 变化而被过滤（例如添加 item 后触发的 auto-range 会改变 view）。
                # 数据范围的有效性已由下方的 CurveInfo.x_min/x_max 检查保证。
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
                    # CurveInfo 已缓存数据端点（x_data 变更处必调 update_x_range）：
                    # 这里再 min()/max() 就是每帧对每条曲线做两次全量扫描
                    if x < curve_data["x_min"] or x > curve_data["x_max"]:
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
                    # Y 出界：circle 仍不画（维持现状）；开启边缘吸附时标签
                    # 仍入列并标记 pinned，ideal 在布局阶段改写为视口边缘。
                    y_pinned = y_val < y_min or y_val > y_max
                    if y_pinned and not self.CURSOR_EDGE_PIN:
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
                            "pinned": y_pinned,      # 交点 y 出界（标签吸附边缘）
                            "up": y_val > y_max,     # pinned 方向：上方出界
                        }
                    )

                    if y_pinned:
                        continue  # 出界交点不建圆点，仅保留边缘标签

                    circle = self._get_circle_from_pool(circle_index)
                    circle_index += 1  # 独立计数器：pinned 项不占池索引
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
                x_str = format_fixed_trim(float(x), self._x_decimals())
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
            # 强制触发 ViewBox 重绘（v5.3：需阻断级联信号，见 _repaint_view_box）
            self._repaint_view_box(view_box)

        except Exception:
            logger.warning("更新多曲线光标标签失败", exc_info=True)

    def _position_labels_avoid_overlap(
        self,
        cursor_values: list,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        """标签定位算法入口（全局排布 + 多列自适应）"""
        self._position_labels_new(cursor_values, x_min, x_max, y_min, y_max)

    # ========================================================================
    # 标签布局算法 (Phase 1 → Phase 2 → Phase 3)
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

        # 缓存 font metrics
        if not hasattr(self, '_cached_font_metrics'):
            sample_text_item = self._get_label_from_pool(0)
            text_font = sample_text_item.textItem.font()
            self._cached_font_metrics = QFontMetrics(text_font)
            # 标签实际高度由 QTextDocument 默认 4px 边距决定，比 fm.height()
            # 多 8px；再留 2px 给描边外沿。旧公式 +6 少算 2px，使
            # 当时的 LABEL_VERTICAL_GAP=4 实际只剩 2px，相邻标签近乎粘连。
            if not sample_text_item.toPlainText():
                sample_text_item.setText("0")
            self._cached_label_height_pixels = (
                sample_text_item.boundingRect().height() + 2
            )

        label_height_pixels = self._cached_label_height_pixels
        font_metrics = self._cached_font_metrics

        # 按 cursor_id 分组（保留原始索引）
        cursor_groups = {}
        for orig_idx, item in enumerate(cursor_values):
            cid = item.get('cursor_id', 0)
            if cid not in cursor_groups:
                cursor_groups[cid] = []
            cursor_groups[cid].append({**item, '_orig_idx': orig_idx})

        # 对每个 cursor 独立处理（组内布局抽成可重入函数，
        # 跨组避让命中后可带 force_side 二次调用）。
        # 先做 Phase 1 过滤并收集非空组：交替判据必须用「过滤后可见组数」
        # ——cursor_groups 在 x 过滤前建立，另一游标出界时会留下空组，
        # 按原组数会误判为双游标而禁用交替。
        pending_groups = []
        all_filtered_out_indices = set()
        for cid, group_values in sorted(cursor_groups.items()):
            # --- Phase 1: 数据过滤 ---
            visible_labels, filtered_out_indices = self._filter_labels_phase1(
                group_values, x_min, x_max,
            )
            all_filtered_out_indices.update(filtered_out_indices)
            if not visible_labels:
                continue

            pending_groups.append({
                'cid': cid,
                'visible': visible_labels,
                'cursor_scene': view_box.mapViewToScene(
                    QPointF(group_values[0]['x_pos'], group_values[0]['y_pos'])
                ),
            })

        # 单游标（唯一可见组）列方向左右交替；双游标保持「右先左后 +
        # 跨组向外分裂」，中间缝隙为禁区，任何情况下不向内塞列。
        alternate_sides = len(pending_groups) == 1

        groups = []
        for pending in pending_groups:
            # --- Phase 2: 布局计算 ---
            layouts, layout_hide_indices, meta = self._layout_cursor_group(
                pending['cid'], pending['visible'], pending['cursor_scene'],
                view_box, x_min, x_max, y_min, y_max,
                label_height_pixels, font_metrics,
                alternate_sides=alternate_sides,
            )
            all_filtered_out_indices.update(layout_hide_indices)
            groups.append({
                **pending,
                'layouts': layouts,
                'hide': layout_hide_indices,
                'meta': meta,
            })

        # 回收已消失 cursor 的滞回状态，防止字典键泄漏
        alive_ids = set(cursor_groups)
        self._column_state = {
            cid: state for cid, state in self._column_state.items()
            if cid in alive_ids
        }

        # --- 跨组避让：碰撞命中且两侧空间预算通过 → 向外分裂 ---
        if (self.CURSOR_CROSS_AVOID and len(groups) == 2
                and groups[0]['layouts'] and groups[1]['layouts']
                and self._layouts_collide(
                    groups[0]['layouts'], groups[1]['layouts'],
                    label_height_pixels,
                )):
            self._apply_cross_group_avoidance(
                groups, view_box, x_min, x_max, y_min, y_max,
                label_height_pixels, font_metrics,
            )

        all_layouts = [layout for group in groups for layout in group['layouts']]
        for group in groups:
            all_filtered_out_indices.update(group['hide'])

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

    def _layout_cursor_group(
        self,
        cursor_id: int,
        visible_labels: list,
        cursor_scene_pos,
        view_box,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        label_height_pixels: float,
        font_metrics,
        force_side: str | None = None,
        feed_hysteresis: bool = True,
        alternate_sides: bool = False,
    ) -> tuple:
        """可重入的组内布局：返回 (layouts, hide_indices, meta)。

        跨组避让命中后带 force_side 再次调用，不重复收集/过滤标签。
        alternate_sides: 单游标列方向左右交替；跨组重布局传 False
        （force_side 优先，重布局不交替）。
        """
        return self._compute_label_layout(
            visible_labels, cursor_scene_pos, view_box,
            x_min, x_max, y_min, y_max,
            label_height_pixels, font_metrics,
            cursor_id=cursor_id,
            force_side=force_side,
            feed_hysteresis=feed_hysteresis,
            alternate_sides=alternate_sides,
        )

    @staticmethod
    def _layouts_collide(a: list, b: list, label_height_pixels: float) -> bool:
        """两组标签矩形是否相交（跨组避让判定）。"""
        for la in a:
            wa = la.w or 1.0
            for lb in b:
                if abs(la.layout_y - lb.layout_y) >= label_height_pixels:
                    continue
                wb = lb.w or 1.0
                if la.layout_x < lb.layout_x + wb and lb.layout_x < la.layout_x + wa:
                    return True
        return False

    @staticmethod
    def _layouts_within_view(layouts: list, meta: dict, h: float) -> bool:
        """标签矩形是否全部落在视口内（x∈[plotL,plotR]、y∈[plotT,plotB]）。"""
        if not meta:
            return False
        x0, x1 = meta['x_min_scene'], meta['x_max_scene']
        y0, y1 = meta['y_min_scene'], meta['y_max_scene']
        for layout in layouts:
            width = layout.w or 1.0
            if layout.layout_x < x0 - 0.5 or layout.layout_x + width > x1 + 0.5:
                return False
            if (layout.layout_y - h / 2 < y0 - 0.5
                    or layout.layout_y + h / 2 > y1 + 0.5):
                return False
        return True

    def _apply_cross_group_avoidance(
        self,
        groups: list,
        view_box,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        label_height_pixels: float,
        font_metrics,
    ) -> None:
        """跨组碰撞 → 向外分裂：光标靠左的组强制左、靠右的组强制右。

        先校验两侧空间预算（所需列总宽 vs 光标到视口边缘的可用宽），
        通过且重布局后零重叠、全部在视口内才接受；否则回退原布局并留
        debug 日志（重叠优于消失，指标条如实显示）。重布局跳过滞回喂入。
        按光标实际 scene x 排序（而非 cursor_id），保证「向外」不跨过邻光标。
        """
        left_group, right_group = sorted(
            groups, key=lambda g: g['cursor_scene'].x()
        )
        common = (
            view_box, x_min, x_max, y_min, y_max,
            label_height_pixels, font_metrics,
        )
        l_left, hide_left, meta_left = self._layout_cursor_group(
            left_group['cid'], left_group['visible'],
            left_group['cursor_scene'], *common,
            force_side='left', feed_hysteresis=False,
            alternate_sides=False,
        )
        l_right, hide_right, meta_right = self._layout_cursor_group(
            right_group['cid'], right_group['visible'],
            right_group['cursor_scene'], *common,
            force_side='right', feed_hysteresis=False,
            alternate_sides=False,
        )
        if meta_left is None or meta_right is None:
            logger.debug("[CURSOR] 跨组避让回退：重布局无可用标签")
            return

        left_need = meta_left['left_need']
        left_avail = meta_left['left_available']
        right_need = meta_right['right_need']
        right_avail = meta_right['right_available']
        if left_need > left_avail + 1e-6 or right_need > right_avail + 1e-6:
            logger.debug(
                "[CURSOR] 跨组避让回退：预算不足 left %.1f/%.1f right %.1f/%.1f",
                left_need, left_avail, right_need, right_avail,
            )
            return

        if self._layouts_collide(l_left, l_right, label_height_pixels):
            logger.debug("[CURSOR] 跨组避让回退：向外分裂后仍重叠")
            return
        if (not self._layouts_within_view(l_left, meta_left, label_height_pixels)
                or not self._layouts_within_view(
                    l_right, meta_right, label_height_pixels
                )):
            logger.debug("[CURSOR] 跨组避让回退：向外分裂后标签越出视口")
            return

        left_group['layouts'] = l_left
        left_group['hide'] = hide_left
        right_group['layouts'] = l_right
        right_group['hide'] = hide_right
        logger.debug(
            "[CURSOR] 跨组避让：向外分裂 left %.1f/%.1f right %.1f/%.1f",
            left_need, left_avail, right_need, right_avail,
        )

    def _filter_labels_phase1(
        self,
        cursor_values: list,
        x_min: float,
        x_max: float,
    ) -> tuple:
        """Phase 1: 在上游过滤基础上二次过滤（仅 x 方向）

        返回 (visible_labels, filtered_out_indices)。
        visible_labels: 按 y_pos 降序排列的可见标签列表（高 y 值在上，保留原始索引）。
        filtered_out_indices: 被过滤掉的标签 pool index 集合（延迟到所有 cursor group
        处理完毕后统一隐藏，避免多 cursor 共享 pool 时相互干扰）。
        注意：cursor_values 中的 item 应已包含 '_orig_idx' 字段。

        y 方向不做死区过滤：交点靠上/下边缘时标签由 _layout_column 的
        边界 clamp 吸附到边缘内侧完整显示（设计原则：交点在视图内时
        标签必须可见，宁可在视口内重叠也不画出视口消失）；y 出界的
        交点在开启 CURSOR_EDGE_PIN 时同样入列并标记 pinned，ideal 由
        列布局改写为视口上/下边缘（circle 仍不画）。
        """
        visible = [
            v for v in cursor_values
            if x_min < v['x_pos'] < x_max
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
        cursor_id: int = 0,
        force_side: str | None = None,
        feed_hysteresis: bool = True,
        alternate_sides: bool = False,
    ) -> tuple:
        """Phase 2: 计算所有标签的布局位置（纯数学，不操作 UI）

        返回 (layouts, hide_indices, meta)。
        hide_indices: 因窗口极小无法放置而需要隐藏的标签 pool index 集合。
        meta: 跨组避让预算信息（左/右所需宽与可用宽）；无标签时为 None。
        force_side: 'left'/'right' 强制单侧布局（跨组向外分裂用）；
        feed_hysteresis=False 用于重布局，避免计数器双喂。
        alternate_sides: 列方向左右交替（单游标）；默认 False 保持旧
        「右先左后」，双游标/重布局均走该路径。
        """
        if not visible_labels:
            return [], set(), None

        # 计算标签宽度（pinned 项按带 ▲/▼ 前缀的最终文本测量，
        # 否则边缘列宽不足会被 elide）
        max_label_width_pixels = 200  # 最大标签宽度（像素）
        label_widths_pixels = []
        for item in visible_labels:
            text_width = font_metrics.horizontalAdvance(
                self._label_display_text(item)
            )
            w = min(text_width + 12, max_label_width_pixels)
            label_widths_pixels.append(w)
            # 宽度随 item 传给 _layout_column，由其写入 LabelLayout.w，
            # 供列宽累计与跨组碰撞矩形使用。
            item['_label_w'] = w

        max_label_width_px = max(label_widths_pixels)

        # 获取 y 轴占用宽度
        view_width_pixels = max(1, view_box.width())
        view_height_pixels = max(1, view_box.height())

        # Scene 坐标边界。mapViewToScene(x_min) 已是 ViewBox 左界（轴区在
        # ViewBox 之外），不能再叠加左轴宽度——重复计入会让左侧可用宽
        # 凭空缩小，跨组避让误判「预算不足」而回退。
        x_min_scene = view_box.mapViewToScene(QPointF(x_min, 0)).x()
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

        # 计算列容量（精确口径：n 个标签需 n*h+(n-1)*gap，
        # 不能按 n*(h+gap) 估，否则临界行会多拆一列）
        margin_pixels_scene = label_height_pixels * 0.1
        available_height = scene_height - 2 * margin_pixels_scene
        raw_capacity = self._column_capacity(
            available_height, label_height_pixels, vertical_gap_pixels
        )

        # 极限保护：图高连一行标签都放不下（< 标签高+间距+2×margin）→
        # 返回该 cursor 组全部 pool index 供统一隐藏。
        # 必须位于 _apply_column_hysteresis 之前：若在滞回计算之后早退，
        # 滞回状态已被喂入 new_column_count = n_labels，窗口恢复正常
        # 尺寸后需连续 3 帧（HYSTERESIS_THRESHOLD）才收敛。
        if raw_capacity < 1:
            hide_indices = {item['_orig_idx'] for item in visible_labels}
            return [], hide_indices, None

        per_column_capacity = max(1, raw_capacity)

        # 列数计算（带滞回，按 cursor_id 分状态）
        n_labels = len(visible_labels)
        new_column_count = max(1, int(np.ceil(n_labels / per_column_capacity)))
        if feed_hysteresis:
            column_count = self._apply_column_hysteresis(
                new_column_count, cursor_id
            )
        else:
            # 跨组重布局：沿用原布局的列数，不双喂滞回计数器
            column_count = self._column_state.get(cursor_id, (1, 0))[0]

        # 计算列放置方向
        cursor_scene_x = cursor_scene_pos.x() if cursor_scene_pos else 0
        right_available = x_max_scene - cursor_scene_x - gap_to_cursor
        left_available = cursor_scene_x - gap_to_cursor - x_min_scene

        # 列数按可用宽度截断：溢出列会被方向分配压到兜底侧、绘制在视口外被
        # 裁剪（消失）。截断与分配共用 _possible_columns（含预算余量
        # COLUMN_BUDGET_MARGIN），避免截断后分配数不足而向两侧追加。
        # 截断后超容量标签由 _layout_column 的防重叠 Pass 压缩间距消化——
        # 最坏情况列内重叠，但全部在视口内（重叠严格优于消失）。
        col_width = max_label_width_px + col_gap_pixels
        max_cols = (
            self._possible_columns(
                right_available, col_width, self.COLUMN_BUDGET_MARGIN
            )
            + self._possible_columns(
                left_available, col_width, self.COLUMN_BUDGET_MARGIN
            )
        )
        column_count = max(1, min(column_count, max_cols))

        # 分配列到两侧
        column_sides = self._compute_column_positions(
            column_count, right_available, left_available,
            max_label_width_px, col_gap_pixels,
            force_side=force_side,
            alternate=alternate_sides,
        )
        # column_sides: list of ('right', col_idx) or ('left', col_idx)

        # 分配标签到各列（均匀分配）
        labels_per_column = self._distribute_labels(
            visible_labels, column_count, per_column_capacity
        )

        # 逐列取列内实际最大宽（分配后再定宽，避免列数估算同帧循环依赖；
        # 列数仍用 max 宽保守估算）。
        column_widths = [
            max((col.get('_label_w', 0.0) for col in col_labels), default=0.0)
            for col_labels in labels_per_column
        ]

        # 计算每列的场景 x 坐标（anchor=(0,0.5)，cx 即标签左边缘）。
        # 累计偏移：offset_k = GAP + Σ_{j<k}(w_j + COL_GAP)，左右侧各自累计；
        # 同侧从光标向外递增，宽列后接窄列不会自重叠。
        # first_col 语义为「该侧最内列」：两侧各有一列（depth 0）用直线引导线，
        # 外侧列（depth ≥ 1）才走 elbow 折线。
        column_scene_x = []
        first_col_flags = []
        side_offset = {'right': gap_to_cursor, 'left': gap_to_cursor}
        side_depth = {'right': 0, 'left': 0}
        for i, (side, _col_idx) in enumerate(column_sides):
            col_width = (
                column_widths[i] if i < len(column_widths)
                else max_label_width_px
            )
            offset = side_offset[side]
            if side == 'right':
                cx = cursor_scene_x + offset
            else:
                cx = cursor_scene_x - offset - col_width
            column_scene_x.append(cx)
            first_col_flags.append(side_depth[side] == 0)
            side_depth[side] += 1
            side_offset[side] = offset + col_width + col_gap_pixels

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
                first_col=first_col_flags[col_idx],
            )
            all_layouts.extend(col_layouts)

        # 跨组避让预算：同侧已用总宽（含列间距）与可用宽
        meta = {
            'left_need': side_offset['left'] - gap_to_cursor,
            'right_need': side_offset['right'] - gap_to_cursor,
            'left_available': left_available,
            'right_available': right_available,
            'x_min_scene': x_min_scene,
            'x_max_scene': x_max_scene,
            'y_min_scene': y_min_scene,
            'y_max_scene': y_max_scene,
        }
        return all_layouts, set(), meta

    def _apply_column_hysteresis(
        self, new_column_count: int, cursor_id: int = 0
    ) -> int:
        """列数切换滞回（按 cursor_id 独立）：连续 3 次调用才切换。

        单游标固定 key=0，初值 (1, 0) 与旧单份状态一致；多游标组间不再
        互相喂入计数器导致永远到不了 3 帧。
        """
        HYSTERESIS_THRESHOLD = 3
        cols, counter = self._column_state.get(cursor_id, (1, 0))
        if new_column_count != cols:
            counter += 1
            if counter >= HYSTERESIS_THRESHOLD:
                cols = new_column_count
                counter = 0
        else:
            counter = 0
        self._column_state[cursor_id] = (cols, counter)
        return cols

    @staticmethod
    def _possible_columns(
        available: float, col_width: float, margin: float = 0.0
    ) -> int:
        """单侧可容纳的列数（与 _assign_column_sides 同口径）。

        margin: 预算余量（像素）。可用宽至少要有「列宽 + margin」才算
        一列，避免刚好放得下的列在累计偏移后被浮点/描边吃掉半列。
        调用方（max_cols 截断）必须传入同一个 margin 值。
        """
        need = col_width + margin
        if available < need:
            return 0
        return int(available / need)

    @staticmethod
    def _column_capacity(available: float, h: float, gap: float) -> int:
        """单列可容纳的标签数（精确口径）。

        n 个标签实际需要 ``n*h + (n-1)*gap``，不能按 ``n*(h+gap)`` 估：
        后者会多要 1 个 gap，临界行高时误判容量 n-1、白白多拆一列。
        极矮兜底沿用旧阈值 ``available < h+gap`` → 0（连一行都放不下）。
        """
        if available < h + gap:
            return 0
        return int((available - h) / (h + gap) + 1e-9) + 1

    def _compute_column_positions(
        self,
        column_count: int,
        right_available: float,
        left_available: float,
        max_label_width: float,
        col_gap: float,
        force_side: str | None = None,
        alternate: bool = True,
    ) -> list:
        """决定每列的放置方向（委托 _assign_column_sides）。

        alternate=True（单游标）：右/左/右… 交替；False：旧「右先左后」。
        force_side='left'/'right'：跨组向外分裂时强制单侧。
        返回 list of (side, col_idx)。
        """
        return self._assign_column_sides(
            column_count, right_available, left_available,
            max_label_width, col_gap,
            force_side=force_side, alternate=alternate,
        )

    @staticmethod
    def _assign_column_sides(
        column_count: int,
        right_available: float,
        left_available: float,
        max_label_width: float,
        col_gap: float,
        force_side: str | None = None,
        alternate: bool = True,
    ) -> list:
        """列方向分配：返回 list of (side, col_idx)。

        force_side: 'left'/'right' 强制单侧，优先级最高（跨组向外分裂）。
        alternate=True（单游标）：偏好序列 右/左/右/左…，让标签贴住光标
        两侧而非单侧堆叠；偏好侧无预算时换另一侧；两侧都无预算时追加到
        占用较少的一侧（允许列重叠——重叠优于消失，不新增隐藏路径）。
        alternate=False（双游标）：沿用旧「右先左后 + 越界压右」，中间
        缝隙为禁区，双游标任何情况下都不向内塞列。

        两侧预算（possible）与调用方 max_cols 截断共用
        _possible_columns + COLUMN_BUDGET_MARGIN 同口径。
        """
        col_width = max_label_width + col_gap
        margin = CursorManager.COLUMN_BUDGET_MARGIN
        right_possible = CursorManager._possible_columns(
            right_available, col_width, margin
        )
        left_possible = CursorManager._possible_columns(
            left_available, col_width, margin
        )

        if force_side:
            return [(force_side, i) for i in range(column_count)]

        if not alternate:
            # 旧行为：优先填充右侧，剩余列放左侧，仍超出则压到右侧
            sides = []
            right_used = min(column_count, right_possible)
            for i in range(right_used):
                sides.append(('right', i))

            remaining = column_count - right_used
            left_used = min(remaining, left_possible)
            for i in range(left_used):
                sides.append(('left', i))

            # 超出可用空间：其余列压到右侧（列 x 由调用方按同侧累计
            # 偏移分配，不依赖 col_idx 连续）。
            for i in range(len(sides), column_count):
                sides.append(('right', i - right_used))
            return sides

        # 交错分配：k 偶数偏好右、奇数偏好左
        sides = []
        used = {'right': 0, 'left': 0}
        possible = {'right': right_possible, 'left': left_possible}
        for k in range(column_count):
            pref = 'right' if k % 2 == 0 else 'left'
            other = 'left' if pref == 'right' else 'right'
            if used[pref] < possible[pref]:
                side = pref
            elif used[other] < possible[other]:
                side = other
            else:
                # 两侧都无预算：追加到占用较少侧（并列取右），允许重叠
                side = 'right' if used['right'] <= used['left'] else 'left'
            sides.append((side, used[side]))
            used[side] += 1
        return sides

    @staticmethod
    def _distribute_labels(
        visible_labels: list, column_count: int, capacity: int
    ) -> list[list]:
        """按列容量填满分配：前列装满 capacity，剩余顺延。

        与平均分配相比，能在容量有余时把后续标签留在最靠近光标的
        第一列（如 6 个 / 容量 5 → 5+1，而不是 3+3）。

        列数被可用宽度截断、或滞回暂持更少列时可能出现
        ``n > capacity*column_count``，此时 PAV 已无法保证容量，
        退回平均分配，避免末列被塞成 4+6。
        """
        n = len(visible_labels)
        if n == 0:
            return [[] for _ in range(column_count)]

        if n > capacity * column_count:
            base = n // column_count
            remainder = n % column_count
            result = []
            start = 0
            for col in range(column_count):
                size = base + (1 if col < remainder else 0)
                result.append(visible_labels[start:start + size])
                start += size
            return result

        result = []
        start = 0
        for col in range(column_count):
            left_cols = column_count - col
            remaining = n - start
            size = remaining if left_cols == 1 else min(
                capacity, max(1, remaining - (left_cols - 1))
            )
            result.append(visible_labels[start:start + size])
            start += size
        return result

    @staticmethod
    def _clamp_label_y(
        target_y: float,
        col_y_min: float,
        col_y_max: float,
        half_height: float,
    ) -> float:
        """标签垂直位置边界 clamp（含倒挂防御）

        正常情况夹到 [col_y_min+half, col_y_max-half]；极矮图（图高 <
        标签高+2×margin）时区间倒挂（下界 > 上界），返回列垂直中心，
        保证标签中心不出界（至多半截贴边，好于完全按倒挂区间推出界外）。
        """
        lo = col_y_min + half_height
        hi = col_y_max - half_height
        if lo > hi:
            return (col_y_min + col_y_max) / 2
        return max(lo, min(hi, target_y))

    @staticmethod
    def _label_display_text(item: dict) -> str:
        """标签最终显示文本：pinned 项带 ▲/▼ 方向前缀。"""
        if item.get("pinned"):
            return ("▲ " if item.get("up") else "▼ ") + item["y_value"]
        return item["y_value"]

    def _ideal_label_scene_y(
        self,
        label: dict,
        view_box: Any,
        col_y_min: float,
        col_y_max: float,
        half_height: float,
    ) -> float:
        """标签理想中心 y：正常项贴交点上方，pinned 项吸附视口边缘内侧。"""
        if label.get("pinned"):
            if label.get("up"):
                return col_y_min + half_height + 2
            return col_y_max - half_height - 2
        scene_pos = view_box.mapViewToScene(
            QPointF(label["x_pos"], label["y_pos"])
        )
        return scene_pos.y() - half_height - self.LABEL_BOTTOM_GAP

    def _layout_column(
        self,
        labels: list,
        col_scene_x: float,
        col_y_min: float,
        col_y_max: float,
        label_height_pixels: float,
        gap_pixels: float,
        first_col: bool = False,
    ) -> list[LabelLayout]:
        """列内垂直堆叠：PAV 等渗回归求最小位移且保持最小间距"""
        n = len(labels)
        if n == 0:
            return []

        half_height = label_height_pixels / 2
        view_box = self.pw.plot_item.getViewBox()

        # 单标签：放在交点上方，底边留 LABEL_BOTTOM_GAP 净空；
        # pinned 项 ideal 吸附边缘内侧
        if n == 1:
            label = labels[0]
            target_y = self._ideal_label_scene_y(
                label, view_box, col_y_min, col_y_max, half_height
            )
            target_y = self._clamp_label_y(
                target_y, col_y_min, col_y_max, half_height
            )
            return [LabelLayout(
                layout_x=col_scene_x, layout_y=target_y,
                index=label['_orig_idx'],
                x_pos=label['x_pos'], y_pos=label['y_pos'],
                y_value=label['y_value'], color=label['color'],
                pinned=label.get('pinned', False), up=label.get('up', False),
                w=label.get('_label_w', 0.0), first_col=first_col,
            )]

        # 多标签：PAV 等渗回归直解（最小位移 + 最小间距）。
        # 标签已按 y_pos 降序（scene y 升序），ideal 为交点上方、
        # 底边留 LABEL_BOTTOM_GAP 净空。
        items = []
        for label in labels:
            items.append({
                'ideal': self._ideal_label_scene_y(
                    label, view_box, col_y_min, col_y_max, half_height
                )
            })
        stacked_y = self._pav_stack(
            items, col_y_min, col_y_max, label_height_pixels, gap_pixels
        )

        return [
            LabelLayout(
                layout_x=col_scene_x, layout_y=stacked_y[i],
                index=label['_orig_idx'],
                x_pos=label['x_pos'], y_pos=label['y_pos'],
                y_value=label['y_value'], color=label['color'],
                pinned=label.get('pinned', False), up=label.get('up', False),
                w=label.get('_label_w', 0.0), first_col=first_col,
            )
            for i, label in enumerate(labels)
        ]

    @staticmethod
    def _pav_stack(
        items: list,
        top: float,
        bottom: float,
        h: float,
        gap: float,
    ) -> list[float]:
        """PAV（pool adjacent violators）等渗回归：列内垂直堆叠直解。

        最小化 Σ(y_i − ideal_i)²，约束 y_i ≥ y_{i−1} + h + gap（标签不重叠）。
        items 为按 scene y 升序排列的 {'ideal': float}；返回各标签中心 y。

        变换 z_i = ideal_i − i·step 后约束变为 z 非降，PAV 合并违序块取均值
        即得全局最优解（未触发边界 clamp 时）。簇围绕中心上下对称分裂，
        避免单向下推的级联位移。边界 clamp 复用 ``_clamp_label_y``（含倒挂
        防御），随后仅做有界正/反推，保证结果不越出列区域。
        """
        n = len(items)
        if n == 0:
            return []

        half_height = h / 2
        step = h + gap

        if n == 1:
            return [
                CursorManager._clamp_label_y(
                    items[0]['ideal'], top, bottom, half_height
                )
            ]

        # 违序块合并：每块携带 (Σz, 块内标签数, 起始下标)
        blocks: list[list[float]] = []
        for i, item in enumerate(items):
            blocks.append([item['ideal'] - i * step, 1, i])
            while len(blocks) > 1:
                last, prev = blocks[-1], blocks[-2]
                if last[0] / last[1] >= prev[0] / prev[1] - 1e-9:
                    break
                prev[0] += last[0]
                prev[1] += last[1]
                blocks.pop()

        y = [0.0] * n
        for block_sum, count, start in blocks:
            mean = block_sum / count
            for i in range(start, start + int(count)):
                y[i] = mean + i * step

        y = [
            CursorManager._clamp_label_y(v, top, bottom, half_height) for v in y
        ]

        # 有界前推：不把标签推出视口下边缘
        for i in range(1, n):
            y[i] = min(max(y[i], y[i - 1] + step), bottom - half_height)
        # 有界后推：不把标签推出视口上边缘
        for i in range(n - 2, -1, -1):
            y[i] = max(min(y[i], y[i + 1] - step), top + half_height)
        return y

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

            # 标签文本截断（防止过长文本导致列宽计算失准）。
            # pinned 项按带 ▲/▼ 前缀的最终文本测宽，与布局阶段同口径。
            display_text = layout.y_value
            if layout.pinned:
                display_text = ("▲ " if layout.up else "▼ ") + display_text
            text_width = font_metrics.horizontalAdvance(display_text) + 12
            if text_width > MAX_LABEL_WIDTH_PIXELS:
                display_text = font_metrics.elidedText(
                    display_text, Qt.ElideRight, MAX_LABEL_WIDTH_PIXELS - 12
                )
                text_width = MAX_LABEL_WIDTH_PIXELS

            text_item.setText(display_text)

            # 色条(4px)与文字留 3px 间隙。setPlainText 会重置 root frame 边距，
            # 故每次 setText 后重设（默认 4 + 3 = 7）；只加左 margin，不改高度。
            doc = text_item.textItem.document()
            fmt = doc.rootFrame().frameFormat()
            if fmt.leftMargin() != 7:
                fmt.setLeftMargin(7)
                doc.rootFrame().setFrameFormat(fmt)

            # 身份色：左侧 4px 全色条 + 90α 白底 + 曲线色描边（A1' 调整版）。
            # 色条不透明全色承担身份；白底 90α(≈35% 白) 让曲线/网格透上来；
            # 描边用曲线色 1.5px（main 原色），把文本框从白底里勾出来。
            # pinned 项描边改虚线；缓存键必须并入 pinned，否则颜色不变时
            # 虚线/实线切换不会生效。
            chip_key = (layout.color, layout.pinned)
            if (not hasattr(text_item, '_cached_chip_key')
                    or text_item._cached_chip_key != chip_key):
                c = pg.mkColor(layout.color)
                white = pg.mkColor("#FFFFFF")
                white.setAlpha(90)
                grad = QLinearGradient(0.0, 0.0, 5.0, 0.0)
                grad.setColorAt(0.0, c)
                grad.setColorAt(0.79, c)
                grad.setColorAt(0.80, white)
                text_item.fill = QBrush(grad)
                text_item.setColor(pg.mkColor("#000000"))
                border_style = Qt.DashLine if layout.pinned else Qt.SolidLine
                text_item.border = pg.mkPen(c, width=1.5, style=border_style)
                text_item._cached_chip_key = chip_key

            text_item.setOpacity(0.8 if layout.pinned else 1.0)
            text_item.setVisible(True)

            # 将 scene 坐标转回数据坐标（TextItem 添加到 plot_item 时使用数据坐标）
            data_pos = view_box.mapSceneToView(
                QPointF(layout.layout_x, layout.layout_y)
            )
            text_item.setPos(data_pos.x(), data_pos.y())
            text_item.setZValue(201)
            pw.multi_cursor_items.append(text_item)

            # 引导线：交点 → 标签（A2）。数据坐标，随视图缩放与标签同动。
            # pinned 项贴边显示，不画引导线（避免横穿视口的噪音线）。
            leader = self._get_leader_from_pool(layout.index)
            leader_scene = leader.scene()
            if leader_scene != plot_scene:
                if leader_scene is not None:
                    leader_scene.removeItem(leader)
                pw.plot_item.addItem(leader, ignoreBounds=True)
            p0 = view_box.mapViewToScene(QPointF(layout.x_pos, layout.y_pos))
            # 端点取标签朝向光标的边缘（左侧列取右缘，否则线会压进标签下方）
            on_right = layout.layout_x >= p0.x()
            p1 = QPointF(
                layout.layout_x if on_right else layout.layout_x + text_width,
                layout.layout_y,
            )
            if not layout.pinned and (p1 - p0).manhattanLength() > 8:
                if self.CURSOR_LEADER_ELBOW and not layout.first_col:
                    # 外侧列：四点正交折线（直线会横穿该侧最内列标签）。
                    mx = p1.x() - 6 if on_right else p1.x() + 6
                    scene_pts = [
                        p0,
                        QPointF(mx, p0.y()),
                        QPointF(mx, p1.y()),
                        p1,
                    ]
                else:
                    # 每侧最内列（或关闭 elbow 开关）：直线，斜率区分归属
                    scene_pts = [p0, p1]
                view_pts = [view_box.mapSceneToView(pt) for pt in scene_pts]
                leader.setData(
                    [pt.x() for pt in view_pts], [pt.y() for pt in view_pts]
                )
                lc = pg.mkColor(layout.color)
                lc.setAlpha(200)
                leader.setPen(pg.mkPen(lc, width=1.25))
                leader.setVisible(True)
                pw.multi_cursor_items.append(leader)
            else:
                leader.setVisible(False)  # 贴住交点或 pinned 时不画，避免噪音

    # ========================================================================

    def _show_x_position_only(self, x_positions=None):
        """仅显示 x 位置标签（隐藏光标数值），同时在图上绘制 x_label 元素"""
        try:
            if not self._has_visible_curve_data():
                self._clear_cursor_items()
                return
            x_positions = (
                x_positions
                if x_positions is not None
                else self._get_cursor_x_positions()
            )
            if not x_positions:
                return

            pw = self.pw
            view_box = pw.view_box
            plot_scene = pw.plot_item.scene()

            (x_min, x_max), (y_min, y_max) = view_box.viewRange()
            self._clear_cursor_items()

            show_x_mode = self._get_cursor_mode()
            for idx, x in enumerate(x_positions):
                # anchored cursor 的 x 位置是用户固定的，不因为 view range 变化而被过滤
                if show_x_mode not in ("1 anchored cursor", "2 anchored cursor"):
                    if x < x_min or x > x_max:
                        continue
                x_str = format_fixed_trim(float(x), self._x_decimals())
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

            # 所有 cursor 可视化元素设置完毕后，
            # 强制触发 ViewBox 重绘，避免部分 item 不可见
            # （P0-2：默认游标路径同样必须阻断级联信号，与多曲线路径对齐）
            self._repaint_view_box(view_box)

        except Exception:
            logger.debug("更新 anchored cursor 可视化失败", exc_info=True)

    def _repaint_view_box(self, view_box) -> None:
        """强制触发 ViewBox 重绘，且阻断级联信号。

        Qt GraphicsView 在某些情况下不会自动调度 paint event（尤其在
        reload / 批量创建 item 后），导致部分 TextItem / ScatterPlotItem
        不可见，直到用户交互才恢复。

        v5.3: 必须用 QSignalBlocker 阻断 sigRangeChanged 等级联信号 ——
        强制重绘触发的级联回调会重新进入 update_cursor_label，
        使 BSP 树在多个回调中交叉修改 → SIGSEGV（不可被 except 捕获）。
        """
        with QSignalBlocker(view_box):
            view_box.update()

    def _has_visible_curve_data(self) -> bool:
        """判断当前 plot 是否有可见且有数据的曲线（统一版）"""
        try:
            for ci in self.pw.curves.values():
                if not ci.visible:
                    continue
                x_data = ci.x_data
                if x_data is not None and len(x_data) > 0:
                    return True
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

    def _update_cursor_after_plot(
        self, min_x_bound: float, max_x_bound: float, *, scene: bool = True
    ):
        """绘图后更新光标边界和可见性。

        ``scene=False``（非活动页的数据重建）只写 vline 边界这一纯数值，
        不改 vline 可见性/标签，并把所属页标脏，留给切页时按全局游标状态补建
        （设计文档 §5.5：隐藏页一律不碰场景）。
        """
        pw = self.pw
        if not scene:
            pw._set_vline_bounds([min_x_bound, max_x_bound])
            pw.mark_dirty_for_activation()
            return
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
        # 单游标模式下 vline2 只是隐藏的备用线。恢复 bounds 时 pyqtgraph 会在
        # setBounds() 内部对它做一次钳制 setValue，这条信号没有语义，却会以
        # cursor_index=1 往 pinned_x_values 塞第二个值，污染单游标状态。
        if (
            line is getattr(self.pw, "vline2", None)
            and self._get_cursor_mode() != "2 anchored cursor"
        ):
            return
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
                            with QSignalBlocker(target_line):
                                target_line.setPos(x_pos)
                        else:
                            logger.debug(
                                "[CURSOR_SYNC] skip: widget=%s y_name=%s cursor_index=%d "
                                "vline=%s vline2=%s is_pinned=%s pinned_x_values=%s",
                                id(widget), getattr(widget, "y_name", "?"), cursor_index,
                                widget.vline is not None,
                                getattr(widget, "vline2", None) is not None,
                                widget.is_cursor_pinned,
                                getattr(widget, "pinned_x_values", None),
                            )
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
        """根据当前绘制的数据更新vline bounds（统一版：始终从 curves 字典）

        有可见曲线：用本图曲线域。
        无曲线 / 全部隐藏：用全局数据 X 域（AxisManager.cursor_x_domain，随当前
        factor/offset 现算）。这里曾经读 pw.xMin/pw.xMax —— 那两个字段只在
        reset_plot 时按当时的 factor 换算过一次，时间修正后再不更新，导致空
        plot 的游标被钉在旧时间轴上（如 factor=0.1 后仍停在 index 域的 1）。
        """
        pw = self.pw
        try:
            if pw.curves:
                x_arrays = pw._collect_visible_curve_arrays('x_data')
                if x_arrays:
                    combined = np.concatenate(x_arrays)
                    min_x, max_x = np.nanmin(combined), np.nanmax(combined)
                    pw._set_vline_bounds([min_x, max_x])
                    return min_x, max_x

            domain = pw._apply_cursor_x_domain()
            return domain if domain is not None else (None, None)
        except Exception as e:
            logger.warning("Error updating vline bounds: %s", e)
            pw._set_vline_bounds([None, None])
            return None, None

    # ── 格式化工具方法 ──

    def sInt_to_fmtStr(self, value: int) -> str:
        """将秒数转换为时间字符串 HH:MM:SS.SS"""
        total = value % (24 * 3600)
        hh = int(total // 3600)
        mm = int((total % 3600) // 60)
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:05.2f}"

    def dateInt_to_fmtStr(self, value: int) -> str:
        """将时间戳转换为日期字符串"""
        try:
            dt = datetime.fromtimestamp(value)
            return dt.strftime('%Y/%m/%d')
        except Exception:
            return str(value)

    def _x_decimals(self) -> int:
        """游标 X 标签的小数位数（两处显示路径共用同一口径）。

        - 分组数据源（MDF/BLF）的时间轴：``factor`` 只是 1.0 的无效倍数，
          精度取可见曲线相邻 x 差中位数中最细的一档；
        - 序号轴（CSV/Excel）：沿用 ``factor`` 语义（1 -> 0 位、0.1 -> 1 位），
          并由 ``decimals_for_step`` 修掉 ``factor < 1e-6`` 退化成 0 位的缺陷。
        - 无可用步长时回退 factor，再回退 2 位。
        """
        pw = self.pw
        loader = getattr(getattr(pw, "plot_context", None), "loader", None)
        if is_grouped_loader(loader):
            curves = list((getattr(pw, "curves", None) or {}).values())
            steps = [
                ci.x_step
                for ci in curves
                if getattr(ci, "visible", True) and getattr(ci, "x_step", 0.0) > 0
            ]
            if not steps:
                # 全部隐藏时仍按数据步长显示：显隐切换不应改变读数精度
                steps = [
                    ci.x_step
                    for ci in curves
                    if getattr(ci, "x_step", 0.0) > 0
                ]
            if steps:
                return decimals_for_step(min(steps))
        factor = abs(self.factor)
        if factor > 0:
            return decimals_for_step(factor)
        return 2

