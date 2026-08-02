"""
MarkRegionManager - 标记区域管理

负责 DraggableGraphicsLayoutWidget 的标记区域功能：
- 添加/移除标记区域
- 更新标记区域位置
- 计算标记区域内的数据统计

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker

from src.core.config import _evaluate_float32_safety
from src.core.data_types import MarkStatEntry

if TYPE_CHECKING:
    from src.ui.widgets.cursor_manager import CursorManager


class MarkRegionManager:
    """负责标记区域的管理和统计计算"""

    def __init__(self, cursor_manager: CursorManager):
        if cursor_manager is None:
            raise ValueError(
                "MarkRegionManager requires a valid CursorManager instance"
            )
        self._cursor_manager = cursor_manager

    @property
    def pw(self) -> Any:
        return self._cursor_manager.pw

    def add_mark_region(self, min_x: float, max_x: float):
        """添加标记区域"""
        self.pw.mark_region = pg.LinearRegionItem([min_x, max_x], movable=True)
        for line in self.pw.mark_region.lines:
            line.setHoverPen(pg.mkPen(color="r", width=10))

        self.pw.plot_item.addItem(self.pw.mark_region)
        self.pw.mark_region.sigRegionChanged.connect(
            self.pw.window().layout_manager.sync_mark_regions
        )

    def remove_mark_region(self):
        """移除标记区域"""
        if self.pw.mark_region and self.pw.mark_region.scene() is not None:
            self.pw.plot_item.removeItem(self.pw.mark_region)
        self.pw.mark_region = None

    def update_mark_region(self):
        """更新标记区域"""
        if self.pw.mark_region:
            old_min, old_max = self.pw.mark_region.getRegion()
            blocker = QSignalBlocker(self.pw.mark_region)
            self.pw.mark_region.setRegion([old_min, old_max])

    def get_mark_stats(self) -> list | None:
        """获取标记区域的统计信息（统一版：始终走 curves 字典路径）

        【NumPy优化】使用NumPy掩码数组批量计算统计值，避免循环过滤
        """
        if not self.pw.mark_region:
            return None

        min_x, max_x = self.pw.mark_region.getRegion()
        return self._get_mark_stats_multi_curve(min_x, max_x)

    def _get_mark_stats_multi_curve(self, min_x: float, max_x: float) -> list | None:
        """多曲线模式的统计计算"""
        stats_list = []

        for var_name, ci in self.pw.curves.items():
            if not ci.visible:
                continue

            if ci.curve is None:
                continue

            if ci.x_data is not None and ci.y_data is not None:
                x_data = ci.x_data
                y_data = ci.y_data
            elif ci.y_data is not None:
                x_data = self.pw.offset + self.pw.factor * np.arange(
                    1, len(ci.y_data) + 1, dtype=np.float32
                )
                y_data = ci.y_data
            else:
                curve = ci.curve
                x_data, y_data = curve.getData()
                if x_data is None or len(x_data) == 0:
                    continue

            x_data = np.asarray(x_data)
            y_data = np.asarray(y_data)
            if x_data.dtype.kind in "iu":
                safe_x, _ = _evaluate_float32_safety(x_data)
                x_dtype = np.float32 if safe_x else np.float64
                x_data = x_data.astype(x_dtype)
            if y_data.dtype.kind in "iu":
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
            slope = float("inf") if dx == 0 else dy / dx

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

            unit = self.pw.units.get(var_name, "")
            label = f"{var_name} ({unit})" if unit else var_name

            stats_list.append(
                MarkStatEntry(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    dx=dx,
                    dy=dy,
                    slope=slope,
                    label=label,
                    y_avg=y_avg,
                    y_max=y_max,
                    y_min=y_min,
                )
            )

        return stats_list if stats_list else None
