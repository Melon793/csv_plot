"""
MarkRegionManager - 标记区域管理

负责 DraggableGraphicsLayoutWidget 的标记区域功能：
- 添加/移除标记区域
- 更新标记区域位置
- 计算标记区域内的数据统计

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, NamedTuple

import pyqtgraph as pg

if TYPE_CHECKING:
    from src.ui.widgets.cursor_manager import CursorManager


class MarkStats(NamedTuple):
    """标记区域统计信息"""

    x1: float
    x2: float
    y1: float
    y2: float
    dx: float
    dy: float
    slope: float
    label: str
    y_avg: float
    y_max: float
    y_min: float


class MarkRegionManager:
    """负责标记区域的管理和统计计算"""

    def __init__(self, cursor_manager: CursorManager):
        import numpy as np
        globals()['np'] = np
        if cursor_manager is None:
            raise ValueError(
                "MarkRegionManager requires a valid CursorManager instance"
            )
        self._cursor_manager = cursor_manager

    @property
    def pw(self) -> Any:
        return self._cursor_manager.pw

    @property
    def mark_region(self) -> Any:
        return getattr(self.pw, "mark_region", None)

    @mark_region.setter
    def mark_region(self, value: Any):
        self.pw.mark_region = value

    def add_mark_region(self, min_x: float, max_x: float):
        """添加标记区域"""
        import pyqtgraph as pg

        self.mark_region = pg.LinearRegionItem([min_x, max_x], movable=True)
        for line in self.mark_region.lines:
            line.setHoverPen(pg.mkPen(color="r", width=10))

        self.pw.plot_item.addItem(self.mark_region)
        if self.pw.plot_context:
            self.mark_region.sigRegionChanged.connect(
                self.pw.plot_context.sync_mark_regions
            )

    def remove_mark_region(self):
        """移除标记区域"""
        if self.mark_region and self.mark_region.scene() is not None:
            self.pw.plot_item.removeItem(self.mark_region)
        self.mark_region = None

    def update_mark_region(self):
        """更新标记区域"""
        if self.mark_region:
            old_min, old_max = self.mark_region.getRegion()
            from PySide6.QtCore import QSignalBlocker

            QSignalBlocker(self.mark_region)
            self.mark_region.setRegion([old_min, old_max])

    def get_mark_stats(self) -> list | None:
        """获取标记区域的统计信息

        使用 NumPy 掩码数组批量计算统计值，避免循环过滤。

        Returns:
            统计信息列表，每个元素为 (x1, x2, y1, y2, dx, dy, slope, label, y_avg, y_max, y_min)
            如果没有标记区域或无数据则返回 None
        """
        if not self.mark_region:
            return None

        min_x, max_x = self.mark_region.getRegion()

        if self.pw.is_multi_curve_mode:
            return self._get_mark_stats_multi_curve(min_x, max_x)
        else:
            return self._get_mark_stats_single_curve(min_x, max_x)

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
                x_data = x_data.astype(np.float32)
            if y_data.dtype.kind in "iu":
                y_data = y_data.astype(np.float32)

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
                    y_avg = float(np.mean(valid_y))
                    y_max = float(np.max(valid_y))
                    y_min = float(np.min(valid_y))
                else:
                    y_avg = y_max = y_min = np.nan

            unit = self.pw.units.get(var_name, "")
            label = f"{var_name} ({unit})" if unit else var_name

            stats_list.append(
                MarkStats(x1, x2, y1, y2, dx, dy, slope, label, y_avg, y_max, y_min)
            )

        return stats_list if stats_list else None

    def _get_mark_stats_single_curve(self, min_x: float, max_x: float) -> list | None:
        """单曲线模式的统计计算"""
        if not self.pw.curve:
            return None

        if (
            hasattr(self.pw, "original_index_x")
            and hasattr(self.pw, "original_y")
            and self.pw.original_index_x is not None
        ):
            x_data = self.pw.offset + self.pw.factor * self.pw.original_index_x
            y_data = self.pw.original_y
        else:
            x_data, y_data = self.pw.curve.getData()
            if x_data is None or len(x_data) == 0:
                return None

        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        if x_data.dtype.kind in "iu":
            x_data = x_data.astype(np.float32)
        if y_data.dtype.kind in "iu":
            y_data = y_data.astype(np.float32)

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
                y_avg = float(np.mean(valid_y))
                y_max = float(np.max(valid_y))
                y_min = float(np.min(valid_y))
            else:
                y_avg = y_max = y_min = np.nan

        return [
            MarkStats(
                x1,
                x2,
                y1,
                y2,
                dx,
                dy,
                slope,
                self.pw.label_left.text(),
                y_avg,
                y_max,
                y_min,
            )
        ]
