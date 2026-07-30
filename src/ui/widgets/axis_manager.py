"""
AxisManager - 坐标轴管理

负责 DraggableGraphicsLayoutWidget 的所有坐标轴相关功能：
- X/Y 轴范围和边界管理
- X 轴标签更新
- 自动缩放（auto_range, auto_y_in_x_range）
- 安全范围计算
- 数据密度计算

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING

import pyqtgraph as pg
from src.core.config import (
    DEFAULT_PADDING_VAL_X,
    DEFAULT_PADDING_VAL_Y,
    MIN_INDEX_LENGTH,
    compute_global_x_limits,
)
from src.core.logger import get_logger
import numpy as np

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.ui.widgets.plot_ui_manager import PlotUIManager


class AxisManager:
    """负责坐标轴范围、边界、标签管理"""

    def __init__(self, plot_ui_manager: PlotUIManager):
        if plot_ui_manager is None:
            raise ValueError("AxisManager requires a valid PlotUIManager instance")
        self._ui_manager = plot_ui_manager

    @property
    def pw(self) -> Any:
        return self._ui_manager.pw

    def update_x_axis_label(self) -> None:
        """更新 X 轴标签文本"""
        pw = self.pw
        if not hasattr(pw, "plot_item"):
            return
        axis = pw.plot_item.getAxis("bottom")
        from src.core.config import DEFAULT_SHOW_X_AXIS_LABEL

        if DEFAULT_SHOW_X_AXIS_LABEL:
            label = pw.time_axis_label if pw.time_axis_label else "Index"
            axis.setLabel(label)
            axis.showLabel(True)
        else:
            axis.showLabel(False)

    def auto_range(
        self,
        external_xmin: float | None = None,
        external_xmax: float | None = None,
    ) -> bool:
        """自动调整视图范围以适应数据

        Args:
            external_xmin: 外部指定的最小 x 值
            external_xmax: 外部指定的最大 x 值

        Returns:
            是否成功设置范围
        """
        pw = self.pw
        is_mdf = (
            pw.plot_context is not None
            and hasattr(pw.plot_context, "loader")
            and pw.plot_context.loader is not None
            and getattr(pw.plot_context.loader, "LOADER_TYPE", "") == "mdf"
        )

        has_own_data = bool(pw.curve or pw.curves)

        x_values = None
        own_min_x = None
        own_max_x = None

        if has_own_data:
            pw.axis_x.setTicks(None)
            pw.axis_y.setTicks(None)

            if pw.is_multi_curve_mode and pw.curves:
                x_arrays = pw._collect_visible_curve_arrays("x_data")
                if x_arrays:
                    x_values = np.concatenate(x_arrays)
            elif pw.original_index_x is not None:
                x_values = pw.offset + pw.factor * pw.original_index_x
            elif pw.curve:
                x_data, _ = pw.curve.getData()
                x_values = x_data if x_data is not None else None

            if x_values is not None:
                own_min_x = np.min(x_values)
                own_max_x = np.max(x_values)
        else:
            pw.axis_x.setTicks(None)
            pw.axis_y.setTicks(None)

        if external_xmin is not None:
            min_x = (
                min(own_min_x, external_xmin)
                if own_min_x is not None
                else external_xmin
            )
        else:
            min_x = own_min_x

        if external_xmax is not None:
            max_x = (
                max(own_max_x, external_xmax)
                if own_max_x is not None
                else external_xmax
            )
        else:
            max_x = own_max_x

        if min_x is None or max_x is None:
            return False

        min_y, max_y = self._get_y_range_for_auto_range(has_own_data, x_values)

        limits_xMin = min_x - DEFAULT_PADDING_VAL_X * (max_x - min_x)
        limits_xMax = max_x + DEFAULT_PADDING_VAL_X * (max_x - min_x)

        # 修复: limits 必须基于全局数据范围（而非 per-plot 曲线范围），
        # 避免 X-link 同步时目标 Plot 的 limits 偏窄导致 viewRange 被钳制
        if not is_mdf and pw.plot_context is not None:
            loader = getattr(pw.plot_context, "loader", None)
            global_result = compute_global_x_limits(
                loader, factor=pw.factor, offset=pw.offset
            )
            if global_result is not None:
                _, _, limits_xMin, limits_xMax = global_result

        pw.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)
        self._set_safe_y_range(min_y, max_y)

        minXRange_val = self._get_min_x_range_value()
        if is_mdf:
            pw.plot_item.setLimits(minXRange=minXRange_val)
        else:
            pw.plot_item.setLimits(
                xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val
            )

        self._set_vline_bounds([min_x, max_x])

        pw._queue_ui_refresh(immediate=True)
        pw.plot_item.update()
        pw._update_cursor_after_plot(min_x, max_x)

        return True

    def _get_y_range_for_auto_range(
        self,
        has_own_data: bool,
        x_values: np.ndarray | None,
    ) -> tuple[float, float]:
        """获取 auto_range 所需的 Y 轴范围"""
        pw = self.pw

        if not has_own_data:
            return 0, 1

        if pw.is_multi_curve_mode and pw.curves:
            y_arrays = pw._collect_visible_curve_arrays("y_data")
            if y_arrays:
                combined = np.concatenate(y_arrays)
                if combined.size:
                    min_y = np.nanmin(combined)
                    max_y = np.nanmax(combined)
                    return min_y, max_y
            return 0, 1
        else:
            if pw.original_y is not None:
                special_limits = pw.handle_single_point_limits(x_values, pw.original_y)
                if special_limits:
                    return special_limits[2], special_limits[3]
                min_y = np.nanmin(pw.original_y)
                max_y = np.nanmax(pw.original_y)
                return min_y, max_y
            elif pw.curve:
                _, y_data = pw.curve.getData()
                if y_data is not None:
                    return np.nanmin(y_data), np.nanmax(y_data)
            elif pw.curves:
                # 兜底：混合状态下从 curves 字典收集数据
                y_arrays = pw._collect_visible_curve_arrays("y_data")
                if y_arrays:
                    combined = np.concatenate(y_arrays)
                    if combined.size:
                        return float(np.nanmin(combined)), float(np.nanmax(combined))
            return 0, 1

    def auto_y_in_x_range(self) -> None:
        """在当前 X 范围内自动调整 Y 轴"""
        pw = self.pw
        vb = pw.view_box
        vb.enableAutoRange(axis=vb.YAxis, enable=True)
        pw.axis_y.setTicks(None)

    def set_xrange_with_link_handling(
        self,
        xmin: float,
        xmax: float,
        padding: float = 0,
    ) -> None:
        pw = self.pw
        plot = pw.plot_item

        linked = plot.getViewBox().linkedView(0)

        if linked is not None:
            plot.setXLink(None)

        plot.getViewBox().enableAutoRange(x=False)
        plot.setXRange(xmin, xmax, padding=max(0, padding))

        if linked is not None:
            plot.setXLink(linked)

        logger.debug(
            "[AXIS] set_xrange_with_link_handling: (%.4f, %.4f) padding=%.4f had_link=%s",
            xmin, xmax, max(0, padding), linked is not None,
        )

    def _get_safe_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        """确保 X 轴范围非零

        Args:
            min_x: X 轴最小值
            max_x: X 轴最大值

        Returns:
            安全（非零）的 X 轴范围
        """
        pw = self.pw
        if min_x == max_x:
            min_x_safe = min_x - 0.5 * pw.factor
            max_x_safe = max_x + 0.5 * pw.factor
            return min_x_safe, max_x_safe
        return min_x, max_x

    def _get_min_x_range_value(self) -> float:
        """计算最小的可缩放 X 范围

        基于全局最大数据点密度计算 minXRange。
        优先从 plot_context 读取 _global_max_density。
        """
        pw = self.pw
        if pw.plot_context is not None and hasattr(
            pw.plot_context, "_global_max_density"
        ):
            density = pw.plot_context._global_max_density
        else:
            density = 0.0

        if density > 0:
            return MIN_INDEX_LENGTH / density
        else:
            return 1.0

    def _set_x_limits_with_min_range(
        self,
        limits_xMin: float | None,
        limits_xMax: float | None,
    ) -> None:
        """统一设置 X 轴的 limits 和 minXRange"""
        pw = self.pw
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            stack = traceback.extract_stack(limit=4)
            caller = stack[-2] if len(stack) >= 2 else None
            caller_info = (
                f"{caller.filename.split('/')[-1]}:{caller.lineno} {caller.name}"
                if caller else "unknown"
            )
            old_limits = (
                pw.view_box.state.get('limits', {}).get('xLimits', [None, None])
                if hasattr(pw, 'view_box') else [None, None]
            )
            plot_id = getattr(pw, 'y_name', '') or str(list(getattr(pw, 'curves', {}).keys())[:2])
            logger.debug(
                "[XLIMITS] plot=%s set xMin=%s xMax=%s (old: %s) caller=%s",
                plot_id, limits_xMin, limits_xMax, old_limits, caller_info,
            )
        minXRange_val = self._get_min_x_range_value()
        pw.plot_item.setLimits(
            xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val
        )

    def _set_min_x_range(self, minXRange: float) -> None:
        """设置 X 轴的最小范围"""
        pw = self.pw
        pw.plot_item.setLimits(minXRange=minXRange)

    def _recalc_max_point_density(self) -> None:
        """重新计算当前 plot 的最大数据点密度"""
        pw = self.pw
        densities: list[float] = []
        for ci in pw.curves.values():
            if ci.point_density > 0:
                densities.append(ci.point_density)
        if not densities and pw.curve is not None and pw.original_y is not None:
            n = len(pw.original_y)
            if n > 1 and pw.original_index_x is not None:
                x_span = (
                    pw.offset
                    + pw.factor * float(np.max(pw.original_index_x))
                    - (pw.offset + pw.factor * float(np.min(pw.original_index_x)))
                )
                if x_span > 0:
                    densities.append(n / x_span)
        pw._max_point_density = max(densities) if densities else 0.0

    def _set_safe_y_range(
        self,
        min_y: float,
        max_y: float,
        set_limits: bool = True,
    ) -> None:
        """设置 Y 轴的 viewRange 和 limits

        自动处理 NaN 或恒定值的情况。

        Args:
            min_y: Y 轴最小值
            max_y: Y 轴最大值
            set_limits: 是否同时设置 y 轴 limits
        """
        pw = self.pw
        padding_yVal_limit = 0.5

        if np.isnan(min_y) or np.isnan(max_y) or min_y == max_y:
            y_center = min_y if not np.isnan(min_y) else 0
            y_range_half = 1.0 if y_center == 0 else abs(y_center) * 0.2

            y_min_view = y_center - y_range_half
            y_max_view = y_center + y_range_half
            y_min_limit = y_min_view
            y_max_limit = y_max_view
        else:
            y_min_view = min_y
            y_max_view = max_y
            y_range = max_y - min_y
            y_min_limit = min_y - padding_yVal_limit * y_range
            y_max_limit = max_y + padding_yVal_limit * y_range

        if set_limits:
            pw.plot_item.setLimits(yMin=y_min_limit, yMax=y_max_limit)
        pw.view_box.setYRange(y_min_view, y_max_view, padding=DEFAULT_PADDING_VAL_Y)

    def _reset_plot_limits(self) -> None:
        """重置绘图限制"""
        pw = self.pw
        try:
            pw.plot_item.setLimits(yMin=None, yMax=None)
            pw.view_box.setYRange(0, 1, padding=DEFAULT_PADDING_VAL_Y)
            self._set_vline_bounds([None, None])  # [None, None] 表示无边界限制
        except Exception:
            logger.debug("重置绘图限制失败", exc_info=True)

    def _set_vline_bounds(self, bounds: list) -> None:
        """设置光标垂直线的边界"""
        pw = self.pw
        if hasattr(pw, "vline"):
            pw.vline.setBounds(bounds)
        if hasattr(pw, "vline2"):
            pw.vline2.setBounds(bounds)
