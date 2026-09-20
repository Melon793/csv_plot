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
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

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


def set_y_autorange(vb: pg.ViewBox) -> None:
    """开启「Y 轴跟随当前 X 可见段」，并保证这一帧就重算。

    autoVisibleOnly 的配对原本只在 ``plot_ui_manager._setup_plot_area`` 里设过一次，
    少这一步就是靠别处的默认值成立：配对一旦丢了，Y 会按全量数据 autoRange，
    Ctrl+Y 静默退化成「缩放到全部」。重算同理不能指望 ``enableAutoRange`` —— 标记
    已经是真时它整段跳过，什么都不做。
    """
    vb.setAutoVisible(x=False, y=True)
    vb.enableAutoRange(axis=vb.YAxis, enable=True)
    vb.queueUpdateAutoRange()


@contextmanager
def y_autorange_preserved(vb: pg.ViewBox) -> Iterator[None]:
    """程序化改范围期间保住用户显式开启的 Y autoRange。

    只复原标记、不强制重算：滚轮每一步都排队 auto-range 会平白多出重绘。
    """
    was_on = bool(vb.state["autoRange"][1])
    try:
        yield
    finally:
        if was_on:
            vb.enableAutoRange(axis=vb.YAxis, enable=True)


class AxisManager:
    """负责坐标轴范围、边界、标签管理"""

    def __init__(self, plot_ui_manager: PlotUIManager):
        if plot_ui_manager is None:
            raise ValueError("AxisManager requires a valid PlotUIManager instance")
        self._ui_manager = plot_ui_manager

    @property
    def pw(self) -> Any:
        ui = self._ui_manager
        if ui is None:
            raise RuntimeError(
                "AxisManager: dependency chain broken (_ui_manager is None)"
            )
        return ui.pw

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

        has_own_data = bool(pw.curves)  # 统一：只看 curves 字典

        x_values = None
        own_min_x = None
        own_max_x = None

        if has_own_data:
            pw.axis_x.setTicks(None)
            pw.axis_y.setTicks(None)

            # 统一：始终从 curves 字典收集
            x_arrays = pw._collect_visible_curve_arrays("x_data")
            if x_arrays:
                x_values = np.concatenate(x_arrays)

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
        logger.debug(
            "[AUTO_RANGE] auto_range: has_own_data=%s, X=(%s, %s), Y=(%.6g, %.6g), "
            "curves=%s",
            has_own_data,
            f"{min_x:.4f}" if min_x is not None else None,
            f"{max_x:.4f}" if max_x is not None else None,
            min_y, max_y, list(pw.curves.keys()),
        )

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

        # 先设 limits（确保新 limits 生效后再设 range，避免被旧 limits 钳制）
        minXRange_val = self._get_min_x_range_value()
        if is_mdf:
            pw.plot_item.setLimits(minXRange=minXRange_val)
        else:
            pw.plot_item.setLimits(
                xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val
            )

        # 再设 range（此时 limits 已是新值）
        pw.view_box.setXRange(min_x, max_x, padding=DEFAULT_PADDING_VAL_X)
        self._set_safe_y_range(min_y, max_y)

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
        """获取 auto_range 所需的 Y 轴范围（统一版：始终从 curves 字典收集）"""
        pw = self.pw
    
        if not has_own_data:
            return 0, 1
    
        y_arrays = pw._collect_visible_curve_arrays("y_data")
        if y_arrays:
            combined = np.concatenate(y_arrays)
            if combined.size:
                special_limits = pw.handle_single_point_limits(x_values, combined)
                if special_limits:
                    return special_limits[2], special_limits[3]
                min_y = np.nanmin(combined)
                max_y = np.nanmax(combined)
                return min_y, max_y
        return 0, 1

    def auto_y_in_x_range(self) -> None:
        """在当前 X 范围内自动调整 Y 轴（Ctrl+Y 的唯一入口）"""
        pw = self.pw
        set_y_autorange(pw.view_box)
        pw.axis_y.setTicks(None)

    def _xlink_target(self, vb: pg.ViewBox) -> pg.ViewBox:
        """X 范围应该写入的 ViewBox：本视图被 XLink 接管时必须是源视图。

        ``ViewBox.linkView`` 在重连末尾会无条件回调 ``linkedViewChanged``，用源
        的 range 覆盖子视图，所以「unlink → setXRange → relink」只要目标范围与
        源不同就会被静默回收（表现为第 2~N 个子图滚轮无反应）。写源则由 XLink
        原生向整组广播，既生效又保持共享 X 轴一致。
        """
        linked = vb.linkedView(0)
        return vb if linked is None else linked

    def _map_x_to_viewbox(
        self, vb: pg.ViewBox, target: pg.ViewBox, x: float
    ) -> float:
        """把 ``vb`` 视图坐标 x 换算成 ``target`` 的视图坐标，按屏幕像素对齐。

        各子图是各自独立的 GraphicsLayoutWidget（scene 不共用），因此不能走
        mapToScene；这里用 ``screenGeometry`` + ``viewRect`` 做线性映射，与
        pyqtgraph ``linkedViewChanged`` 的 XLink 像素映射同源。
        """
        own = vb.screenGeometry()
        dest = target.screenGeometry()
        src_rect = vb.viewRect()
        dst_rect = target.viewRect()
        if own is None or dest is None:
            return x
        if min(own.width(), dest.width(), src_rect.width(), dst_rect.width()) <= 0:
            return x
        frac = (x - src_rect.left()) / src_rect.width()
        global_x = own.left() + frac * own.width()
        return dst_rect.left() + (global_x - dest.left()) * dst_rect.width() / dest.width()

    def zoom_x(self, factor: float, center_x: float) -> None:
        """以 center_x 为中心把 X 轴缩放 factor 倍，且不触碰 Y 轴 autoRange。

        pyqtgraph 的 ViewBox.scaleBy 会经 setRange 同时关掉两轴 autoRange，
        导致 Ctrl+Y 开启的「Y 跟随可见段」在一次滚轮后被静默废除。
        """
        vb = self.pw.view_box
        target = self._xlink_target(vb)
        if target is not vb:
            # center_x 是鼠标所在子图的视图坐标，而缩放写进的是 XLink 源；
            # 不换算锚点的话，跨列/不等宽时缩放中心会按像素映射整体偏移。
            center_x = self._map_x_to_viewbox(vb, target, center_x)
        left, right = target.viewRange()[0]
        with y_autorange_preserved(target):
            self.set_xrange_with_link_handling(
                center_x - (center_x - left) * factor,
                center_x + (right - center_x) * factor,
            )

    def set_xrange_with_link_handling(
        self,
        xmin: float,
        xmax: float,
        padding: float = 0,
    ) -> None:
        """设置 X 范围；本视图被 XLink 接管时写入源视图，由其广播到整组。"""
        vb = self.pw.view_box
        target = self._xlink_target(vb)

        target.enableAutoRange(x=False)
        target.setXRange(xmin, xmax, padding=max(0, padding))

        logger.debug(
            "[AXIS] set_xrange_with_link_handling: (%.4f, %.4f) padding=%.4f "
            "link_source=%s",
            xmin, xmax, max(0, padding), target is not vb,
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
        """统一设置 X 轴的 limits 和 minXRange

        保证 limits 宽度 >= minXRange，避免 pyqtgraph 约束求解器
        在 xMin/xMax 与 minXRange 冲突时产生不稳定的 viewRange。
        """
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
            plot_id = str(list(getattr(pw, 'curves', {}).keys())[:2])
            logger.debug(
                "[XLIMITS] plot=%s set xMin=%s xMax=%s (old: %s) caller=%s",
                plot_id, limits_xMin, limits_xMax, old_limits, caller_info,
            )
        minXRange_val = self._get_min_x_range_value()
        # 确保 limits 宽度 >= minXRange，防止约束冲突导致 viewRange 振荡
        if limits_xMin is not None and limits_xMax is not None:
            limits_width = limits_xMax - limits_xMin
            if limits_width < minXRange_val:
                center = (limits_xMin + limits_xMax) / 2.0
                limits_xMin = center - minXRange_val / 2.0
                limits_xMax = center + minXRange_val / 2.0
                logger.debug(
                    "[XLIMITS] limits width %.4f < minXRange %.4f, "
                    "expanded limits to [%.4f, %.4f]",
                    limits_width, minXRange_val, limits_xMin, limits_xMax,
                )
        pw.plot_item.setLimits(
            xMin=limits_xMin, xMax=limits_xMax, minXRange=minXRange_val
        )

    def _set_min_x_range(self, minXRange: float) -> None:
        """设置 X 轴的最小范围

        同时确保 xMin/xMax limits 宽度 >= minXRange，
        防止 pyqtgraph 约束求解器在二者冲突时产生不稳定的 viewRange。
        """
        pw = self.pw
        # 获取当前的 xMin/xMax limits，必要时扩展以容纳 minXRange
        try:
            current_limits = pw.view_box.state.get('limits', {})
            x_limits = current_limits.get('xLimits', [None, None])
            cur_xMin = x_limits[0] if x_limits else None
            cur_xMax = x_limits[1] if len(x_limits) > 1 else None
        except Exception:
            cur_xMin, cur_xMax = None, None

        if cur_xMin is not None and cur_xMax is not None:
            limits_width = cur_xMax - cur_xMin
            if limits_width < minXRange:
                center = (cur_xMin + cur_xMax) / 2.0
                cur_xMin = center - minXRange / 2.0
                cur_xMax = center + minXRange / 2.0
                logger.debug(
                    "[XLIMITS] _set_min_x_range: limits width %.4f < minXRange %.4f, "
                    "expanded limits to [%.4f, %.4f]",
                    limits_width, minXRange, cur_xMin, cur_xMax,
                )
                pw.plot_item.setLimits(
                    xMin=cur_xMin, xMax=cur_xMax, minXRange=minXRange
                )
                return

        pw.plot_item.setLimits(minXRange=minXRange)

    def _recalc_max_point_density(self) -> None:
        """重新计算当前 plot 的最大数据点密度（统一版：仅从 curves 字典）"""
        pw = self.pw
        densities: list[float] = []
        for ci in pw.curves.values():
            if ci.point_density > 0:
                densities.append(ci.point_density)
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

        # 防御性检查：Y 范围跨度过大（> 1e20）会导致 QTransform 浮点精度丢失，
        # 使 InfiniteLine 的 mapFromScene 失效，cursor 无法被鼠标选中。
        # 通常由未过滤的 CAN 哨兵值（≈ -1.7e38）引起，回退到有限值的真实范围。
        Y_RANGE_ABS_LIMIT = 1e20
        if abs(min_y) > Y_RANGE_ABS_LIMIT or abs(max_y) > Y_RANGE_ABS_LIMIT:
            logger.warning(
                "[Y_RANGE] 检测到超大 Y 范围 [%s, %s]，可能存在未过滤的哨兵值",
                min_y,
                max_y,
            )
            if getattr(pw, "curves", None):
                try:
                    y_arrays = pw._collect_visible_curve_arrays("y_data")
                except Exception:
                    logger.warning(
                        "收集可见曲线 y_data 失败，使用空列表作为回退",
                        exc_info=True,
                    )
                    y_arrays = []
                if y_arrays:
                    combined = np.concatenate(y_arrays)
                    finite_mask = np.isfinite(combined) & (
                        np.abs(combined) < Y_RANGE_ABS_LIMIT
                    )
                    if finite_mask.any():
                        finite_vals = combined[finite_mask]
                        min_y = float(np.min(finite_vals))
                        max_y = float(np.max(finite_vals))
                        logger.info(
                            "[Y_RANGE] 已限制为有限值范围 [%s, %s]",
                            min_y,
                            max_y,
                        )

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
            # 游标域回到全局数据域（而非无界）；无权威源时内部回退 [None, None]
            self.apply_cursor_x_domain()
        except Exception:
            logger.debug("重置绘图限制失败", exc_info=True)

    def cursor_x_domain(self) -> tuple[float, float] | None:
        """游标可移动区间的唯一权威来源：全局数据 X 域（已套当前 factor/offset）。

        与 ViewBox 的 xLimits 同源（都走 compute_global_x_limits），区别是本函数
        返回不含 ±5% padding 的裸数据域：limits 管「视窗能看到哪」，本函数管
        「游标能到哪」。

        Returns:
            (min_x, max_x)；loader 无效 / 未加载数据时返回 None，调用方需自行回退。
        """
        pw = self.pw
        ctx = pw.plot_context
        loader = getattr(ctx, "loader", None) if ctx is not None else None
        result = compute_global_x_limits(loader, factor=pw.factor, offset=pw.offset)
        if result is None:
            return None
        return (float(result[0]), float(result[1]))

    def apply_cursor_x_domain(self) -> tuple[float, float] | None:
        """按 cursor_x_domain() 设置 vline bounds。

        用于「本图没有可见曲线」的所有回退场景（空 plot、删掉最后一条曲线、
        全部隐藏、reset/clear）：域来自全局数据轴，而不是本图曲线的历史快照。
        无权威源时回退 [None, None]（= pyqtgraph 的无界语义）。
        """
        domain = self.cursor_x_domain()
        self._set_vline_bounds(list(domain) if domain is not None else [None, None])
        return domain

    def _set_vline_bounds(self, bounds: list) -> None:
        """设置光标垂直线的边界"""
        pw = self.pw
        if hasattr(pw, "vline"):
            pw.vline.setBounds(bounds)
        if hasattr(pw, "vline2"):
            pw.vline2.setBounds(bounds)
