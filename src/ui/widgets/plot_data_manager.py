"""
PlotDataManager - 绘图和数据管理（统一版）

负责 DraggableGraphicsLayoutWidget 的绘图和数据管理功能：
- 曲线绘制（统一路径：始终写入 curves 字典）
- 数据准备、验证
- 时间修正（统一从 CurveInfo.original_index 重算）
- 数据清除

单线/多线模式统一后，所有曲线均以 CurveInfo 存储在 pw.curves 字典中。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker

from src.core.config import (
    DEFAULT_LINE_WIDTH,
    DEFAULT_PADDING_VAL_X,
    DEFAULT_PADDING_VAL_Y,
    _evaluate_float32_safety,
    compute_global_x_limits,
    safe_qt_op,
)
from src.core.data_types import CurveInfo
from src.core.logger import get_logger

logger = get_logger("widget.plot_data")

if TYPE_CHECKING:
    from src.ui.widgets.axis_manager import AxisManager


class PlotDataManager:
    """负责绘图和数据管理（统一路径：始终写入 curves 字典）"""

    def __init__(self, axis_manager: AxisManager):
        if axis_manager is None:
            raise ValueError("PlotDataManager requires a valid AxisManager instance")
        self._axis_manager = axis_manager

    @property
    def pw(self) -> Any:
        return self._axis_manager.pw

    def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
        """绘制变量到图表（统一版）
    
        空 plot 时创建首条曲线（蓝色）写入 curves 字典；
        已有曲线时转交 add_variable_to_plot 路径。
    
        Args:
            var_name: 要绘制的变量名称
            show_duplicate_warning: 是否显示重复变量警告
    
        Returns:
            bool: 绘制是否成功
        """
        pw = self.pw
        logger.debug(
            "[PLOT_VAR] plot_variable 入口: var_name=%s, 当前 curves=%s",
            var_name, list(pw.curves.keys()),
        )
    
        is_valid, error_msg = self._validate_plot_data(var_name)
        if not is_valid:
            from PySide6.QtWidgets import QMessageBox
    
            QMessageBox.warning(pw, "错误", error_msg)
            return False
    
        success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(
            var_name
        )
        if not success:
            from PySide6.QtWidgets import QMessageBox
    
            QMessageBox.warning(pw, "错误", error_msg)
            return False
    
        try:
            # 已有曲线时，转交 add_variable_to_plot 路径（由后者负责 emit curves_changed）
            if pw.curves:
                x_values = pw.offset + pw.factor * x_array
                return pw.add_variable_to_plot(
                    var_name,
                    x_values,
                    y_array,
                    y_format,
                    show_duplicate_warning=show_duplicate_warning,
                )
    
            # === 统一路径：始终写入 curves 字典 ===
            pw._clear_cursor_items(hide_only=False)
            self._safe_clear_plot_items()
            pw.curves.clear()
            pw.current_color_index = 0
    
            # 性能优化：保持 ascontiguousarray + float32 动态选择
            original_index = np.ascontiguousarray(x_array, dtype=np.float32)
            safe_for_float32, abs_max_plot = _evaluate_float32_safety(y_array)
            keep_float64 = (
                y_format in ["s", "date"]
                or not safe_for_float32
                or (abs_max_plot is not None and abs_max_plot > 1e8)
            )
            target_y_dtype = np.float64 if keep_float64 else np.float32
            y_contiguous = np.ascontiguousarray(y_array, dtype=target_y_dtype)
            x_values = pw.offset + pw.factor * original_index
    
            # 首条曲线保持蓝色（向后兼容）
            color = "blue"
            _pen = pg.mkPen(color=color, width=DEFAULT_LINE_WIDTH)
            curve = pw.plot_item.plot(
                x_values, y_contiguous, pen=_pen, name=var_name,
                skipFiniteCheck=True, connect="all",
            )
    
            # 写入 curves 字典（统一数据源）
            pw.curves[var_name] = CurveInfo(
                var_name=var_name,
                curve=curve,
                x_data=x_values,
                y_data=y_contiguous,
                original_index=original_index,
                color=color,
                y_format=y_format or '',
                visible=True,
            )
            pw.current_color_index = 1  # 后续曲线从颜色循环第 2 色开始
            logger.debug(
                "[PLOT_VAR] CurveInfo 已创建: var_name=%s, x_shape=%s, y_shape=%s, "
                "y_dtype=%s, color=%s",
                var_name, x_values.shape, y_contiguous.shape,
                y_contiguous.dtype, color,
            )
    
            pw._queue_ui_refresh()
    
            pw._update_header_for_curves()
    
            special_limits = self.handle_single_point_limits(x_values, y_contiguous)
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
                self._axis_manager._set_safe_y_range(min_y, max_y)
            else:
                data_min_y = np.nanmin(y_contiguous)
                data_max_y = np.nanmax(y_contiguous)
                self._axis_manager._set_safe_y_range(data_min_y, data_max_y, set_limits=True)
    
                # v5.x 修复问题 B：Y viewRange 基于“用户当前 viewRange 与新数据范围的交集”计算。
                # 保留用户的 X viewRange 不变，但 Y 范围只反映用户可见窗口内实际存在的数据。
                # 交集为空时 _get_y_range_in_x_window 内部会回退到全数据范围。
                data_x_min = float(np.min(x_values))
                data_x_max = float(np.max(x_values))
                view_x_min, view_x_max = pw.view_box.viewRange()[0]
                x_min = max(view_x_min, data_x_min)
                x_max = min(view_x_max, data_x_max)
                min_y, max_y = self._get_y_range_in_x_window(
                    x_values, y_contiguous, x_min, x_max
                )
                self._axis_manager._set_safe_y_range(min_y, max_y, set_limits=False)
                logger.debug(
                    "[PLOT_VAR] Y轴范围计算: limits=(%.6g, %.6g), view=(%.6g, %.6g)",
                    data_min_y, data_max_y, min_y, max_y,
                )
    
            min_x, max_x = np.min(x_values), np.max(x_values)
            self._axis_manager._set_vline_bounds([min_x, max_x])
            pw.plot_item.update()
            pw._update_cursor_after_plot(min_x, max_x)
    
            self._axis_manager._recalc_max_point_density()
            if not getattr(pw, "_is_updating_data", False):
                main_window = pw.window()
                if main_window is not None and hasattr(
                    main_window, "cursor_sync_manager"
                ):
                    main_window.cursor_sync_manager._sync_min_xrange()
    
            logger.debug("[PLOT_VAR] plot_variable 完成: var_name=%s, 数据点=%d", var_name, len(y_contiguous))
            return True
    
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
    
            QMessageBox.critical(pw, "绘图错误", f"绘制变量时发生错误: {str(e)}")
            return False

    def _validate_plot_data(self, var_name: str) -> tuple[bool, str]:
        """验证绘图数据的有效性

        Args:
            var_name: 要验证的变量名称

        Returns:
            tuple: (是否有效, 错误信息)
        """
        pw = self.pw

        if not isinstance(var_name, str) or not var_name.strip():
            return False, "变量名无效"

        if (
            pw.plot_context
            and hasattr(pw.plot_context, "loader")
            and pw.plot_context.loader is not None
        ):
            loader = pw.plot_context.loader
            if getattr(loader, "LOADER_TYPE", "") == "mdf":
                return True, ""

        if not hasattr(pw, "data") or pw.data is None:
            return False, "没有可用的数据"

        if not hasattr(pw.data, "columns"):
            return False, "数据格式无效"

        if var_name not in pw.data.columns:
            return False, f"变量 {var_name} 不存在"

        return True, ""

    def _get_x_data_for_variable(self, y_len: int) -> np.ndarray:
        return np.arange(1, y_len + 1, dtype=np.float32)

    def _prepare_plot_data(
        self, var_name: str
    ) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
        """准备绘图数据

        Args:
            var_name: 变量名称

        Returns:
            tuple: (是否成功, 错误信息, x数组, y数组, y格式)
        """
        pw = self.pw

        try:
            y_values, y_format = self.get_value_from_name(var_name=var_name)

            if y_values is None or len(y_values) == 0:
                return False, f"变量 {var_name} 没有有效数据", None, None, ""

            if isinstance(y_values, pd.Series):
                array_source = y_values.to_numpy()
                safety_source = y_values
            else:
                array_source = np.asarray(y_values)
                safety_source = array_source

            float32_safe, abs_max = _evaluate_float32_safety(safety_source)
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

            # 过滤 CAN/Automotive 哨兵值（约 -2^127 ≈ -1.7e38，表示"无效"信号状态）
            # 这类值会将 Y 轴范围撑到 1e38 量级，导致 QTransform 浮点精度丢失，
            # 进而使 InfiniteLine 的 mapFromScene 失效，cursor 无法被鼠标选中。
            SENTINEL_THRESHOLD = -1e30
            if y_array.dtype.kind == "f":
                sentinel_mask = y_array < SENTINEL_THRESHOLD
                if np.any(sentinel_mask):
                    sentinel_count = int(np.sum(sentinel_mask))
                    logger.debug(
                        "变量 %s 检测到 %d 个哨兵值（< %.1e），已替换为 NaN",
                        var_name,
                        sentinel_count,
                        SENTINEL_THRESHOLD,
                    )
                    y_array = y_array.copy()
                    y_array[sentinel_mask] = np.nan

            if np.all(np.isnan(y_array)):
                return False, f"变量 {var_name} 的数据全为无效值", None, None, ""

            if (
                pw.plot_context
                and hasattr(pw.plot_context, "loader")
                and hasattr(pw.plot_context.loader, "get_value_from_name")
            ):
                try:
                    x_array, _, _, _ = pw.plot_context.loader.get_value_from_name(
                        var_name
                    )
                    x_array = x_array[: len(y_array)]
                except KeyError:
                    x_array = self._get_x_data_for_variable(len(y_array))
            else:
                x_array = self._get_x_data_for_variable(len(y_array))

            return True, "", x_array, y_array, y_format

        except Exception as e:
            logger.warning(
                "准备绘图数据时出错: %s", e, exc_info=True
            )
            return False, f"处理数据时出错: {str(e)}", None, None, ""

    def _compute_valid_min_max(self, values) -> tuple[float | None, float | None]:
        """Safely compute min/max ignoring NaN/INF values."""
        if values is None:
            return None, None

        try:
            if isinstance(values, pd.Series):
                arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            try:
                arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
                    dtype=np.float64
                )
            except Exception:
                return None, None

        if arr.size == 0:
            return None, None

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return None, None

        finite_values = arr[finite_mask]
        return float(np.min(finite_values)), float(np.max(finite_values))

    def _get_y_range_in_x_window(
        self, x_values: np.ndarray, y_values: np.ndarray, x_min: float, x_max: float
    ):
        """计算在指定x轴范围内的y值范围

        Args:
            x_values: X轴数据数组
            y_values: Y轴数据数组
            x_min: X轴范围最小值
            x_max: X轴范围最大值

        Returns:
            tuple: (min_y, max_y)
        """
        try:
            mask = (x_values >= x_min) & (x_values <= x_max)
            if not np.any(mask):
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
            logger.warning(
                "Y 范围计算失败，回退默认值 (0.0, 1.0)", exc_info=True
            )
            return 0.0, 1.0

    def handle_single_point_limits(
        self, x_values: np.ndarray, y_values: np.ndarray
    ) -> tuple | None:
        """处理单点或所有点x坐标相同的特殊情况，避免x轴范围为0

        Args:
            x_values: x坐标数组
            y_values: y坐标数组

        Returns:
            tuple: (min_x, max_x, min_y, max_y) 或 None（正常情况不需要特殊处理）
        """
        pw = self.pw

        if (
            x_values is None
            or len(x_values) == 0
            or y_values is None
            or len(y_values) == 0
        ):
            return None

        if len(x_values) == 1:
            x = x_values[0]
            min_x, max_x = self._axis_manager._get_safe_x_range(x, x)
            if len(y_values) == 1:
                y = y_values[0]
                min_y = y - 0.5 if y != 0 else -0.5
                max_y = y + 0.5 if y != 0 else 0.5
            else:
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
            return min_x, max_x, min_y, max_y
        else:
            unique_x = set(x_values)
            if len(unique_x) == 1:
                x = list(unique_x)[0]
                min_x, max_x = self._axis_manager._get_safe_x_range(x, x)
                min_y = np.nanmin(y_values)
                max_y = np.nanmax(y_values)
                return min_x, max_x, min_y, max_y
            else:
                return None

    def clear_value_cache(self):
        """清除值缓存"""
        pw = self.pw
        if pw.plot_context and hasattr(pw.plot_context, "value_cache"):
            pw.plot_context.value_cache.clear()

    def datetime_to_unix_seconds(self, series: pd.Series) -> pd.Series:
        """将datetime Series转换为Unix时间戳（秒，float64精度）"""
        dtype_str = str(series.dtype)
        if "ns" in dtype_str:
            return (series.astype("int64") / 10**9).astype("float64")
        elif "us" in dtype_str:
            return (series.astype("int64") / 10**6).astype("float64")
        elif "ms" in dtype_str:
            return (series.astype("int64") / 10**3).astype("float64")
        else:
            raise ValueError(f"Unsupported datetime dtype: {series.dtype}")

    def get_value_from_name(self, var_name: str) -> tuple | None:
        """根据变量名获取值和格式"""
        pw = self.pw

        if not pw.plot_context:
            return None, None
        if var_name in pw.plot_context.value_cache:
            return pw.plot_context.value_cache[var_name]

        if hasattr(pw.plot_context, "loader") and pw.plot_context.loader is not None:
            loader = pw.plot_context.loader
            if getattr(loader, "LOADER_TYPE", "") == "mdf":
                raw_values = loader.get_series(var_name)
            else:
                raw_values = pw.data[var_name]
        else:
            raw_values = pw.data[var_name]

        if (
            hasattr(pw.plot_context, "loader")
            and pw.plot_context.loader is not None
            and hasattr(pw.plot_context.loader, "get_value_from_name")
            and getattr(pw.plot_context.loader, "LOADER_TYPE", "") == "mdf"
        ):
            try:
                _, _, _, text_map = pw.plot_context.loader.get_value_from_name(var_name)
                if text_map:
                    y_values = raw_values
                    y_format = "enum"
                    if pw.plot_context:
                        if hasattr(pw.plot_context, "_enum_text_maps"):
                            pw.plot_context._enum_text_maps[var_name] = text_map
                        if hasattr(pw.plot_context, "value_cache"):
                            pw.plot_context.value_cache[var_name] = (y_values, y_format)
                    return y_values, y_format
            except KeyError:
                pass  # var_name 无枚举映射，继续正常取值流程

        dtype_kind = raw_values.dtype.kind
        y_values = None
        y_format = "number"

        if dtype_kind in "iuf":
            y_values = raw_values
        elif dtype_kind == "b":
            y_values = raw_values.astype(np.int32)
        elif var_name in pw.time_channels_info:
            fmt = pw.time_channels_info[var_name]
            try:
                if "%H:%M:%S" in fmt:
                    times = pd.to_datetime(raw_values, format=fmt, errors="coerce")
                    today = pd.Timestamp.today().normalize()
                    time_deltas = times - times.dt.normalize()
                    dt_values = today + time_deltas
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = "s"
                else:
                    dt_values = pd.to_datetime(raw_values, format=fmt, errors="coerce")
                    y_values = self.datetime_to_unix_seconds(dt_values)
                    y_format = "date"
            except (ValueError, TypeError):
                return None, None
        else:
            try:
                numeric_values = pd.to_numeric(raw_values, errors="coerce")
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

        if pw.plot_context:
            if hasattr(pw.plot_context, "value_cache"):
                pw.plot_context.value_cache[var_name] = (y_values, y_format)
        return y_values, y_format

    def update_time_correction(self, new_factor: float, new_offset: float):
        """更新时间修正参数（统一版：始终从 CurveInfo.original_index 重算）"""
        pw = self.pw
        pw._suppress_pin_update = True
        try:
            old_factor = pw.factor
            old_offset = pw.offset
            pw.factor = new_factor
            pw.offset = new_offset
            logger.debug(
                "[TIME_CORR] update_time_correction: factor %.6g -> %.6g, "
                "offset %.6g -> %.6g, 影响曲线数=%d",
                old_factor, new_factor, old_offset, new_offset, len(pw.curves),
            )

            # 统一路径：遍历所有曲线，从 original_index 重算 x_data（消除反算精度问题）
            for var_name, ci in pw.curves.items():
                if ci.curve is None or ci.y_data is None:
                    continue
                if ci.original_index is not None:
                    new_x = pw.offset + pw.factor * ci.original_index
                else:
                    # 兜底：无 original_index 时用 arange 重建（不应发生）
                    logger.warning(
                        "[TIME_CORR] 曲线 %s 缺少 original_index，使用 arange 兜底", var_name
                    )
                    new_x = pw.offset + pw.factor * np.arange(
                        1, len(ci.y_data) + 1, dtype=np.float32
                    )
                ci.curve.setData(new_x, ci.y_data)
                ci.x_data = new_x
                ci.update_x_range()

            # 统一 X limits 计算：始终基于全局数据范围（loader.datalength / global_time_range），
            # 避免使用 per-plot 曲线长度导致与 reload/auto_range 路径不一致
            loader = (
                pw.plot_context.loader
                if pw.plot_context is not None
                and hasattr(pw.plot_context, "loader")
                else None
            )
            global_result = compute_global_x_limits(
                loader, factor=pw.factor, offset=pw.offset
            )
            if global_result is not None:
                _, _, limits_xMin, limits_xMax = global_result
                self._axis_manager._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            # 统一走 CursorManager 版本（含可见曲线过滤、NaN 保护与异常处理），
            # 避免与 plot_widget 转发路径行为不一致
            self.pw._cursor_manager._update_vline_bounds_from_data()
            if (
                pw.mark_region is not None
                and pw.plot_context
                and pw is pw.plot_context.plot_widgets[0].plot_widget
            ):
                old_min, old_max = pw.mark_region.getRegion()
                if old_factor != 0:
                    index_min = (old_min - old_offset) / old_factor
                    index_max = (old_max - old_offset) / old_factor
                    new_min = new_offset + new_factor * index_min
                    new_max = new_offset + new_factor * index_max
                    # 用 with 持有 QSignalBlocker，确保阻塞覆盖 setRegion 全过程，
                    # 避免临时对象被提前回收导致 sigRegionChanged 触发递归同步
                    with QSignalBlocker(pw.mark_region):
                        pw.mark_region.setRegion([new_min, new_max])
                        pw.plot_context.sync_mark_regions(pw.mark_region)
        finally:
            if pw.plot_context is not None:
                if not getattr(pw, "_is_being_destroyed", False):
                    pw.plot_context.request_mark_stats_refresh()
            pw._suppress_pin_update = False

    def _safe_clear_plot_items(self):
        """安全地清理所有plot items"""
        pw = self.pw
        if not hasattr(pw, "plot_item") or pw.plot_item is None:
            return

        safe_qt_op(lambda: self._clear_items_from_scene(pw.plot_item))

    def _clear_items_from_scene(self, plot_item):
        """从场景中逐个清理 plot item"""
        current_scene = plot_item.scene()
        if current_scene is None:
            return

        all_items = current_scene.items()
        for item in all_items:
            safe_qt_op(lambda: self._clear_single_item(item, current_scene))

    def _clear_single_item(self, item, current_scene):
        """清理单个 plot item"""
        item_scene = item.scene()
        if item_scene != current_scene:
            return
        should_remove = False
        if hasattr(item, "getData") and hasattr(item, "opts"):
            if not hasattr(item, "setLabel"):
                should_remove = True
                if hasattr(item, "_cached_pen_key"):
                    delattr(item, "_cached_pen_key")
                if hasattr(item, "_has_symbols"):
                    delattr(item, "_has_symbols")
                try:
                    item.clear()
                except Exception:
                    logger.debug("清理 plot item.clear() 异常", exc_info=True)
        if should_remove:
            current_scene.removeItem(item)

    def _clear_plot_data(self):
        """清除绘图数据（统一版：仅操作 curves 字典）"""
        pw = self.pw
        try:
            cleared_count = len(pw.curves)
            pw._clear_cursor_items(hide_only=False)
            self._safe_clear_plot_items()
            pw.axis_y.setLabel(text="")
            pw.update_legend_label("channel name")

            for var_name, ci in pw.curves.items():
                if ci.curve is not None:
                    curve = ci.curve
                    if hasattr(curve, "_cached_pen_key"):
                        delattr(curve, "_cached_pen_key")
                    if hasattr(curve, "_has_symbols"):
                        delattr(curve, "_has_symbols")
                    try:
                        curve.clear()
                    except Exception:
                        logger.debug("清理曲线时异常: %s", var_name)

            pw.curves.clear()
            pw.current_color_index = 0
            logger.debug("[CLEAR] _clear_plot_data: 已清除 %d 条曲线", cleared_count)

            self._axis_manager._recalc_max_point_density()
            if not getattr(pw, "_is_updating_data", False):
                main_window = pw.window()
                if main_window is not None and hasattr(main_window, "cursor_sync_manager"):
                    main_window.cursor_sync_manager._sync_min_xrange()

            self._axis_manager._set_vline_bounds([None, None])
        except Exception:
            logger.debug("清理绘图数据时异常", exc_info=True)

    def clear_plot_item(self):
        """清除绘图项"""
        self._axis_manager._reset_plot_limits()
        self._clear_plot_data()

    def reset_plot(self, index_xMin: float, index_xMax: float):
        """重置绘图"""
        pw = self.pw

        pw.plot_item.setLimits(xMin=None, xMax=None)
        pw.plot_item.setLimits(yMin=None, yMax=None)

        xMin = pw.offset + pw.factor * index_xMin
        xMax = pw.offset + pw.factor * index_xMax

        if not (np.isnan(xMax) or np.isinf(xMax)):
            xMin, xMax = self._axis_manager._get_safe_x_range(xMin, xMax)

            # 先设 limits
            padding_xVal = DEFAULT_PADDING_VAL_X
            limits_xMin = xMin - padding_xVal * (xMax - xMin)
            limits_xMax = xMax + padding_xVal * (xMax - xMin)
            self._axis_manager._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            # 再设 range
            pw.view_box.setXRange(xMin, xMax, padding=DEFAULT_PADDING_VAL_X)

        pw.view_box.setYRange(0, 1, padding=DEFAULT_PADDING_VAL_Y)
        self._axis_manager._set_vline_bounds([None, None])

        pw.xMin = xMin
        pw.xMax = xMax
        pw._clear_cursor_items(hide_only=False)
        self._safe_clear_plot_items()
        pw.axis_y.setLabel(text="")
        pw.update_legend_label("channel name")

        pw.curves.clear()
        pw.current_color_index = 0
