"""
PlotDataManager - 单曲线绘图和数据管理

负责 DraggableGraphicsLayoutWidget 的单曲线绘图和数据管理功能：
- 单曲线绘制
- 数据准备、验证
- 时间修正
- 数据清除
- 原始数据缓存

此模块从 csv_plot_pyqt6.py 迁移而来。
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
)
from src.core.logger import get_logger

logger = get_logger("widget.plot_data")

if TYPE_CHECKING:
    from src.ui.widgets.axis_manager import AxisManager


class PlotDataManager:
    """负责单曲线绘图和数据管理"""

    def __init__(self, axis_manager: AxisManager):
        if axis_manager is None:
            raise ValueError("PlotDataManager requires a valid AxisManager instance")
        self._axis_manager = axis_manager

    @property
    def pw(self) -> Any:
        return self._axis_manager.pw

    def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
        """绘制变量到图表

        Args:
            var_name: 要绘制的变量名称
            show_duplicate_warning: 是否显示重复变量警告

        Returns:
            bool: 绘制是否成功
        """
        pw = self.pw

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
            if pw.is_multi_curve_mode:
                x_values = pw.offset + pw.factor * x_array
                return pw.add_variable_to_plot(
                    var_name,
                    x_values,
                    y_array,
                    y_format,
                    show_duplicate_warning=show_duplicate_warning,
                )

            pw.y_format = y_format
            pw.y_name = var_name
            pw.original_index_x = np.asarray(x_array, dtype=np.float32)
            safe_for_float32, abs_max_plot = _evaluate_float32_safety(y_array)
            keep_float64 = (
                y_format in ["s", "date"]
                or not safe_for_float32
                or (abs_max_plot is not None and abs_max_plot > 1e8)
            )
            target_y_dtype = np.float64 if keep_float64 else np.float32
            pw.original_y = np.asarray(y_array, dtype=target_y_dtype)
            x_values = pw.offset + pw.factor * pw.original_index_x

            pw._clear_cursor_items(hide_only=False)
            self._safe_clear_plot_items()
            pw.curves.clear()

            _pen = pg.mkPen(color="blue", width=DEFAULT_LINE_WIDTH)
            pw.curve = pw.plot_item.plot(
                x_values, pw.original_y, pen=_pen, name=var_name, skipFiniteCheck=True
            )

            pw._queue_ui_refresh()

            full_title = f"{var_name} ({pw.units.get(var_name, '')})".strip()
            pw.update_left_header(full_title)

            special_limits = self.handle_single_point_limits(x_values, pw.original_y)
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
                self._set_safe_y_range(min_y, max_y)
            else:
                data_min_y = np.nanmin(pw.original_y)
                data_max_y = np.nanmax(pw.original_y)
                self._set_safe_y_range(data_min_y, data_max_y, set_limits=True)

                current_x_range = pw.view_box.viewRange()[0]
                x_min, x_max = current_x_range
                min_y, max_y = self._get_y_range_in_x_window(
                    x_values, pw.original_y, x_min, x_max
                )
                self._set_safe_y_range(min_y, max_y, set_limits=False)

            min_x, max_x = np.min(x_values), np.max(x_values)
            self._set_vline_bounds([min_x, max_x])
            pw.plot_item.update()
            pw._update_cursor_after_plot(min_x, max_x)

            self._recalc_max_point_density()
            if pw.plot_context is not None and hasattr(
                pw.plot_context, "_sync_min_xrange"
            ):
                pw.plot_context._sync_min_xrange()

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
            return 0.0, 1.0

    def handle_single_point_limits(
        self, x_values: np.ndarray, y_values: np.ndarray
    ) -> tuple | None:
        """处理单点或所有点x坐标相同的特殊情况"""
        pw = self.pw
        if (
            x_values is None
            or len(x_values) == 0
            or y_values is None
            or len(y_values) == 0
        ):
            return None

        unique_x = np.unique(x_values)
        if len(unique_x) <= 1:
            x_min = np.min(x_values)
            x_max = np.max(x_values)
            y_min = np.nanmin(y_values)
            y_max = np.nanmax(y_values)

            if y_min == y_max:
                y_range_half = 0.5
                y_min -= y_range_half
                y_max += y_range_half

            if x_min == x_max:
                x_range_half = 0.5 * pw.factor if pw.factor != 0 else 0.5
                x_min -= x_range_half
                x_max += x_range_half

            return (x_min, x_max, y_min, y_max)

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
        """更新时间修正参数"""
        pw = self.pw
        pw._suppress_pin_update = True
        try:
            old_factor = pw.factor
            old_offset = pw.offset
            pw.factor = new_factor
            pw.offset = new_offset

            is_mdf = (
                pw.plot_context is not None
                and hasattr(pw.plot_context, "loader")
                and pw.plot_context.loader is not None
                and getattr(pw.plot_context.loader, "LOADER_TYPE", "") == "mdf"
            )

            if pw.is_multi_curve_mode:
                for var_name, ci in pw.curves.items():
                    if ci.curve is not None and ci.y_data is not None:
                        curve = ci.curve
                        y_data = ci.y_data
                        if is_mdf:
                            old_x_data = ci.x_data
                            if old_x_data is not None and old_factor != 0:
                                original_time = (old_x_data - old_offset) / old_factor
                            else:
                                original_time = np.arange(1, len(y_data) + 1)
                        else:
                            original_time = np.arange(1, len(y_data) + 1)
                        new_x = pw.offset + pw.factor * original_time
                        curve.setData(new_x, y_data)
                        ci.x_data = new_x
                        ci.update_x_range()
            else:
                if pw.original_index_x is not None:
                    new_x = pw.offset + pw.factor * pw.original_index_x
                    pw.curve.setData(new_x, pw.original_y)

            if pw.is_multi_curve_mode and pw.curves:
                first_curve_info = next(iter(pw.curves.values()))
                datalength = (
                    len(first_curve_info.y_data)
                    if first_curve_info.y_data is not None
                    else 0
                )
            elif pw.original_index_x is not None:
                datalength = len(pw.original_index_x)
            else:
                datalength = (
                    pw.plot_context.loader.datalength
                    if hasattr(pw.plot_context, "loader")
                    and pw.plot_context.loader is not None
                    else 0
                )

            padding_xVal = DEFAULT_PADDING_VAL_X
            if (
                is_mdf
                and pw.plot_context is not None
                and hasattr(pw.plot_context.loader, "global_time_range")
            ):
                x_min, x_max = pw.plot_context.loader.global_time_range
                data_min_x = pw.offset + pw.factor * x_min
                data_max_x = pw.offset + pw.factor * x_max
            else:
                index_min = 1 - padding_xVal * datalength
                index_max = datalength + padding_xVal * datalength
                data_min_x = pw.offset + pw.factor * index_min
                data_max_x = pw.offset + pw.factor * index_max
            limits_xMin = data_min_x - padding_xVal * (data_max_x - data_min_x)
            limits_xMax = data_max_x + padding_xVal * (data_max_x - data_min_x)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)
            self._update_vline_bounds_from_data()
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
                    QSignalBlocker(pw.mark_region)
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
        try:
            if not hasattr(pw, "plot_item") or pw.plot_item is None:
                return

            current_scene = pw.plot_item.scene()

            if current_scene is not None:
                all_items = current_scene.items()
                for item in all_items:
                    try:
                        item_scene = item.scene()
                        if item_scene == current_scene:
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
                                        logger.debug("清理 plot item.clear() 异常")
                            if should_remove:
                                current_scene.removeItem(item)
                    except (RuntimeError, AttributeError):
                        logger.debug("C++ 对象已销毁，跳过该 item 清理")
        except (RuntimeError, AttributeError):
            logger.debug("plot_item 场景已销毁，跳过批量清理")

    def _clear_plot_data(self):
        """清除绘图数据"""
        pw = self.pw
        try:
            pw._clear_cursor_items(hide_only=False)
            self._safe_clear_plot_items()
            pw.axis_y.setLabel(text="")
            pw.y_name = ""
            pw.y_format = ""
            pw.update_left_header("channel name")
            pw.update_right_header("")

            if pw.curve:
                if hasattr(pw.curve, "_cached_pen_key"):
                    delattr(pw.curve, "_cached_pen_key")
                if hasattr(pw.curve, "_has_symbols"):
                    delattr(pw.curve, "_has_symbols")
                try:
                    pw.curve.clear()
                except Exception:
                    logger.debug("清理 curve 时异常")

            pw.curve = None
            pw.original_index_x = None
            pw.original_y = None

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
                        logger.debug("清理多曲线 curve 时异常: %s", var_name)
            
            # 清除 vline bounds
            self._set_vline_bounds([None, None])
        except Exception:
            logger.debug("清理绘图数据时异常", exc_info=True)

    def clear_plot_item(self):
        """清除单个plot item"""
        self._clear_plot_data()

    def reset_plot(self, index_xMin: float, index_xMax: float):
        """重置绘图"""
        pw = self.pw

        pw.plot_item.setLimits(xMin=None, xMax=None)
        pw.plot_item.setLimits(yMin=None, yMax=None)

        xMin = pw.offset + pw.factor * index_xMin
        xMax = pw.offset + pw.factor * index_xMax

        if not (np.isnan(xMax) or np.isinf(xMax)):
            xMin, xMax = self._get_safe_x_range(xMin, xMax)

            pw.view_box.setXRange(xMin, xMax, padding=DEFAULT_PADDING_VAL_X)
            padding_xVal = DEFAULT_PADDING_VAL_X
            limits_xMin = xMin - padding_xVal * (xMax - xMin)
            limits_xMax = xMax + padding_xVal * (xMax - xMin)
            self._set_x_limits_with_min_range(limits_xMin, limits_xMax)

        pw.view_box.setYRange(0, 1, padding=DEFAULT_PADDING_VAL_Y)
        self._set_vline_bounds([None, None])

        pw.xMin = xMin
        pw.xMax = xMax
        pw.y_name = ""
        pw.y_format = ""
        pw._clear_cursor_items(hide_only=False)
        self._safe_clear_plot_items()
        pw.axis_y.setLabel(text="")
        pw.update_left_header("channel name")
        pw.update_right_header("")

        pw.curve = None
        pw.original_index_x = None
        pw.original_y = None

    def _recalc_max_point_density(self):
        """重新计算最大点密度"""
        self._axis_manager._recalc_max_point_density()

    def _get_safe_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        """获取安全的X轴范围"""
        return self._axis_manager._get_safe_x_range(min_x, max_x)

    def _set_safe_y_range(self, min_y: float, max_y: float, set_limits: bool = True):
        """设置安全的Y轴范围"""
        self._axis_manager._set_safe_y_range(min_y, max_y, set_limits)

    def _set_x_limits_with_min_range(
        self, limits_xMin: float | None, limits_xMax: float | None
    ):
        """设置X轴限制"""
        self._axis_manager._set_x_limits_with_min_range(limits_xMin, limits_xMax)

    def _set_vline_bounds(self, bounds: list):
        """设置光标线边界"""
        self._axis_manager._set_vline_bounds(bounds)

    def _update_vline_bounds_from_data(self):
        """从数据更新光标线边界"""
        pw = self.pw
        updated = False
        if pw.is_multi_curve_mode and pw.curves:
            all_x = []
            for ci in pw.curves.values():
                if ci.x_data is not None:
                    all_x.extend(ci.x_data)
            if all_x:
                self._set_vline_bounds([min(all_x), max(all_x)])
                updated = True
        elif pw.original_index_x is not None:
            x_values = pw.offset + pw.factor * pw.original_index_x
            self._set_vline_bounds([np.min(x_values), np.max(x_values)])
            updated = True
        if not updated:
            self._set_vline_bounds([None, None])
