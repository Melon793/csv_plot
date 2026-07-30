"""MainWindow 光标同步与图表同步管理器"""

from __future__ import annotations
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QMessageBox

from src.core.config import (
    DEFAULT_LINE_WIDTH,
    DEFAULT_PADDING_VAL_X,
    RATIO_RESET_PLOTS,
    MIN_INDEX_LENGTH,
    safe_qt_op,
)
from src.core.logger import get_logger
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.ui.table_dialog import DataTableDialog

logger = get_logger(__name__)


class CursorSyncManager(MainWindowBaseManager):

    def filter_variables(self):
        if self.mw.var_names is None:
            return

        name_text = self.mw.filter_input.text().lower()
        unit_text = self.mw.unit_filter_input.text().lower()
        name_keywords = name_text.split() if name_text else []
        unit_keywords = unit_text.split() if unit_text else []

        # 无过滤条件：显示全部行
        if not name_keywords and not unit_keywords:
            self.mw.list_widget.show_all_rows()
            return

        # 有关键词：通过隐藏行来过滤
        self.mw.list_widget.hide_non_matching(name_keywords, unit_keywords, self.mw.units)

    def reset_plots_after_loading(
        self, index_xMin, index_xMax, *, reason: str | None = None
    ):
        try:
            for container in self.mw.plot_widgets:
                container.plot_widget.clear_plot_item()
                container.plot_widget.reset_plot(index_xMin, index_xMax)
                container.plot_widget.clear_value_cache()
                container.plot_widget.reset_pin_state()

            self.mw.cursor_mode = "1 free cursor"
            self.mw.pinned_x_values = []
            self.mw.saved_mark_range = None
            if self.mw.mark_stats_window:
                self.mw.mark_stats_window.hide()
                self.mw.mark_stats_window.tree.clear()

            if self.mw.mark_region_btn.isChecked():
                self.mw.mark_region_btn.setChecked(False)
                self.mw.layout_manager.toggle_mark_region(False)

        finally:
            for container in self.mw.plot_widgets:
                widget = container.plot_widget
                try:
                    has_data = (widget.curve is not None) or (
                        widget.is_multi_curve_mode and widget.curves
                    )
                    if has_data:
                        widget._queue_ui_refresh(immediate=True, stats=False)
                except Exception:
                    logger.debug("刷新 cursor UI 失败", exc_info=True)

    def _get_cursor_source_plot(self, source_plot=None):
        if source_plot is not None and hasattr(source_plot, "view_box"):
            return source_plot
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget is not None and container.isVisible():
                return widget
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget is not None:
                return widget
        return None

    def _get_cursor_view_range(self, source_plot=None):
        plot = self._get_cursor_source_plot(source_plot)
        if plot is None or not hasattr(plot, "view_box"):
            return None, None
        try:
            view_min, view_max = plot.view_box.viewRange()[0]
            return view_min, view_max
        except Exception as e:
            logger.debug("获取视图范围失败: %s", e)
            return None, None

    @staticmethod
    def _clamp_value(value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def _calc_second_cursor_position(self, pinned_x, view_min, view_max):
        if view_min is None or view_max is None:
            return pinned_x
        if view_min > view_max:
            view_min, view_max = view_max, view_min
        clamped = self._clamp_value(pinned_x, view_min, view_max)
        threshold = view_min + 0.6 * (view_max - view_min)
        if clamped <= threshold:
            return clamped + (view_max - clamped) / 2
        return view_min + (clamped - view_min) / 2

    def _select_farthest_cursor_index(self, context_x):
        if not self.mw.pinned_x_values:
            return None
        if context_x is None:
            return len(self.mw.pinned_x_values) - 1
        distances = [abs(x - context_x) for x in self.mw.pinned_x_values]
        return int(np.argmax(distances))

    def _apply_cursor_mode_to_plots(self):
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget is None:
                continue
            widget.apply_cursor_mode(self.mw.cursor_mode, self.mw.pinned_x_values)

    def set_cursor_mode(self, mode, *, source_plot=None, context_x=None):
        if mode == "off":
            if self.mw.cursor_btn.isChecked():
                self.toggle_cursor_all(False)
            return
        if mode not in ("1 free cursor", "1 anchored cursor", "2 anchored cursor"):
            return
        if not hasattr(self.mw, "cursor_btn") or not self.mw.cursor_btn.isChecked():
            self.toggle_cursor_all(True)

        self.mw.last_valid_cursor_mode = mode

        prev_mode = getattr(self.mw, "cursor_mode", "1 free cursor")
        view_min, view_max = self._get_cursor_view_range(source_plot)

        if mode == "1 free cursor":
            self.mw.cursor_mode = mode
            self.mw.pinned_x_values = []
        elif mode == "1 anchored cursor":
            if prev_mode == "2 anchored cursor":
                remove_idx = self._select_farthest_cursor_index(context_x)
                if remove_idx is not None:
                    remaining = [
                        x
                        for idx, x in enumerate(self.mw.pinned_x_values)
                        if idx != remove_idx
                    ]
                    self.mw.pinned_x_values = remaining[:1]
            if not self.mw.pinned_x_values:
                pinned = context_x
                if pinned is None and source_plot is not None and hasattr(source_plot, "vline"):
                    pinned = source_plot.vline.value()
                if pinned is not None:
                    self.mw.pinned_x_values = [pinned]
            self.mw.cursor_mode = mode
        elif mode == "2 anchored cursor":
            if prev_mode == "1 free cursor" or not self.mw.pinned_x_values:
                pinned = context_x
                if (
                    pinned is None
                    and source_plot is not None
                    and hasattr(source_plot, "vline")
                ):
                    pinned = source_plot.vline.value()
                if pinned is not None:
                    second = self._calc_second_cursor_position(
                        pinned, view_min, view_max
                    )
                    self.mw.pinned_x_values = [pinned, second]
            elif prev_mode == "1 anchored cursor":
                pinned = self.mw.pinned_x_values[0] if self.mw.pinned_x_values else None
                if (
                    pinned is None
                    and source_plot is not None
                    and hasattr(source_plot, "vline")
                ):
                    pinned = source_plot.vline.value()
                if pinned is not None:
                    second = self._calc_second_cursor_position(
                        pinned, view_min, view_max
                    )
                    self.mw.pinned_x_values = [pinned, second]
            else:
                if len(self.mw.pinned_x_values) == 1:
                    second = self._calc_second_cursor_position(
                        self.mw.pinned_x_values[0], view_min, view_max
                    )
                    self.mw.pinned_x_values = [self.mw.pinned_x_values[0], second]
            self.mw.cursor_mode = mode

        self._apply_cursor_mode_to_plots()
        for container in getattr(self.mw, "plot_widgets", []):
            widget = getattr(container, "plot_widget", None)
            if widget is not None:
                widget._last_cursor_update_time = 0
                widget.update_cursor_label()

    def set_cursor_enabled(self, enabled: bool) -> None:
        if self.mw.cursor_btn:
            self.mw.cursor_btn.setChecked(enabled)

    def is_cursor_enabled(self) -> bool:
        if self.mw.cursor_btn:
            return self.mw.cursor_btn.isChecked()
        return False

    def toggle_cursor_all(self, checked):
        if not self.mw.plot_widgets:
            return
        for container in self.mw.plot_widgets:
            widget = container.plot_widget
            if checked and self.mw.cursor_values_hidden:
                widget.toggle_cursor(False, hide_values_only=True)
            else:
                widget.toggle_cursor(checked)
        if checked:
            self.mw.cursor_mode = "1 free cursor"
            self.mw.pinned_x_values = []
            self._apply_cursor_mode_to_plots()
        else:
            self.mw.cursor_mode = "1 free cursor"
            self.mw.pinned_x_values = []
        self.mw.cursor_btn.setChecked(checked)
        self.mw.cursor_btn.setText("隐藏光标" if checked else "显示光标")

    def _realign_pinned_cursor_after_time_correction(
        self, old_factor, old_offset, new_factor, new_offset
    ):
        if not self.mw.plot_widgets:
            return

        if getattr(self.mw, "cursor_mode", "1 free cursor") == "1 free cursor":
            return

        pinned_indices = list(
            getattr(self.mw, "_time_correction_pinned_index_values", []) or []
        )
        if not pinned_indices:
            pinned_values = list(getattr(self.mw, "pinned_x_values", []) or [])
            if not pinned_values:
                return
            if old_factor == 0:
                return
            for pinned_value in pinned_values:
                if pinned_value is None or not np.isfinite(pinned_value):
                    continue
                index_pos = (pinned_value - old_offset) / old_factor
                if np.isfinite(index_pos):
                    pinned_indices.append(index_pos)
        if not pinned_indices:
            return

        datalength = 0
        if hasattr(self.mw, "loader") and self.mw.loader is not None:
            datalength = max(int(self.mw.loader.datalength), 0)
        elif self.mw.plot_widgets[0].plot_widget.original_index_x is not None:
            datalength = len(self.mw.plot_widgets[0].plot_widget.original_index_x)

        new_display_values = []
        for index_pos in pinned_indices:
            if index_pos is None or not np.isfinite(index_pos):
                continue
            if datalength > 0:
                index_pos = min(max(index_pos, 1), datalength)
            new_display_x = new_offset + new_factor * index_pos
            if np.isfinite(new_display_x):
                new_display_values.append(new_display_x)

        if not new_display_values:
            return

        self.mw.pinned_x_values = new_display_values
        self.mw.pinned_index_values = list(pinned_indices)

        for container in self.mw.plot_widgets:
            widget = container.plot_widget

            if (
                hasattr(widget, "original_index_x")
                and widget.original_index_x is not None
                and len(widget.original_index_x) > 0
            ):
                min_index = np.min(widget.original_index_x)
                max_index = np.max(widget.original_index_x)
                new_min_x = widget.offset + widget.factor * min_index
                new_max_x = widget.offset + widget.factor * max_index
            elif widget.is_multi_curve_mode and widget.curves:
                first_curve_info = next(iter(widget.curves.values()), None)
                if first_curve_info is not None and first_curve_info.y_data is not None:
                    data_len = len(first_curve_info.y_data)
                    new_min_x = widget.offset + widget.factor * 1
                    new_max_x = widget.offset + widget.factor * data_len
                else:
                    new_min_x = widget.offset + widget.factor * 1
                    new_max_x = widget.offset + widget.factor * datalength
            else:
                new_min_x = widget.offset + widget.factor * 1
                new_max_x = widget.offset + widget.factor * datalength

            if hasattr(widget, "_set_vline_bounds"):
                widget._set_vline_bounds([new_min_x, new_max_x])
            else:
                widget.vline.setBounds([new_min_x, new_max_x])

            widget.apply_cursor_mode(self.mw.cursor_mode, new_display_values)
            if hasattr(widget.view_box, "is_cursor_pinned"):
                widget.view_box.is_cursor_pinned = True
            if hasattr(widget, "_last_cursor_update_time"):
                widget._last_cursor_update_time = 0
            widget.update_cursor_label()

    def sync_crosshair(self, x, sender_widget):
        if not self.mw.cursor_btn.isChecked():
            return
        if getattr(self.mw, "cursor_mode", "1 free cursor") != "1 free cursor":
            return
        if getattr(self.mw, "_is_loading_new_data", False):
            return
        if self.mw._is_syncing_crosshair:
            return

        if sender_widget and getattr(sender_widget, "_is_interacting", False):
            return

        if self.mw._pending_crosshair_x is not None:
            if abs(x - self.mw._pending_crosshair_x) < 0.0001:
                return

        self.mw._is_syncing_crosshair = True
        try:
            has_pinned_plot = any(
                c.plot_widget.is_cursor_pinned
                for c in self.mw.plot_widgets
                if c.isVisible() and hasattr(c.plot_widget, "is_cursor_pinned")
            )

            if has_pinned_plot:
                return

            for container in self.mw.plot_widgets:
                if not container.isVisible():
                    continue
                w = container.plot_widget
                if getattr(w, "_is_interacting", False):
                    continue
                if getattr(w, "_is_updating_data", False):
                    continue
                w.vline.setVisible(True)
                with QSignalBlocker(w.vline):
                    w.vline.setPos(x)

            self.mw._pending_crosshair_x = x
            if not self.mw._crosshair_update_timer.isActive():
                self.mw._crosshair_update_timer.start(16)

        finally:
            self.mw._is_syncing_crosshair = False

    def _flush_crosshair_updates(self):
        if self.mw._is_loading_new_data:
            self.mw._pending_crosshair_x = None
            return

        self.mw._pending_crosshair_x = None

        for container in self.mw.plot_widgets:
            if not container.isVisible():
                continue
            w = container.plot_widget
            if getattr(w, "_is_interacting", False):
                continue
            if getattr(w, "_is_updating_data", False):
                continue
            safe_qt_op(w.update_cursor_label)

    def reset_all_pin_states(self):
        self.mw.cursor_mode = "1 free cursor"
        self.mw.pinned_x_values = []
        for container in self.mw.plot_widgets:
            container.plot_widget.reset_pin_state()

    def clear_all_plots(self):
        for container in self.mw.plot_widgets:
            widget = container.plot_widget
            widget.clear_plot_item()
            widget.reset_pin_state()
        self.mw.saved_mark_range = None
        self.mw.layout_manager.request_mark_stats_refresh(immediate=True)

    def collect_global_x_range(
        self, curves_filter: str = "visible"
    ) -> tuple[float | None, float | None]:
        all_mins: list[float] = []
        all_maxs: list[float] = []

        for container in self.mw.plot_widgets:
            if not container.isVisible():
                continue
            x_min, x_max = container.plot_widget.get_curve_x_limits(curves_filter)
            if x_min is not None and x_max is not None:
                all_mins.append(x_min)
                all_maxs.append(x_max)

        if not all_mins:
            if self.mw.loader and hasattr(self.mw.loader, "global_time_range"):
                fallback = self.mw.loader.global_time_range
                logger.debug("[GLOBAL_X] collect_global_x_range: no visible plot data, fallback=time_range %s", fallback)
                return fallback
            elif self.mw.loader and self.mw.loader.datalength > 0:
                fallback = (1.0, float(self.mw.loader.datalength))
                logger.debug("[GLOBAL_X] collect_global_x_range: no visible plot data, fallback=datalength %s", fallback)
                return fallback
            logger.debug("[GLOBAL_X] collect_global_x_range: no data available")
            return (None, None)

        result = (min(all_mins), max(all_maxs))

        if result[0] == result[1]:
            expand_val = 0.5
            result = (result[0] - expand_val, result[1] + expand_val)

        logger.debug(
            "[GLOBAL_X] collect_global_x_range: filter=%s result=(%.4f, %.4f) "
            "from %d visible plots",
            curves_filter, result[0], result[1], len(all_mins),
        )
        return result

    def _compute_baseline_density(self):
        if not self.mw.loader or self.mw.loader.datalength == 0:
            self.mw._baseline_density = 0.0
            return

        if hasattr(self.mw.loader, "global_time_range"):
            t_min, t_max = self.mw.loader.global_time_range
        else:
            t_min, t_max = 1.0, float(self.mw.loader.datalength)

        span = t_max - t_min
        if span > 0:
            self.mw._baseline_density = float(self.mw.loader.datalength) / span
        else:
            self.mw._baseline_density = 0.0

    def _sync_min_xrange(self):

        new_max = max(
            (
                container.plot_widget._max_point_density
                for container in self.mw.plot_widgets
                if container.isVisible()
                and container.plot_widget._max_point_density > 0
            ),
            default=0.0,
        )

        if new_max == 0.0:
            new_max = self.mw._baseline_density

        if new_max != self.mw._global_max_density and new_max > 0:
            prev = self.mw._global_max_density
            self.mw._global_max_density = new_max
            min_range = MIN_INDEX_LENGTH / new_max
            logger.debug(
                "[MIN_XRANGE] _sync_min_xrange: density %.4f -> %.4f, min_range=%.4f, "
                "syncing to %d visible plots",
                prev, new_max, min_range,
                sum(1 for c in self.mw.plot_widgets if c.isVisible()),
            )
            for container in self.mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget._set_min_x_range(min_range)
        else:
            logger.debug(
                "[MIN_XRANGE] _sync_min_xrange: no change, density=%.4f unchanged",
                self.mw._global_max_density,
            )

    def auto_range_all_plots(self):
        if not self.mw.loader or self.mw.loader.datalength == 0:
            return

        global_min_x, global_max_x = self.collect_global_x_range(
            curves_filter="visible"
        )

        for container in self.mw.plot_widgets:
            if container.isVisible():
                container.plot_widget.auto_range(
                    external_xmin=global_min_x,
                    external_xmax=global_max_x,
                )

    def auto_y_in_x_range(self):
        for container in self.mw.plot_widgets:
            widget = container.plot_widget
            widget.auto_y_in_x_range()

    def replots_after_loading(self, skip_pin_reset: bool = False):
        for container in self.mw.plot_widgets:
            container.plot_widget._is_updating_data = True
            if hasattr(container.plot_widget, "_cancel_ui_refresh"):
                container.plot_widget._cancel_ui_refresh()

        try:
            if self.mw.loader.datalength == 0:
                return

            # v5.11: reload 场景下跳过 pin 状态重置，因为 _restore_cursor_state_after_reload
            # 已经恢复了正确的 cursor_mode 和 pinned_x_values
            if not skip_pin_reset:
                self.reset_all_pin_states()

            all_y_names = []
            for container in self.mw.plot_widgets:
                widget = container.plot_widget
                if widget.y_name:
                    all_y_names.append(widget.y_name)
                if widget.is_multi_curve_mode and widget.curves:
                    all_y_names.extend(widget.curves.keys())

            if DataTableDialog._instance is not None:
                all_y_names.extend(DataTableDialog._instance._df.columns.tolist())

            is_mdf = getattr(self.mw.loader, "LOADER_TYPE", "") == "mdf"

            unique_y_names = set(all_y_names)
            skip_var_restore = False

            if not unique_y_names:
                if is_mdf:
                    x_min, x_max = self.mw.loader.global_time_range
                else:
                    x_min, x_max = 1, self.mw.loader.datalength
                self.reset_plots_after_loading(x_min, x_max, reason="no tracked curves")
                skip_var_restore = True

            cleared = []

            if not skip_var_restore:
                var_names_set = set(self.mw.loader.var_names)
                in_var_names = [y for y in unique_y_names if y in var_names_set]

                if in_var_names:
                    # 仅排除已知无效(INVALID=-1)的变量，保留 UNKNOWN(-2) 的变量。
                    # MDF 使用懒加载(lazy loading)不会在加载时扫描全部数据，
                    # 其 df_validity 统一返回 UNKNOWN(-2)；若使用 >=0 会将所有
                    # MDF 变量误判为无效，导致重载后已绘变量被错误清除。
                    # CSV 加载器会主动扫描数据并返回 VALID(1)/CONST(0)/INVALID(-1)，
                    # 对 CSV 场景 != -1 等价于旧逻辑 >=0，行为不变。
                    # 注：后期 MDF 有效性检查将改为用户主动触发。
                    validity_values = [
                        self.mw.loader.df_validity.get(y, -1) for y in in_var_names
                    ]
                    validity_array = np.array(validity_values)
                    valid_mask = validity_array != -1
                    found = [in_var_names[i] for i in np.where(valid_mask)[0]]
                else:
                    found = []

                ratio = len(found) / len(unique_y_names) if unique_y_names else 0

                if ratio <= RATIO_RESET_PLOTS or len(found) < 1:
                    if is_mdf:
                        x_min, x_max = self.mw.loader.global_time_range
                    else:
                        x_min, x_max = 1, self.mw.loader.datalength
                    self.reset_plots_after_loading(
                        x_min, x_max, reason="insufficient valid vars"
                    )
                else:
                    self.mw.value_cache = {}
                    for idx, container in enumerate(self.mw.plot_widgets):
                        widget = container.plot_widget

                        if is_mdf:
                            x_min, x_max = self.mw.loader.global_time_range
                            min_x = widget.offset + widget.factor * x_min
                            max_x = widget.offset + widget.factor * x_max
                        else:
                            original_index_x = np.arange(
                                1, self.mw.loader.datalength + 1, dtype=np.float32
                            )
                            min_x = widget.offset + widget.factor * np.min(original_index_x)
                            max_x = widget.offset + widget.factor * np.max(original_index_x)
                        min_x, max_x = widget._get_safe_x_range(min_x, max_x)
                        limits_xMin = min_x - DEFAULT_PADDING_VAL_X * (max_x - min_x)
                        limits_xMax = max_x + DEFAULT_PADDING_VAL_X * (max_x - min_x)
                        widget._set_x_limits_with_min_range(limits_xMin, limits_xMax)
                        if hasattr(widget, "_set_vline_bounds"):
                            widget._set_vline_bounds([min_x, max_x])
                        else:
                            widget.vline.setBounds([min_x, max_x])

                        if widget.is_multi_curve_mode:
                            current_curves = dict(widget.curves)

                            widget.curves.clear()
                            widget.is_multi_curve_mode = False
                            widget.current_color_index = 0

                            widget._clear_cursor_items(hide_only=False)
                            widget._safe_clear_plot_items()
                            widget.curve = None
                            widget.y_name = ""
                            widget.original_index_x = None
                            widget.original_y = None

                            curves_added = 0
                            visibility_to_restore = {}

                            for var_name, ci in current_curves.items():
                                var_exists = (
                                    (var_name in self.mw.loader.var_names)
                                    if is_mdf
                                    else (var_name in self.mw.loader.df.columns)
                                )
                                if (
                                    var_exists
                                    and self.mw.loader.df_validity.get(var_name, -1) != -1
                                ):
                                    preferred_color = ci.color
                                    success = widget.add_variable_to_plot(
                                        var_name,
                                        skip_existence_check=True,
                                        preferred_color=preferred_color,
                                    )
                                    if success:
                                        curves_added += 1
                                        visibility_to_restore[var_name] = ci.visible

                            widget.update_multi_curve_mode()

                            # 修复：仅剩1条曲线时，转换为规范化单曲线状态，
                            # 避免 curves 字典有数据但 curve/original_y 为空的混合状态
                            if curves_added == 1:
                                single_var = next(iter(widget.curves.keys()))
                                saved_color = widget.curves[single_var].color
                                widget.plot_variable(single_var)
                                # 恢复原有颜色（plot_variable 会重置为蓝色）
                                try:
                                    if widget.curve is not None and hasattr(widget.curve, 'opts'):
                                        pen = pg.mkPen(color=saved_color, width=DEFAULT_LINE_WIDTH)
                                        widget.curve.setPen(pen)
                                except Exception:
                                    logger.debug("重载归一化后恢复曲线颜色失败", exc_info=True)
                            elif curves_added > 1:
                                # 原有的 visibility 恢复逻辑（仅多曲线时执行）
                                for var_name, original_visible in visibility_to_restore.items():
                                    if var_name in widget.curves:
                                        widget.curves[var_name].visible = original_visible
                                        if widget.curves[var_name].curve is not None:
                                            try:
                                                widget.curves[var_name].curve.setVisible(
                                                    original_visible
                                                )
                                            except Exception:
                                                logger.debug("恢复曲线可见性失败", exc_info=True)
                                widget.update_legend()

                            if curves_added == 0:
                                cleared.append((idx + 1, "所有变量无效"))
                        else:
                            y_name = widget.y_name
                            if not y_name:
                                continue
                            var_exists = (
                                (y_name in self.mw.loader.var_names)
                                if is_mdf
                                else (y_name in self.mw.loader.df.columns)
                            )
                            if (
                                var_exists
                                and self.mw.loader.df_validity.get(y_name, -1) != -1
                            ):
                                success = widget.plot_variable(y_name)
                                if not success:
                                    widget.clear_plot_item()
                                    cleared.append((idx + 1, "无效数据"))
                            else:
                                widget.clear_plot_item()
                                reason = (
                                    f"未找到变量:{y_name}"
                                    if not var_exists
                                    else f"无效数据:{y_name}"
                                )
                                cleared.append((idx + 1, reason))

            if self.mw.plot_widgets:
                first_plot = self.mw.plot_widgets[0].plot_widget
                curr_min, curr_max = first_plot.view_box.viewRange()[0]
                first_plot.view_box.setXRange(curr_min, curr_max, padding=0)

        finally:
            for container in self.mw.plot_widgets:
                container.plot_widget._is_updating_data = False

            for container in self.mw.plot_widgets:
                widget = container.plot_widget
                try:
                    if hasattr(widget, "view_box") and hasattr(widget, "plot_item"):
                        has_data = (widget.curve is not None) or (
                            widget.is_multi_curve_mode and widget.curves
                        )
                        if has_data:
                            widget._queue_ui_refresh(immediate=True, stats=False)
                except Exception:
                    logger.debug("刷新 cursor UI 失败", exc_info=True)

        if cleared:
            # 安全防护：数据重载期间严禁调用 processEvents()，
            # 否则会泵出 paint 事件导致 QGraphicsView 在 items 重建期间崩溃 (SIGSEGV)
            if not getattr(self.mw, '_is_loading_new_data', False):
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
            msg = "以下图表被清除：\n"
            for plot_idx, reason in cleared:
                msg += f"Plot {plot_idx}: {reason}\n"
            QMessageBox.information(self.mw, "更新通知", msg)
