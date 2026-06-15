"""
MultiCurveManager - 多曲线绘图管理

负责 DraggableGraphicsLayoutWidget 的多曲线绘图、样式和可见性管理功能：
- 多曲线添加和移除
- 曲线样式管理
- 可见性切换
- 图例管理

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.widgets.plot_data_manager import PlotDataManager


class MultiCurveManager:
    """负责多曲线绘图和样式管理"""

    def __init__(self, plot_data_manager: PlotDataManager):
        if plot_data_manager is None:
            raise ValueError(
                "MultiCurveManager requires a valid PlotDataManager instance"
            )
        self._data_manager = plot_data_manager

    @property
    def pw(self) -> Any:
        return self._data_manager.pw

    def update_multi_curve_mode(self):
        """更新多曲线模式状态"""
        pw = self.pw
        curve_count = len(pw.curves)

        if not hasattr(pw, "_batch_adding"):
            pw._batch_adding = False

        if not pw._batch_adding:
            pw.is_multi_curve_mode = curve_count > 1

        if pw.is_multi_curve_mode:
            self.update_legend()
        else:
            if curve_count == 1:
                var_name = list(pw.curves.keys())[0]
                full_title = f"{var_name} ({pw.units.get(var_name, '')})".strip()
                pw.update_left_header(full_title)
            else:
                pw.update_left_header("channel name")
                pw.update_right_header("")

    def update_legend(self):
        """更新图例显示

        在多曲线模式下，在左上角显示所有曲线的图例。
        图例样式：
        - 可见曲线：实心方块(■) + 曲线颜色 + 变量名(单位)
        - 隐藏曲线：空心方块(□) + 半透明颜色 + 灰色文字

        点击图例中的曲线名可以切换该曲线的显示/隐藏状态。
        """
        pw = self.pw

        if not pw.is_multi_curve_mode:
            return

        legend_items = []
        for var_name, ci in pw.curves.items():
            color = ci.color
            unit = pw.units.get(var_name, "")
            legend_text = f"{var_name} ({unit})" if unit else var_name

            if ci.visible:
                legend_items.append(
                    f"<span style='color: {color}; font-weight: bold;'>■</span> {legend_text}"
                )
            else:
                legend_items.append(
                    f"<span style='color: {color}; opacity: 0.5;'>□</span>"
                    f" <span style='color: gray;'>{legend_text}</span>"
                )

        if legend_items:
            legend_text = " | ".join(legend_items)
            pw.update_left_header(legend_text)
        else:
            pw.update_left_header("channel name")

    def toggle_curve_visibility_by_name(self, var_name: str):
        """通过变量名切换曲线可见性

        点击图例中的曲线名时调用，切换该曲线的显示/隐藏状态。
        如果曲线对象失效（不在scene中），会尝试重新创建。

        Args:
            var_name: 要切换可见性的变量名
        """
        pw = self.pw

        if var_name not in pw.curves:
            return

        ci = pw.curves[var_name]
        ci.visible = not ci.visible
        new_visible = ci.visible

        if ci.curve is not None:
            curve_obj = ci.curve
            try:
                if curve_obj.scene() is not None:
                    curve_obj.setVisible(new_visible)
                else:
                    self._recreate_curve(var_name)
            except Exception:
                self._recreate_curve(var_name)

        self.update_legend()

        if pw.is_multi_curve_mode:
            self._update_axes_for_multi_curve(update_x_range=False)

        if pw.vline.isVisible():
            pw.update_cursor_label()

    def _recreate_curve(self, var_name: str):
        """重新创建失效的曲线"""
        pw = self.pw
        try:
            if var_name in pw.curves:
                ci = pw.curves[var_name]
                pw.add_variable_to_plot(
                    var_name,
                    skip_existence_check=True,
                    preferred_color=ci.color,
                )
        except Exception:
            pass

    def _collect_visible_curve_arrays(self, key: str) -> list:
        """收集可见曲线的数据数组"""
        import numpy as np

        pw = self.pw
        arrays = []
        if not getattr(pw, "curves", None):
            return arrays
        for ci in pw.curves.values():
            if not ci.visible:
                continue
            data = getattr(ci, key, None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.size == 0:
                continue
            arrays.append(arr)
        return arrays

    def _collect_visible_curve_pairs(self) -> list:
        """收集可见曲线的 x-y 数据对"""
        import numpy as np

        pw = self.pw
        pairs = []
        if not getattr(pw, "curves", None):
            return pairs
        for ci in pw.curves.values():
            if not ci.visible:
                continue
            x_data = ci.x_data
            y_data = ci.y_data
            if x_data is None or y_data is None:
                continue
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            if x_arr.size == 0 or y_arr.size == 0:
                continue
            pairs.append((x_arr, y_arr))
        return pairs

    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple:
        """获取曲线 X 轴限制

        Args:
            curves_filter: "visible" — 仅可见曲线；"all" — 所有曲线（含隐藏）

        Returns:
            (min_x, max_x) 或 (None, None) 当无数据时
        """
        import numpy as np

        pw = self.pw
        mins = []
        maxs = []

        if pw.curves:
            for ci in pw.curves.values():
                if curves_filter == "visible" and not ci.visible:
                    continue
                mins.append(ci.x_min)
                maxs.append(ci.x_max)
        elif pw.curve and pw.y_name:
            if pw.original_index_x is not None:
                x_data = pw.offset + pw.factor * pw.original_index_x
            else:
                x_data, _ = pw.curve.getData()
                if x_data is None:
                    return (None, None)
            mins.append(float(np.min(x_data)))
            maxs.append(float(np.max(x_data)))

        if not mins:
            return (None, None)
        return (min(mins), max(maxs))

    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        """为多曲线更新坐标轴范围

        计算所有可见曲线的数据范围，并更新坐标轴显示范围。
        只考虑visible=True的曲线，忽略隐藏的曲线。

        Args:
            update_x_range: 是否更新X轴范围。默认为False，保持当前x轴范围不变。
        """
        import numpy as np

        pw = self.pw

        if not pw.curves:
            return

        pairs = self._collect_visible_curve_pairs()
        if not pairs:
            return
        x_values = np.concatenate([p[0] for p in pairs])
        y_values = np.concatenate([p[1] for p in pairs])
        if x_values.size == 0 or y_values.size == 0:
            return

        if update_x_range:
            pw._setup_plot_axes(x_values, y_values, update_x_range=True)
        else:
            all_data_min_y = np.nanmin(y_values)
            all_data_max_y = np.nanmax(y_values)
            self._data_manager._set_safe_y_range(
                all_data_min_y, all_data_max_y, set_limits=True
            )

            special_limits = self._data_manager.handle_single_point_limits(
                x_values, y_values
            )
            if special_limits:
                min_x, max_x, min_y, max_y = special_limits
                self._data_manager._set_safe_y_range(min_y, max_y, set_limits=False)
            else:
                current_x_range = pw.view_box.viewRange()[0]
                x_min, x_max = current_x_range

                all_y_in_range = []
                for x_arr, y_arr in pairs:
                    min_y, max_y = pw._get_y_range_in_x_window(
                        x_arr, y_arr, x_min, x_max
                    )
                    all_y_in_range.extend([min_y, max_y])

                if all_y_in_range:
                    final_min_y = np.nanmin(all_y_in_range)
                    final_max_y = np.nanmax(all_y_in_range)
                    self._data_manager._set_safe_y_range(
                        final_min_y, final_max_y, set_limits=False
                    )

    def _on_legend_clicked(self, event):
        """Legend点击事件处理

        使用QTextDocument进行精确的hitTest，定位用户点击的是哪条曲线，
        然后切换该曲线的显示/隐藏状态。

        Args:
            event: 鼠标点击事件
        """
        pw = self.pw

        if not pw.is_multi_curve_mode:
            return

        pos = event.pos()
        click_x = pos.x()

        if not pw.curves:
            return

        curve_list = list(pw.curves.items())
        if not curve_list:
            return

        from PySide6.QtGui import QTextDocument
        from PySide6.QtCore import QPointF, Qt

        legend_parts = []
        for var_name, ci in curve_list:
            color = ci.color
            unit = pw.units.get(var_name, "")
            legend_text = f"{var_name} ({unit})" if unit else var_name

            if ci.visible:
                legend_parts.append(
                    f"<span style='color: {color}; font-weight: bold;'>■</span> {legend_text}"
                )
            else:
                legend_parts.append(
                    f"<span style='color: {color}; opacity: 0.5;'>□</span>"
                    f" <span style='color: gray;'>{legend_text}</span>"
                )

        full_html = " | ".join(legend_parts)

        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(pw.label_left.font())
        doc.setHtml(full_html)

        layout = doc.documentLayout()
        hit_pos = layout.hitTest(QPointF(click_x, pos.y()), Qt.HitTestAccuracy.ExactHit)

        clicked_index = -1
        char_pos = 0
        item_ranges = []

        for i, part in enumerate(legend_parts):
            if i > 0:
                char_pos += 3

            part_start = char_pos
            part_doc = QTextDocument()
            part_doc.setHtml(part)
            part_len = len(part_doc.toPlainText())
            part_end = part_start + part_len

            item_ranges.append(
                {
                    "index": i,
                    "start": part_start,
                    "end": part_end,
                    "var_name": curve_list[i][0],
                }
            )

            if part_start <= hit_pos < part_end:
                clicked_index = i
                break

            char_pos = part_end

        if clicked_index == -1:
            if hit_pos < 0:
                total_text_width = doc.size().width()
                if click_x < total_text_width / 2:
                    clicked_index = 0
                else:
                    clicked_index = len(curve_list) - 1
            else:
                min_distance = float("inf")
                for item in item_ranges:
                    if hit_pos < item["start"]:
                        distance = item["start"] - hit_pos
                    elif hit_pos >= item["end"]:
                        distance = hit_pos - item["end"]
                    else:
                        distance = 0

                    if distance < min_distance:
                        min_distance = distance
                        clicked_index = item["index"]

        clicked_index = max(0, min(clicked_index, len(curve_list) - 1))

        target_name, _ = curve_list[clicked_index]
        self.toggle_curve_visibility_by_name(target_name)

    def _apply_plot_style(self, show_symbols: bool = False):
        """应用绘图样式"""
        pw = self.pw

        for ci in pw.curves.values():
            if ci.curve is not None:
                if show_symbols:
                    ci.curve.setSymbol("o")
                else:
                    ci.curve.setSymbol(None)
